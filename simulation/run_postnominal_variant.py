"""Postnominal variant of the controlled random-scene simulation.

Regenerates the 24 preserved scene sets from the sweep seed, then evaluates the
size-first pragmatic-listener advantage with composition running either against
the surface order (prenominal, as in the paper) or with it (postnominal: the
noun-adjacent adjective is also the first adjective). Prefix evaluation always
follows the surface order. Order, salience, length and selection terms are zero.

Usage: python simulation/run_postnominal_variant.py --out <dir> [--scenes 1000]
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "true"

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/production"))
sys.path.insert(0, str(ROOT / "simulation"))
import modelSpecification as ms  # noqa: E402
import run_controlled_random_scenes as rc  # noqa: E402
import run_reproducible_sweep as sweep  # noqa: E402

RULES = [("terminal", None), ("prefix_k0", 0.0), ("prefix_k05", 0.5), ("prefix_k1", 1.0)]
CONFIGS = [
    (support, rule, kappa)
    for support in ["two", "four", "fifteen"]
    for rule, kappa in RULES
    if support == "fifteen" or rule in ("terminal", "prefix_k0")
]
F0 = np.array([[0.59, 0.5, 0.6856]])


def literal_all(scene: jnp.ndarray, params: jnp.ndarray, recursive: bool, postnominal: bool) -> jnp.ndarray:
    """Copy of rc.literal_all with a switch for the direction of composition."""
    colour, k, wf = params
    prior = jnp.ones(scene.shape[0], dtype=jnp.float64) / scene.shape[0]
    sizes = scene[:, 0]
    idx = jnp.argsort(sizes)

    def size(post: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(ms.compute_size_semantics_fast_presorted(sizes, idx, sizes[idx], post, k, wf), 1e-8)

    fixed = size(prior)
    color = jnp.where(scene[:, 1] == 1, colour, 1 - colour) + 1e-8
    form = jnp.full_like(color, 0.5 + 1e-8)

    def interpret(tokens: jnp.ndarray) -> jnp.ndarray:
        def step(post, token):
            m = jnp.stack([size(post) if recursive else fixed, color, form])[jnp.maximum(token, 0)]
            updated = post * m
            updated = updated / updated.sum()
            return jnp.where(token >= 0, updated, post), None

        # Prenominal: noun-adjacent adjective is the last token, so compose right to left.
        # Postnominal: noun-adjacent adjective is the first token, so compose left to right.
        order = tokens if postnominal else tokens[::-1]
        return jax.lax.scan(step, prior, order)[0]

    return jax.vmap(interpret)(ms.utterance_list)


def evaluate(scene: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Size-first advantage (DC minus CD) for every direction x semantics x config."""
    out = []
    for postnominal in (False, True):
        for recursive in (False, True):
            literal = literal_all(scene, params, recursive, postnominal)
            for support, rule, kappa in CONFIGS:
                probs = rc.structural_probs(literal, support, rule, kappa if kappa is not None else 0.0, 0.0)
                listener = rc.metric(probs, [1, 6])
                out.append(listener[0] - listener[1])
    return jnp.stack(out)


def regenerate_scenes() -> list[tuple[int, float, np.ndarray]]:
    rng = np.random.default_rng(sweep.SEED)
    groups = []
    for nobj in sweep.NOBJ:
        for spread in sweep.SPREAD:
            sizes = truncnorm.rvs((1 - 15.5) / spread, (30 - 15.5) / spread, loc=15.5, scale=spread,
                                  size=(1000, nobj), random_state=rng)
            scenes = np.stack([sizes, rng.binomial(1, 0.5, (1000, nobj)), rng.binomial(1, 0.5, (1000, nobj))], axis=-1)
            maxima = np.argmax(sizes, axis=1)
            scenes[np.arange(1000), 0, 0], scenes[np.arange(1000), maxima, 0] = (
                sizes[np.arange(1000), maxima].copy(), sizes[:, 0].copy())
            scenes[:, 0, 1:] = 1
            groups.append((nobj, spread, scenes))
    return groups


def labels() -> list[dict]:
    return [dict(direction=d, semantics=s, support=sup, rule=r)
            for d in ("prenominal", "postnominal") for s in ("fixed", "updating") for sup, r, _ in CONFIGS]


def main(n_scenes: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fn = jax.jit(jax.vmap(jax.vmap(evaluate, in_axes=(0, None)), in_axes=(None, 0)))
    rows = []
    start = time.monotonic()
    for nobj, spread, scenes in regenerate_scenes():
        sub = jnp.asarray(scenes[:n_scenes])
        for name, params in (("original_grid", sweep.GRID), ("fitted_constants", F0)):
            res = np.asarray(fn(sub, jnp.asarray(params)))  # params x scenes x configs
            by_scene = res.mean(axis=0)
            for j, lab in enumerate(labels()):
                y = by_scene[:, j]
                rows.append(dict(**lab, parameter_set=name, nobj=nobj, spread=spread, mean=float(y.mean()),
                                 mcse=float(y.std(ddof=1) / np.sqrt(len(y))), scenes=len(y)))
        print(f"n={nobj} sd={spread} t={time.monotonic() - start:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / "postnominal_context_summary.csv", index=False)
    frame = pd.DataFrame(rows)
    frame["variance"] = frame.mcse ** 2
    keys = ["parameter_set", "direction", "semantics", "support", "rule"]
    agg = frame.groupby(keys).agg(mean=("mean", "mean"), variance=("variance", "sum"), groups=("mean", "size")).reset_index()
    agg["mcse"] = np.sqrt(agg.variance) / agg.groups
    agg.drop(columns="variance").to_csv(out / "postnominal_overall.csv", index=False)
    print(agg.to_string(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.scenes, a.out)
