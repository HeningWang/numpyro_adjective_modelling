"""Listener size-first preference with the stable order term and salience.

Extends the controlled simulation (run_controlled_random_scenes.py) on the same
preserved random scenes. Every speaker model is crossed with the stable order
term (off / on) and salience (off / on), under the semantic grid and at the
production constants, so that the effect of sequential context updating can be
read with and without the convention and the availability preference.

Salience follows the production model: a fixed property baseline plus the
display contrast of each property, centred across properties. In the pragmatic
listener, each object is in turn treated as the intended referent, so the
display contrast is computed with that object as the target.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import run_controlled_random_scenes as base

ms = base.ms
ARCHITECTURES = (("global", 0.0), ("intermediate", 0.5), ("incremental", 1.0))
SIZE_FIRST, COLOUR_FIRST = 1, 6  # DC and CD in the fifteen-utterance list


@dataclass(frozen=True)
class Weights:
    """Weights of the auxiliary components relative to informativeness (alpha = 1)."""

    beta: float
    salience: float
    is_sharp: float


def referent_salience(scene: jnp.ndarray, is_sharp: float) -> jnp.ndarray:
    """Salience scores (size, colour, form) with each object as the target."""
    n = scene.shape[0]

    def one(o: jnp.ndarray) -> jnp.ndarray:
        index = jnp.arange(n)
        order = jnp.argsort(jnp.where(index == o, -1, index))  # o first, others in place
        return ms._visual_salience_scores(scene[order], is_sharp).astype(jnp.float64)

    return jax.vmap(one)(jnp.arange(n))


def path_parts(log_target: jnp.ndarray, salience: jnp.ndarray, support: str):
    """Summed prefix utility and local normalisers, with salience in each step."""
    scores = log_target[base.PFX] + salience[None, None, :]
    scores = jnp.where(base.MASKS[support], scores, -jnp.inf)
    inert = jnp.full_like(scores, -jnp.inf).at[:, :, 0].set(0.0)
    scores = jnp.where(base.ACTIVE[:, :, None], scores, inert)
    chosen = jnp.sum(jnp.where(base.ONEHOT > 0, scores, 0.0), axis=-1)
    norms = jax.scipy.special.logsumexp(scores, axis=-1)
    chosen = jnp.where(base.ACTIVE, chosen, 0.0)
    norms = jnp.where(base.ACTIVE, norms, 0.0)
    probabilities = jnp.sum(jax.nn.softmax(scores, axis=-1) * base.ONEHOT, axis=-1)
    probabilities = jnp.where(base.ACTIVE, probabilities, 1.0)
    floor = jnp.where(base.ACTIVE, jnp.log(jnp.clip(probabilities, 1e-8)), 0.0) - (chosen - norms)
    return jnp.sum(chosen, axis=0), jnp.sum(norms - floor, axis=0)


def speaker_probs(literal, salience, support, kappa, beta):
    """Speaker distribution over utterances for every object as referent."""
    logs = jnp.log(jnp.clip(literal.T, 1e-8))
    raw, normalizers = jax.vmap(lambda v, s: path_parts(v, s, support))(logs, salience)
    mask = base.OUTPUT_MASKS[support]
    utility = jnp.where(mask, raw, 0.0) - kappa * jnp.where(mask, normalizers, 0.0)
    utility = utility + beta * ms.LOG_LM_ORDER_ONLY_15
    return jax.nn.softmax(jnp.where(mask, utility, -jnp.inf), axis=-1)


def configs() -> list[dict]:
    return [
        dict(semantics=sem, architecture=arch, support=support, order=order, salience=sal)
        for sem in ("fixed", "updating")
        for arch, _ in ARCHITECTURES
        for support in base.SUPPORTS
        for order in ("neutral", "stable")
        for sal in ("none", "salient")
    ]


def evaluate(scene, params, weights: Weights):
    """Listener target probability after DC and CD, and speaker P(DC | DC or CD)."""
    z = referent_salience(scene, weights.is_sharp)
    values = []
    for recursive in (False, True):
        literal = base.literal_all(scene, params, recursive)
        for _, kappa in ARCHITECTURES:
            for support in base.SUPPORTS:
                for beta in (0.0, weights.beta):
                    for lam in (0.0, weights.salience):
                        prob = speaker_probs(literal, lam * z, support, kappa, beta)
                        values.append(base.metric(prob, [SIZE_FIRST, COLOUR_FIRST]))
    return jnp.stack(values)


def check_reduction(scene: np.ndarray, params: np.ndarray) -> float:
    """With salience off, the speaker must equal the existing controlled simulation."""
    error = 0.0
    zeros = jnp.zeros((scene.shape[0], 3))
    for recursive in (False, True):
        literal = base.literal_all(jnp.asarray(scene), jnp.asarray(params), recursive)
        for (rule, kappa), (_, k2) in zip(base.RULES[1:], ARCHITECTURES):
            assert kappa == k2
            for support in base.SUPPORTS:
                for beta in (0.0, base.BETA):
                    new = speaker_probs(literal, zeros, support, kappa, beta)
                    old = base.structural_probs(literal, support, rule, kappa, beta)
                    error = max(error, float(jnp.max(jnp.abs(new - old))))
    assert error < 1e-10, error
    return error


def summarize(result, cfg, nobj, spread, parameter_set, rows, paired):
    """Scene-level means and paired contrasts (semantic settings averaged within scenes)."""
    advantage = (result[:, :, :, 0] - result[:, :, :, 1]).mean(axis=0)  # scenes x configs
    for j, c in enumerate(cfg):
        m, se = base.mean_se(advantage[:, j])
        rows.append(dict(**c, nobj=nobj, spread=spread, parameter_set=parameter_set,
                         metric="listener_advantage", mean=m, mcse=se))
    lookup = {tuple(c.values()): j for j, c in enumerate(cfg)}
    for j, c in enumerate(cfg):
        key = dict(c)
        for factor, level, reference in (("semantics", "updating", "fixed"),
                                         ("order", "stable", "neutral"),
                                         ("salience", "salient", "none")):
            if c[factor] == level:
                i = lookup[tuple({**key, factor: reference}.values())]
                m, se = base.mean_se(advantage[:, j] - advantage[:, i])
                paired.append(dict(contrast=f"{level}_minus_{reference}", **c, nobj=nobj,
                                   spread=spread, parameter_set=parameter_set, mean=m, mcse=se))


def pool(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Equal weight to the 24 scene groups, as in the controlled simulation."""
    frame = frame.assign(variance=frame.mcse ** 2)
    out = frame.groupby(keys, dropna=False).agg(
        mean=("mean", "mean"), variance=("variance", "sum"), groups=("mean", "size")).reset_index()
    out["mcse"] = np.sqrt(out.variance) / out.groups
    out["lower_mc95"] = out["mean"] - 1.96 * out.mcse
    out["upper_mc95"] = out["mean"] + 1.96 * out.mcse
    return out.drop(columns="variance")


def main(args: argparse.Namespace) -> None:
    assert jax.config.x64_enabled
    if args.require_gpu:
        assert jax.default_backend() == "gpu"
    weights = Weights(beta=args.beta, salience=args.salience, is_sharp=args.is_sharp)
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(args.scenes.glob("scenes_n*_sd*.npz"),
                   key=lambda p: (int(p.stem.split("_")[1][1:]), float(p.stem.split("_sd")[1])))
    if args.limit:
        files = files[: args.limit]
    cfg = configs()
    rows, paired, checks = [], [], []
    start = time.monotonic()
    fn = jax.jit(jax.vmap(jax.vmap(lambda s, p: evaluate(s, p, weights), in_axes=(0, None)),
                          in_axes=(None, 0)))
    for file in files:
        nobj, spread = int(file.stem.split("_")[1][1:]), float(file.stem.split("_sd")[1])
        with np.load(file) as z:
            scenes, grid = z["scenes"][: args.scenes_per_file], z["grid"]
        checks.append(dict(file=file.name, reduction_error=check_reduction(scenes[0], grid[0])))
        for parameter_set, params in (("grid", grid), ("production_constants", base.F0)):
            result = np.concatenate(
                [np.asarray(fn(jnp.asarray(scenes[b:b + args.batch]), jnp.asarray(params)))
                 for b in range(0, len(scenes), args.batch)], axis=1)
            assert result.shape == (len(params), len(scenes), len(cfg), 3)
            assert np.isfinite(result).all()
            summarize(result, cfg, nobj, spread, parameter_set, rows, paired)
        jax.clear_caches()
        print(json.dumps(dict(scene=file.name, seconds=round(time.monotonic() - start))), flush=True)
    keys = list(cfg[0]) + ["parameter_set"]
    pd.DataFrame(rows).to_csv(args.output / "context_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(args.output / "paired_contrasts.csv", index=False)
    pool(pd.DataFrame(rows), keys + ["metric"]).to_csv(args.output / "context_summary_overall.csv", index=False)
    pool(pd.DataFrame(paired), ["contrast"] + keys).to_csv(args.output / "paired_contrasts_overall.csv", index=False)
    receipt = dict(complete=True, seconds=time.monotonic() - start, backend=jax.default_backend(),
                   weights=weights.__dict__, scene_files=len(files),
                   scenes_per_file=args.scenes_per_file, checks=checks)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--beta", type=float, default=base.BETA)
    p.add_argument("--salience", type=float, required=True,
                   help="salience coefficient relative to informativeness (alpha = 1)")
    p.add_argument("--is-sharp", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=25)
    p.add_argument("--scenes-per-file", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0, help="number of scene files (0: all)")
    p.add_argument("--require-gpu", action="store_true")
    main(p.parse_args())
