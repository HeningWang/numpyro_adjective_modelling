"""Reproduce the two-order sweep with paired scenes and scene-level MC error."""
from __future__ import annotations
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import jax
import jax.numpy as jnp
import core_rsa

SEED = 20260906
NOBJ = list(range(2, 31, 4))
SPREAD = [2., 7.75, 15.]
COLOR = [.90, .92, .94, .96, .98]
K = [.1, .3, .5, .7, .9]
WF = [.1, .2, .3, .5, .8]
GRID = np.array(np.meshgrid(COLOR, K, WF)).reshape(3, -1).T
KEYS = [(s, a) for s in ["static", "recursive"] for a in ["global_speaker", "incremental_speaker"]]


def main(out, batch):
    assert jax.default_backend() == "gpu" and jax.config.x64_enabled
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, main_rows, marginal = [], [], []
    start = time.monotonic()
    checks = []
    for nobj in NOBJ:
        functions = []
        for semantics, speaker in KEYS:
            key = speaker + ("_static" if semantics == "static" else "")
            def one(scene, params, key=key):
                p = core_rsa.pragmatic_listener(scene, 1., 0., params[0], None,
                    params[2], params[1], key, 2)
                return p[:, 0]
            functions.append(jax.jit(jax.vmap(jax.vmap(one, in_axes=(0, None)), in_axes=(None, 0))))
        for spread in SPREAD:
            sizes = truncnorm.rvs((1-15.5)/spread, (30-15.5)/spread,
                loc=15.5, scale=spread, size=(1000, nobj), random_state=rng)
            scenes = np.stack([sizes, rng.binomial(1, .5, (1000, nobj)),
                               rng.binomial(1, .5, (1000, nobj))], axis=-1)
            maxima = np.argmax(sizes, axis=1)
            scenes[np.arange(1000), 0, 0], scenes[np.arange(1000), maxima, 0] = (
                sizes[np.arange(1000), maxima].copy(), sizes[:, 0].copy())
            scenes[:, 0, 1:] = 1
            all_results = []
            for (semantics, speaker), fn in zip(KEYS, functions):
                result = np.concatenate([np.asarray(fn(jnp.asarray(scenes[b:b+batch]), jnp.asarray(GRID)))
                                         for b in range(0, 1000, batch)], axis=1)
                assert result.shape == (125, 1000, 2) and np.isfinite(result).all()
                assert result.min() >= 0 and result.max() <= 1
                advantage = result[:, :, 0] - result[:, :, 1]
                if semantics == "static" and speaker == "global_speaker":
                    assert np.max(np.abs(advantage)) < 1e-10
                all_results.append(result)
                for params, values, adv in zip(GRID, result, advantage):
                    rows.append(dict(semantics=semantics, speaker=speaker, nobj=nobj, sd_spread=spread,
                        color_semvalue=params[0], k=params[1], wf=params[2], n_scenes=1000,
                        mean_big_blue=float(values[:, 0].mean()), mean_blue_big=float(values[:, 1].mean()),
                        mean_advantage=float(adv.mean()), mcse_advantage=float(adv.std(ddof=1)/np.sqrt(1000))))
                scene_average = advantage.mean(axis=0)
                main_rows.append(dict(semantics=semantics, speaker=speaker, nobj=nobj, sd_spread=spread,
                    mean_advantage=float(scene_average.mean()), mcse_advantage=float(scene_average.std(ddof=1)/np.sqrt(1000)),
                    n_scenes=1000, parameter_combinations=125))
                for j, name in enumerate(["color_semvalue", "k", "wf"]):
                    for value in np.unique(GRID[:, j]):
                        by_scene = result[GRID[:, j] == value].mean(axis=0)
                        for code, order in enumerate(["big blue", "blue big"]):
                            marginal.append(dict(semantics=semantics, speaker=speaker, parameter=name, value=value,
                                nobj=nobj, sd_spread=spread, order=order, mean=float(by_scene[:, code].mean()),
                                variance_of_mean=float(by_scene[:, code].var(ddof=1)/1000)))
            np.savez_compressed(out/f"scenes_n{nobj}_sd{spread:g}.npz", scenes=scenes, results=np.array(all_results), grid=GRID)
            checks.append(dict(nobj=nobj, sd_spread=spread, complete=True))
            pd.DataFrame(rows).to_csv(out/"simulation_cell_summary.csv", index=False)
            pd.DataFrame(main_rows).to_csv(out/"simulation_size_first_advantage_summary.csv", index=False)
            print(json.dumps(dict(nobj=nobj, spread=spread, seconds=time.monotonic()-start)), flush=True)
    marginal = pd.DataFrame(marginal)
    aggregate = marginal.groupby(["semantics", "speaker", "parameter", "value", "order"]).agg(
        mean=("mean", "mean"), sum_variance=("variance_of_mean", "sum"), scene_groups=("mean", "size")).reset_index()
    aggregate["mcse"] = np.sqrt(aggregate.sum_variance) / aggregate.scene_groups
    aggregate.to_csv(out/"simulation_parameter_summary.csv", index=False)
    manifest = dict(complete=True, seed=SEED, scenes_per_context=1000, independent_scene_groups=24,
        evaluated_rows=len(rows)*1000, backend=jax.default_backend(), x64=jax.config.x64_enabled,
        jax=jax.__version__, seconds=time.monotonic()-start, paired_across_models_and_parameters=True,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(core_rsa.__file__)]},
        checks=checks)
    (out/"simulation_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=50)
    a = p.parse_args()
    main(a.output, a.batch)
