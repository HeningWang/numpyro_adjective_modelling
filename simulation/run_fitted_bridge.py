"""Compare speaker order and pragmatic recovery with the fitted response model.

Counterfactual referents retain the named adjective extensions and task context.
The result is a posterior-derived simulation on the size-colour displays.
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import jax
import jax.numpy as jnp
from numpyro import handlers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/production"))
import run_corrected_primary as run


def main(campaign, output, name):
    assert jax.default_backend() == "gpu" and jax.config.x64_enabled
    assert run.sha256(run.DATA) == run.DATA_SHA256
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    data = pd.read_csv(run.DATA)
    ix = np.flatnonzero(data.combination.eq("dimension_color"))
    selected_data = data.iloc[ix].reset_index(drop=True)
    with np.load(campaign / "encoded_model_input.npz") as z:
        d = dict(z)
    states = d["states_train"][ix]
    n, objects = states.shape[:2]
    counterfactual = []
    for referent in range(objects):
        order = np.arange(objects)
        order[0], order[referent] = referent, 0
        counterfactual.append(states[:, order])
    all_states = jnp.asarray(np.concatenate(counterfactual))
    def repeat(field):
        return jnp.asarray(np.tile(d[field][ix], objects))
    kwargs = dict(states=all_states, empirical=repeat("empirical_seq_flat"),
        participant_idx=repeat("participant_idx"), n_participants=int(d["n_participants"]),
        sufficient_dim=repeat("sufficient_dim"), has_one_word_solution=repeat("has_one_word_solution"),
        is_sharp=repeat("sharpness_idx"), is_colour_sufficient=repeat("is_colour_sufficient"))
    kwargs["precomputed_features"] = run.ms.precompute_principled_discovery_features(
        all_states, kwargs["is_sharp"], recursive="UPD" in name)
    model = run.MODELS[name]
    tr = handlers.trace(handlers.seed(model, 0)).get_trace(**kwargs)
    keys = [k for k, v in tr.items() if v["type"] == "sample" and not v["is_observed"]]
    artifact = campaign / ("recovery" if name in {"G-HO", "G-UPD-HO"} else "primary") / f"{name}.nc"
    manifest = json.loads(artifact.with_name(f"{name}_manifest.json").read_text())
    assert run.sha256(artifact) == manifest["artifact_sha256"]
    assert run.sha256(ROOT / "models/production/modelSpecification.py") == manifest["source_hashes"]["modelSpecification.py"]
    with xr.open_dataset(artifact, group="posterior", engine="h5netcdf") as p:
        draw_ix = np.linspace(0, p.sizes["draw"]-1, 50, dtype=int)
        posterior = {k: jnp.asarray(p[k].values[:, draw_ix].reshape(200, *p[k].shape[2:])) for k in keys}
    with xr.open_dataset(artifact, group="log_likelihood", engine="h5netcdf") as p:
        reference_ll = p.obs.values[:, draw_ix].reshape(200, -1)[:, ix]
    def predict(values):
        tr = handlers.trace(handlers.condition(model, data=values)).get_trace(**kwargs)
        dist = tr["obs"]["fn"]
        return dist.probs[:, [1, 6]], dist.log_prob(kwargs["empirical"])[:n]
    fn = jax.jit(jax.vmap(predict))
    metrics = []
    error = 0.
    for begin in range(0, 200, 8):
        probs, ll = fn({k: v[begin:begin+8] for k, v in posterior.items()})
        probs = np.asarray(probs).reshape(-1, objects, n, 2)
        error = max(error, float(np.max(np.abs(np.asarray(ll) - reference_ll[begin:begin+8]))))
        listener = probs / probs.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(listener.sum(axis=1), 1, atol=1e-12)
        target = probs[:, 0]
        order = target[:, :, 0] / target.sum(axis=2)
        target_listener = listener[:, 0]
        metrics.append(np.stack([order, target_listener[:, :, 0], target_listener[:, :, 1],
                                 target_listener[:, :, 0]-target_listener[:, :, 1]], axis=-1))
    if error >= 1e-8:
        raise ValueError(f"Original-target replay failure: {error}")
    metrics = np.concatenate(metrics)
    names = ["speaker_size_first_given_set", "listener_size_first", "listener_colour_first", "listener_advantage"]
    rows = []
    groups = [(('all', 'all'), np.arange(n)), *selected_data.groupby(["relevant_property", "sharpness"]).groups.items()]
    for condition, indices in groups:
        for j, metric in enumerate(names):
            draws = metrics[:, np.asarray(indices), j].mean(axis=1)
            lo, hi = np.quantile(draws, [.025, .975])
            rows.append(dict(model=name, relevant_property=condition[0], sharpness=condition[1],
                metric=metric, mean=float(draws.mean()), lower_95=float(lo), upper_95=float(hi),
                trials=len(indices), posterior_draws=200))
    pd.DataFrame(rows).to_csv(output / f"{name}_bridge_summary.csv", index=False)
    np.savez_compressed(output / f"{name}_bridge_draws.npz", metrics=metrics, canonical_rows=ix,
                        retained_draw_indices=draw_ix)
    receipt = dict(complete=True, model=name, original_target_replay_error=error,
        posterior_draws_per_chain=50, draw_indices=draw_ix.tolist(), trials=n, referents=objects,
        response_support=run.ms.UTTERANCE_LABELS, fixed_word_extensions=True,
        task_covariates_held_fixed=True, artifact_sha256=manifest["artifact_sha256"],
        source_sha256=run.sha256(Path(__file__)), seconds=time.monotonic()-start)
    (output / f"{name}_bridge_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", choices=run.MODELS, required=True)
    a = p.parse_args()
    main(a.campaign, a.output, a.model)
