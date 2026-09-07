"""Factor response-LOO prediction into length, content, and conditional order.

All factors use weights for the same full held-out response. Their logarithms
sum to the existing response score; no new leave-out target is introduced.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
import jax
import jax.numpy as jnp
from numpyro import handlers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/production"))
import run_corrected_primary as run

LABELS = np.array(run.ms.UTTERANCE_LABELS)
LENGTH = np.array([len(u) for u in LABELS])
SETS = np.array(["".join(sorted(u)) for u in LABELS])
PAIRS = [("G-HO", "G-UPD-HO"), ("I-HO", "I-UPD-HO"),
         ("K-HO", "K-UPD-HO"), ("K-HKO", "K-UPD-HKO")]
FACTORS = ["response", "length", "set_given_length", "order_given_set"]


def decompose(p, observed):
    """Exact chain rule after integrating the predictive distribution."""
    same_length = LENGTH[None, :] == LENGTH[observed, None]
    same_set = SETS[None, :] == SETS[observed, None]
    full = p[np.arange(len(observed)), observed]
    length = (p * same_length).sum(axis=1)
    content = (p * same_set).sum(axis=1)
    result = np.stack([np.log(full), np.log(length), np.log(content / length),
                       np.log(full / content)], axis=1)
    np.testing.assert_allclose(result[:, 0], result[:, 1:].sum(axis=1), atol=1e-12)
    return result


def analyze_model(name, campaign, output, data, batch_size):
    start = time.monotonic()
    selected = "recovery" if name in {"G-HO", "G-UPD-HO"} else "primary"
    artifact = campaign / selected / f"{name}.nc"
    manifest = json.loads(artifact.with_name(f"{name}_manifest.json").read_text())
    assert run.sha256(artifact) == manifest["artifact_sha256"]
    assert run.sha256(ROOT / "models/production/modelSpecification.py") == manifest["source_hashes"]["modelSpecification.py"]
    with np.load(campaign / "encoded_model_input.npz") as z:
        d = {k: jnp.asarray(v) for k, v in z.items()}
    obs = np.asarray(d["empirical_seq_flat"])
    np.testing.assert_array_equal(LABELS[obs], data.annotation)
    kwargs = dict(states=d["states_train"], empirical=d["empirical_seq_flat"],
        participant_idx=d["participant_idx"], n_participants=int(d["n_participants"]),
        sufficient_dim=d["sufficient_dim"], has_one_word_solution=d["has_one_word_solution"],
        is_sharp=d["sharpness_idx"], is_colour_sufficient=d["is_colour_sufficient"])
    kwargs["precomputed_features"] = run.ms.precompute_principled_discovery_features(
        kwargs["states"], kwargs["is_sharp"], recursive="UPD" in name)
    model = run.MODELS[name]
    trace = handlers.trace(handlers.seed(model, 0)).get_trace(**kwargs)
    stochastic = [k for k, v in trace.items() if v["type"] == "sample" and not v["is_observed"]]
    with xr.open_dataset(artifact, group="posterior", engine="h5netcdf") as ds:
        posterior = ds.load()
    n = posterior.sizes["chain"] * posterior.sizes["draw"]
    flat = {k: jnp.asarray(posterior[k].values.reshape(n, *posterior[k].shape[2:])) for k in stochastic}
    with xr.open_dataset(artifact, group="log_likelihood", engine="h5netcdf") as ds:
        ll = ds.obs.values.reshape(n, len(data)).copy()
    # Match ArviZ loo's relative-efficiency calculation, including deterministics.
    ess = az.ess(posterior, method="mean")
    reff = np.concatenate([v.values.ravel() for v in ess.data_vars.values()]).mean() / n
    print(json.dumps(dict(model=name, stage="psis", draws=n, reff=float(reff))), flush=True)
    weights, pareto = az.psislw(-ll.T, reff=reff)
    weights = np.exp(weights.T)
    diagnostic_dir = campaign / ("recovery_diagnostics" if selected == "recovery" else "diagnostics")
    reference = pd.read_csv(diagnostic_dir / f"{name}_pointwise_loo.csv")
    np.testing.assert_allclose(pareto, reference.pareto_k, atol=1e-8)

    def predict(values):
        tr = handlers.trace(handlers.condition(model, data=values)).get_trace(**kwargs)
        return tr["obs"]["fn"].probs
    predict_batch = jax.jit(jax.vmap(predict))
    loo = np.zeros((len(data), 15))
    posterior_mean = np.zeros_like(loo)
    max_replay = 0.0
    for begin in range(0, n, batch_size):
        end = min(begin + batch_size, n)
        p = np.asarray(predict_batch({k: v[begin:end] for k, v in flat.items()}))
        replay = np.log(p[:, np.arange(len(data)), obs])
        max_replay = max(max_replay, float(np.max(np.abs(replay - ll[begin:end]))))
        loo += np.einsum("sn,snu->nu", weights[begin:end], p)
        posterior_mean += p.sum(axis=0) / n
        if begin % (batch_size * 50) == 0:
            print(json.dumps(dict(model=name, stage="prediction", done=end, total=n)), flush=True)
    if max_replay >= 1e-8:
        raise ValueError(f"Likelihood replay error {max_replay}")
    np.testing.assert_allclose(loo.sum(axis=1), 1, atol=1e-10)
    factors = decompose(loo, obs)
    np.testing.assert_allclose(factors[:, 0], reference[name], atol=1e-8)
    frame = data[["canonical_row_position", "id", "combination", "relevant_property", "sharpness", "annotation"]].copy()
    frame["length_observed"] = LENGTH[obs]
    frame["set_observed"] = SETS[obs]
    frame[FACTORS] = factors
    frame.to_csv(output / f"{name}_pointwise_factors.csv", index=False)
    np.savez_compressed(output / f"{name}_predictive_probabilities.npz", loo=loo, posterior_mean=posterior_mean)
    rows = []
    for condition, indices in data.groupby(["combination", "relevant_property", "sharpness"]).groups.items():
        idx = np.asarray(indices)
        for code, response in enumerate(LABELS):
            codes = np.flatnonzero(SETS == SETS[code])
            observed_set = np.isin(obs[idx], codes)
            mean_p = posterior_mean[idx]
            rows.append(dict(zip(["combination", "relevant_property", "sharpness"], condition),
                model=name, response=response, response_mean=float(mean_p[:, code].mean()),
                order_given_set_mean=float(mean_p[:, code].sum() / mean_p[:, codes].sum()),
                observed_response=float((obs[idx] == code).mean()),
                observed_order_given_set=float((obs[idx] == code).sum() / observed_set.sum()) if observed_set.any() else np.nan,
                n=len(idx), observed_set_n=int(observed_set.sum())))
    pd.DataFrame(rows).to_csv(output / f"{name}_condition_predictions.csv", index=False)
    receipt = dict(model=name, complete=True, artifact_sha256=manifest["artifact_sha256"],
        reff=float(reff), max_replay=max_replay, max_pareto_k=float(np.max(pareto)),
        factorization_error=float(np.max(np.abs(factors[:, 0] - factors[:, 1:].sum(axis=1)))),
        response_reconstruction_error=float(np.max(np.abs(factors[:, 0] - reference[name]))),
        seconds=time.monotonic()-start)
    (output / f"{name}_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


def summarize(output):
    rows, partitions = [], []
    for fixed, updating in PAIRS:
        a, b = [pd.read_csv(output / f"{name}_pointwise_factors.csv") for name in (fixed, updating)]
        np.testing.assert_array_equal(a.canonical_row_position, b.canonical_row_position)
        difference = b[FACTORS] - a[FACTORS]
        participant = difference.groupby(a.id, sort=True).sum().to_numpy()
        rng = np.random.default_rng(20260906)
        boot = np.concatenate([rng.dirichlet(np.ones(len(participant)), size=5000) @ participant * len(participant)
                               for _ in range(40)])
        for j, factor in enumerate(FACTORS):
            lo, hi = np.quantile(boot[:, j], [.025, .975])
            rows.append(dict(fixed=fixed, updating=updating, factor=factor,
                delta_elpd=float(difference[factor].sum()), lower_95=lo, upper_95=hi,
                prob_positive=float((boot[:, j] > 0).mean())))
        for group in ["combination", "length_observed", "annotation"]:
            for level, idx in a.groupby(group, sort=True).groups.items():
                for factor in FACTORS:
                    partitions.append(dict(fixed=fixed, updating=updating, partition=group, level=level,
                        factor=factor, n=len(idx), delta_elpd=float(difference.loc[idx, factor].sum())))
    pd.DataFrame(rows).to_csv(output / "semantic_factor_contrasts.csv", index=False)
    pd.DataFrame(partitions).to_csv(output / "semantic_loss_partitions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=run.MODELS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.output)
    else:
        assert jax.default_backend() == "gpu" and jax.config.x64_enabled
        assert run.sha256(run.DATA) == run.DATA_SHA256
        analyze_model(args.model, args.campaign, args.output, pd.read_csv(run.DATA), args.batch)
