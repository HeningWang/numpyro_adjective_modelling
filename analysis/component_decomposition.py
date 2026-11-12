"""Factor the response LOO of any fit, including ablations, into length, set, and order.

Extends semantic_diagnostics.py to the stable-order and informativeness
ablations: the same PSIS-LOO predictive distribution is split by the chain rule
into response length, adjective set given length, and order given set.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from numpyro import handlers

from semantic_diagnostics import FACTORS, LABELS, LENGTH, SETS, decompose, run
import tuned_model as tuned  # resolved through the models/production path set by run


def load_inputs(encoded: Path, recursive: bool, spec=None,
                colour_semval: float = .59) -> tuple[dict, np.ndarray]:
    with np.load(encoded) as z:
        d = {k: jnp.asarray(v) for k, v in z.items()}
    kwargs = dict(states=d["states_train"], empirical=d["empirical_seq_flat"],
        participant_idx=d["participant_idx"], n_participants=int(d["n_participants"]),
        sufficient_dim=d["sufficient_dim"], has_one_word_solution=d["has_one_word_solution"],
        is_sharp=d["sharpness_idx"], is_colour_sufficient=d["is_colour_sufficient"])
    if spec is None:
        kwargs["precomputed_features"] = run.ms.precompute_principled_discovery_features(
            kwargs["states"], kwargs["is_sharp"], recursive=recursive)
    else:
        kwargs["precomputed_features"] = tuned.precompute_tuned_features(
            kwargs["states"], kwargs["is_sharp"], recursive=recursive, spec=spec,
            color_semval=colour_semval)
    return kwargs, np.asarray(d["empirical_seq_flat"])


def loo_predictive(model, kwargs: dict, artifact: Path, n_rows: int,
                   batch_size: int) -> tuple[np.ndarray, np.ndarray, float]:
    """PSIS-weighted predictive distribution over the 15 responses, per trial."""
    trace = handlers.trace(handlers.seed(model, 0)).get_trace(**kwargs)
    stochastic = [k for k, v in trace.items() if v["type"] == "sample" and not v["is_observed"]]
    with xr.open_dataset(artifact, group="posterior", engine="h5netcdf") as ds:
        posterior = ds.load()
    n = posterior.sizes["chain"] * posterior.sizes["draw"]
    flat = {k: jnp.asarray(posterior[k].values.reshape(n, *posterior[k].shape[2:]))
            for k in stochastic}
    with xr.open_dataset(artifact, group="log_likelihood", engine="h5netcdf") as ds:
        ll = ds.obs.values.reshape(n, n_rows).copy()
    # Match ArviZ loo's relative-efficiency calculation, including deterministics.
    ess = az.ess(posterior, method="mean")
    reff = np.concatenate([v.values.ravel() for v in ess.data_vars.values()]).mean() / n
    log_weights, pareto = az.psislw(-ll.T, reff=reff)
    weights = np.exp(log_weights.T)

    def predict(values):
        tr = handlers.trace(handlers.condition(model, data=values)).get_trace(**kwargs)
        return tr["obs"]["fn"].probs
    predict_batch = jax.jit(jax.vmap(predict))
    obs = np.asarray(kwargs["empirical"])
    loo = np.zeros((n_rows, 15))
    max_replay = 0.0
    for begin in range(0, n, batch_size):
        end = min(begin + batch_size, n)
        p = np.asarray(predict_batch({k: v[begin:end] for k, v in flat.items()}))
        replay = np.log(p[:, np.arange(n_rows), obs])
        max_replay = max(max_replay, float(np.max(np.abs(replay - ll[begin:end]))))
        loo += np.einsum("sn,snu->nu", weights[begin:end], p)
    if max_replay >= 1e-8:
        raise ValueError(f"{artifact.name}: likelihood replay error {max_replay}")
    np.testing.assert_allclose(loo.sum(axis=1), 1, atol=1e-10)
    return loo, np.asarray(pareto), max_replay


def analyse(label: str, model_name: str, artifact: Path, encoded: Path, output: Path,
            data: pd.DataFrame, *, drop_order_term: bool, drop_informativeness: bool,
            drop_prefix_salience: bool, batch_size: int,
            tuned_fields: tuple[str, ...] = (), colour_semval: float = .59) -> None:
    start = time.monotonic()
    spec = tuned.TuningSpec(**{k: True for k in tuned_fields}) if tuned_fields else None
    if spec is not None and (drop_order_term or drop_informativeness or drop_prefix_salience):
        raise ValueError("Tuned specs carry their own switches")
    model = (tuned.make_tuned_model(model_name, spec) if spec is not None else
             run.model_for_run(model_name, drop_order_term=drop_order_term,
                               drop_informativeness=drop_informativeness,
                               drop_prefix_salience=drop_prefix_salience))
    if spec is None and colour_semval != .59:
        raise ValueError("A colour semantic value other than .59 requires a tuned spec")
    kwargs, obs = load_inputs(encoded, recursive="UPD" in model_name, spec=spec,
                              colour_semval=colour_semval)
    np.testing.assert_array_equal(LABELS[obs], data.annotation)
    loo, pareto, max_replay = loo_predictive(model, kwargs, artifact, len(data), batch_size)
    factors = decompose(loo, obs)
    reference = az.loo(az.from_netcdf(artifact), pointwise=True)
    response_error = float(np.max(np.abs(factors[:, 0] - np.asarray(reference.loo_i))))
    if response_error >= 1e-6:
        raise ValueError(f"{label}: response LOO differs from ArviZ by {response_error}")
    frame = data[["canonical_row_position", "id", "combination", "annotation"]].copy()
    frame[FACTORS] = factors
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"{label}_pointwise_factors.csv", index=False)
    receipt = dict(label=label, model=model_name, artifact=str(artifact),
                   drop_order_term=drop_order_term, drop_informativeness=drop_informativeness,
                   drop_prefix_salience=drop_prefix_salience, tuned=list(tuned_fields),
                   max_replay=max_replay, max_pareto_k=float(pareto.max()),
                   response_reconstruction_error=response_error,
                   seconds=time.monotonic() - start)
    (output / f"{label}_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--model", choices=run.MODELS, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--encoded", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--drop-order-term", action="store_true")
    p.add_argument("--drop-informativeness", action="store_true")
    p.add_argument("--drop-prefix-salience", action="store_true")
    p.add_argument("--tuned", nargs="*", default=[],
                   help="Boolean TuningSpec fields of a tuned posterior")
    p.add_argument("--colour-reliability", type=float, default=.59)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    assert jax.default_backend() == "gpu" and jax.config.x64_enabled
    analyse(args.label, args.model, args.artifact, args.encoded, args.output,
            pd.read_csv(run.DATA), drop_order_term=args.drop_order_term,
            drop_informativeness=args.drop_informativeness,
            drop_prefix_salience=args.drop_prefix_salience, batch_size=args.batch,
            tuned_fields=tuple(args.tuned), colour_semval=args.colour_reliability)
