"""Predict slider order ratings from production-fitted posteriors, without refitting.

For each slider trial the production model gives a distribution over the fifteen
ordered utterances.  The predicted order preference is the share of the canonical
two-adjective order among the two orders of the trial's adjective pair,
P(first) / (P(first) + P(reversed)).  New raters are represented by sampling
participant effects from the fitted population distribution.

Run from models/production of the code version that produced the posteriors.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from numpyro import handlers

sys.path.insert(0, str(Path.cwd()))
import modelSpecification as ms  # noqa: E402
import tuned_model as tm  # noqa: E402
from helper import SYMBOL_TO_INDEX, import_dataset, import_dataset_hier  # noqa: E402

PAIR_ORDERS = {  # canonical (rated as "first") order and its reversal
    "dimension_color": ("DC", "CD"),
    "dimension_form": ("DF", "FD"),
    "color_form": ("CF", "FC"),
}
PARTICIPANT_SITES = ("z_alpha", "z_order", "z_kappa")
CELL = ["study", "combination", "relevant_property", "sharpness"]


def utterance_index(production_csv: Path) -> dict[str, int]:
    """Map utterance strings to the categorical index used by the production fits."""
    unique = np.asarray(import_dataset_hier(file_path=production_csv)["unique_utterances"])
    symbols = {v: k for k, v in SYMBOL_TO_INDEX.items()}
    index = {"".join(symbols[s] for s in row if s >= 0): i for i, row in enumerate(unique)}
    if len(index) != 15:
        raise ValueError(f"Expected 15 utterances, found {len(index)}")
    return index


def load_slider(path: Path, study: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["prefer_first_1st"].notna() & df["combination"].isin(PAIR_ORDERS)].copy()
    df["sharpness"] = df["sharpness"].astype(str)
    if not set(df["sharpness"]) <= {"sharp", "blurred"}:
        raise ValueError(f"{study}: unexpected sharpness labels {set(df['sharpness'])}")
    df["study"] = study
    return df.reset_index(drop=True)


def encode(df: pd.DataFrame) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """Encode unique displays with the production loader; return row -> unique map."""
    keys = ([f"size_{c}" for c in "ABCDEF"] + [f"color_{c}" for c in "ABCDEF"]
            + [f"form_{c}" for c in "ABCDEF"] + ["conditions", "sharpness", "relevant_property"])
    work = df[keys].astype(str)
    unique = work.drop_duplicates().reset_index(drop=True)
    inverse = work.merge(unique.reset_index(), on=keys, how="left")["index"].to_numpy()
    if np.isnan(inverse.astype(float)).any():
        raise ValueError("Display mapping failed")
    table = unique.copy()
    for c in keys[:6]:
        table[c] = table[c].astype(float)
    table["annotation"] = "D"  # placeholder; the prediction never conditions on it
    table["item"], table["list"] = np.arange(len(table)), 0  # loader bookkeeping only
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        table.to_csv(f.name, index=False)
        enc = import_dataset(f.name)
    if len(enc["states_train"]) != len(unique):
        raise ValueError("Loader dropped displays")
    return enc, inverse.astype(int), unique


def tile_inputs(enc: dict, recursive: bool, n_virtual: int, form_semval: float = .50,
                spec: "tm.TuningSpec | None" = None) -> dict:
    """Repeat every display for each virtual participant."""
    n = len(enc["states_train"])
    rep = lambda x: jnp.tile(jnp.asarray(x), (n_virtual,) + (1,) * (jnp.ndim(x) - 1))
    # Placeholder observations keep the response site observed; only its
    # probabilities are read, so the values never enter the prediction.
    kwargs = dict(states=rep(enc["states_train"]),
                  empirical=jnp.zeros(n * n_virtual, dtype=jnp.int32),
                  participant_idx=jnp.repeat(jnp.arange(n_virtual), n),
                  n_participants=n_virtual, sufficient_dim=rep(enc["sufficient_dim"]),
                  has_one_word_solution=rep(enc["has_one_word_solution"]),
                  is_sharp=rep(enc["sharpness_idx"]),
                  is_colour_sufficient=rep(enc["is_colour_sufficient"]))
    if spec is None:
        kwargs["precomputed_features"] = ms.precompute_principled_discovery_features(
            kwargs["states"], kwargs["is_sharp"], recursive=recursive, form_semval=form_semval)
    else:
        kwargs["precomputed_features"] = tm.precompute_tuned_features(
            kwargs["states"], kwargs["is_sharp"], recursive=recursive, spec=spec,
            form_semval=form_semval)
    return kwargs


def select_model(name: str, form_semval: float | None, ablation: dict[str, bool],
                 spec: "tm.TuningSpec | None"):
    """Pick the production model that produced the posterior being replayed."""
    chosen = sum([form_semval is not None, any(ablation.values()), spec is not None])
    if chosen > 1:
        raise ValueError("Choose one of form reliability, utility ablation, or tuned spec")
    if spec is not None:
        return tm.make_tuned_model(name, spec)
    if any(ablation.values()):
        return ms.utility_ablation_model(name, **ablation)
    if form_semval is not None:
        return ms.form_reliability_model(name, form_semval)
    return {**ms.V10_ORDER_MODELS, **ms.V10_ORDER_UPD_MODELS,
            **ms.V11_JOINT_PARTICIPANT_MODELS,
            **ms.V12_UPDATING_JOINT_PARTICIPANT_MODELS}[name]


def predict_model(name: str, artifact: Path, enc: dict, n_draws: int, n_virtual: int,
                  batch: int, seed: int, form_semval: float | None = None,
                  ablation: dict[str, bool] | None = None,
                  spec: "tm.TuningSpec | None" = None) -> np.ndarray:
    """Return (draws, displays, 15) utterance probabilities averaged over raters."""
    model = select_model(name, form_semval, ablation or {}, spec)
    kwargs = tile_inputs(enc, "UPD" in name, n_virtual,
                         .50 if form_semval is None else form_semval, spec)
    trace = handlers.trace(handlers.seed(model, 0)).get_trace(**kwargs)
    stochastic = [k for k, v in trace.items()
                  if v["type"] == "sample" and not v["is_observed"] and k != "obs"]
    with xr.open_dataset(artifact, group="posterior", engine="h5netcdf") as ds:
        post = ds.load()
    total = post.sizes["chain"] * post.sizes["draw"]
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(total, size=n_draws, replace=False))
    globals_ = {k: jnp.asarray(post[k].values.reshape(total, *post[k].shape[2:])[pick])
                for k in stochastic if k not in PARTICIPANT_SITES}
    local = {k: jnp.asarray(rng.standard_normal((n_draws, n_virtual)))
             for k in stochastic if k in PARTICIPANT_SITES}
    missing = set(stochastic) - set(globals_) - set(local)
    if missing:
        raise ValueError(f"{name}: unmapped sites {missing}")

    def probs(values):
        tr = handlers.trace(handlers.condition(model, data=values)).get_trace(**kwargs)
        return tr["obs"]["fn"].probs

    run = jax.jit(jax.vmap(probs))
    n_disp = len(enc["states_train"])
    out = np.empty((n_draws, n_disp, 15))
    for b in range(0, n_draws, batch):
        e = min(b + batch, n_draws)
        p = np.asarray(run({k: v[b:e] for k, v in {**globals_, **local}.items()}))
        out[b:e] = p.reshape(e - b, n_virtual, n_disp, 15).mean(axis=1)
    np.testing.assert_allclose(out.sum(-1), 1, atol=1e-8)
    return out


def cell_predictions(probs: np.ndarray, inverse: np.ndarray, rows: pd.DataFrame,
                     uindex: dict[str, int]) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the cell table and the (draws, cells) matrix of predicted shares."""
    first = np.array([uindex[PAIR_ORDERS[c][0]] for c in rows["combination"]])
    rev = np.array([uindex[PAIR_ORDERS[c][1]] for c in rows["combination"]])
    p = probs[:, inverse, :]
    idx = np.arange(len(rows))
    pf, pr = p[:, idx, first], p[:, idx, rev]
    share = pf / (pf + pr)
    records, draws = [], []
    for key, g in rows.groupby(CELL):
        s = share[:, g.index].mean(axis=1)
        draws.append(s)
        records.append(dict(zip(CELL, key), n=len(g), observed=g["prefer_first_1st"].mean() / 100,
                            pred_mean=s.mean(), pred_lower=np.quantile(s, .025),
                            pred_upper=np.quantile(s, .975),
                            two_adj_mass=(pf + pr)[:, g.index].mean()))
    return pd.DataFrame(records), np.stack(draws, axis=1)


def main(args):
    uindex = utterance_index(args.production_csv)
    rows = pd.concat([load_slider(args.slider, "original"),
                      load_slider(args.replication, "replication")], ignore_index=True)
    enc, inverse, _ = encode(rows)
    print(json.dumps({"slider_rows": len(rows), "displays": len(enc["states_train"])}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    ablation = dict(drop_order_term=args.drop_order_term,
                    drop_informativeness=args.drop_informativeness,
                    drop_prefix_salience=args.drop_prefix_salience)
    spec = (tm.TuningSpec(**{k: True for k in args.tuned}) if args.tuned else None)
    for name in args.models:
        t0 = time.perf_counter()
        probs = predict_model(name, args.posteriors / f"{name}.nc", enc, args.draws,
                              args.virtual, args.batch, args.seed, args.form_reliability,
                              ablation, spec)
        cells, draws = cell_predictions(probs, inverse, rows, uindex)
        cells.insert(0, "model", name)
        cells.to_csv(args.output / f"{name}_cells.csv", index=False)
        np.save(args.output / f"{name}_cell_draws.npy", draws)  # columns follow the CSV rows
        print(json.dumps({"model": name, "seconds": time.perf_counter() - t0,
                          "r": float(np.corrcoef(cells.observed, cells.pred_mean)[0, 1])}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--posteriors", type=Path, required=True)
    p.add_argument("--production-csv", type=Path, required=True)
    p.add_argument("--slider", type=Path, required=True)
    p.add_argument("--replication", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--drop-order-term", action="store_true")
    p.add_argument("--drop-informativeness", action="store_true")
    p.add_argument("--drop-prefix-salience", action="store_true")
    p.add_argument("--tuned", nargs="*", default=[],
                   help="Boolean TuningSpec fields of a tuned posterior, e.g. split_salience")
    p.add_argument("--models", nargs="+", default=["G-HO", "K-HO", "I-HO", "G-UPD-HO",
                                                    "K-UPD-HO", "I-UPD-HO", "K-HKO"])
    p.add_argument("--draws", type=int, default=400)
    p.add_argument("--virtual", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--form-reliability", type=float,
                   help="nu_F of posteriors fitted with the form-reliability override")
    main(p.parse_args())
