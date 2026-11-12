"""Compare production fits with informative form semantics (nu_F) against nu_F = .50.

For every model and nu_F the pointwise PSIS-LOO is computed from the stored
log-likelihood.  Differences are summed per adjective pair and overall, with a
participant Dirichlet(1, ..., 1) bootstrap scaled to the number of participants,
as in the other production contrasts.  Run from models/production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

MODELS = ["G-HO", "K-HO", "I-HO", "G-UPD-HO", "K-UPD-HO", "I-UPD-HO"]
PAIRS = {"dc": "size-colour", "df": "size-form", "cf": "colour-form"}


def pointwise(artifact: Path) -> tuple[np.ndarray, float]:
    idata = az.from_netcdf(artifact)
    loo = az.loo(idata, pointwise=True)
    return np.asarray(loo.loo_i), float(np.max(loo.pareto_k))


def bootstrap(diff: np.ndarray, participant: np.ndarray, draws: int, seed: int) -> dict:
    ids, inv = np.unique(participant, return_inverse=True)
    per = np.bincount(inv, weights=diff, minlength=len(ids))
    w = np.random.default_rng(seed).dirichlet(np.ones(len(ids)), size=draws) * len(ids)
    totals = w @ per
    return dict(estimate=float(per.sum()), lower=float(np.quantile(totals, .025)),
                upper=float(np.quantile(totals, .975)), p_positive=float((totals > 0).mean()))


def main(args: argparse.Namespace) -> None:
    data = pd.read_csv(args.data)
    if len(data) != 9100:
        raise ValueError("Expected the 9,100 modelled production rows")
    participant = data["id"].to_numpy()
    pair = data["conditions"].astype(str).str[-2:].to_numpy()
    loo, rows, diag = {}, [], []
    for m in MODELS:
        loo[(m, .50)], k = pointwise(args.primary / f"{m}.nc")
        diag.append(dict(model=m, nu_F=.50, max_pareto_k=k))
        for nu in args.nu:
            artifact = args.fits / f"nuF{nu:.2f}" / f"{m}.nc"
            loo[(m, nu)], k = pointwise(artifact)
            diag.append(dict(model=m, nu_F=nu, max_pareto_k=k, elpd=float(loo[(m, nu)].sum())))
    for m in MODELS:
        for nu in args.nu:
            diff = loo[(m, nu)] - loo[(m, .50)]
            subsets = {"all": np.ones(len(diff), bool), **{v: pair == k for k, v in PAIRS.items()}}
            for name, mask in subsets.items():
                rows.append(dict(contrast=f"{m}: nu_F {nu:.2f} - 0.50", subset=name,
                                 **bootstrap(np.where(mask, diff, 0), participant,
                                             args.draws, args.seed)))
    for nu in [.50, *args.nu]:  # architecture and semantics contrasts within each nu_F
        for a, b in [("K-HO", "G-HO"), ("K-HO", "I-HO"), ("K-UPD-HO", "K-HO"),
                     ("G-UPD-HO", "G-HO"), ("I-UPD-HO", "I-HO")]:
            rows.append(dict(contrast=f"nu_F {nu:.2f}: {a} - {b}", subset="all",
                             **bootstrap(loo[(a, nu)] - loo[(b, nu)], participant,
                                         args.draws, args.seed)))
    out = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output / "elpd_contrasts.csv", index=False)
    pd.DataFrame(diag).to_csv(args.output / "diagnostics.csv", index=False)
    print(out.round(1).to_string(index=False))
    print(json.dumps(diag))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--fits", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--nu", type=float, nargs="+", default=[.59, .75])
    p.add_argument("--draws", type=int, default=200000)
    p.add_argument("--seed", type=int, default=20260906)
    main(p.parse_args())
