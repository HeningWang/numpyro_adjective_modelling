"""Summarise cross-task predictions of slider ratings from production posteriors.

Reads the per-model cell tables written by slider_cross_task_prediction.py and reports,
for each production model:
  * r: correlation of predicted and observed cell means (36 cells, no free parameters);
  * r_within: the same after removing each adjective pair's mean (context gradient only);
  * held-out RMSE of a two-parameter logit calibration fitted on one slider study and
    evaluated on the other;
  * the predicted and observed context gradient (first- minus second-sufficient,
    logit scale) for each adjective pair.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibrated_rmse(cells: pd.DataFrame, train: str, test: str) -> float:
    fit, held = cells[cells.study == train], cells[cells.study == test]
    b, a = np.polyfit(logit(fit.pred_mean.to_numpy()), logit(fit.observed.to_numpy()), 1)
    pred = 1 / (1 + np.exp(-(a + b * logit(held.pred_mean.to_numpy()))))
    return float(np.sqrt(np.mean((pred - held.observed.to_numpy()) ** 2)))


def summarise(cells: pd.DataFrame) -> dict:
    centred = cells.assign(
        obs_c=cells.observed - cells.groupby(["study", "combination"])["observed"].transform("mean"),
        pred_c=logit(cells.pred_mean) - cells.assign(l=logit(cells.pred_mean))
        .groupby(["study", "combination"]).l.transform("mean"))
    row = dict(model=cells.model.iloc[0],
               r=np.corrcoef(cells.observed, cells.pred_mean)[0, 1],
               r_within=np.corrcoef(centred.obs_c, centred.pred_c)[0, 1],
               rmse_orig_to_repl=calibrated_rmse(cells, "original", "replication"),
               rmse_repl_to_orig=calibrated_rmse(cells, "replication", "original"))
    for pair, g in cells.groupby("combination"):
        m = g.groupby("relevant_property")[["observed", "pred_mean"]].mean()
        row[f"grad_obs_{pair}"] = logit(m.observed["first"]) - logit(m.observed["second"])
        row[f"grad_pred_{pair}"] = logit(m.pred_mean["first"]) - logit(m.pred_mean["second"])
    return row


def main(args: argparse.Namespace) -> None:
    tables = sorted(args.cells.glob("*_cells.csv"))
    if not tables:
        raise FileNotFoundError(f"No cell tables in {args.cells}")
    cells = pd.concat([pd.read_csv(t) for t in tables], ignore_index=True)
    summary = pd.DataFrame([summarise(g) for _, g in cells.groupby("model", sort=False)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    cells.to_csv(args.output.with_name("slider_cross_task_cells.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cells", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
