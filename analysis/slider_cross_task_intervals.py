"""Credible intervals for the cross-task slider predictions and model contrasts.

Inputs: per-model `*_cells.csv` and `*_cell_draws.npy` from slider_cross_task_prediction.py.
For each posterior draw it computes the context gradient of each adjective pair
(logit of the mean predicted share when the first property is sufficient minus that
when the second is sufficient, pooled over studies and discriminability) and the
within-pair correlation of predicted and observed cells.  Model contrasts pair draws
by index; the posteriors are independent, so this is a difference of independent
posterior distributions.  The observed gradient gets a participant bootstrap.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6
PAIRS = ("dimension_color", "dimension_form", "color_form")
CONTRASTS = [("I-HO", "K-HO"), ("K-HO", "G-HO"), ("I-HO", "G-HO"), ("K-HKO", "K-HO"),
             ("G-UPD-HO", "G-HO"), ("K-UPD-HO", "K-HO"), ("I-UPD-HO", "I-HO")]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def gradients(cells: pd.DataFrame, values: np.ndarray) -> dict[str, np.ndarray]:
    """values: (draws, cells) or (cells,) shares aligned with the cell table."""
    values = np.atleast_2d(values)
    out = {}
    for pair in PAIRS:
        first = (cells.combination == pair) & (cells.relevant_property == "first")
        second = (cells.combination == pair) & (cells.relevant_property == "second")
        out[pair] = logit(values[:, first.to_numpy()].mean(1)) - logit(values[:, second.to_numpy()].mean(1))
    out["average"] = np.mean([out[p] for p in PAIRS], axis=0)
    return out


def within_r(cells: pd.DataFrame, draws: np.ndarray) -> np.ndarray:
    group = cells.study + "|" + cells.combination
    obs = cells.observed - cells.groupby(group)["observed"].transform("mean")
    pred = logit(draws)
    for g in group.unique():
        m = (group == g).to_numpy()
        pred[:, m] -= pred[:, m].mean(1, keepdims=True)
    o = (obs - obs.mean()).to_numpy()
    p = pred - pred.mean(1, keepdims=True)
    return (p @ o) / np.sqrt((p ** 2).sum(1) * (o ** 2).sum())


def observed_bootstrap(slider: dict[str, Path], cells: pd.DataFrame, reps: int,
                       seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    frames = []
    for study, path in slider.items():
        d = pd.read_csv(path)
        d = d[d.prefer_first_1st.notna() & d.combination.isin(PAIRS)].assign(study=study)
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    data["id"] = data.study + "|" + data.id.astype(str)
    key = ["study", "combination", "relevant_property", "sharpness"]
    sums = data.groupby(["id"] + key).prefer_first_1st.agg(["sum", "count"]).reset_index()
    order = cells[key].astype(str).agg("|".join, axis=1)
    sums["cell"] = sums[key].astype(str).agg("|".join, axis=1)
    ids = sums.id.unique()
    s_mat = sums.pivot_table(index="id", columns="cell", values="sum", fill_value=0)[order]
    n_mat = sums.pivot_table(index="id", columns="cell", values="count", fill_value=0)[order]
    studies = np.array([i.split("|")[0] for i in s_mat.index])
    boot = np.empty((reps, len(order)))
    for r in range(reps):
        w = np.zeros(len(ids))
        for st in np.unique(studies):  # resample participants within each study
            idx = np.flatnonzero(studies == st)
            w[idx] = np.bincount(rng.choice(len(idx), len(idx)), minlength=len(idx))
        boot[r] = (w @ s_mat.to_numpy()) / (w @ n_mat.to_numpy()) / 100
    return gradients(cells, boot)


def interval(x: np.ndarray) -> dict:
    return dict(mean=x.mean(), lower=np.quantile(x, .025), upper=np.quantile(x, .975),
                p_positive=(x > 0).mean())


def main(args: argparse.Namespace) -> None:
    models = [p.name.removesuffix("_cells.csv") for p in sorted(args.draws.glob("*_cells.csv"))]
    cells = {m: pd.read_csv(args.draws / f"{m}_cells.csv") for m in models}
    draws = {m: np.load(args.draws / f"{m}_cell_draws.npy") for m in models}
    ref = cells[models[0]]
    for m in models:
        if not cells[m][["study", "combination", "relevant_property", "sharpness"]].equals(
                ref[["study", "combination", "relevant_property", "sharpness"]]):
            raise ValueError(f"{m}: cell order differs")
    grads = {m: gradients(ref, draws[m]) for m in models}
    rs = {m: within_r(ref, draws[m]) for m in models}
    obs = observed_bootstrap({"original": args.slider, "replication": args.replication},
                             ref, args.reps, args.seed)
    rows = [dict(source="observed", metric=f"gradient_{k}", **interval(v)) for k, v in obs.items()]
    for m in models:
        rows += [dict(source=m, metric=f"gradient_{k}", **interval(v)) for k, v in grads[m].items()]
        rows.append(dict(source=m, metric="r_within", **interval(rs[m])))
    for a, b in [(a, b) for a, b in CONTRASTS if a in grads and b in grads]:
        rows += [dict(source=f"{a} - {b}", metric=f"gradient_{k}", **interval(grads[a][k] - grads[b][k]))
                 for k in grads[a]]
        rows.append(dict(source=f"{a} - {b}", metric="r_within", **interval(rs[a] - rs[b])))
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--draws", type=Path, required=True)
    p.add_argument("--slider", type=Path, required=True)
    p.add_argument("--replication", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261110)
    main(p.parse_args())
