"""Cross-task rating prediction with a linear link fitted on an independent study.

For each production model, the predicted preference for the conventional order
(logits, per rating cell) is mapped to the observed ratings by an intercept and
a slope. The link is estimated on the original rating study and tested on the
replication study, separately for each posterior draw. The script reports the
test error of each model and the reduction in test error for each contrast
(error of B minus error of A; positive values favour A).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    "G": ("G-HO", "no_contrast"), "K": ("K-HO", "no_contrast"), "I": ("I-HO", "no_contrast"),
    "G no order term": ("G-HO", "base_no_order"),
    "K no order term": ("K-HO", "base_no_order_modeA"),
    "I no order term": ("I-HO", "base_no_order"),
    "G no salience": ("G-HO", "no_sal"), "K no salience": ("K-HO", "no_sal_K"),
    "I no salience": ("I-HO", "no_sal"),
    "G no informativeness": ("G-HO", "base_no_info"),
    "K no informativeness": ("K-HO", "base_no_info"),
    "I no informativeness": ("I-HO", "base_no_info"),
    "G updating": ("G-UPD-HO", "base_upd"), "K updating": ("K-UPD-HO", "base_upd"),
    "I updating": ("I-UPD-HO", "base_upd"),
}
CONTRASTS = [(f"{a}: {name}", f"{a}{x}", f"{a}{y}") for a in "GKI" for name, x, y in (
    ("stable order term", "", " no order term"), ("salience", "", " no salience"),
    ("informativeness", "", " no informativeness"), ("updating", " updating", ""))]
CONTRASTS += [("plan-guided vs global", "K", "G"), ("plan-guided vs fully incremental", "K", "I")]


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def test_errors(slider: Path, model: str, label: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-draw test RMSE and slope, and the RMSE of predicting the training mean."""
    cells = pd.read_csv(slider / label / f"{model}_cells.csv")
    predicted = logit(np.load(slider / label / f"{model}_cell_draws.npy"))
    observed = logit(cells.observed.to_numpy())
    train = (cells.study == "original").to_numpy()
    test = ~train
    if not train.any() or not test.any():
        raise ValueError(f"{model} {label}: both studies are required")
    design = np.stack([np.ones(train.sum()), np.zeros(train.sum())], axis=1)
    errors, slopes = [], []
    for draw in predicted:
        design[:, 1] = draw[train]
        intercept, slope = np.linalg.lstsq(design, observed[train], rcond=None)[0]
        residual = intercept + slope * draw[test] - observed[test]
        errors.append(np.sqrt(np.mean(residual ** 2)))
        slopes.append(slope)
    null = float(np.sqrt(np.mean((observed[test] - observed[train].mean()) ** 2)))
    return np.asarray(errors), np.asarray(slopes), null


def interval(values: np.ndarray) -> dict[str, float]:
    lower, upper = np.quantile(values, [.025, .975])
    return dict(estimate=float(values.mean()), lower=float(lower), upper=float(upper))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slider", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors, rows = {}, []
    for name, (model, label) in MODELS.items():
        errors[name], slopes, null = test_errors(args.slider, model, label)
        rows.append(dict(kind="model", name=name, **interval(errors[name]),
                         slope=float(slopes.mean()),
                         r_squared=float(1 - (errors[name].mean() / null) ** 2), null_rmse=null))
    for name, a, b in CONTRASTS:
        rows.append(dict(kind="contrast", name=name, **interval(errors[b] - errors[a])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
