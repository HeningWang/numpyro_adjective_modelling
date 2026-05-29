import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from analysis.posterior_utils import (
    compute_ppc_correlation_metrics,
    compute_residual_tables,
)


def test_compute_ppc_correlation_metrics_returns_multiple_scopes():
    df = pd.DataFrame({
        "model": ["m"] * 4,
        "human_mean": [0.50, 0.25, 0.00, 0.10],
        "model_mean": [0.45, 0.20, 0.05, 0.15],
    })

    out = compute_ppc_correlation_metrics(df)

    assert set(out["scope"]) == {"all_cells", "nonzero_cells"}
    assert set(["r", "r2", "mae", "rmse", "slope", "intercept"]).issubset(out.columns)


def test_compute_residual_tables_summarizes_worst_cells_by_condition():
    df = pd.DataFrame({
        "model": ["m", "m", "m"],
        "relevant_property": ["first", "first", "second"],
        "sharpness": ["blurred", "blurred", "sharp"],
        "utterance_code": [0, 1, 0],
        "utterance_label": ["D", "DC", "D"],
        "human_mean": [0.50, 0.20, 0.10],
        "human_lo": [0.40, 0.10, 0.05],
        "human_hi": [0.60, 0.30, 0.15],
        "model_mean": [0.35, 0.24, 0.18],
        "model_lo": [0.30, 0.20, 0.12],
        "model_hi": [0.40, 0.28, 0.23],
    })

    by_cell, summary, worst = compute_residual_tables(df, top_n=1)

    assert "signed_residual" in by_cell.columns
    first = summary.query("relevant_property == 'first' and sharpness == 'blurred'").iloc[0]
    assert first["worst_utterance_label"] == "D"
    assert first["max_abs_residual"] == 0.15
    assert len(worst) == 2


if __name__ == "__main__":
    test_compute_ppc_correlation_metrics_returns_multiple_scopes()
    test_compute_residual_tables_summarizes_worst_cells_by_condition()
    print("PASS posterior_utils simplified diagnostics tests")
