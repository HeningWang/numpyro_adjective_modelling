"""Compare observed-minus-predicted proportions from frozen posterior means.

Run from the repository root. Inputs are the archived condition predictions of
the participant-weight plan-guided semantic pair (K-HKO and K-UPD-HKO), exported
by semantic_diagnostics.py. These descriptive residuals use the full-data
posterior means; the separate semantic_factor_contrasts.csv contains response-LOO
scores. No fitting, posterior sampling, or uncertainty estimation is performed.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper/data"
KEYS = ("combination", "relevant_property", "sharpness", "response")


def main():
    groups = defaultdict(dict)
    totals = defaultdict(lambda: [0.0, 0.0])
    with (DATA / "semantic_condition_predictions.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[k] for k in KEYS)
            assert row["model"] not in groups[key], "Duplicate prediction cell"
            groups[key][row["model"]] = row
            total = totals[(row["model"], *key[:3])]
            total[0] += float(row["response_mean"])
            total[1] += float(row["observed_response"])
    assert len(groups) == 270 and len(totals) == 36
    assert all(abs(value - 1) < 1e-10 for pair in totals.values() for value in pair)
    rows = []
    for key, pair in sorted(groups.items()):
        fixed, updating = pair["K-HKO"], pair["K-UPD-HKO"]
        assert fixed["n"] == updating["n"]
        assert fixed["observed_response"] == updating["observed_response"]
        observed = float(fixed["observed_response"])
        f, u = float(fixed["response_mean"]), float(updating["response_mean"])
        rows.append(dict(zip(KEYS, key), n=int(fixed["n"]),
                         observed_percent=100 * observed,
                         fixed_predicted_percent=100 * f,
                         updating_predicted_percent=100 * u,
                         fixed_residual_pp=100 * (observed - f),
                         updating_residual_pp=100 * (observed - u),
                         fixed_absolute_error_advantage_pp=100 * (abs(observed-u)-abs(observed-f))))
    assert sum(r["n"] for r in rows) / 15 == 9100
    with (DATA / "semantic_response_residuals.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    # Select by absolute change, retaining improvements for either semantic model.
    selected = sorted(rows, key=lambda r: abs(r["fixed_absolute_error_advantage_pp"]), reverse=True)[:6]
    print("Six largest changes in absolute residual error (percentage points):")
    for row in selected:
        print(tuple(row[k] for k in KEYS),
              *(round(row[k], 1) for k in ("observed_percent", "fixed_residual_pp", "updating_residual_pp")))
    for model in ("fixed", "updating"):
        residuals = [r[f"{model}_residual_pp"] for r in rows]
        print(model, "270-cell MAE", sum(map(abs, residuals))/len(rows),
              "RMSE", math.sqrt(sum(r*r for r in residuals)/len(rows)))


if __name__ == "__main__":
    main()
