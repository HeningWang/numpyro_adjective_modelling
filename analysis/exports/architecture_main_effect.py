"""Export the architecture main effect for Panel A of the architecture figure.

Each architecture's pointwise PSIS-LOO is averaged over the two semantic regimes (and also taken within each regime)
and contrasted with plan-guided production, using the paired participant
Bayesian bootstrap from ``robustness_utils``. No model is refit here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robustness_utils import DATA, ROOT, bootstrap_totals, load_rows  # noqa: E402

POINTWISE = ROOT / "tmp/robustness/inputs/corrected_inputs"
FIGURE_DATA = ROOT / "paper/data/production_architecture_figure_data.csv"
PANEL = "Architecture main effect"
ARCHITECTURES = {"Global": "G", "Plan-guided": "K", "Fully incremental": "I"}
REGIMES = {"Context-fixed": "-HO", "Sequential context updating": "-UPD-HO"}


def main() -> None:
    models = [f"{code}{suffix}" for code in ARCHITECTURES.values() for suffix in ("-HO", "-UPD-HO")]
    rows = load_rows(DATA, POINTWISE, models)
    scores = {"Average": {code: (rows[f"{code}-HO"] + rows[f"{code}-UPD-HO"]) / 2 for code in ARCHITECTURES.values()}}
    # Within-regime contrasts, drawn alongside the main effect as in the semantic panel.
    for regime, suffix in REGIMES.items():
        scores[regime] = {code: rows[f"{code}{suffix}"] for code in ARCHITECTURES.values()}
    records = []
    for regime, by_code in scores.items():
        for label, code in ARCHITECTURES.items():
            if code == "K":
                records.append(dict(panel=PANEL, label=label, semantic_regime=regime, estimate=0.0, lower=0.0, upper=0.0))
                continue
            totals = (by_code[code] - by_code["K"]).groupby(rows["id"]).sum().to_numpy()
            summary = bootstrap_totals(totals)
            records.append(dict(
                panel=PANEL, label=label, semantic_regime=regime, estimate=summary["estimate_elpd"],
                lower=summary["credible_lower_95"], upper=summary["credible_upper_95"],
            ))
    figure_data = pd.read_csv(FIGURE_DATA)
    figure_data = figure_data[figure_data["panel"] != PANEL]
    pd.concat([figure_data, pd.DataFrame(records)]).to_csv(FIGURE_DATA, index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
