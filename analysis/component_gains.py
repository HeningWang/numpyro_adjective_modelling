"""Per-factor ELPD gains of the stable order term, informativeness, and each model cell.

Reads the pointwise factors written by component_decomposition.py and reports
paired participant-bootstrap gains for response, length, set given length, and
order given set. Cell gains are relative to the global, context-fixed model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from robustness_utils import DRAWS, SEED, bootstrap_totals, fresh_output, write_readme

FACTORS = ("response", "length", "set_given_length", "order_given_set")
CELLS = ("G-HO", "G-UPD-HO", "K-HO", "K-UPD-HO", "I-HO", "I-UPD-HO")
BASELINE = "G-HO__order"


def load(directory: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(directory / f"{label}_pointwise_factors.csv")
    return frame.sort_values("canonical_row_position").reset_index(drop=True)


def gains(directory: Path, contrasts: list[tuple[str, str, str]], *, draws: int,
          seed: int) -> pd.DataFrame:
    rows = []
    for name, minuend, subtrahend in contrasts:
        a, b = load(directory, minuend), load(directory, subtrahend)
        if not a.canonical_row_position.equals(b.canonical_row_position):
            raise ValueError(f"{name}: misaligned rows")
        for factor in FACTORS:
            totals = (a[factor] - b[factor]).groupby(a.id).sum()
            rows.append(dict(contrast=name, minuend=minuend, subtrahend=subtrahend,
                             factor=factor, **bootstrap_totals(totals, draws=draws, seed=seed)))
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--factors", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=DRAWS)
    args = p.parse_args()
    contrasts = [(f"{m}: stable order term", f"{m}__order", f"{m}__no_order") for m in CELLS]
    contrasts += [(f"{m}: informativeness", f"{m}__order", f"{m}__no_info")
                  for m in ("G-HO", "K-HO", "I-HO")]
    contrasts += [(f"{m} vs global context-fixed", f"{m}__order", BASELINE)
                  for m in CELLS if f"{m}__order" != BASELINE]
    table = gains(args.factors, contrasts, draws=args.draws, seed=SEED)
    output = fresh_output(args.output)
    table.to_csv(output / "component_gains.csv", index=False)
    write_readme(output, {"component_gains": table}, [
        "Gains are minuend minus subtrahend; positive values favour the minuend.",
        "Factors: response = length + set_given_length + order_given_set (exact chain rule).",
        f"Participant Dirichlet(1,...,1) bootstrap, N-scaled totals: {args.draws} draws, seed {SEED}."])


if __name__ == "__main__":
    main()
