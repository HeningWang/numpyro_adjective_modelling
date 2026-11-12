"""Hypothesis-driven pairwise contrasts on length, selection, and order.

Each hypothesis is tested by pairwise comparisons that differ only in the
mechanism the account names. Gains are minuend minus subtrahend from the
pointwise factors of component_decomposition.py; positive values favour the
minuend. Participant Dirichlet bootstrap as in component_gains.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from component_gains import gains
from robustness_utils import DRAWS, SEED

ARCH = {"G": "global", "K": "plan-guided", "I": "incremental"}
SALIENCE_OFF = {"G": "G-HO__no_sal", "K": "K-HO__no_sal_w4000", "I": "I-HO__no_sal"}
COMPONENTS = {"length": "Length", "set_given_length": "Selection", "order_given_set": "Order"}


def hypotheses() -> list[tuple[str, str, str, str]]:
    """(hypothesis, comparison, minuend, subtrahend)."""
    rows = []
    for a, name in ARCH.items():
        rows.append(("Convention", f"{name}: with vs without stable term",
                     f"{a}-HO__order", f"{a}-HO__no_order"))
    for a, name in ARCH.items():
        rows.append(("Discriminatory efficiency", f"{name}: with vs without informativeness",
                     f"{a}-HO__order", f"{a}-HO__no_info"))
    for a, name in ARCH.items():
        rows.append(("Composition", f"{name}: updating vs fixed",
                     f"{a}-UPD-HO__order", f"{a}-HO__order"))
    for a, name in ARCH.items():
        rows.append(("Composition without convention", f"{name}: updating vs fixed, no stable term",
                     f"{a}-UPD-HO__no_order", f"{a}-HO__no_order"))
    rows += [("Planning", "plan-guided vs global", "K-HO__order", "G-HO__order"),
             ("Planning", "incremental vs global", "I-HO__order", "G-HO__order"),
             ("Planning", "plan-guided vs incremental", "K-HO__order", "I-HO__order")]
    for a, name in ARCH.items():
        rows.append(("Availability", f"{name}: with vs without salience",
                     f"{a}-HO__order", SALIENCE_OFF[a]))
    return rows


BASELINE = "G-HO__order"


def models_vs_baseline() -> list[tuple[str, str, str, str]]:
    """Every fitted model minus the global, context-fixed model (hypothesis, model, label)."""
    rows = []
    for a, name in ARCH.items():
        rows += [("Convention", f"{name}, with stable term", f"{a}-HO__order"),
                 ("Convention", f"{name}, without stable term", f"{a}-HO__no_order"),
                 ("Discriminatory efficiency", f"{name}, with informativeness", f"{a}-HO__order"),
                 ("Discriminatory efficiency", f"{name}, without informativeness", f"{a}-HO__no_info"),
                 ("Composition", f"{name}, fixed", f"{a}-HO__order"),
                 ("Composition", f"{name}, updating", f"{a}-UPD-HO__order"),
                 ("Planning", name, f"{a}-HO__order"),
                 ("Availability", f"{name}, with salience", f"{a}-HO__order"),
                 ("Availability", f"{name}, without salience", SALIENCE_OFF[a])]
    return [(h, m, lab, BASELINE) for h, m, lab in rows]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--factors", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", action="store_true",
                   help="compare every model with the global, context-fixed model")
    args = p.parse_args()
    spec = models_vs_baseline() if args.baseline else hypotheses()
    unique = {(m, s) for _, _, m, s in spec if m != s}
    table = gains(args.factors, [(f"{m} - {s}", m, s) for m, s in sorted(unique)],
                  draws=DRAWS, seed=SEED)
    meta = pd.DataFrame([dict(hypothesis=h, model=c, contrast=f"{m} - {s}")
                         for h, c, m, s in spec])
    out = (table.merge(meta, on="contrast")
           .assign(component=lambda d: d.factor.map(COMPONENTS).fillna("Response"))
           [["hypothesis", "model", "contrast", "component", "minuend", "subtrahend", "estimate_elpd",
             "credible_lower_95", "credible_upper_95", "probability_positive"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
