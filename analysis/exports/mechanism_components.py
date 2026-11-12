"""Export per-component ELPD gains of each mechanism for the account-landscape figure.

Each row is one mechanism in one architecture (context-fixed semantics), with
its paired participant-bootstrap gain on response length, adjective set, and
order given the set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from component_gains import gains  # noqa: E402
from robustness_utils import DRAWS, SEED  # noqa: E402

ARCHITECTURES = {"G": "Global", "K": "Plan-guided", "I": "Fully incremental"}
COMPONENTS = {"length": "Number of adjectives", "set_given_length": "Which adjectives",
              "order_given_set": "Order"}


def contrasts(salience_labels: dict[str, str]) -> list[tuple[str, str, str, str, str]]:
    """(account, mechanism, architecture, minuend, subtrahend)."""
    rows = []
    for a, name in ARCHITECTURES.items():
        rows += [("Lexical prediction", "Stable order term", name, f"{a}-HO__order", f"{a}-HO__no_order"),
                 ("Discriminatory efficiency", "Informativeness", name,
                  f"{a}-HO__order", f"{a}-HO__no_info"),
                 ("Hierarchical composition", "Context updating", name,
                  f"{a}-UPD-HO__order", f"{a}-HO__order")]
        if a in salience_labels:
            rows.append(("Availability", "Prefix salience", name, f"{a}-HO__order",
                         salience_labels[a]))
        if a != "G":
            rows.append(("Incremental planning", "Architecture vs global", name,
                         f"{a}-HO__order", "G-HO__order"))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--factors", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--salience", action="append", default=[],
                   help="architecture=label of its salience-off factors, e.g. K=K-HO__no_sal_w4000")
    args = p.parse_args()
    salience = dict(item.split("=", 1) for item in args.salience)
    spec = contrasts(salience)
    table = gains(args.factors, [(str(i), m, s) for i, (*_, m, s) in enumerate(spec)],
                  draws=DRAWS, seed=SEED)
    meta = pd.DataFrame([dict(contrast=str(i), account=acc, mechanism=mech, architecture=arch)
                         for i, (acc, mech, arch, _, _) in enumerate(spec)])
    out = (table.merge(meta, on="contrast")
           .loc[lambda d: d.factor.isin(COMPONENTS)]
           .assign(component=lambda d: d.factor.map(COMPONENTS))
           [["account", "mechanism", "architecture", "component", "minuend", "subtrahend",
             "estimate_elpd", "credible_lower_95", "credible_upper_95", "probability_positive"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
