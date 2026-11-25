"""Tidy table of the matched contrasts of the production models.

The reference model is the intermediate speaker with participant-specific kappa
(K-HKO) and primary salience (property baseline plus display contrast); the global
and fully incremental speakers (G-HO, I-HO) use the same components with one kappa.
Each row is one contrast A minus B (positive values favour A, the model with the
component named in the contrast) on one measure: which adjectives speakers produce
(length and set given length), their order (order given set), rated canonical
preference, rated context gradient or variance explained in the ratings. Production
values are participant-bootstrap ELPD differences; rating values are reductions in
the absolute error (logits) of the cross-task predictions (replication study) and
differences in R2 under a linear link fitted on the original study.

All inputs share one layout: factors/{model}__{label}_pointwise_factors.csv and
slider/{label}/{model}_cells.csv, with labels primary, no_order, no_info, no_sal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rating_link_transfer import test_errors
from robustness_utils import bootstrap_totals

SEED = 20261110
DRAWS = 20_000
ARCHITECTURES = {"G": "Global", "K": "Intermediate", "I": "Fully incremental"}
MODELS = {"G": ("G-HO", "G-UPD-HO"), "K": ("K-HKO", "K-UPD-HKO"), "I": ("I-HO", "I-UPD-HO")}
PAIRS = ("dimension_color", "dimension_form", "color_form")


def cell(model: str, label: str) -> tuple[str, str, str]:
    """(model, factor label, slider label); both labels coincide in this layout."""
    return (model, label, label)


def component_contrasts() -> list[dict]:
    """Contrasts evaluated within each architecture."""
    rows = []
    for a, name in ARCHITECTURES.items():
        fixed, upd = MODELS[a]
        full = cell(fixed, "primary")
        rows += [
            dict(contrast="Stable order term", architecture=name, a=full, b=cell(fixed, "no_order")),
            dict(contrast="Salience", architecture=name, a=full, b=cell(fixed, "no_sal")),
            dict(contrast="Informativeness", architecture=name, a=full, b=cell(fixed, "no_info")),
            dict(contrast="Sequential context updating", architecture=name,
                 a=cell(upd, "primary"), b=full),
            dict(contrast="Sequential context updating, no stable order term", architecture=name,
                 a=cell(upd, "no_order"), b=cell(fixed, "no_order")),
        ]
    return rows


def architecture_contrasts() -> list[dict]:
    """The intermediate speaker against each endpoint, under both semantics."""
    rows = []
    for other, claim in (("G", "Local competition (intermediate vs global)"),
                         ("I", "Complete-utterance evaluation (intermediate vs fully incremental)")):
        for index, semantics in ((0, "Context-fixed"), (1, "Updating")):
            rows.append(dict(contrast=claim, architecture=semantics,
                             a=cell(MODELS["K"][index], "primary"),
                             b=cell(MODELS[other][index], "primary")))
    return rows


def load_factors(directory: Path, model: str, label: str) -> pd.DataFrame:
    frame = pd.read_csv(directory / f"{model}__{label}_pointwise_factors.csv")
    return frame.sort_values("canonical_row_position").reset_index(drop=True)


def production(directory: Path, a: tuple, b: tuple) -> dict[str, dict]:
    fa, fb = load_factors(directory, a[0], a[1]), load_factors(directory, b[0], b[1])
    if not fa.canonical_row_position.equals(fb.canonical_row_position):
        raise ValueError(f"misaligned rows: {a} vs {b}")
    content = (fa.length + fa.set_given_length) - (fb.length + fb.set_given_length)
    order = fa.order_given_set - fb.order_given_set
    total = fa.response - fb.response
    out = {}
    for measure, diff in (("Production: total", total), ("Production: which adjectives", content),
                          ("Production: order", order)):
        boot = bootstrap_totals(diff.groupby(fa.id).sum(), draws=DRAWS, seed=SEED)
        out[measure] = dict(estimate=boot["estimate_elpd"], lower=boot["credible_lower_95"],
                            upper=boot["credible_upper_95"])
    return out


def rating_summaries(cells: pd.DataFrame, shares: np.ndarray) -> dict[str, np.ndarray]:
    """Average preference and context gradient (logits) for rows of cell-level shares."""
    logits = np.log(shares / (1 - shares))
    replication = (cells.study == "replication").to_numpy()
    gradients = []
    for pair in PAIRS:
        in_pair = replication & (cells.combination == pair).to_numpy()
        first = logits[:, in_pair & (cells.relevant_property == "first").to_numpy()].mean(axis=1)
        second = logits[:, in_pair & (cells.relevant_property == "second").to_numpy()].mean(axis=1)
        gradients.append(first - second)
    return {"Ratings: canonical preference": logits[:, replication].mean(axis=1),
            "Ratings: context gradient": np.mean(gradients, axis=0)}


def rating_errors(slider: Path, model: str, label: str) -> dict[str, np.ndarray]:
    """Absolute distance of each predicted summary from its observed value, per draw."""
    cells = pd.read_csv(slider / label / f"{model}_cells.csv")
    predicted = rating_summaries(cells, np.load(slider / label / f"{model}_cell_draws.npy"))
    observed = rating_summaries(cells, cells.observed.to_numpy()[None, :])
    return {measure: np.abs(predicted[measure] - observed[measure][0]) for measure in predicted}


def test_r_squared(slider: Path, model: str, label: str) -> np.ndarray:
    """Per-draw R2 on the replication after a linear link fitted on the original study."""
    errors, _, null = test_errors(slider, model, label)
    return 1 - (errors / null) ** 2


def ratings(slider: Path, a: tuple, b: tuple) -> dict[str, dict]:
    """Error of B minus error of A, so that positive values mean A is closer to the data."""
    ea, eb = rating_errors(slider, a[0], a[2]), rating_errors(slider, b[0], b[2])
    diffs = {measure: eb[measure] - ea[measure] for measure in ea}
    diffs["Ratings: R2"] = test_r_squared(slider, a[0], a[2]) - test_r_squared(slider, b[0], b[2])
    out = {}
    for measure, diff in diffs.items():
        lower, upper = np.quantile(diff, [.025, .975])
        out[measure] = dict(estimate=float(diff.mean()), lower=float(lower), upper=float(upper))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--factors", type=Path, required=True,
                        help="directory holding all pointwise-factor files")
    parser.add_argument("--slider", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for spec in component_contrasts() + architecture_contrasts():
        values = {**production(args.factors, spec["a"], spec["b"]),
                  **ratings(args.slider, spec["a"], spec["b"])}
        for measure, value in values.items():
            rows.append(dict(contrast=spec["contrast"], architecture=spec["architecture"],
                             model_a="__".join(spec["a"][:2]), model_b="__".join(spec["b"][:2]),
                             measure=measure, **value))
    table = pd.DataFrame(rows)
    table["credible"] = np.where(table.lower > 0, "A better",
                                 np.where(table.upper < 0, "B better", "no credible difference"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
