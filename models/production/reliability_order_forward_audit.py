"""Forward audit for an order-only planning layer on reliability backup.

This audit starts from fitted reliability-backup 2x2 condition predictions and
reallocates probability only among utterances with the same adjective set.
For example, ``DC`` and ``CD`` may trade probability mass, but their combined
mass is preserved.  This tests whether the slider planned-order improvement can
be made compatible with the production response policy without changing
utterance length or adjective selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from architecture_contrast_audit import (  # noqa: E402
    classify_utterance,
    overinformativeness_class,
)


GROUP_COLS = ["relevant_property", "sharpness"]
DEFAULT_INPUT = (
    SCRIPT_DIR
    / "results_reliabilitybackup_2x2_full"
    / "stats"
    / "production_simplified_ppc_by_condition.csv"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "results_reliability_order_forward_audit" / "stats"
ANCHOR_MODELS = (
    "principled_salience_stop_regularized_responsepolicy_reliabilitybackup_2x2_inc_rec_fixedeps",
    "principled_salience_stop_regularized_responsepolicy_reliabilitybackup_2x2_inc_static_fixedeps",
)
TARGET_RESIDUAL_SPECS = (
    ("utterance", "both", "sharp", "DC"),
    ("utterance", "both", "sharp", "CDF"),
    ("utterance", "both", "blurred", "DCF"),
    ("utterance", "second", "sharp", "CF"),
    ("order", "both", "sharp", "size_initial_multi"),
    ("order", "second", "sharp", "colour_initial_multi"),
)


def utterance_key(label: str) -> str:
    return "".join(sorted(str(label)))


def positional_order_score(label: str, relevant_property: str) -> float:
    """Score orders by putting currently useful dimensions earlier.

    The score is contextual but order-only: all permutations of the same
    adjective set keep the same total probability mass.  The dimension weights
    encode the production interpretation: size is useful for first-property
    contexts, colour for second-property contexts, both are useful when both
    dimensions are required, and form is a reliability backup rather than the
    leading descriptive dimension.
    """

    if relevant_property == "first":
        dim_weight = {"D": 1.0, "C": 0.0, "F": 0.30}
    elif relevant_property == "second":
        dim_weight = {"D": 0.0, "C": 1.0, "F": 0.40}
    elif relevant_property == "both":
        dim_weight = {"D": 1.0, "C": 0.75, "F": 0.30}
    else:
        dim_weight = {"D": 0.0, "C": 0.0, "F": 0.0}

    position_discount = (1.0, 0.45, 0.20)
    return float(
        sum(
            position_discount[pos] * dim_weight.get(char, 0.0)
            for pos, char in enumerate(str(label))
        )
    )


def apply_order_planning(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby(["model"] + GROUP_COLS + ["utterance_set"], sort=False):
        out = group.copy()
        base = np.clip(out["model_mean"].to_numpy(dtype=float), 1e-12, None)
        scores = out["order_score"].to_numpy(dtype=float)
        adjusted = base * np.exp(scale * scores)
        total = out["model_mean"].sum()
        out["candidate_model_mean"] = total * adjusted / adjusted.sum()
        rows.append(out)
    result = pd.concat(rows, ignore_index=True)
    result["candidate_signed_residual"] = (
        result["candidate_model_mean"] - result["human_mean"]
    )
    result["candidate_abs_residual"] = result["candidate_signed_residual"].abs()
    result["order_planning_scale"] = scale
    return result


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(
        [classify_utterance(label) for label in df["utterance_label"]],
        index=df.index,
    )
    out = pd.concat([df.copy(), features], axis=1)
    out["overinformativeness_class"] = [
        overinformativeness_class(label, rel)
        for label, rel in zip(out["utterance_label"], out["relevant_property"])
    ]
    return out


def category_residuals(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    base = add_feature_columns(candidate_rows)
    specs = {
        "length": "length_class",
        "selection": "selection_class",
        "order": "order_class",
        "overinformativeness": "overinformativeness_class",
    }
    records = []
    group_base = [
        "model",
        "order_planning_scale",
        "relevant_property",
        "sharpness",
    ]
    for summary_type, category_col in specs.items():
        grouped = (
            base.groupby(group_base + [category_col], as_index=False)
            .agg(
                human_mean=("human_mean", "sum"),
                model_mean=("model_mean", "sum"),
                candidate_model_mean=("candidate_model_mean", "sum"),
            )
            .rename(columns={category_col: "category"})
        )
        grouped["summary_type"] = summary_type
        grouped["baseline_signed_residual"] = (
            grouped["model_mean"] - grouped["human_mean"]
        )
        grouped["candidate_signed_residual"] = (
            grouped["candidate_model_mean"] - grouped["human_mean"]
        )
        grouped["baseline_abs_residual"] = grouped["baseline_signed_residual"].abs()
        grouped["candidate_abs_residual"] = grouped["candidate_signed_residual"].abs()
        grouped["abs_residual_reduction"] = (
            grouped["baseline_abs_residual"] - grouped["candidate_abs_residual"]
        )
        records.append(grouped)
    return pd.concat(records, ignore_index=True)


def rmse(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(arr))))


def summary_metrics(
    candidate_rows: pd.DataFrame,
    category_rows: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for (model, scale), group in candidate_rows.groupby(
        ["model", "order_planning_scale"]
    ):
        records.append(
            {
                "model": model,
                "order_planning_scale": scale,
                "summary_type": "utterance_cells",
                "scope": "overall",
                "n_cells": len(group),
                "baseline_mae": group["abs_residual"].mean(),
                "candidate_mae": group["candidate_abs_residual"].mean(),
                "baseline_rmse": rmse(group["signed_residual"]),
                "candidate_rmse": rmse(group["candidate_signed_residual"]),
            }
        )
    for (model, scale, summary_type), group in category_rows.groupby(
        ["model", "order_planning_scale", "summary_type"]
    ):
        records.append(
            {
                "model": model,
                "order_planning_scale": scale,
                "summary_type": summary_type,
                "scope": "overall",
                "n_cells": len(group),
                "baseline_mae": group["baseline_abs_residual"].mean(),
                "candidate_mae": group["candidate_abs_residual"].mean(),
                "baseline_rmse": rmse(group["baseline_signed_residual"]),
                "candidate_rmse": rmse(group["candidate_signed_residual"]),
            }
        )
    out = pd.DataFrame(records)
    out["mae_gain"] = out["baseline_mae"] - out["candidate_mae"]
    out["rmse_gain"] = out["baseline_rmse"] - out["candidate_rmse"]
    return out


def target_residual_deltas(
    candidate_rows: pd.DataFrame,
    category_rows: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for summary_type, relevant_property, sharpness, category in TARGET_RESIDUAL_SPECS:
        if summary_type == "utterance":
            sub = candidate_rows[
                candidate_rows["relevant_property"].eq(relevant_property)
                & candidate_rows["sharpness"].eq(sharpness)
                & candidate_rows["utterance_label"].eq(category)
            ].copy()
            sub["summary_type"] = summary_type
            sub["category"] = category
        else:
            sub = category_rows[
                category_rows["summary_type"].eq(summary_type)
                & category_rows["relevant_property"].eq(relevant_property)
                & category_rows["sharpness"].eq(sharpness)
                & category_rows["category"].eq(category)
            ].copy()
        if sub.empty:
            continue
        keep = [
            "model",
            "order_planning_scale",
            "summary_type",
            "relevant_property",
            "sharpness",
            "category",
            "baseline_abs_residual",
            "candidate_abs_residual",
            "abs_residual_reduction",
        ]
        if "baseline_abs_residual" not in sub.columns:
            sub["baseline_abs_residual"] = sub["abs_residual"]
            sub["candidate_abs_residual"] = sub["candidate_abs_residual"]
        if "abs_residual_reduction" not in sub.columns:
            sub["abs_residual_reduction"] = (
                sub["baseline_abs_residual"] - sub["candidate_abs_residual"]
            )
        records.append(sub[keep])
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def gate_decision(
    metrics: pd.DataFrame,
    target_deltas: pd.DataFrame,
    *,
    order_rmse_gate: float,
    utterance_rmse_gate: float,
    non_order_harm_gate: float,
    target_reduction_gate: float,
) -> pd.DataFrame:
    metric_wide = metrics.pivot_table(
        index=["model", "order_planning_scale"],
        columns="summary_type",
        values="rmse_gain",
        aggfunc="first",
    ).reset_index()
    target_summary = (
        target_deltas.groupby(["model", "order_planning_scale"], as_index=False)
        .agg(
            target_abs_residual_reduction_mean=("abs_residual_reduction", "mean"),
            target_worst_abs_residual_harm=("abs_residual_reduction", "min"),
        )
    )
    out = metric_wide.merge(
        target_summary,
        on=["model", "order_planning_scale"],
        how="left",
    )
    for col in ["utterance_cells", "order", "length", "selection", "overinformativeness"]:
        if col not in out.columns:
            out[col] = 0.0
    non_order_harm = -out[["length", "selection", "overinformativeness"]].min(axis=1)
    out["non_order_rmse_harm"] = np.maximum(non_order_harm, 0.0)
    out["order_rmse_gate"] = order_rmse_gate
    out["utterance_rmse_gate"] = utterance_rmse_gate
    out["non_order_harm_gate"] = non_order_harm_gate
    out["target_reduction_gate"] = target_reduction_gate
    out["forward_gate_pass"] = (
        out["order"].ge(order_rmse_gate)
        & out["utterance_cells"].ge(utterance_rmse_gate)
        & out["non_order_rmse_harm"].le(non_order_harm_gate)
        & out["target_abs_residual_reduction_mean"].ge(target_reduction_gate)
    )
    out["rank_score"] = (
        out["order"]
        + 0.50 * out["utterance_cells"]
        + 0.25 * out["target_abs_residual_reduction_mean"].fillna(0.0)
        - out["non_order_rmse_harm"]
    )
    out["recommended_for_gpu_pilot"] = out["forward_gate_pass"]
    return out.sort_values("rank_score", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--scales",
        default="0,0.25,0.5,0.75,1.0,1.5,2.0,3.0",
        help="Comma-separated order planning scales.",
    )
    parser.add_argument("--order-rmse-gate", type=float, default=0.002)
    parser.add_argument("--utterance-rmse-gate", type=float, default=0.0005)
    parser.add_argument("--non-order-harm-gate", type=float, default=1e-6)
    parser.add_argument("--target-reduction-gate", type=float, default=0.002)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = [float(item) for item in args.scales.split(",") if item.strip()]

    raw = pd.read_csv(args.input)
    base = raw[raw["model"].isin(ANCHOR_MODELS)].copy()
    if base.empty:
        raise ValueError(f"No anchor rows found in {args.input}")
    base["utterance_set"] = base["utterance_label"].map(utterance_key)
    base["order_score"] = [
        positional_order_score(label, rel)
        for label, rel in zip(base["utterance_label"], base["relevant_property"])
    ]
    base["baseline_abs_residual"] = base["abs_residual"]

    candidates = pd.concat(
        [apply_order_planning(base, scale) for scale in scales],
        ignore_index=True,
    )
    cats = category_residuals(candidates)
    metrics = summary_metrics(candidates, cats)
    targets = target_residual_deltas(candidates, cats)
    gate = gate_decision(
        metrics,
        targets,
        order_rmse_gate=args.order_rmse_gate,
        utterance_rmse_gate=args.utterance_rmse_gate,
        non_order_harm_gate=args.non_order_harm_gate,
        target_reduction_gate=args.target_reduction_gate,
    )
    standard = pd.DataFrame(
        [
            {
                "order_rmse_gate": args.order_rmse_gate,
                "utterance_rmse_gate": args.utterance_rmse_gate,
                "non_order_harm_gate": args.non_order_harm_gate,
                "target_reduction_gate": args.target_reduction_gate,
                "interpretation": (
                    "Run MCMC only if order-only planning improves order and "
                    "utterance-cell RMSE while preserving non-order summaries."
                ),
            }
        ]
    )

    compact_cols = [
        "model",
        "order_planning_scale",
        "relevant_property",
        "sharpness",
        "utterance_code",
        "utterance_label",
        "human_mean",
        "model_mean",
        "candidate_model_mean",
        "signed_residual",
        "candidate_signed_residual",
        "abs_residual",
        "candidate_abs_residual",
        "utterance_set",
        "order_score",
    ]
    candidates[compact_cols].to_csv(
        out_dir / "production_reliability_order_by_condition.csv",
        index=False,
    )
    cats.to_csv(out_dir / "production_reliability_order_category_residuals.csv", index=False)
    metrics.to_csv(out_dir / "production_reliability_order_summary_metrics.csv", index=False)
    targets.to_csv(out_dir / "production_reliability_order_target_residual_deltas.csv", index=False)
    gate.to_csv(out_dir / "production_reliability_order_gate_decision.csv", index=False)
    standard.to_csv(out_dir / "production_reliability_order_success_standard.csv", index=False)

    print(f"Wrote reliability order forward audit to {out_dir}")
    print(gate.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
