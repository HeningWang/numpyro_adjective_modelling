"""Forward audit for production planned-prefix speaker variants.

This script does not run inference. It deterministically compares the current
production incremental 2x2 anchor with planned-prefix incremental variants over
the production displays and writes compact CSVs used to decide whether GPU MCMC
is warranted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import modelSpecification as ms  # noqa: E402
from architecture_contrast_audit import (  # noqa: E402
    classify_utterance,
    overinformativeness_class,
    safe_kl,
)
from helper import import_dataset_hier  # noqa: E402


GROUP_COLS = ["relevant_property", "sharpness"]
UTTERANCE_LABELS = ms.UTTERANCE_LABELS
ANCHOR_MODEL = "principled_salience_stop_regularized_2x2_inc_static"
RECURSIVE_ANCHOR_MODEL = "principled_salience_stop_regularized_2x2_inc_rec"
PLANNED_STATIC_MODEL = "principled_salience_stop_regularized_plannedprefix_2x2_inc_static"
PLANNED_RECURSIVE_MODEL = "principled_salience_stop_regularized_plannedprefix_2x2_inc_rec"

TARGET_RESIDUAL_SPECS = (
    ("selection", "both", "ALL", "redundant_form"),
    ("selection", "both", "ALL", "colour_only"),
    ("overinformativeness", "second", "ALL", "over_minimal_length"),
    ("selection", "second", "ALL", "colour_only"),
    ("selection", "first", "ALL", "size_only"),
    ("selection", "first", "ALL", "size_colour"),
)


def _read_anchor_params(path: Path, anchor_model: str) -> dict[str, float]:
    params = {
        "alpha": 1.44,
        "beta_order": float(np.exp(0.59)),
        "lambda_salience": 1.57,
        "rho_salience_stop": 0.49,
        "gamma_uncertainty_len": 0.0,
        "epsilon": 0.003,
    }
    if not path.exists():
        return params

    diag = pd.read_csv(path)
    if "model" not in diag.columns or "parameter" not in diag.columns:
        return params
    sub = diag[diag["model"].eq(anchor_model)].copy()
    if sub.empty:
        return params
    means = sub.set_index("parameter")["mean"].to_dict()
    for key in ("alpha", "lambda_salience", "rho_salience_stop", "epsilon"):
        if key in means and np.isfinite(means[key]):
            params[key] = float(means[key])
    if "log_beta_order" in means and np.isfinite(means["log_beta_order"]):
        params["beta_order"] = float(np.exp(means["log_beta_order"]))
    return params


def _filter_conditions(data: dict, condition_subset: str) -> dict:
    if not condition_subset:
        return data
    codes = tuple(c.strip() for c in condition_subset.split(",") if c.strip())
    if not codes:
        return data
    df = data["df"].reset_index(drop=True)
    keep = df["conditions"].isin(codes).to_numpy()
    if not keep.any():
        raise ValueError(f"--condition-subset matched zero rows: {codes}")
    keep_idx = jnp.asarray(np.where(keep)[0])
    out = dict(data)
    for key in (
        "states_train",
        "empirical_seq_flat",
        "empirical_flat",
        "empirical_seq",
        "seq_mask",
        "sharpness_idx",
        "is_colour_sufficient",
        "sufficient_dim",
        "has_one_word_solution",
        "participant_idx",
    ):
        if key in out and out[key] is not None:
            out[key] = out[key][keep_idx]
    out["df"] = df.loc[keep].reset_index(drop=True)
    return out


def _speaker_probabilities(
    data: dict,
    params: dict[str, float],
    *,
    recursive: bool,
    planned_prefix: bool,
    planning_scale: float,
) -> np.ndarray:
    n = len(data["df"])
    alpha = jnp.full((n,), params["alpha"], dtype=jnp.float32)
    common = (
        data["states_train"],
        data["sufficient_dim"],
        data["has_one_word_solution"],
        data["sharpness_idx"],
        alpha,
        jnp.float32(params["beta_order"]),
        jnp.float32(params["lambda_salience"]),
        jnp.float32(params["rho_salience_stop"]),
    )
    if planned_prefix:
        probs = ms.jitted_speaker_principled_planned_hier(
            *common,
            jnp.float32(planning_scale),
            jnp.float32(params["gamma_uncertainty_len"]),
            jnp.float32(0.59),
            jnp.float32(0.50),
            jnp.float32(0.50),
            jnp.float32(0.6856),
            jnp.float32(params["epsilon"]),
            ms.LOG_LM_ORDER_ONLY_15,
            ms.BASE_VISUAL_SALIENCE,
            recursive=recursive,
            size_context_mode="posterior",
        )
    else:
        probs = ms.jitted_speaker_principled_hier(
            *common,
            jnp.float32(params["gamma_uncertainty_len"]),
            jnp.float32(0.59),
            jnp.float32(0.50),
            jnp.float32(0.50),
            jnp.float32(0.6856),
            jnp.float32(params["epsilon"]),
            ms.LOG_LM_ORDER_ONLY_15,
            ms.BASE_VISUAL_SALIENCE,
            recursive=recursive,
            size_context_mode="posterior",
        )
    return np.asarray(probs)


def _empirical_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    conditions = df[GROUP_COLS].drop_duplicates().sort_values(GROUP_COLS)
    grid = conditions.merge(
        pd.DataFrame(
            {
                "utterance_code": list(range(len(UTTERANCE_LABELS))),
                "utterance_label": UTTERANCE_LABELS,
            }
        ),
        how="cross",
    )
    counts = (
        df.groupby(GROUP_COLS + ["annotation_seq_flat"])
        .size()
        .rename("n")
        .reset_index()
        .rename(columns={"annotation_seq_flat": "utterance_code"})
    )
    totals = df.groupby(GROUP_COLS).size().rename("condition_n").reset_index()
    out = grid.merge(counts, on=GROUP_COLS + ["utterance_code"], how="left")
    out = out.merge(totals, on=GROUP_COLS, how="left")
    out["n"] = out["n"].fillna(0).astype(int)
    out["empirical_proportion"] = out["n"] / out["condition_n"]
    return out


def _model_by_condition(data: dict, probs: np.ndarray, model: str, semantics: str,
                        planning_scale: float) -> pd.DataFrame:
    df = data["df"].reset_index(drop=True)
    tmp = df[GROUP_COLS].copy()
    for code, label in enumerate(UTTERANCE_LABELS):
        tmp[label] = probs[:, code]
    long = tmp.melt(
        id_vars=GROUP_COLS,
        var_name="utterance_label",
        value_name="model_proportion",
    )
    out = (
        long.groupby(GROUP_COLS + ["utterance_label"], as_index=False)["model_proportion"]
        .mean()
    )
    out["utterance_code"] = out["utterance_label"].map(
        {label: code for code, label in enumerate(UTTERANCE_LABELS)}
    )
    out["model"] = model
    out["semantics"] = semantics
    out["planning_scale"] = planning_scale
    return out


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
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


def _condition_residuals(empirical: pd.DataFrame, model_rows: pd.DataFrame) -> pd.DataFrame:
    merged = model_rows.merge(
        empirical[
            GROUP_COLS
            + ["utterance_code", "utterance_label", "empirical_proportion", "condition_n"]
        ],
        on=GROUP_COLS + ["utterance_code", "utterance_label"],
        how="left",
    )
    merged["signed_residual"] = merged["model_proportion"] - merged["empirical_proportion"]
    merged["abs_residual"] = merged["signed_residual"].abs()
    return merged


def _category_residuals(condition_residuals: pd.DataFrame) -> pd.DataFrame:
    base = _add_features(condition_residuals)
    records = []
    specs = {
        "length": "length_class",
        "selection": "selection_class",
        "order": "order_class",
        "overinformativeness": "overinformativeness_class",
    }
    group_base = ["model", "semantics", "planning_scale"] + GROUP_COLS
    for summary_type, category_col in specs.items():
        grouped = (
            base.groupby(group_base + [category_col], as_index=False)
            .agg(
                empirical_proportion=("empirical_proportion", "sum"),
                model_proportion=("model_proportion", "sum"),
                condition_n=("condition_n", "max"),
            )
            .rename(columns={category_col: "category"})
        )
        grouped["summary_type"] = summary_type
        grouped["signed_residual"] = (
            grouped["model_proportion"] - grouped["empirical_proportion"]
        )
        grouped["abs_residual"] = grouped["signed_residual"].abs()
        records.append(grouped)
    return pd.concat(records, ignore_index=True)


def _metric_rows(condition_residuals: pd.DataFrame, category_residuals: pd.DataFrame) -> pd.DataFrame:
    frames = [
        condition_residuals.assign(summary_type="utterance_cells", category="utterance"),
        category_residuals,
    ]
    rows = []
    for frame in frames:
        for keys, sub in frame.groupby(
            ["model", "semantics", "planning_scale", "summary_type"],
            dropna=False,
        ):
            model, semantics, planning_scale, summary_type = keys
            rows.append(
                {
                    "model": model,
                    "semantics": semantics,
                    "planning_scale": planning_scale,
                    "summary_type": summary_type,
                    "scope": "overall",
                    "n_cells": int(len(sub)),
                    "mae": float(sub["abs_residual"].mean()),
                    "rmse": float(np.sqrt(np.mean(sub["signed_residual"] ** 2))),
                    "kl_empirical_to_model": safe_kl(
                        sub["empirical_proportion"],
                        sub["model_proportion"],
                    ),
                }
            )
    return pd.DataFrame(rows)


def _target_residual_deltas(category_residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = category_residuals.set_index(
        [
            "model",
            "semantics",
            "planning_scale",
            "summary_type",
            "relevant_property",
            "sharpness",
            "category",
        ]
    )
    candidates = category_residuals[
        category_residuals["model"].isin([PLANNED_STATIC_MODEL, PLANNED_RECURSIVE_MODEL])
        & category_residuals["planning_scale"].gt(0)
    ][["model", "semantics", "planning_scale"]].drop_duplicates()
    baseline_by_semantics = {
        "context_fixed": ANCHOR_MODEL,
        "context_updating": RECURSIVE_ANCHOR_MODEL,
    }
    for _, candidate in candidates.iterrows():
        baseline = baseline_by_semantics[candidate["semantics"]]
        for summary_type, rel, sharpness, category in TARGET_RESIDUAL_SPECS:
            sharp_values = (
                ["blurred", "sharp"]
                if sharpness == "ALL"
                else [sharpness]
            )
            for sharp in sharp_values:
                key_base = (
                    baseline,
                    candidate["semantics"],
                    0.0,
                    summary_type,
                    rel,
                    sharp,
                    category,
                )
                key_cand = (
                    candidate["model"],
                    candidate["semantics"],
                    candidate["planning_scale"],
                    summary_type,
                    rel,
                    sharp,
                    category,
                )
                if key_base not in indexed.index or key_cand not in indexed.index:
                    continue
                base = indexed.loc[key_base]
                cand = indexed.loc[key_cand]
                rows.append(
                    {
                        "candidate": candidate["model"],
                        "baseline": baseline,
                        "semantics": candidate["semantics"],
                        "planning_scale": candidate["planning_scale"],
                        "summary_type": summary_type,
                        "relevant_property": rel,
                        "sharpness": sharp,
                        "category": category,
                        "baseline_signed_residual": float(base["signed_residual"]),
                        "candidate_signed_residual": float(cand["signed_residual"]),
                        "baseline_abs_residual": float(base["abs_residual"]),
                        "candidate_abs_residual": float(cand["abs_residual"]),
                        "abs_residual_reduction": float(
                            base["abs_residual"] - cand["abs_residual"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _gate_decision(metrics: pd.DataFrame, target_deltas: pd.DataFrame,
                   args: argparse.Namespace) -> pd.DataFrame:
    pivot = metrics.pivot_table(
        index=["model", "semantics", "planning_scale"],
        columns="summary_type",
        values="rmse",
        aggfunc="first",
    ).reset_index()
    baseline = pivot[pivot["model"].isin([ANCHOR_MODEL, RECURSIVE_ANCHOR_MODEL])].copy()
    baseline = baseline.set_index("semantics")
    candidates = pivot[
        pivot["model"].isin([PLANNED_STATIC_MODEL, PLANNED_RECURSIVE_MODEL])
        & pivot["planning_scale"].gt(0)
    ].copy()
    records = []
    for _, cand in candidates.iterrows():
        base = baseline.loc[cand["semantics"]]
        target = target_deltas[
            target_deltas["candidate"].eq(cand["model"])
            & target_deltas["planning_scale"].eq(cand["planning_scale"])
        ]
        target_mean_gain = (
            float(target["abs_residual_reduction"].mean()) if not target.empty else np.nan
        )
        target_worst_harm = (
            float(np.maximum(-target["abs_residual_reduction"], 0).max())
            if not target.empty else np.nan
        )
        records.append(
            {
                "candidate": cand["model"],
                "baseline": base["model"],
                "semantics": cand["semantics"],
                "planning_scale": cand["planning_scale"],
                "utterance_rmse_gain": float(base["utterance_cells"] - cand["utterance_cells"]),
                "category_rmse_gain": float(
                    np.mean(
                        [
                            base[col] - cand[col]
                            for col in ("length", "selection", "order", "overinformativeness")
                            if col in cand.index and col in base.index
                        ]
                    )
                ),
                "target_abs_residual_reduction_mean": target_mean_gain,
                "target_worst_abs_residual_harm": target_worst_harm,
            }
        )
    scored = pd.DataFrame(records)
    if scored.empty:
        return pd.DataFrame()
    scored["forward_gate_pass"] = (
        (scored["utterance_rmse_gain"] >= args.utterance_rmse_gate)
        | (scored["category_rmse_gain"] >= args.category_rmse_gate)
        | (
            (scored["target_abs_residual_reduction_mean"] >= args.target_residual_gate)
            & (scored["target_worst_abs_residual_harm"] <= args.max_target_harm)
        )
    )
    scored["rank_score"] = (
        scored["utterance_rmse_gain"]
        + scored["category_rmse_gain"]
        + scored["target_abs_residual_reduction_mean"].fillna(0)
    )
    scored = scored.sort_values("rank_score", ascending=False).reset_index(drop=True)
    scored["recommended_for_gpu_pilot"] = False
    if bool(scored["forward_gate_pass"].any()):
        scored.loc[scored["forward_gate_pass"].idxmax(), "recommended_for_gpu_pilot"] = True
    scored["utterance_rmse_gate"] = args.utterance_rmse_gate
    scored["category_rmse_gate"] = args.category_rmse_gate
    scored["target_residual_gate"] = args.target_residual_gate
    scored["max_target_harm"] = args.max_target_harm
    return scored


def _success_standard(args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "forward_audit",
                "criterion": "run_gpu_pilot_if_any_gate_passes",
                "utterance_rmse_gain": args.utterance_rmse_gate,
                "category_rmse_gain": args.category_rmse_gate,
                "target_abs_residual_reduction_mean": args.target_residual_gate,
                "max_target_worst_abs_residual_harm": args.max_target_harm,
            },
            {
                "stage": "posterior_pilot",
                "criterion": "advance_if_diagnostic_clean_and_elpd_or_pareto_ppc",
                "max_r_hat": args.max_r_hat,
                "max_divergences": 0,
                "positive_delta_elpd_over_matched_anchor": 0.0,
                "ppc_rmse_gain": args.posterior_ppc_rmse_gate,
                "target_abs_residual_reduction_mean": args.posterior_target_residual_gate,
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results_planned_prefix_forward_audit/stats")
    parser.add_argument(
        "--anchor-diagnostics",
        default=(
            "results_principled_2x2_regularized_local/stats/"
            "production_simplified_mcmc_diagnostics.csv"
        ),
    )
    parser.add_argument("--anchor-model", default=ANCHOR_MODEL)
    parser.add_argument("--condition-subset", default="")
    parser.add_argument("--planning-grid", default="0,0.25,0.5,1,1.5,2,3")
    parser.add_argument("--utterance-rmse-gate", type=float, default=0.005)
    parser.add_argument("--category-rmse-gate", type=float, default=0.02)
    parser.add_argument("--target-residual-gate", type=float, default=0.03)
    parser.add_argument("--max-target-harm", type=float, default=0.05)
    parser.add_argument("--max-r-hat", type=float, default=1.05)
    parser.add_argument("--posterior-ppc-rmse-gate", type=float, default=0.005)
    parser.add_argument("--posterior-target-residual-gate", type=float, default=0.03)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = SCRIPT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_path = Path(args.anchor_diagnostics)
    if not anchor_path.is_absolute():
        anchor_path = SCRIPT_DIR / anchor_path
    params = _read_anchor_params(anchor_path, args.anchor_model)
    planning_grid = [
        float(x.strip()) for x in args.planning_grid.split(",") if x.strip()
    ]

    data = _filter_conditions(
        import_dataset_hier(state_encoding="target_match"),
        args.condition_subset,
    )
    empirical = _empirical_by_condition(data["df"])

    model_frames = []
    for recursive, semantics, base_model, planned_model in (
        (False, "context_fixed", ANCHOR_MODEL, PLANNED_STATIC_MODEL),
        (True, "context_updating", RECURSIVE_ANCHOR_MODEL, PLANNED_RECURSIVE_MODEL),
    ):
        base_probs = _speaker_probabilities(
            data,
            params,
            recursive=recursive,
            planned_prefix=False,
            planning_scale=0.0,
        )
        model_frames.append(
            _model_by_condition(data, base_probs, base_model, semantics, 0.0)
        )
        for scale in planning_grid:
            planned_probs = _speaker_probabilities(
                data,
                params,
                recursive=recursive,
                planned_prefix=True,
                planning_scale=scale,
            )
            model_frames.append(
                _model_by_condition(data, planned_probs, planned_model, semantics, scale)
            )

    model_rows = pd.concat(model_frames, ignore_index=True)
    condition_residuals = _condition_residuals(empirical, model_rows)
    category_residuals = _category_residuals(condition_residuals)
    metrics = _metric_rows(condition_residuals, category_residuals)
    target_deltas = _target_residual_deltas(category_residuals)
    gate = _gate_decision(metrics, target_deltas, args)

    empirical.to_csv(out_dir / "production_planned_prefix_empirical.csv", index=False)
    condition_residuals.to_csv(
        out_dir / "production_planned_prefix_by_condition.csv",
        index=False,
    )
    category_residuals.to_csv(
        out_dir / "production_planned_prefix_category_residuals.csv",
        index=False,
    )
    metrics.to_csv(out_dir / "production_planned_prefix_summary_metrics.csv", index=False)
    target_deltas.to_csv(
        out_dir / "production_planned_prefix_target_residual_deltas.csv",
        index=False,
    )
    gate.to_csv(out_dir / "production_planned_prefix_gate_decision.csv", index=False)
    pd.DataFrame([{**params, "condition_subset": args.condition_subset}]).to_csv(
        out_dir / "production_planned_prefix_anchor_params.csv",
        index=False,
    )
    _success_standard(args).to_csv(
        out_dir / "production_planned_prefix_success_standard.csv",
        index=False,
    )

    print(f"Wrote production planned-prefix forward audit CSVs to {out_dir}")
    if not gate.empty:
        print(gate.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
