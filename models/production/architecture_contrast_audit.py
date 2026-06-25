"""Descriptive audit of the production 2x2 architecture contrast.

This script does not run inference. It reads the existing production 2x2
posterior-analysis CSV artifacts and exports derived condition-level summaries
for the incremental-vs-global contrast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent / "results_principled_2x2_regularized_local"
)
DEFAULT_MEMO_PATH = (
    REPO_ROOT / "paper" / "memos" / "2026-06-25_architecture_contrast_audit.md"
)

GROUP_COLS = ["relevant_property", "sharpness"]
MODEL_SUFFIX_META = {
    "inc_rec": {
        "architecture": "incremental",
        "architecture_short": "inc",
        "semantics": "context_updating",
        "semantics_short": "rec",
    },
    "inc_static": {
        "architecture": "incremental",
        "architecture_short": "inc",
        "semantics": "context_fixed",
        "semantics_short": "static",
    },
    "glob_rec": {
        "architecture": "global",
        "architecture_short": "glob",
        "semantics": "context_updating",
        "semantics_short": "rec",
    },
    "glob_static": {
        "architecture": "global",
        "architecture_short": "glob",
        "semantics": "context_fixed",
        "semantics_short": "static",
    },
}


def classify_utterance(label: str) -> dict[str, str]:
    """Return derived audit classes for one production utterance label."""
    label = str(label)
    length = len(label)
    length_class = {
        1: "one_adjective",
        2: "two_adjectives",
        3: "three_adjectives",
    }.get(length, "other_length")

    if label == "D":
        selection_class = "size_only"
    elif label == "C":
        selection_class = "colour_only"
    elif "F" in label:
        selection_class = "redundant_form"
    elif set(label) == {"D", "C"}:
        selection_class = "size_colour"
    else:
        selection_class = "other_selection"

    if length == 1:
        order_class = "single_adjective"
    elif label.startswith("D"):
        order_class = "size_initial_multi"
    elif label.startswith("C"):
        order_class = "colour_initial_multi"
    elif label.startswith("F"):
        order_class = "form_initial_multi"
    else:
        order_class = "other_order"

    return {
        "length_class": length_class,
        "selection_class": selection_class,
        "order_class": order_class,
    }


def overinformativeness_class(label: str, relevant_property: str) -> str:
    """Classify response length relative to the condition's minimal length."""
    minimal_length = 2 if relevant_property == "both" else 1
    length = len(str(label))
    if length < minimal_length:
        return "under_minimal_length"
    if length == minimal_length:
        return "minimal_length"
    return "over_minimal_length"


def parse_2x2_model_name(model: str) -> dict[str, str]:
    """Extract architecture and semantic-regime labels from a 2x2 model name."""
    for suffix, meta in MODEL_SUFFIX_META.items():
        if str(model).endswith(suffix):
            return dict(meta)
    raise ValueError(f"Cannot parse 2x2 model name: {model}")


def _safe_corr(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) < 2 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def safe_kl(empirical: Iterable[float], predicted: Iterable[float], eps: float = 1e-12) -> float:
    """KL(empirical || predicted) for categorical proportions."""
    p = np.asarray(list(empirical), dtype=float)
    q = np.asarray(list(predicted), dtype=float)
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    q = np.clip(q, eps, None)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required audit artifact not found: {path}")
    return pd.read_csv(path)


def _add_utterance_features(df: pd.DataFrame) -> pd.DataFrame:
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


def build_input_condition_totals(empirical: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Record source condition totals before audit-time normalization."""
    emp = (
        empirical.groupby(GROUP_COLS, as_index=False)["human_mean"]
        .sum()
        .rename(columns={"human_mean": "total_before_normalization"})
    )
    emp["source"] = "empirical"
    emp["model"] = "empirical"
    emp["architecture"] = "empirical"
    emp["semantics"] = "empirical"

    pred = (
        predictions.groupby(["model"] + GROUP_COLS, as_index=False)["model_mean"]
        .sum()
        .rename(columns={"model_mean": "total_before_normalization"})
    )
    metadata = pred["model"].map(parse_2x2_model_name).apply(pd.Series)
    pred = pd.concat([pred, metadata[["architecture", "semantics"]]], axis=1)
    pred["source"] = "model"

    keep = [
        "source",
        "model",
        "architecture",
        "semantics",
        "relevant_property",
        "sharpness",
        "total_before_normalization",
    ]
    return pd.concat([emp[keep], pred[keep]], ignore_index=True)


def build_long_utterance_rows(empirical: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Stack empirical and model utterance proportions in a common schema."""
    n_by_condition = (
        empirical.groupby(GROUP_COLS, as_index=False)["n"]
        .max()
        .rename(columns={"n": "condition_n"})
    )

    emp = empirical.copy()
    emp["source"] = "empirical"
    emp["model"] = "empirical"
    emp["architecture"] = "empirical"
    emp["architecture_short"] = "empirical"
    emp["semantics"] = "empirical"
    emp["semantics_short"] = "empirical"
    emp["proportion"] = emp["human_mean"]
    emp_total = emp.groupby(GROUP_COLS)["proportion"].transform("sum")
    emp["proportion"] = np.where(emp_total > 0, emp["proportion"] / emp_total, 0.0)
    emp["condition_n"] = emp["n"]

    pred = predictions.merge(n_by_condition, on=GROUP_COLS, how="left")
    metadata = pred["model"].map(parse_2x2_model_name).apply(pd.Series)
    pred = pd.concat([pred, metadata], axis=1)
    pred["source"] = "model"
    pred["proportion"] = pred["model_mean"]
    pred_total = pred.groupby(["model"] + GROUP_COLS)["proportion"].transform("sum")
    pred["proportion"] = np.where(pred_total > 0, pred["proportion"] / pred_total, 0.0)

    keep = [
        "source",
        "model",
        "architecture",
        "architecture_short",
        "semantics",
        "semantics_short",
        "relevant_property",
        "sharpness",
        "utterance_code",
        "utterance_label",
        "proportion",
        "condition_n",
    ]
    rows = pd.concat([emp[keep], pred[keep]], ignore_index=True)
    return _add_utterance_features(rows)


def aggregate_category_proportions(rows: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """Aggregate utterance proportions into a derived category column."""
    rows = rows.copy()
    if category_col not in rows.columns and "utterance_label" in rows.columns:
        rows = _add_utterance_features(rows)
    if "architecture_short" not in rows.columns:
        rows["architecture_short"] = rows["architecture"]
    if "semantics_short" not in rows.columns:
        rows["semantics_short"] = rows["semantics"]
    n_col = "condition_n" if "condition_n" in rows.columns else "n"
    group_cols = [
        "source",
        "model",
        "architecture",
        "architecture_short",
        "semantics",
        "semantics_short",
        "relevant_property",
        "sharpness",
        category_col,
    ]
    out = (
        rows.groupby(group_cols, dropna=False)
        .agg(
            proportion=("proportion", "sum"),
            n=(n_col, "max"),
        )
        .reset_index()
        .rename(columns={category_col: "category"})
    )
    return out


def _add_summary_type(df: pd.DataFrame, summary_type: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "summary_type", summary_type)
    return out


def build_category_tables(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {
        "length": _add_summary_type(
            aggregate_category_proportions(rows, "length_class"),
            "length",
        ),
        "selection": _add_summary_type(
            aggregate_category_proportions(rows, "selection_class"),
            "selection",
        ),
        "order": _add_summary_type(
            aggregate_category_proportions(rows, "order_class"),
            "order",
        ),
        "overinformativeness": _add_summary_type(
            aggregate_category_proportions(rows, "overinformativeness_class"),
            "overinformativeness",
        ),
    }

    order = tables["order"].copy()
    keys = [
        "source",
        "model",
        "architecture",
        "architecture_short",
        "semantics",
        "semantics_short",
        "relevant_property",
        "sharpness",
    ]
    multi = order[order["category"] != "single_adjective"]
    denominators = (
        multi.groupby(keys, as_index=False)["proportion"]
        .sum()
        .rename(columns={"proportion": "multi_adjective_total"})
    )
    order = order.merge(denominators, on=keys, how="left")
    order["conditional_multi_proportion"] = np.where(
        (order["category"] != "single_adjective")
        & (order["multi_adjective_total"] > 0),
        order["proportion"] / order["multi_adjective_total"],
        np.nan,
    )
    tables["order"] = order
    return tables


def build_residuals(category_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    residual_frames = []
    for summary_type, table in category_tables.items():
        empirical = (
            table[table["source"] == "empirical"]
            [["relevant_property", "sharpness", "category", "proportion", "n"]]
            .rename(columns={"proportion": "empirical_proportion"})
        )
        models = table[table["source"] == "model"].copy()
        merged = models.merge(
            empirical,
            on=["relevant_property", "sharpness", "category"],
            how="left",
            suffixes=("", "_empirical"),
        )
        merged["model_proportion"] = merged["proportion"]
        merged["signed_residual"] = (
            merged["model_proportion"] - merged["empirical_proportion"]
        )
        merged["abs_residual"] = merged["signed_residual"].abs()
        merged["summary_type"] = summary_type
        residual_frames.append(merged[[
            "model",
            "architecture",
            "architecture_short",
            "semantics",
            "semantics_short",
            "summary_type",
            "relevant_property",
            "sharpness",
            "category",
            "empirical_proportion",
            "model_proportion",
            "signed_residual",
            "abs_residual",
            "n_empirical",
        ]].rename(columns={"n_empirical": "n"}))
    return pd.concat(residual_frames, ignore_index=True)


def build_fit_metrics(residuals: pd.DataFrame) -> pd.DataFrame:
    records = []
    condition_cols = [
        "model",
        "architecture",
        "architecture_short",
        "semantics",
        "semantics_short",
        "summary_type",
        "relevant_property",
        "sharpness",
    ]
    for keys, sub in residuals.groupby(condition_cols, dropna=False):
        row = dict(zip(condition_cols, keys))
        row.update({
            "scope": "condition",
            "n_categories": int(len(sub)),
            "mae": float(sub["abs_residual"].mean()),
            "rmse": float(np.sqrt(np.mean(sub["signed_residual"] ** 2))),
            "kl_empirical_to_model": safe_kl(
                sub["empirical_proportion"],
                sub["model_proportion"],
            ),
        })
        records.append(row)

    condition_metrics = pd.DataFrame.from_records(records)
    overall_records = []
    overall_cols = [
        "model",
        "architecture",
        "architecture_short",
        "semantics",
        "semantics_short",
        "summary_type",
    ]
    for keys, sub in residuals.groupby(overall_cols, dropna=False):
        row = dict(zip(overall_cols, keys))
        cond = condition_metrics
        for col, value in row.items():
            cond = cond[cond[col] == value]
        row.update({
            "scope": "overall",
            "relevant_property": "ALL",
            "sharpness": "ALL",
            "n_categories": int(len(sub)),
            "mae": float(sub["abs_residual"].mean()),
            "rmse": float(np.sqrt(np.mean(sub["signed_residual"] ** 2))),
            "kl_empirical_to_model": float(cond["kl_empirical_to_model"].mean()),
        })
        overall_records.append(row)

    return pd.concat(
        [condition_metrics, pd.DataFrame.from_records(overall_records)],
        ignore_index=True,
    )


def build_ppc_correlation_by_utterance(rows: pd.DataFrame) -> pd.DataFrame:
    empirical = (
        rows[rows["source"] == "empirical"]
        [GROUP_COLS + ["utterance_label", "proportion"]]
        .rename(columns={"proportion": "empirical_proportion"})
    )
    models = rows[rows["source"] == "model"].copy()
    merged = models.merge(empirical, on=GROUP_COLS + ["utterance_label"], how="left")
    merged["signed_residual"] = merged["proportion"] - merged["empirical_proportion"]
    records = []
    group_cols = [
        "model",
        "architecture",
        "architecture_short",
        "semantics",
        "semantics_short",
        "utterance_label",
    ]
    for keys, sub in merged.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update(classify_utterance(row["utterance_label"]))
        row.update({
            "n_conditions": int(len(sub)),
            "r": _safe_corr(sub["empirical_proportion"], sub["proportion"]),
            "mae": float(np.mean(np.abs(sub["signed_residual"]))),
            "rmse": float(np.sqrt(np.mean(sub["signed_residual"] ** 2))),
            "max_abs_residual": float(np.max(np.abs(sub["signed_residual"]))),
        })
        records.append(row)
    return pd.DataFrame.from_records(records)


def build_architecture_deltas(category_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    model_rows = pd.concat(category_tables.values(), ignore_index=True)
    model_rows = model_rows[model_rows["source"] == "model"].copy()
    keys = ["semantics", "summary_type", "relevant_property", "sharpness", "category"]
    indexed = model_rows.set_index(keys + ["architecture"])
    proportions = indexed["proportion"].unstack("architecture").reset_index()
    model_names = indexed["model"].unstack("architecture").reset_index()

    out = proportions.merge(model_names, on=keys, suffixes=("_proportion", "_model"))
    rename = {
        "incremental_proportion": "incremental_proportion",
        "global_proportion": "global_proportion",
        "incremental_model": "incremental_model",
        "global_model": "global_model",
    }
    out = out.rename(columns=rename)
    if {"incremental_proportion", "global_proportion"}.issubset(out.columns):
        out["delta_incremental_minus_global"] = (
            out["incremental_proportion"] - out["global_proportion"]
        )
        out["abs_delta_incremental_global"] = out[
            "delta_incremental_minus_global"
        ].abs()
    return out.sort_values(
        ["semantics", "summary_type", "abs_delta_incremental_global"],
        ascending=[True, True, False],
    )


def build_global_misses(residuals: pd.DataFrame) -> pd.DataFrame:
    keys = ["semantics", "summary_type", "relevant_property", "sharpness", "category"]
    incremental = residuals[residuals["architecture"] == "incremental"].copy()
    global_rows = residuals[residuals["architecture"] == "global"].copy()

    inc_cols = keys + [
        "model",
        "model_proportion",
        "signed_residual",
        "abs_residual",
    ]
    glob_cols = keys + [
        "model",
        "model_proportion",
        "signed_residual",
        "abs_residual",
        "empirical_proportion",
    ]
    merged = global_rows[glob_cols].merge(
        incremental[inc_cols],
        on=keys,
        how="inner",
        suffixes=("_global", "_incremental"),
    )
    merged["global_minus_incremental_abs_residual"] = (
        merged["abs_residual_global"] - merged["abs_residual_incremental"]
    )
    return merged.sort_values(
        "global_minus_incremental_abs_residual",
        ascending=False,
    ).reset_index(drop=True)


def _parse_semicolon_flags(value: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if not isinstance(value, str):
        return out
    for item in value.split(";"):
        if ":" not in item:
            continue
        key, flag = item.split(":", 1)
        out[key.strip()] = flag.strip().lower() == "true"
    return out


def _parse_jsonish(value: str) -> dict:
    if not isinstance(value, str) or not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def build_loo_diagnostics(
    comparison: pd.DataFrame,
    mcmc_summary: pd.DataFrame | None = None,
    robustness: pd.DataFrame | None = None,
) -> pd.DataFrame:
    comp = comparison.copy()
    if comp.columns[0].startswith("Unnamed"):
        comp = comp.rename(columns={comp.columns[0]: "model"})
    elif "model" not in comp.columns:
        comp = comp.reset_index().rename(columns={"index": "model"})

    meta = comp["model"].map(parse_2x2_model_name).apply(pd.Series)
    comp = pd.concat([comp, meta], axis=1)

    if mcmc_summary is not None and not mcmc_summary.empty:
        comp = comp.merge(
            mcmc_summary[["model", "max_r_hat", "diagnostic_status"]],
            on="model",
            how="left",
        )
    else:
        comp["max_r_hat"] = np.nan
        comp["diagnostic_status"] = np.nan

    bad_pareto = {}
    robust_warnings = {}
    robust_rhats = {}
    if robustness is not None and not robustness.empty:
        row = robustness.loc[robustness["set"] == "new_principled_regularized"]
        if not row.empty:
            row0 = row.iloc[0]
            bad_pareto = _parse_jsonish(row0.get("bad_pareto_k_ge_0.7", ""))
            robust_warnings = _parse_semicolon_flags(row0.get("warnings", ""))
            robust_rhats = _parse_jsonish(row0.get("rhats", ""))

    model_key_by_suffix = {
        "inc_rec": "principled_salience_stop_regularized_2x2_inc_rec",
        "inc_static": "principled_salience_stop_regularized_2x2_inc_static",
        "glob_rec": "principled_salience_stop_regularized_2x2_glob_rec",
        "glob_static": "principled_salience_stop_regularized_2x2_glob_static",
    }
    pareto_by_model = {
        model_key_by_suffix[key]: value
        for key, value in bad_pareto.items()
        if key in model_key_by_suffix
    }
    warning_by_model = {
        model_key_by_suffix[key]: value
        for key, value in robust_warnings.items()
        if key in model_key_by_suffix
    }
    rhat_by_model = {
        model_key_by_suffix[key]: value
        for key, value in robust_rhats.items()
        if key in model_key_by_suffix
    }

    comp["bad_pareto_k_ge_0_7"] = comp["model"].map(pareto_by_model)
    comp["robustness_loo_warning"] = comp["model"].map(warning_by_model)
    comp["robustness_max_rhat"] = comp["model"].map(rhat_by_model)

    rows = []
    for semantics, sub in comp.groupby("semantics", dropna=False):
        inc = sub[sub["architecture"] == "incremental"]
        glob = sub[sub["architecture"] == "global"]
        if inc.empty or glob.empty:
            continue
        inc_row = inc.iloc[0]
        glob_row = glob.iloc[0]
        rows.append({
            "semantics": semantics,
            "incremental_model": inc_row["model"],
            "global_model": glob_row["model"],
            "incremental_elpd_loo": inc_row["elpd_loo"],
            "global_elpd_loo": glob_row["elpd_loo"],
            "delta_elpd_incremental_minus_global": (
                inc_row["elpd_loo"] - glob_row["elpd_loo"]
            ),
            "incremental_loo_warning": inc_row.get("warning"),
            "global_loo_warning": glob_row.get("warning"),
            "incremental_bad_pareto_k_ge_0_7": inc_row.get("bad_pareto_k_ge_0_7"),
            "global_bad_pareto_k_ge_0_7": glob_row.get("bad_pareto_k_ge_0_7"),
            "incremental_max_rhat": inc_row.get("max_r_hat"),
            "global_max_rhat": glob_row.get("max_r_hat"),
            "incremental_diagnostic_status": inc_row.get("diagnostic_status"),
            "global_diagnostic_status": glob_row.get("diagnostic_status"),
            "incremental_elpd_diff_vs_best": inc_row.get("elpd_diff"),
            "global_elpd_diff_vs_best": glob_row.get("elpd_diff"),
            "incremental_dse_vs_best": inc_row.get("dse"),
            "global_dse_vs_best": glob_row.get("dse"),
        })
    return pd.DataFrame.from_records(rows).sort_values("semantics")


def _write_csv(df: pd.DataFrame, path: Path, outputs: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    outputs.append(path)


def write_memo(
    memo_path: Path,
    loo_diagnostics: pd.DataFrame,
    fit_metrics: pd.DataFrame,
    global_misses: pd.DataFrame,
    outputs: list[Path],
) -> None:
    fixed = loo_diagnostics[loo_diagnostics["semantics"] == "context_fixed"]
    fixed_delta = float(fixed["delta_elpd_incremental_minus_global"].iloc[0])
    fixed_global_warning = bool(fixed["global_loo_warning"].iloc[0])
    fixed_global_bad_k = fixed["global_bad_pareto_k_ge_0_7"].iloc[0]

    overall = fit_metrics[
        (fit_metrics["scope"] == "overall")
        & (fit_metrics["semantics"] == "context_fixed")
    ].copy()
    overall = overall.sort_values(["summary_type", "architecture"])

    top_misses = global_misses[
        (global_misses["semantics"] == "context_fixed")
        & (global_misses["global_minus_incremental_abs_residual"] > 0)
    ].head(10)

    lines = [
        "# Production Architecture Contrast Audit",
        "",
        "Date: 2026-06-25",
        "",
        "This descriptive audit uses existing production 2x2 CSV artifacts only.",
        "It does not change semantic code and does not rerun inference.",
        "Model prediction proportions are normalized within each model-condition before category aggregation; the pre-normalization totals are saved in the input-total CSV.",
        "",
        "## LOO Contrast",
        "",
        (
            "Within context-fixed semantics, incremental beats global by "
            f"{fixed_delta:.2f} ELPD."
        ),
        (
            "The current global context-fixed cell carries a PSIS-LOO warning "
            f"flag of {fixed_global_warning}; the robustness CSV reports "
            f"{fixed_global_bad_k} observations with Pareto k >= 0.7."
        ),
        "",
        "## Overall Fit By Derived Summary",
        "",
    ]
    for _, row in overall.iterrows():
        lines.append(
            "- "
            f"{row['architecture']} {row['summary_type']}: "
            f"RMSE={row['rmse']:.3f}, KL={row['kl_empirical_to_model']:.3f}"
        )

    lines.extend([
        "",
        "## Largest Context-Fixed Global Misses",
        "",
    ])
    for _, row in top_misses.iterrows():
        lines.append(
            "- "
            f"{row['summary_type']} / {row['relevant_property']} / "
            f"{row['sharpness']} / {row['category']}: "
            f"empirical={row['empirical_proportion']:.3f}, "
            f"global={row['model_proportion_global']:.3f}, "
            f"incremental={row['model_proportion_incremental']:.3f}, "
            "global-minus-incremental absolute residual="
            f"{row['global_minus_incremental_abs_residual']:.3f}"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The architecture contrast is not primarily a context-updating semantics contrast here.",
        "The context-fixed incremental model succeeds because it can distribute probability across stopping, continuation, and prefix-sensitive order choices.",
        "A successful incremental production model therefore needs more than greedy local informativeness.",
        "It needs a stopping and length mechanism, a stable order prior, and separable pressures for referential usefulness and perceptual reliability.",
        "Those mechanisms are visible in the residuals: the global model misses one-word solution contexts and multi-adjective continuation patterns even when semantics are held fixed.",
        "",
        "## CSV Outputs",
        "",
    ])
    for path in outputs:
        lines.append(f"- `{path.relative_to(REPO_ROOT)}`")

    memo_path.parent.mkdir(parents=True, exist_ok=True)
    memo_path.write_text("\n".join(lines) + "\n")


def run_audit(results_dir: Path, out_dir: Path | None, memo_path: Path) -> list[Path]:
    stats_dir = results_dir / "stats"
    out_dir = out_dir or (stats_dir / "architecture_contrast")

    empirical = _read_csv(stats_dir / "production_empirical.csv")
    predictions = _read_csv(stats_dir / "production_predictions.csv")
    comparison = _read_csv(stats_dir / "production_loo_comparison.csv")

    mcmc_path = stats_dir / "production_simplified_mcmc_model_summary.csv"
    robustness_path = results_dir / "robustness" / "loo_interaction_robustness.csv"
    mcmc_summary = pd.read_csv(mcmc_path) if mcmc_path.exists() else pd.DataFrame()
    robustness = pd.read_csv(robustness_path) if robustness_path.exists() else pd.DataFrame()

    input_totals = build_input_condition_totals(empirical, predictions)
    rows = build_long_utterance_rows(empirical, predictions)
    category_tables = build_category_tables(rows)
    residuals = build_residuals(category_tables)
    fit_metrics = build_fit_metrics(residuals)
    ppc_by_utterance = build_ppc_correlation_by_utterance(rows)
    architecture_deltas = build_architecture_deltas(category_tables)
    global_misses = build_global_misses(residuals)
    loo_diagnostics = build_loo_diagnostics(comparison, mcmc_summary, robustness)

    outputs: list[Path] = []
    _write_csv(
        input_totals,
        out_dir / "architecture_contrast_input_condition_totals.csv",
        outputs,
    )
    _write_csv(
        category_tables["length"],
        out_dir / "architecture_contrast_length_by_condition.csv",
        outputs,
    )
    _write_csv(
        category_tables["selection"],
        out_dir / "architecture_contrast_selection_by_condition.csv",
        outputs,
    )
    _write_csv(
        category_tables["order"],
        out_dir / "architecture_contrast_order_by_condition.csv",
        outputs,
    )
    _write_csv(
        category_tables["overinformativeness"],
        out_dir / "architecture_contrast_overinformativeness_by_condition.csv",
        outputs,
    )
    _write_csv(
        fit_metrics,
        out_dir / "architecture_contrast_fit_metrics.csv",
        outputs,
    )
    _write_csv(
        ppc_by_utterance,
        out_dir / "architecture_contrast_ppc_correlation_by_utterance.csv",
        outputs,
    )
    _write_csv(
        residuals,
        out_dir / "architecture_contrast_residuals.csv",
        outputs,
    )
    _write_csv(
        global_misses,
        out_dir / "architecture_contrast_global_misses.csv",
        outputs,
    )
    _write_csv(
        architecture_deltas,
        out_dir / "architecture_contrast_architecture_deltas.csv",
        outputs,
    )
    _write_csv(
        loo_diagnostics,
        out_dir / "architecture_contrast_loo_diagnostics.csv",
        outputs,
    )

    write_memo(memo_path, loo_diagnostics, fit_metrics, global_misses, outputs)
    outputs.append(memo_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit existing production 2x2 architecture contrast CSVs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Production 2x2 result directory containing stats/ and robustness/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for architecture audit CSVs.",
    )
    parser.add_argument(
        "--memo-path",
        type=Path,
        default=DEFAULT_MEMO_PATH,
        help="Ignored internal memo path.",
    )
    args = parser.parse_args()

    outputs = run_audit(args.results_dir, args.out_dir, args.memo_path)
    print("Architecture contrast audit outputs:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
