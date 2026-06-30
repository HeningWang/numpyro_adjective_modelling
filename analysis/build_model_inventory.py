"""Build compact inventories of fitted slider and production models.

The inputs are the existing post-inference summary CSVs under
``models/slider/results_*`` and ``models/production/results_*``.  The script
does not read posterior samples or prediction dumps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "models" / "model_inventory" / "stats"

SCORING_RULE = (
    {
        "component": "fit_score",
        "weight": 0.40,
        "definition": (
            "Percentile rank of the selected diagnostic-passing model within its "
            "dataset and primary metric. Slider heldout ELPD is preferred when "
            "available; production uses LOO ELPD."
        ),
    },
    {
        "component": "ppc_score",
        "weight": 0.35,
        "definition": (
            "Average of the normalized PPC RMSE score and PPC R2 percentile rank "
            "within diagnostic-passing rows for the dataset; higher is better."
        ),
    },
    {
        "component": "alignment_score",
        "weight": 0.25,
        "definition": (
            "Manual theoretical-alignment score for how much the candidate can be "
            "described as one model family across datasets, allowing only "
            "response-space reductions."
        ),
    },
    {
        "component": "missing_dataset_penalty",
        "weight": -0.20,
        "definition": "Applied per missing dataset counterpart.",
    },
    {
        "component": "slider_nonheldout_penalty",
        "weight": -0.15,
        "definition": (
            "Applied when a slider candidate is available only through full-sample "
            "LOO rather than heldout ELPD."
        ),
    },
)

CANDIDATE_GROUPS = {
    "shared_reliability_backup": {
        "description": (
            "Current production reliability-backup speaker and slider reduction; "
            "only response-space/hierarchy reductions should differ."
        ),
        "same_model_claim_ok": True,
        "alignment_score": 0.95,
        "dataset_families": {
            "production": ["reliability_backup_shared"],
            "slider": [
                "reliability_backup_shared_logalpha_slider",
                "reliability_backup_shared",
            ],
        },
        "next_inference_if_missing": "",
    },
    "planned_order_shared_candidate": {
        "description": (
            "Planned-order/usefulness policy as a shared speaker improvement; "
            "slider has fitted planned-usefulness variants and production has a "
            "reliability-backup order-planning pilot."
        ),
        "same_model_claim_ok": False,
        "alignment_score": 0.80,
        "dataset_families": {
            "production": ["reliability_backup_order_planning", "planned_prefix"],
            "slider": [
                "planned_usefulness_order",
                "planned_usefulness_signed_order",
                "planned_usefulness_mixture",
                "planned_usefulness_anchored_mixture",
            ],
        },
        "next_inference_if_missing": (
            "Do not run the existing production planned-prefix cell unless a new "
            "forward audit passes. First test a reliability-backup-compatible "
            "order-planning variant."
        ),
    },
    "minimal_greedy_baseline": {
        "description": (
            "Older greedy incremental/static baseline family; useful as a fit "
            "reference but not the current production-anchor theory."
        ),
        "same_model_claim_ok": False,
        "alignment_score": 0.60,
        "dataset_families": {
            "production": ["principled_salience_stop", "response_policy"],
            "slider": ["slider_greedy_incremental"],
        },
        "next_inference_if_missing": "",
    },
    "production_specific_policy": {
        "description": (
            "Production-only response-policy extensions such as bounded-form, "
            "sharp-form, or size-sharp."
        ),
        "same_model_claim_ok": False,
        "alignment_score": 0.35,
        "dataset_families": {
            "production": ["bounded_form", "sharp_form", "size_sharp"],
            "slider": [],
        },
        "next_inference_if_missing": "No slider counterpart without changing the response space.",
    },
}

ONE_MODEL_CANDIDATES = {
    "strict_reliability_backup": {
        "claim_strength": "strict_shared_family",
        "same_model_claim_ok": True,
        "description": (
            "Reliability-backup speaker in both datasets, with slider-only reductions "
            "forced by the DC/CD response format and slider likelihood."
        ),
        "differentiation_notes": (
            "Production has the full utterance response space; slider reduces the same "
            "speaker family to the DC/CD order contrast."
        ),
        "alignment_score": 1.00,
        "dataset_families": {
            "production": ["reliability_backup_shared"],
            "slider": [
                "reliability_backup_shared_logalpha_slider",
                "reliability_backup_shared",
            ],
        },
    },
    "strict_order_planning": {
        "claim_strength": "strict_shared_family_missing_slider",
        "same_model_claim_ok": True,
        "description": (
            "Reliability-backup plus order-planning as the same shared speaker family."
        ),
        "differentiation_notes": (
            "Production order-planning has been fitted; no strict slider counterpart "
            "with the same family label is summarized yet."
        ),
        "alignment_score": 0.95,
        "dataset_families": {
            "production": ["reliability_backup_order_planning"],
            "slider": ["reliability_backup_order_planning"],
        },
    },
    "loose_order_planning_bridge": {
        "claim_strength": "loose_theoretical_bridge",
        "same_model_claim_ok": False,
        "description": (
            "Production reliability-backup order-planning paired with slider "
            "planned-usefulness/order variants."
        ),
        "differentiation_notes": (
            "Useful as evidence that order-sensitive planning helps, but too different "
            "to report as literally one model across datasets."
        ),
        "alignment_score": 0.65,
        "dataset_families": {
            "production": ["reliability_backup_order_planning"],
            "slider": [
                "planned_usefulness_order",
                "planned_usefulness_signed_order",
                "planned_usefulness_mixture",
                "planned_usefulness_anchored_mixture",
            ],
        },
    },
    "loose_greedy_baseline": {
        "claim_strength": "loose_baseline",
        "same_model_claim_ok": False,
        "description": (
            "Older greedy incremental/static baseline family."
        ),
        "differentiation_notes": (
            "Useful as a reference baseline, but it is not the current production-anchor "
            "speaker family."
        ),
        "alignment_score": 0.55,
        "dataset_families": {
            "production": ["principled_salience_stop", "response_policy"],
            "slider": ["slider_greedy_incremental"],
        },
    },
}

SUMMARY_FILES = {
    "slider": {
        "heldout": "slider_heldout_elpd_model_summary.csv",
        "loo": "slider_loo_comparison.csv",
        "ppc": "slider_ppc_correlation.csv",
        "diagnostics": ("slider_mcmc_model_summary.csv", "slider_mcmc_diagnostics.csv"),
    },
    "production": {
        "heldout": None,
        "loo": "production_loo_comparison.csv",
        "ppc": ("production_simplified_ppc_correlation.csv", "production_correlation.csv"),
        "diagnostics": (
            "production_simplified_mcmc_model_summary.csv",
            "production_simplified_mcmc_diagnostics.csv",
        ),
    },
}


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def first_existing(stats_dir: Path, filenames: str | tuple[str, ...] | None) -> Path | None:
    if filenames is None:
        return None
    if isinstance(filenames, str):
        filenames = (filenames,)
    for filename in filenames:
        path = stats_dir / filename
        if path.exists():
            return path
    return None


def ensure_model_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model" in out.columns:
        return out
    unnamed = [col for col in out.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        return out.rename(columns={unnamed[0]: "model"})
    if "variant" in out.columns:
        return out.rename(columns={"variant": "model"})
    raise ValueError("Could not infer a model column.")


def normalize_heldout(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_model_column(df)
    out["primary_metric"] = pd.to_numeric(out["total_heldout_elpd"], errors="coerce")
    out["primary_metric_name"] = "heldout_elpd"
    keep = [
        "model",
        "speaker_type",
        "primary_metric",
        "primary_metric_name",
        "mean_fold_elpd",
        "se_fold_elpd",
        "n_heldout",
        "n_folds",
        "n_parameters",
        "max_r_hat",
        "n_divergent",
        "diagnostic_status",
        "diagnostics_ok",
    ]
    keep = [col for col in keep if col in out.columns]
    return out[keep].copy()


def normalize_loo(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_model_column(df)
    out = out.rename(columns={"warning": "loo_warning", "se": "elpd_loo_se"})
    out["primary_metric"] = pd.to_numeric(out["elpd_loo"], errors="coerce")
    out["primary_metric_name"] = "loo_elpd"
    keep = [
        "model",
        "rank",
        "primary_metric",
        "primary_metric_name",
        "elpd_loo",
        "p_loo",
        "elpd_diff",
        "elpd_loo_se",
        "dse",
        "loo_warning",
    ]
    keep = [col for col in keep if col in out.columns]
    return out[keep].copy()


def normalize_ppc(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_model_column(df)
    if "scope" in out.columns:
        all_cells = out[out["scope"].astype(str).eq("all_cells")].copy()
        if not all_cells.empty:
            out = all_cells
    rename = {
        "r": "ppc_r",
        "r2": "ppc_r2",
        "mae": "ppc_mae",
        "rmse": "ppc_rmse",
        "correlation": "ppc_r",
        "condition_rmse": "ppc_rmse",
        "condition_mae": "ppc_mae",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    keep = ["model", "n_points", "ppc_r", "ppc_r2", "ppc_mae", "ppc_rmse"]
    keep = [col for col in keep if col in out.columns]
    for col in keep:
        if col != "model":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[keep].groupby("model", as_index=False).first()


def normalize_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_model_column(df)
    out = out.rename(columns={"max_rhat": "max_r_hat"})
    keep = [
        "model",
        "n_parameters",
        "max_r_hat",
        "n_r_hat_gt_1_01",
        "n_r_hat_gt_1_05",
        "min_ess_bulk",
        "min_ess_tail",
        "n_divergent",
        "divergence_rate",
        "diagnostic_status",
    ]
    keep = [col for col in keep if col in out.columns]
    out = out[keep].copy()
    for col in keep:
        if col not in {"model", "diagnostic_status"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.groupby("model", as_index=False).first()


def infer_architecture(model: str) -> str:
    name = model.lower()
    if "glob" in name or "global" in name:
        return "global"
    if "inc" in name or "incremental" in name:
        return "incremental"
    return "unspecified"


def infer_semantics(model: str) -> str:
    name = model.lower()
    if "static" in name:
        return "context_fixed"
    if "rec" in name or "recursive" in name:
        return "context_updating"
    return "unspecified"


def infer_family(model: str, dataset: str) -> str:
    name = model.lower()
    if "production_anchor_reliabilitybackup_logalpha" in name:
        return "reliability_backup_shared_logalpha_slider"
    if "reliabilitybackup_orderplan" in name:
        return "reliability_backup_order_planning"
    if "reliabilitybackup" in name:
        return "reliability_backup_shared"
    if "production_anchor_sizesharp" in name or "sizesharp" in name or "size_sharp" in name:
        return "size_sharp"
    if "sharpform" in name or "sharp_form" in name:
        return "sharp_form"
    if "boundedform" in name or "bounded_form" in name:
        return "bounded_form"
    if "planned_usefulness_mixture_anchored" in name:
        return "planned_usefulness_anchored_mixture"
    if "planned_usefulness_mixture" in name:
        return "planned_usefulness_mixture"
    if "planned_usefulness_signed_order" in name:
        return "planned_usefulness_signed_order"
    if "planned_usefulness_order" in name:
        return "planned_usefulness_order"
    if "plannedprefix" in name or "planned_prefix" in name:
        return "planned_prefix"
    if "responsepolicy" in name or "response_policy" in name:
        return "response_policy"
    if "principled_salience_stop" in name:
        return "principled_salience_stop"
    if dataset == "slider" and name in {"incremental_recursive", "incremental_static"}:
        return "slider_greedy_incremental"
    return "other"


def family_alignment_class(family: str) -> str:
    if family in {"reliability_backup_shared", "reliability_backup_shared_logalpha_slider"}:
        return "shared_production_anchor"
    if family == "reliability_backup_order_planning":
        return "shared_order_planning_candidate"
    if family in {"bounded_form", "size_sharp", "sharp_form"}:
        return "production_specific_extension"
    if family.startswith("planned_usefulness"):
        return "slider_specific_extension"
    if family == "slider_greedy_incremental":
        return "older_slider_baseline"
    return "not_aligned"


def load_result_dir(dataset: str, result_dir: Path) -> pd.DataFrame | None:
    stats_dir = result_dir / "stats"
    if not stats_dir.exists():
        return None
    files = SUMMARY_FILES[dataset]

    metric = None
    heldout_path = first_existing(stats_dir, files["heldout"])
    if heldout_path is not None:
        metric = normalize_heldout(pd.read_csv(heldout_path))
    else:
        loo_path = first_existing(stats_dir, files["loo"])
        if loo_path is not None:
            metric = normalize_loo(pd.read_csv(loo_path))
    if metric is None:
        return None

    ppc_path = first_existing(stats_dir, files["ppc"])
    if ppc_path is not None:
        metric = metric.merge(normalize_ppc(pd.read_csv(ppc_path)), on="model", how="left")

    diag_path = first_existing(stats_dir, files["diagnostics"])
    if diag_path is not None:
        diagnostics = normalize_diagnostics(pd.read_csv(diag_path))
        suffix_cols = [col for col in diagnostics.columns if col != "model" and col in metric.columns]
        diagnostics = diagnostics.rename(columns={col: f"{col}_diag" for col in suffix_cols})
        metric = metric.merge(diagnostics, on="model", how="left")
        for col in suffix_cols:
            diag_col = f"{col}_diag"
            metric[col] = metric[col].combine_first(metric[diag_col])
            metric = metric.drop(columns=[diag_col])

    metric["dataset"] = dataset
    metric["result_dir"] = str(result_dir.relative_to(ROOT))
    metric["architecture"] = metric["model"].map(infer_architecture)
    metric["semantics"] = metric["model"].map(infer_semantics)
    metric["model_family"] = [
        infer_family(model, dataset) for model in metric["model"].astype(str)
    ]
    metric["alignment_class"] = metric["model_family"].map(family_alignment_class)
    metric["diagnostics_ok"] = metric.apply(row_diagnostics_ok, axis=1)
    metric["delta_primary_from_result_best"] = (
        metric["primary_metric"] - metric["primary_metric"].max()
    )
    return metric


def row_diagnostics_ok(row: pd.Series) -> bool:
    if "diagnostics_ok" in row and pd.notna(row["diagnostics_ok"]):
        if str(row["diagnostics_ok"]).lower() in {"true", "1", "yes"}:
            return True
        if str(row["diagnostics_ok"]).lower() in {"false", "0", "no"}:
            return False
    status = str(row.get("diagnostic_status", "")).lower()
    if status == "fail":
        return False
    n_divergent = row.get("n_divergent", np.nan)
    if pd.notna(n_divergent) and float(n_divergent) > 0:
        return False
    max_r_hat = row.get("max_r_hat", np.nan)
    if pd.notna(max_r_hat) and float(max_r_hat) > 1.05:
        return False
    return True


def build_inventory() -> pd.DataFrame:
    rows = []
    for dataset in ("slider", "production"):
        base = ROOT / "models" / dataset
        for result_dir in sorted(base.glob("results_*")):
            loaded = load_result_dir(dataset, result_dir)
            if loaded is not None and not loaded.empty:
                rows.append(loaded)
    if not rows:
        raise RuntimeError("No model summary CSVs found.")
    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out.drop_duplicates(["dataset", "result_dir", "model"])
    out["evaluation_scope"] = np.where(
        out["primary_metric_name"].eq("heldout_elpd"),
        "heldout",
        "full_sample",
    )
    ppc_cols = ["ppc_r", "ppc_r2", "ppc_rmse", "ppc_mae", "n_points"]
    available_ppc_cols = [col for col in ppc_cols if col in out.columns]
    if available_ppc_cols:
        ppc_lookup = (
            out.dropna(subset=["ppc_rmse"])
            .sort_values(["dataset", "model", "result_dir"])
            .groupby(["dataset", "model"], as_index=False)[available_ppc_cols]
            .first()
        )
        out = out.merge(
            ppc_lookup,
            on=["dataset", "model"],
            how="left",
            suffixes=("", "_model_level"),
        )
        for col in available_ppc_cols:
            fill_col = f"{col}_model_level"
            out[col] = out[col].combine_first(out[fill_col])
            out = out.drop(columns=[fill_col])
    leading = [
        "dataset",
        "result_dir",
        "model",
        "model_family",
        "alignment_class",
        "architecture",
        "semantics",
        "evaluation_scope",
        "primary_metric_name",
        "primary_metric",
        "delta_primary_from_result_best",
        "ppc_r",
        "ppc_r2",
        "ppc_rmse",
        "ppc_mae",
        "diagnostics_ok",
        "diagnostic_status",
        "max_r_hat",
        "n_divergent",
        "n_parameters",
    ]
    ordered = [col for col in leading if col in out.columns]
    remaining = [col for col in out.columns if col not in ordered]
    return out[ordered + remaining].sort_values(
        ["dataset", "alignment_class", "model_family", "result_dir", "model"]
    )


def build_best_by_family(inventory: pd.DataFrame) -> pd.DataFrame:
    candidates = inventory[inventory["diagnostics_ok"].fillna(False)].copy()
    candidates["has_ppc"] = candidates["ppc_rmse"].notna()
    candidates["metric_preference"] = np.where(
        (candidates["dataset"].eq("slider"))
        & (candidates["primary_metric_name"].eq("heldout_elpd")),
        0,
        1,
    )
    candidates = candidates.sort_values(
        [
            "dataset",
            "model_family",
            "metric_preference",
            "has_ppc",
            "primary_metric",
            "ppc_rmse",
        ],
        ascending=[True, True, True, False, False, True],
    )
    best = candidates.groupby(["dataset", "model_family"], as_index=False).first()
    best["dataset_best_primary_metric"] = best.groupby(
        ["dataset", "primary_metric_name"]
    )["primary_metric"].transform("max")
    best["delta_primary_from_dataset_best"] = (
        best["primary_metric"] - best["dataset_best_primary_metric"]
    )
    return best.drop(columns=["has_ppc", "metric_preference"])


def build_sweet_spot_summary(best_by_family: pd.DataFrame) -> pd.DataFrame:
    rows = []
    family_groups = {
        "shared_reliability_backup": [
            "reliability_backup_shared",
            "reliability_backup_shared_logalpha_slider",
        ],
        "slider_planned_usefulness": [
            "planned_usefulness_order",
            "planned_usefulness_signed_order",
            "planned_usefulness_mixture",
            "planned_usefulness_anchored_mixture",
        ],
        "production_specific_policy": ["bounded_form", "sharp_form", "size_sharp"],
        "older_slider_greedy": ["slider_greedy_incremental"],
    }
    for group, families in family_groups.items():
        subset = best_by_family[best_by_family["model_family"].isin(families)].copy()
        if subset.empty:
            continue
        for dataset in ("slider", "production"):
            ds = subset[subset["dataset"].eq(dataset)].copy()
            if ds.empty:
                row = {
                    "candidate_family": group,
                    "dataset": dataset,
                    "available": False,
                }
            else:
                ds = ds.sort_values(
                    ["primary_metric", "ppc_rmse"],
                    ascending=[False, True],
                )
                best = ds.iloc[0].to_dict()
                row = {
                    "candidate_family": group,
                    "dataset": dataset,
                    "available": True,
                    "model_family": best.get("model_family"),
                    "model": best.get("model"),
                    "result_dir": best.get("result_dir"),
                    "primary_metric_name": best.get("primary_metric_name"),
                    "primary_metric": best.get("primary_metric"),
                    "delta_primary_from_dataset_best": best.get(
                        "delta_primary_from_dataset_best"
                    ),
                    "ppc_r2": best.get("ppc_r2"),
                    "ppc_rmse": best.get("ppc_rmse"),
                    "max_r_hat": best.get("max_r_hat"),
                    "n_divergent": best.get("n_divergent"),
                    "diagnostics_ok": best.get("diagnostics_ok"),
                }
            rows.append(row)
    out = pd.DataFrame(rows)
    order = {
        "shared_reliability_backup": 0,
        "slider_planned_usefulness": 1,
        "production_specific_policy": 2,
        "older_slider_greedy": 3,
    }
    out["candidate_order"] = out["candidate_family"].map(order)
    return out.sort_values(["candidate_order", "dataset"]).drop(columns=["candidate_order"])


def _score_series_high_is_good(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(1.0, index=values.index)
    return numeric.rank(method="average", pct=True)


def _score_series_low_is_good(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if len(valid) <= 1:
        return pd.Series(1.0, index=values.index)
    span = valid.max() - valid.min()
    if span == 0:
        return pd.Series(1.0, index=values.index)
    return 1.0 - ((numeric - valid.min()) / span)


def add_row_scores(inventory: pd.DataFrame) -> pd.DataFrame:
    out = inventory[inventory["diagnostics_ok"].fillna(False)].copy()
    out["fit_score"] = np.nan
    for _, idx in out.groupby(["dataset", "primary_metric_name"]).groups.items():
        out.loc[idx, "fit_score"] = _score_series_high_is_good(
            out.loc[idx, "primary_metric"]
        )
    out["ppc_rmse_score"] = np.nan
    out["ppc_r2_score"] = np.nan
    for _, idx in out.groupby("dataset").groups.items():
        out.loc[idx, "ppc_rmse_score"] = _score_series_low_is_good(
            out.loc[idx, "ppc_rmse"]
        )
        out.loc[idx, "ppc_r2_score"] = _score_series_high_is_good(
            out.loc[idx, "ppc_r2"]
        )
    out["ppc_score"] = out[["ppc_rmse_score", "ppc_r2_score"]].mean(
        axis=1,
        skipna=True,
    )
    out["slider_metric_penalty"] = np.where(
        out["dataset"].eq("slider") & ~out["primary_metric_name"].eq("heldout_elpd"),
        0.15,
        0.0,
    )
    return out


def select_candidate_row(
    scored_inventory: pd.DataFrame,
    dataset: str,
    families: list[str],
) -> pd.Series | None:
    if not families:
        return None
    subset = scored_inventory[
        scored_inventory["dataset"].eq(dataset)
        & scored_inventory["model_family"].isin(families)
    ].copy()
    if subset.empty:
        return None
    subset["metric_preference"] = np.where(
        subset["dataset"].eq("slider") & subset["primary_metric_name"].eq("heldout_elpd"),
        0,
        1,
    )
    subset["combined_dataset_score"] = (
        0.55 * subset["fit_score"].fillna(0.0)
        + 0.45 * subset["ppc_score"].fillna(0.0)
        - subset["slider_metric_penalty"].fillna(0.0)
    )
    subset = subset.sort_values(
        ["metric_preference", "combined_dataset_score", "primary_metric", "ppc_rmse"],
        ascending=[True, False, False, True],
    )
    return subset.iloc[0]


def build_cross_dataset_scores(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = add_row_scores(inventory)
    rows = []
    missing = []
    for candidate, config in CANDIDATE_GROUPS.items():
        selected_rows = []
        available_datasets = 0
        missing_datasets = 0
        slider_nonheldout_penalty = 0.0
        for dataset in ("slider", "production"):
            families = config["dataset_families"].get(dataset, [])
            selected = select_candidate_row(scored, dataset, families)
            if selected is None:
                missing_datasets += 1
                missing.append(
                    {
                        "candidate_family": candidate,
                        "dataset": dataset,
                        "expected_model_families": ";".join(families),
                        "reason": "No diagnostic-passing summarized result found.",
                        "recommended_next_step": config["next_inference_if_missing"],
                    }
                )
                continue
            available_datasets += 1
            slider_nonheldout_penalty += float(selected.get("slider_metric_penalty", 0.0))
            selected_rows.append(selected)
            rows.append(
                {
                    "candidate_family": candidate,
                    "candidate_description": config["description"],
                    "dataset": dataset,
                    "available": True,
                    "model_family": selected["model_family"],
                    "model": selected["model"],
                    "result_dir": selected["result_dir"],
                    "evaluation_scope": selected["evaluation_scope"],
                    "primary_metric_name": selected["primary_metric_name"],
                    "primary_metric": selected["primary_metric"],
                    "fit_score": selected["fit_score"],
                    "ppc_score": selected["ppc_score"],
                    "ppc_rmse_score": selected.get("ppc_rmse_score", np.nan),
                    "ppc_r2_score": selected.get("ppc_r2_score", np.nan),
                    "ppc_r2": selected.get("ppc_r2", np.nan),
                    "ppc_rmse": selected.get("ppc_rmse", np.nan),
                    "alignment_score": config["alignment_score"],
                    "same_model_claim_ok": config.get("same_model_claim_ok", False),
                    "max_r_hat": selected.get("max_r_hat", np.nan),
                    "n_divergent": selected.get("n_divergent", np.nan),
                    "diagnostics_ok": selected.get("diagnostics_ok", np.nan),
                }
            )

        if selected_rows:
            mean_fit = float(np.nanmean([row["fit_score"] for row in selected_rows]))
            mean_ppc = float(np.nanmean([row["ppc_score"] for row in selected_rows]))
            mean_ppc_r2 = float(np.nanmean([row.get("ppc_r2", np.nan) for row in selected_rows]))
            mean_ppc_rmse = float(
                np.nanmean([row.get("ppc_rmse", np.nan) for row in selected_rows])
            )
        else:
            mean_fit = 0.0
            mean_ppc = 0.0
            mean_ppc_r2 = np.nan
            mean_ppc_rmse = np.nan
        missing_dataset_penalty = 0.20 * missing_datasets
        total = (
            0.40 * mean_fit
            + 0.35 * mean_ppc
            + 0.25 * config["alignment_score"]
            - missing_dataset_penalty
            - slider_nonheldout_penalty
        )
        rows.append(
            {
                "candidate_family": candidate,
                "candidate_description": config["description"],
                "dataset": "combined",
                "available": available_datasets == 2,
                "model_family": "",
                "model": "",
                "result_dir": "",
                "evaluation_scope": "",
                "primary_metric_name": "sweet_spot_score",
                "primary_metric": total,
                "fit_score": mean_fit,
                "ppc_score": mean_ppc,
                "ppc_rmse_score": np.nan,
                "ppc_r2_score": np.nan,
                "ppc_r2": mean_ppc_r2,
                "ppc_rmse": mean_ppc_rmse,
                "alignment_score": config["alignment_score"],
                "same_model_claim_ok": config.get("same_model_claim_ok", False),
                "max_r_hat": np.nan,
                "n_divergent": np.nan,
                "diagnostics_ok": available_datasets == 2,
                "missing_dataset_count": missing_datasets,
                "missing_dataset_penalty": missing_dataset_penalty,
                "slider_nonheldout_penalty": slider_nonheldout_penalty,
                "ready_for_paper": (
                    available_datasets == 2
                    and missing_datasets == 0
                    and slider_nonheldout_penalty == 0.0
                    and config.get("same_model_claim_ok", False)
                ),
            }
        )
    ranked = pd.DataFrame(rows)
    if "missing_dataset_count" not in ranked.columns:
        ranked["missing_dataset_count"] = np.nan
    if "ready_for_paper" not in ranked.columns:
        ranked["ready_for_paper"] = np.nan
    combined = ranked["dataset"].eq("combined")
    ranked.loc[combined, "rank"] = (
        ranked.loc[combined, "primary_metric"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    ranked = ranked.sort_values(
        ["dataset", "rank", "candidate_family"],
        na_position="last",
    )
    return ranked, pd.DataFrame(missing)


def build_shared_model_scorecard(ranked_scores: pd.DataFrame) -> pd.DataFrame:
    dataset_rows = ranked_scores[ranked_scores["dataset"].isin(["slider", "production"])].copy()
    combined_rows = ranked_scores[ranked_scores["dataset"].eq("combined")].copy()
    records = []
    for _, combined in combined_rows.sort_values("rank").iterrows():
        candidate = combined["candidate_family"]
        row = {
            "candidate_family": candidate,
            "candidate_description": combined["candidate_description"],
            "available_in_both_datasets": bool(combined["available"]),
            "ready_for_paper": bool(combined.get("ready_for_paper", False)),
            "sweet_spot_score": combined["primary_metric"],
            "mean_fit_score": combined["fit_score"],
            "mean_ppc_score": combined["ppc_score"],
            "mean_ppc_r2": combined["ppc_r2"],
            "mean_ppc_rmse": combined["ppc_rmse"],
            "alignment_score": combined["alignment_score"],
            "same_model_claim_ok": bool(combined.get("same_model_claim_ok", False)),
            "missing_dataset_count": combined.get("missing_dataset_count", np.nan),
            "slider_nonheldout_penalty": combined.get("slider_nonheldout_penalty", np.nan),
            "rank": combined.get("rank", np.nan),
        }
        for dataset in ("production", "slider"):
            match = dataset_rows[
                dataset_rows["candidate_family"].eq(candidate)
                & dataset_rows["dataset"].eq(dataset)
            ]
            prefix = f"{dataset}_"
            if match.empty:
                row.update(
                    {
                        prefix + "available": False,
                        prefix + "model_family": "",
                        prefix + "model": "",
                        prefix + "primary_metric_name": "",
                        prefix + "primary_metric": np.nan,
                        prefix + "ppc_r2": np.nan,
                        prefix + "ppc_rmse": np.nan,
                        prefix + "diagnostics_ok": np.nan,
                    }
                )
                continue
            selected = match.iloc[0]
            row.update(
                {
                    prefix + "available": True,
                    prefix + "model_family": selected["model_family"],
                    prefix + "model": selected["model"],
                    prefix + "primary_metric_name": selected["primary_metric_name"],
                    prefix + "primary_metric": selected["primary_metric"],
                    prefix + "ppc_r2": selected["ppc_r2"],
                    prefix + "ppc_rmse": selected["ppc_rmse"],
                    prefix + "max_r_hat": selected["max_r_hat"],
                    prefix + "n_divergent": selected["n_divergent"],
                    prefix + "diagnostics_ok": selected["diagnostics_ok"],
                }
            )
        records.append(row)
    return pd.DataFrame(records)


def build_one_model_scorecard(inventory: pd.DataFrame) -> pd.DataFrame:
    """Score candidates by how defensible the one-model-across-datasets claim is."""
    scored = add_row_scores(inventory)
    records = []
    for candidate, config in ONE_MODEL_CANDIDATES.items():
        selected_by_dataset = {}
        missing_datasets = []
        for dataset in ("production", "slider"):
            families = config["dataset_families"].get(dataset, [])
            selected = select_candidate_row(scored, dataset, families)
            if selected is None:
                missing_datasets.append(dataset)
                continue
            selected_by_dataset[dataset] = selected

        available_in_both = len(selected_by_dataset) == 2
        mean_fit = (
            float(np.nanmean([row["fit_score"] for row in selected_by_dataset.values()]))
            if selected_by_dataset
            else 0.0
        )
        mean_ppc = (
            float(np.nanmean([row["ppc_score"] for row in selected_by_dataset.values()]))
            if selected_by_dataset
            else 0.0
        )
        mean_ppc_r2 = (
            float(np.nanmean([row.get("ppc_r2", np.nan) for row in selected_by_dataset.values()]))
            if selected_by_dataset
            else np.nan
        )
        mean_ppc_rmse = (
            float(np.nanmean([row.get("ppc_rmse", np.nan) for row in selected_by_dataset.values()]))
            if selected_by_dataset
            else np.nan
        )
        slider_nonheldout_penalty = sum(
            float(row.get("slider_metric_penalty", 0.0))
            for row in selected_by_dataset.values()
        )
        missing_dataset_penalty = 0.25 * len(missing_datasets)
        loose_claim_penalty = 0.20 if not config["same_model_claim_ok"] else 0.0
        score = (
            0.35 * mean_fit
            + 0.35 * mean_ppc
            + 0.30 * config["alignment_score"]
            - missing_dataset_penalty
            - slider_nonheldout_penalty
            - loose_claim_penalty
        )
        ready_for_one_model_claim = (
            available_in_both
            and bool(config["same_model_claim_ok"])
            and slider_nonheldout_penalty == 0.0
        )
        row = {
            "candidate_family": candidate,
            "claim_strength": config["claim_strength"],
            "same_model_claim_ok": config["same_model_claim_ok"],
            "ready_for_one_model_claim": ready_for_one_model_claim,
            "available_in_both_datasets": available_in_both,
            "missing_datasets": ";".join(missing_datasets),
            "one_model_score": score,
            "mean_fit_score": mean_fit,
            "mean_ppc_score": mean_ppc,
            "mean_ppc_r2": mean_ppc_r2,
            "mean_ppc_rmse": mean_ppc_rmse,
            "alignment_score": config["alignment_score"],
            "missing_dataset_penalty": missing_dataset_penalty,
            "slider_nonheldout_penalty": slider_nonheldout_penalty,
            "loose_claim_penalty": loose_claim_penalty,
            "candidate_description": config["description"],
            "differentiation_notes": config["differentiation_notes"],
        }
        for dataset in ("production", "slider"):
            prefix = f"{dataset}_"
            selected = selected_by_dataset.get(dataset)
            if selected is None:
                row.update(
                    {
                        prefix + "available": False,
                        prefix + "model_family": "",
                        prefix + "model": "",
                        prefix + "result_dir": "",
                        prefix + "primary_metric_name": "",
                        prefix + "primary_metric": np.nan,
                        prefix + "ppc_r2": np.nan,
                        prefix + "ppc_rmse": np.nan,
                        prefix + "max_r_hat": np.nan,
                        prefix + "n_divergent": np.nan,
                    }
                )
            else:
                row.update(
                    {
                        prefix + "available": True,
                        prefix + "model_family": selected["model_family"],
                        prefix + "model": selected["model"],
                        prefix + "result_dir": selected["result_dir"],
                        prefix + "primary_metric_name": selected["primary_metric_name"],
                        prefix + "primary_metric": selected["primary_metric"],
                        prefix + "ppc_r2": selected.get("ppc_r2", np.nan),
                        prefix + "ppc_rmse": selected.get("ppc_rmse", np.nan),
                        prefix + "max_r_hat": selected.get("max_r_hat", np.nan),
                        prefix + "n_divergent": selected.get("n_divergent", np.nan),
                    }
                )
        records.append(row)
    out = pd.DataFrame(records)
    out["rank"] = out["one_model_score"].rank(method="min", ascending=False).astype(int)
    return out.sort_values(["rank", "candidate_family"])


def planned_prefix_forward_status() -> dict[str, object]:
    path = (
        ROOT
        / "models"
        / "production"
        / "results_planned_prefix_forward_audit"
        / "stats"
        / "production_planned_prefix_gate_decision.csv"
    )
    if not path.exists():
        return {
            "forward_audit_available": False,
            "forward_gate_pass": np.nan,
            "recommended_for_gpu_pilot": np.nan,
            "best_rank_score": np.nan,
        }
    gate = pd.read_csv(path)
    if gate.empty:
        return {
            "forward_audit_available": True,
            "forward_gate_pass": False,
            "recommended_for_gpu_pilot": False,
            "best_rank_score": np.nan,
        }
    best = gate.sort_values("rank_score", ascending=False).iloc[0]
    return {
        "forward_audit_available": True,
        "forward_gate_pass": bool(gate["forward_gate_pass"].fillna(False).any()),
        "recommended_for_gpu_pilot": bool(
            gate["recommended_for_gpu_pilot"].fillna(False).any()
        ),
        "best_rank_score": float(best["rank_score"]),
        "best_candidate": best["candidate"],
        "best_planning_scale": float(best["planning_scale"]),
        "best_utterance_rmse_gain": float(best["utterance_rmse_gain"]),
        "best_category_rmse_gain": float(best["category_rmse_gain"]),
    }


def reliability_order_forward_status() -> dict[str, object]:
    path = (
        ROOT
        / "models"
        / "production"
        / "results_reliability_order_forward_audit"
        / "stats"
        / "production_reliability_order_gate_decision.csv"
    )
    if not path.exists():
        return {
            "forward_audit_available": False,
            "forward_gate_pass": np.nan,
            "recommended_for_gpu_pilot": np.nan,
            "best_rank_score": np.nan,
        }
    gate = pd.read_csv(path)
    if gate.empty:
        return {
            "forward_audit_available": True,
            "forward_gate_pass": False,
            "recommended_for_gpu_pilot": False,
            "best_rank_score": np.nan,
        }
    best = gate.sort_values("rank_score", ascending=False).iloc[0]
    return {
        "forward_audit_available": True,
        "forward_gate_pass": bool(gate["forward_gate_pass"].fillna(False).any()),
        "recommended_for_gpu_pilot": bool(
            gate["recommended_for_gpu_pilot"].fillna(False).any()
        ),
        "best_rank_score": float(best["rank_score"]),
        "best_model": best["model"],
        "best_order_planning_scale": float(best["order_planning_scale"]),
        "best_order_rmse_gain": float(best["order"]),
        "best_utterance_rmse_gain": float(best["utterance_cells"]),
        "best_target_reduction_mean": float(
            best["target_abs_residual_reduction_mean"]
        ),
    }


def build_decision_summary(
    ranked_scores: pd.DataFrame,
    one_model_scorecard: pd.DataFrame,
) -> pd.DataFrame:
    combined = ranked_scores[ranked_scores["dataset"].eq("combined")].copy()
    strict_ready = one_model_scorecard[
        one_model_scorecard["ready_for_one_model_claim"].fillna(False)
    ].sort_values(
        "one_model_score", ascending=False
    )
    top_ready = strict_ready.iloc[0] if not strict_ready.empty else None
    planned_status = planned_prefix_forward_status()
    order_status = reliability_order_forward_status()
    records = []
    if top_ready is not None:
        records.append(
            {
                "decision_item": "current_best_ready_shared_model",
                "candidate_family": top_ready["candidate_family"],
                "status": "best_strict_one_model_candidate",
                "evidence": (
                    "Diagnostic-passing candidate with both datasets available, "
                    "slider heldout support, and a defensible one-model claim."
                ),
                "sweet_spot_score": top_ready["one_model_score"],
                "next_action": (
                    "Use as the current paper-facing shared model unless a strict "
                    "order-planning slider counterpart is fitted and improves the "
                    "same one-model score."
                ),
            }
        )
    order_strict = one_model_scorecard[
        one_model_scorecard["candidate_family"].eq("strict_order_planning")
    ]
    order_loose = one_model_scorecard[
        one_model_scorecard["candidate_family"].eq("loose_order_planning_bridge")
    ]
    order_evidence = (
        "Strict order-planning missing."
        if order_strict.empty
        else (
            f"Strict order-planning missing_datasets="
            f"{order_strict.iloc[0]['missing_datasets']}; "
            f"score={order_strict.iloc[0]['one_model_score']}."
        )
    )
    if not order_loose.empty:
        order_evidence += (
            f" Loose bridge score={order_loose.iloc[0]['one_model_score']}; "
            f"same_model_claim_ok={order_loose.iloc[0]['same_model_claim_ok']}."
        )
    records.append(
        {
            "decision_item": "reliability_order_planning",
            "candidate_family": "strict_order_planning",
            "status": (
                "pilot_complete_but_missing_strict_slider_counterpart"
                if order_status.get("recommended_for_gpu_pilot")
                else "needs_forward_audit"
            ),
            "evidence": (
                f"Forward gate pass={order_status.get('forward_gate_pass')}; "
                f"recommended_for_gpu_pilot={order_status.get('recommended_for_gpu_pilot')}; "
                f"best_rank_score={order_status.get('best_rank_score')}; "
                f"best_order_rmse_gain={order_status.get('best_order_rmse_gain')}; "
                f"best_utterance_rmse_gain={order_status.get('best_utterance_rmse_gain')}. "
                f"{order_evidence}"
            ),
            "sweet_spot_score": np.nan,
            "next_action": (
                "Do not report the loose bridge as one model. If the order-planning "
                "idea is kept, fit a strict slider counterpart and re-score."
            ),
        }
    )
    records.append(
        {
            "decision_item": "old_production_planned_prefix",
            "candidate_family": "planned_order_shared_candidate",
            "status": (
                "do_not_run_existing_inference"
                if planned_status.get("forward_audit_available")
                and not planned_status.get("recommended_for_gpu_pilot")
                else "needs_forward_audit"
            ),
            "evidence": (
                f"Forward gate pass={planned_status.get('forward_gate_pass')}; "
                f"recommended_for_gpu_pilot={planned_status.get('recommended_for_gpu_pilot')}; "
                f"best_rank_score={planned_status.get('best_rank_score')}."
            ),
            "sweet_spot_score": np.nan,
            "next_action": (
                "Do not spend server time on the existing planned-prefix production cell. "
                "If pursuing the slider PPC gain, implement a reliability-backup-compatible "
                "order-planning forward audit first."
            ),
        }
    )
    records.append(
        {
            "decision_item": "production_specific_policy",
            "candidate_family": "production_specific_policy",
            "status": "not_a_one_model_solution",
            "evidence": (
                "Best production score is size-sharp, but there is no slider counterpart "
                "without adding dataset-specific response-space machinery."
            ),
            "sweet_spot_score": combined.loc[
                combined["candidate_family"].eq("production_specific_policy"),
                "primary_metric",
            ].max(),
            "next_action": "Keep as production fit reference, not as shared paper model.",
        }
    )
    return pd.DataFrame(records)


def parse_nc_artifacts() -> pd.DataFrame:
    records = []
    pattern = (
        r"mcmc_results_(?P<model>.+?)"
        r"(?:_speaker_hier|_warmup|_fold|$)"
    )
    for dataset in ("slider", "production"):
        base = ROOT / "models" / dataset / "inference_data"
        if not base.exists():
            continue
        for path in sorted(base.glob("*.nc")):
            name = path.name
            model_match = __import__("re").search(pattern, name)
            fold_match = __import__("re").search(r"_fold(?P<fold>\d+)of(?P<n_folds>\d+)_", name)
            warmup_match = __import__("re").search(r"_warmup(?P<warmup>\d+)", name)
            samples_match = __import__("re").search(r"_samples(?P<samples>\d+)", name)
            chains_match = __import__("re").search(r"_chains(?P<chains>\d+)", name)
            records.append(
                {
                    "dataset": dataset,
                    "path": str(path.relative_to(ROOT)),
                    "filename": name,
                    "parsed_model": model_match.group("model") if model_match else "",
                    "heldout_fold": int(fold_match.group("fold")) if fold_match else np.nan,
                    "n_folds": int(fold_match.group("n_folds")) if fold_match else np.nan,
                    "warmup": int(warmup_match.group("warmup")) if warmup_match else np.nan,
                    "samples": int(samples_match.group("samples")) if samples_match else np.nan,
                    "chains": int(chains_match.group("chains")) if chains_match else np.nan,
                    "size_mb": path.stat().st_size / (1024 * 1024),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory()
    best_by_family = build_best_by_family(inventory)
    sweet_spots = build_sweet_spot_summary(best_by_family)
    ranked_scores, missing_inference = build_cross_dataset_scores(inventory)
    shared_scorecard = build_shared_model_scorecard(ranked_scores)
    one_model_scorecard = build_one_model_scorecard(inventory)
    decision_summary = build_decision_summary(ranked_scores, one_model_scorecard)
    nc_inventory = parse_nc_artifacts()

    inventory.to_csv(out_dir / "fitted_model_inventory.csv", index=False)
    best_by_family.to_csv(out_dir / "best_model_by_dataset_family.csv", index=False)
    sweet_spots.to_csv(out_dir / "cross_dataset_sweet_spot_candidates.csv", index=False)
    ranked_scores.to_csv(out_dir / "cross_dataset_sweet_spot_ranked.csv", index=False)
    shared_scorecard.to_csv(out_dir / "shared_model_scorecard.csv", index=False)
    one_model_scorecard.to_csv(out_dir / "one_model_scorecard.csv", index=False)
    missing_inference.to_csv(out_dir / "missing_inference_cells.csv", index=False)
    decision_summary.to_csv(out_dir / "sweet_spot_decision_summary.csv", index=False)
    pd.DataFrame(SCORING_RULE).to_csv(out_dir / "sweet_spot_scoring_rule.csv", index=False)
    nc_inventory.to_csv(out_dir / "local_nc_artifact_inventory.csv", index=False)

    print(f"Wrote {len(inventory)} inventory rows to {out_dir}")
    print(f"Wrote {len(best_by_family)} dataset-family rows")
    print(f"Wrote {len(sweet_spots)} sweet-spot candidate rows")
    print(f"Wrote {len(ranked_scores)} ranked sweet-spot rows")
    print(f"Wrote {len(shared_scorecard)} shared model scorecard rows")
    print(f"Wrote {len(one_model_scorecard)} one-model scorecard rows")
    print(f"Wrote {len(missing_inference)} missing-inference rows")
    print(f"Wrote {len(decision_summary)} decision-summary rows")
    print(f"Wrote {len(nc_inventory)} local nc artifact rows")


if __name__ == "__main__":
    main()
