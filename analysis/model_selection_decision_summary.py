"""Summarize final model-selection evidence from compact post-inference CSVs.

This script intentionally consumes CSV summaries only. It can be run before
Vast artifacts exist, in which case it writes pending evidence rows instead of
attempting to read NetCDF inference artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_bool(value, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass"}


def model_architecture(model: str) -> str:
    text = str(model)
    if "_inc_" in text or "incremental" in text:
        return "incremental"
    if "_glob_" in text or "global" in text:
        return "global"
    if "planned_usefulness_mixture" in text:
        return "planned_usefulness_mixture"
    if "planned_usefulness_order" in text:
        return "planned_usefulness_order"
    return "unknown"


def model_semantics(model: str) -> str:
    text = str(model)
    if "static" in text:
        return "context_fixed"
    if "_rec" in text or "recursive" in text:
        return "context_updating"
    return "unspecified"


def numeric_value(row: pd.Series, candidates: Iterable[str], default=np.nan) -> float:
    for candidate in candidates:
        if candidate in row.index:
            value = pd.to_numeric(row[candidate], errors="coerce")
            if pd.notna(value):
                return float(value)
    return default


def bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index)
    return df[column].map(lambda value: parse_bool(value, default=default))


class EvidenceReader:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def read(
        self,
        path: Path,
        evidence_source: str,
        file_role: str,
        required: bool = True,
    ) -> pd.DataFrame | None:
        row = {
            "evidence_source": evidence_source,
            "file_role": file_role,
            "path": str(path),
            "required": required,
            "exists": path.exists(),
            "n_rows": pd.NA,
            "status": "pending" if required else "optional_missing",
            "note": "",
        }
        if not path.exists():
            row["note"] = "file not found"
            self.rows.append(row)
            return None
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive CLI path
            row["status"] = "fail"
            row["note"] = f"could not read CSV: {exc}"
            self.rows.append(row)
            return None
        row["n_rows"] = len(df)
        if len(df) == 0 and required:
            row["status"] = "fail"
            row["note"] = "CSV has no rows"
        else:
            row["status"] = "ready"
        self.rows.append(row)
        return df

    def evidence_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def join_values(values: Iterable[str]) -> str:
    cleaned = [str(value) for value in values if pd.notna(value) and str(value)]
    return ";".join(dict.fromkeys(cleaned))


def summarize_slider_stage(
    reader: EvidenceReader,
    stats_dir: Path,
    prefix: str,
    evidence_source: str,
    heldout: bool,
) -> tuple[dict, pd.DataFrame]:
    pairwise = reader.read(
        stats_dir / f"{prefix}_pairwise_decisions.csv",
        evidence_source,
        "pairwise_decisions",
        required=True,
    )
    model_summary = reader.read(
        stats_dir / f"{prefix}_model_decision_summary.csv",
        evidence_source,
        "model_decision_summary",
        required=False,
    )
    scores = reader.read(
        stats_dir / f"{prefix}_pareto_scores.csv",
        evidence_source,
        "pareto_scores",
        required=False,
    )
    frontier = reader.read(
        stats_dir / f"{prefix}_pareto_frontier.csv",
        evidence_source,
        "pareto_frontier",
        required=False,
    )

    decision = {
        "decision_stage": evidence_source,
        "status": "pending",
        "selected_model": pd.NA,
        "selected_architecture": pd.NA,
        "selected_semantics": pd.NA,
        "criterion": "heldout_elpd_and_ppc" if heldout else "posterior_loo_and_ppc",
        "n_candidates": 0,
        "max_delta_elpd": np.nan,
        "best_ppc_rmse_gain": np.nan,
        "best_second_property_abs_residual_reduction": np.nan,
        "evidence": "waiting for pairwise decision CSV",
        "blocker": "",
    }

    if pairwise is None:
        return decision, standardize_scores(model_summary, scores, frontier, evidence_source)
    if pairwise.empty:
        decision.update(
            status="fail",
            evidence="pairwise decision CSV is empty",
            blocker="No candidate-baseline comparisons were exported.",
        )
        return decision, standardize_scores(model_summary, scores, frontier, evidence_source)

    pairwise = pairwise.copy()
    decision["n_candidates"] = int(len(pairwise))
    delta_col = (
        "delta_heldout_elpd_candidate_minus_baseline"
        if heldout else "delta_elpd_candidate_minus_baseline"
    )
    if delta_col in pairwise.columns:
        pairwise[delta_col] = pd.to_numeric(pairwise[delta_col], errors="coerce")
        decision["max_delta_elpd"] = float(pairwise[delta_col].max())
    if "ppc_rmse_gain" in pairwise.columns:
        pairwise["ppc_rmse_gain"] = pd.to_numeric(pairwise["ppc_rmse_gain"], errors="coerce")
        decision["best_ppc_rmse_gain"] = float(pairwise["ppc_rmse_gain"].max())
    if "second_property_abs_residual_reduction" in pairwise.columns:
        pairwise["second_property_abs_residual_reduction"] = pd.to_numeric(
            pairwise["second_property_abs_residual_reduction"],
            errors="coerce",
        )
        decision["best_second_property_abs_residual_reduction"] = float(
            pairwise["second_property_abs_residual_reduction"].max()
        )

    recommended = pairwise[bool_series(pairwise, "recommended_for_full_run")]
    ppc_success = pairwise[bool_series(pairwise, "ppc_success")]
    frontier_candidates: set[str] = set()
    if frontier is not None and "model" in frontier.columns:
        frontier_candidates = set(frontier["model"].astype(str))

    if not recommended.empty:
        sort_cols = [col for col in [delta_col, "ppc_rmse_gain"] if col in recommended.columns]
        best = (
            recommended.sort_values(sort_cols, ascending=False).iloc[0]
            if sort_cols else recommended.iloc[0]
        )
        selected_model = str(best["candidate"])
        decision.update(
            status="pass",
            selected_model=selected_model,
            selected_architecture=model_architecture(selected_model),
            selected_semantics=model_semantics(selected_model),
            evidence=(
                f"{selected_model} is recommended against "
                f"{best.get('baseline', 'baseline')}."
            ),
        )
    elif frontier_candidates:
        selected_model = sorted(frontier_candidates)[0]
        decision.update(
            status="mixed",
            selected_model=selected_model,
            selected_architecture=model_architecture(selected_model),
            selected_semantics=model_semantics(selected_model),
            evidence=(
                "No pairwise recommendation passed all gates, but at least one "
                "candidate is Pareto-nondominated."
            ),
            blocker="Interpret PPC gate failures before treating this as a final architecture.",
        )
    elif not ppc_success.empty:
        selected_model = str(ppc_success.sort_values("ppc_rmse_gain", ascending=False).iloc[0]["candidate"])
        decision.update(
            status="mixed",
            selected_model=selected_model,
            selected_architecture=model_architecture(selected_model),
            selected_semantics=model_semantics(selected_model),
            evidence="PPC gate passed for at least one pair, but no full recommendation was exported.",
            blocker="Check ELPD/diagnostic gate before full inference.",
        )
    else:
        decision.update(
            status="fail",
            evidence="No candidate passed recommendation, PPC, or Pareto frontier gates.",
            blocker="Speaker ablation did not support a more complex speaker under current gates.",
        )

    return decision, standardize_scores(model_summary, scores, frontier, evidence_source)


def summarize_production_stage(
    reader: EvidenceReader,
    architecture_dir: Path,
    prefix: str,
) -> tuple[dict, pd.DataFrame]:
    evidence_source = "production_2x2"
    scores = reader.read(
        architecture_dir / f"{prefix}_pareto_scores.csv",
        evidence_source,
        "pareto_scores",
        required=True,
    )
    frontier = reader.read(
        architecture_dir / f"{prefix}_pareto_frontier.csv",
        evidence_source,
        "pareto_frontier",
        required=True,
    )
    loo = reader.read(
        architecture_dir / "architecture_contrast_loo_diagnostics.csv",
        evidence_source,
        "architecture_loo_diagnostics",
        required=True,
    )
    reader.read(
        architecture_dir / "architecture_contrast_fit_metrics.csv",
        evidence_source,
        "architecture_fit_metrics",
        required=False,
    )
    reader.read(
        architecture_dir / "architecture_contrast_global_misses.csv",
        evidence_source,
        "architecture_global_misses",
        required=False,
    )

    decision = {
        "decision_stage": evidence_source,
        "status": "pending",
        "selected_model": pd.NA,
        "selected_architecture": pd.NA,
        "selected_semantics": pd.NA,
        "criterion": "loo_ppc_diagnostics_architecture_contrast",
        "n_candidates": 0,
        "max_delta_elpd": np.nan,
        "best_ppc_rmse_gain": np.nan,
        "best_second_property_abs_residual_reduction": np.nan,
        "evidence": "waiting for production 2x2 CSVs",
        "blocker": "",
    }

    if scores is None or frontier is None or loo is None:
        return decision, standardize_scores(None, scores, frontier, evidence_source)
    if frontier.empty:
        decision.update(
            status="fail",
            evidence="Production Pareto frontier is empty.",
            blocker="No production model is eligible for the frontier.",
        )
        return decision, standardize_scores(None, scores, frontier, evidence_source)

    frontier = frontier.copy()
    frontier["architecture"] = frontier["model"].map(model_architecture)
    frontier["semantics"] = frontier["model"].map(model_semantics)
    decision["n_candidates"] = int(len(scores)) if scores is not None else int(len(frontier))

    inc_frontier = frontier[frontier["architecture"].eq("incremental")]
    glob_frontier = frontier[frontier["architecture"].eq("global")]
    selected_pool = inc_frontier if not inc_frontier.empty else frontier
    sort_cols = [col for col in ["rank", "elpd_loo"] if col in selected_pool.columns]
    selected_row = (
        selected_pool.sort_values(
            sort_cols,
            ascending=[True, False][:len(sort_cols)],
        ).iloc[0]
        if sort_cols else selected_pool.iloc[0]
    )

    loo = loo.copy()
    if "delta_elpd_incremental_minus_global" in loo.columns:
        loo["delta_elpd_incremental_minus_global"] = pd.to_numeric(
            loo["delta_elpd_incremental_minus_global"],
            errors="coerce",
        )
        decision["max_delta_elpd"] = float(loo["delta_elpd_incremental_minus_global"].max())
        all_incremental_elpd_better = bool(
            loo["delta_elpd_incremental_minus_global"].dropna().gt(0).all()
        )
    else:
        all_incremental_elpd_better = False
    inc_diag_ok = (
        loo.get("incremental_diagnostic_status", pd.Series("", index=loo.index))
        .astype(str)
        .str.lower()
        .isin({"pass", "warn"})
        .all()
    )
    global_diag_bad = (
        loo.get("global_diagnostic_status", pd.Series("", index=loo.index))
        .astype(str)
        .str.lower()
        .eq("fail")
        .any()
    )

    selected_model = str(selected_row["model"])
    base_update = {
        "selected_model": selected_model,
        "selected_architecture": model_architecture(selected_model),
        "selected_semantics": model_semantics(selected_model),
    }
    if not inc_frontier.empty and glob_frontier.empty and all_incremental_elpd_better and inc_diag_ok:
        decision.update(
            status="pass",
            evidence=(
                "Incremental model is on the production Pareto frontier, "
                "global models are off the frontier, and incremental ELPD is higher "
                "within both semantic regimes."
            ),
            blocker="" if global_diag_bad else "Global diagnostics were not clearly worse; interpret contrast directly.",
            **base_update,
        )
    elif not inc_frontier.empty:
        decision.update(
            status="mixed",
            evidence="Incremental model is Pareto-nondominated, but not all architecture gates passed.",
            blocker="Inspect LOO diagnostics and semantic-regime deltas before manuscript use.",
            **base_update,
        )
    else:
        decision.update(
            status="fail",
            evidence="No incremental model is on the production Pareto frontier.",
            blocker="Current production 2x2 does not support the incremental anchor.",
            **base_update,
        )
    return decision, standardize_scores(None, scores, frontier, evidence_source)


def standardize_scores(
    model_summary: pd.DataFrame | None,
    scores: pd.DataFrame | None,
    frontier: pd.DataFrame | None,
    evidence_source: str,
) -> pd.DataFrame:
    source = scores if scores is not None else model_summary
    if source is None:
        source = frontier
    if source is None or source.empty:
        return pd.DataFrame()
    out = source.copy()
    if "model" not in out.columns:
        return pd.DataFrame()

    frontier_models = set()
    if frontier is not None and "model" in frontier.columns:
        frontier_models = set(frontier["model"].astype(str))

    out["evidence_source"] = evidence_source
    out["architecture"] = out["model"].map(model_architecture)
    out["semantics"] = out["model"].map(model_semantics)
    out["on_frontier"] = out["model"].astype(str).isin(frontier_models)
    keep = [
        "evidence_source", "model", "architecture", "semantics", "rank",
        "elpd_loo", "total_heldout_elpd", "delta_elpd_from_best",
        "ppc_rmse", "ppc_r", "n_parameters", "complexity",
        "diagnostic_status", "max_r_hat", "n_divergent",
        "diagnostics_ok", "on_frontier", "posterior_pareto_frontier",
        "heldout_pareto_frontier",
    ]
    keep = [col for col in keep if col in out.columns]
    return out[keep].copy()


def final_decision(stage_rows: list[dict]) -> dict:
    by_stage = {row["decision_stage"]: row for row in stage_rows}
    required = ["slider_posterior_ablation", "slider_heldout_ablation", "production_2x2"]
    statuses = {stage: str(by_stage.get(stage, {}).get("status", "pending")) for stage in required}
    production = by_stage.get("production_2x2", {})
    selected_model = production.get("selected_model", pd.NA)

    if any(status == "pending" for status in statuses.values()):
        status = "pending"
        blocker = "One or more post-Vast evidence CSV sets are not available yet."
    elif statuses["production_2x2"] == "fail" or any(status == "fail" for status in statuses.values()):
        status = "fail"
        blocker = "At least one required model-selection gate failed."
    elif all(status == "pass" for status in statuses.values()):
        status = "pass"
        blocker = ""
    else:
        status = "mixed"
        blocker = "Some evidence is Pareto-nondominated or interpretable but did not pass all gates."

    return {
        "decision_stage": "final_interpretable_2x2",
        "status": status,
        "selected_model": selected_model,
        "selected_architecture": model_architecture(selected_model),
        "selected_semantics": model_semantics(selected_model),
        "criterion": "slider_ablation_plus_heldout_plus_production_2x2",
        "n_candidates": pd.NA,
        "max_delta_elpd": production.get("max_delta_elpd", np.nan),
        "best_ppc_rmse_gain": pd.NA,
        "best_second_property_abs_residual_reduction": pd.NA,
        "evidence": (
            "Final decision combines slider speaker ablation, heldout slider ELPD/PPC, "
            "and production 2x2 LOO/PPC/diagnostics."
        ),
        "blocker": blocker,
    }


def build_decision_summary(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reader = EvidenceReader()
    stage_rows: list[dict] = []
    score_frames: list[pd.DataFrame] = []

    row, scores = summarize_slider_stage(
        reader,
        args.slider_posterior_stats_dir,
        args.slider_posterior_prefix,
        "slider_posterior_ablation",
        heldout=False,
    )
    stage_rows.append(row)
    if not scores.empty:
        score_frames.append(scores)

    row, scores = summarize_slider_stage(
        reader,
        args.slider_heldout_stats_dir,
        args.slider_heldout_prefix,
        "slider_heldout_ablation",
        heldout=True,
    )
    stage_rows.append(row)
    if not scores.empty:
        score_frames.append(scores)

    row, scores = summarize_production_stage(
        reader,
        args.production_architecture_dir,
        args.production_prefix,
    )
    stage_rows.append(row)
    if not scores.empty:
        score_frames.append(scores)

    stage_rows.append(final_decision(stage_rows))
    stage_decisions = pd.DataFrame(stage_rows)
    evidence = reader.evidence_frame()
    candidate_scores = (
        pd.concat(score_frames, ignore_index=True)
        if score_frames else pd.DataFrame(
            columns=["evidence_source", "model", "architecture", "semantics", "on_frontier"]
        )
    )
    return evidence, stage_decisions, candidate_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slider-posterior-stats-dir",
        type=Path,
        default=REPO_ROOT / "models" / "slider" / "results_speaker_ablation_pilot" / "stats",
    )
    parser.add_argument("--slider-posterior-prefix", type=str, default="slider_speaker_ablation_eval")
    parser.add_argument(
        "--slider-heldout-stats-dir",
        type=Path,
        default=REPO_ROOT / "models" / "slider" / "results_heldout_pilot" / "stats",
    )
    parser.add_argument("--slider-heldout-prefix", type=str, default="slider_heldout_eval")
    parser.add_argument(
        "--production-architecture-dir",
        type=Path,
        default=(
            REPO_ROOT / "models" / "production" / "results_final_2x2" /
            "stats" / "architecture_contrast"
        ),
    )
    parser.add_argument("--production-prefix", type=str, default="production_2x2")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "analysis" / "results_model_selection" / "stats",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    evidence, stage_decisions, candidate_scores = build_decision_summary(args)

    evidence_path = args.out_dir / "model_selection_evidence_status.csv"
    stage_path = args.out_dir / "model_selection_stage_decisions.csv"
    scores_path = args.out_dir / "model_selection_candidate_scores.csv"
    evidence.to_csv(evidence_path, index=False)
    stage_decisions.to_csv(stage_path, index=False)
    candidate_scores.to_csv(scores_path, index=False)

    print(f"Wrote {evidence_path}")
    print(f"Wrote {stage_path}")
    print(f"Wrote {scores_path}")
    print(stage_decisions.to_string(index=False))


if __name__ == "__main__":
    main()
