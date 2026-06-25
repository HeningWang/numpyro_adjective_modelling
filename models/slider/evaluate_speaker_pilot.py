"""Evaluate slider speaker-pilot posterior summaries.

This script consumes compact CSVs already exported by posterior_analysis.py
and writes pilot-level decision summaries. It does not read .nc artifacts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.model_pareto_frontier import build_frontier_from_csvs  # noqa: E402


DEFAULT_STATS_DIR = Path(__file__).resolve().parent / "results_planned_usefulness_pilot" / "stats"

PAIR_SPECS = (
    ("planned_usefulness_order", "incremental_recursive", "planned_vs_greedy_recursive"),
    ("planned_usefulness_order_static", "incremental_static", "planned_vs_greedy_static"),
    ("planned_usefulness_mixture", "planned_usefulness_order", "mixture_vs_planned_recursive"),
    (
        "planned_usefulness_mixture_static",
        "planned_usefulness_order_static",
        "mixture_vs_planned_static",
    ),
)

PROPERTY_LABELS = {
    "second": "zrdc_like_second_property",
    "first": "erdc_like_first_property",
    "both": "brdc_like_both_properties",
}


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    return pd.read_csv(path)


def normalize_loo(path: Path) -> pd.DataFrame:
    loo = pd.read_csv(path)
    first_col = loo.columns[0]
    if first_col.startswith("Unnamed") or first_col == "":
        loo = loo.rename(columns={first_col: "model"})
    elif first_col != "model":
        loo = loo.rename(columns={first_col: "model"})
    loo = loo.rename(columns={"warning": "loo_warning", "se": "elpd_loo_se"})
    numeric = ["rank", "elpd_loo", "p_loo", "elpd_diff", "elpd_loo_se", "dse"]
    for col in numeric:
        if col in loo.columns:
            loo[col] = pd.to_numeric(loo[col], errors="coerce")
    return loo


def normalize_ppc_correlation(path: Path, scope: str) -> pd.DataFrame:
    ppc = pd.read_csv(path)
    if "scope" in ppc.columns:
        ppc = ppc[ppc["scope"].astype(str).eq(scope)].copy()
    ppc = ppc.rename(columns={"rmse": "ppc_rmse", "mae": "ppc_mae", "r": "ppc_r", "r2": "ppc_r2"})
    for col in ["ppc_rmse", "ppc_mae", "ppc_r", "ppc_r2"]:
        if col in ppc.columns:
            ppc[col] = pd.to_numeric(ppc[col], errors="coerce")
    return ppc


def summarize_model_level(stats_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    loo = normalize_loo(stats_dir / "slider_loo_comparison.csv")
    ppc = normalize_ppc_correlation(stats_dir / "slider_ppc_correlation.csv", args.ppc_scope)
    diag = _read_csv(stats_dir / "slider_mcmc_model_summary.csv")
    pareto = _read_csv(stats_dir / "slider_loo_pareto_diagnostics.csv", required=False)

    out = loo.merge(ppc, on="model", how="outer")
    if diag is not None:
        out = out.merge(diag, on="model", how="left")
    if pareto is not None:
        out = out.merge(pareto, on="model", how="left", suffixes=("", "_pareto"))

    out["diagnostics_ok"] = (
        out.get("diagnostic_status", pd.Series("missing", index=out.index)).astype(str).eq("pass")
        & pd.to_numeric(out.get("n_divergent", 1), errors="coerce").fillna(1).eq(0)
        & pd.to_numeric(out.get("max_r_hat", np.inf), errors="coerce").fillna(np.inf).le(args.max_r_hat)
    )

    max_pareto = pd.to_numeric(out.get("max_pareto_k", np.inf), errors="coerce")
    frac_bad = pd.to_numeric(out.get("frac_pareto_k_gt_0_7", 1.0), errors="coerce")
    n_gt_1 = pd.to_numeric(out.get("n_pareto_k_gt_1_0", 1), errors="coerce")
    loo_warning = out.get("loo_warning", True).astype(str).str.lower().isin({"true", "1", "yes"})
    out["psis_loo_reliable"] = (
        np.isfinite(max_pareto)
        & max_pareto.le(args.max_pareto_k)
        & frac_bad.fillna(1.0).le(args.max_frac_pareto_gt_0_7)
        & n_gt_1.fillna(1).eq(0)
        & ~loo_warning
    )
    out["loo_interpretation"] = np.where(
        out["psis_loo_reliable"],
        "usable_for_elpd_ranking",
        "diagnostic_only_bad_pareto_k",
    )
    return out.sort_values(["ppc_rmse", "elpd_loo"], ascending=[True, False])


def residual_changes(ppc_by_condition: pd.DataFrame, candidate: str, baseline: str) -> pd.DataFrame:
    cols = ["model", "relevant_property", "sharpness", "human_mean", "model_mean", "signed_residual", "abs_residual"]
    sub = ppc_by_condition[ppc_by_condition["model"].isin([candidate, baseline])][cols].copy()
    wide = sub.pivot_table(
        index=["relevant_property", "sharpness"],
        columns="model",
        values=["model_mean", "signed_residual", "abs_residual"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    wide["candidate"] = candidate
    wide["baseline"] = baseline
    wide["property_label"] = wide["relevant_property"].map(PROPERTY_LABELS).fillna(wide["relevant_property"])
    wide["baseline_model_mean"] = wide[f"model_mean_{baseline}"]
    wide["candidate_model_mean"] = wide[f"model_mean_{candidate}"]
    wide["baseline_signed_residual"] = wide[f"signed_residual_{baseline}"]
    wide["candidate_signed_residual"] = wide[f"signed_residual_{candidate}"]
    wide["baseline_abs_residual"] = wide[f"abs_residual_{baseline}"]
    wide["candidate_abs_residual"] = wide[f"abs_residual_{candidate}"]
    wide["abs_residual_reduction"] = wide["baseline_abs_residual"] - wide["candidate_abs_residual"]
    wide["signed_residual_change"] = wide["candidate_signed_residual"] - wide["baseline_signed_residual"]
    return wide[
        [
            "candidate",
            "baseline",
            "relevant_property",
            "property_label",
            "sharpness",
            "baseline_model_mean",
            "candidate_model_mean",
            "baseline_signed_residual",
            "candidate_signed_residual",
            "baseline_abs_residual",
            "candidate_abs_residual",
            "abs_residual_reduction",
            "signed_residual_change",
        ]
    ]


def pairwise_summary(
    model_summary: pd.DataFrame,
    ppc_by_condition: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    residual_frames = []
    indexed = model_summary.set_index("model")
    for candidate, baseline, label in PAIR_SPECS:
        if candidate not in indexed.index or baseline not in indexed.index:
            continue
        cand = indexed.loc[candidate]
        base = indexed.loc[baseline]
        residual = residual_changes(ppc_by_condition, candidate, baseline)
        residual_frames.append(residual.assign(pair=label))

        second = residual[residual["relevant_property"].eq("second")]
        first = residual[residual["relevant_property"].eq("first")]
        both = residual[residual["relevant_property"].eq("both")]

        delta_elpd = float(cand["elpd_loo"] - base["elpd_loo"])
        ppc_rmse_gain = float(base["ppc_rmse"] - cand["ppc_rmse"])
        ppc_r_gain = float(cand["ppc_r"] - base["ppc_r"])
        second_gain = float(second["abs_residual_reduction"].mean()) if not second.empty else np.nan
        first_harm = float(-first["abs_residual_reduction"].min()) if not first.empty else np.nan
        both_harm = float(-both["abs_residual_reduction"].min()) if not both.empty else np.nan
        worst_new_harm = np.nanmax([first_harm, both_harm])

        loo_reliable = bool(cand["psis_loo_reliable"] and base["psis_loo_reliable"])
        loo_success = bool(delta_elpd > args.elpd_gate) if loo_reliable else pd.NA
        ppc_success = bool(
            ppc_rmse_gain >= args.ppc_rmse_gate
            and second_gain >= args.second_residual_gate
            and (not np.isfinite(worst_new_harm) or worst_new_harm <= args.max_new_residual_harm)
        )
        rows.append(
            {
                "pair": label,
                "candidate": candidate,
                "baseline": baseline,
                "candidate_diagnostics_ok": bool(cand["diagnostics_ok"]),
                "baseline_diagnostics_ok": bool(base["diagnostics_ok"]),
                "psis_loo_reliable_for_pair": loo_reliable,
                "delta_elpd_candidate_minus_baseline": delta_elpd,
                "loo_success": loo_success,
                "baseline_ppc_rmse": base["ppc_rmse"],
                "candidate_ppc_rmse": cand["ppc_rmse"],
                "ppc_rmse_gain": ppc_rmse_gain,
                "baseline_ppc_r": base["ppc_r"],
                "candidate_ppc_r": cand["ppc_r"],
                "ppc_r_gain": ppc_r_gain,
                "second_property_abs_residual_reduction": second_gain,
                "worst_first_or_both_abs_residual_harm": worst_new_harm,
                "ppc_success": ppc_success,
                "recommended_for_full_run": bool(
                    cand["diagnostics_ok"]
                    and (
                        (loo_reliable and loo_success)
                        or ((not loo_reliable) and ppc_success)
                    )
                ),
                "decision_basis": (
                    "elpd_and_ppc" if loo_reliable else "ppc_only_due_bad_pareto_k"
                ),
            }
        )
    pairwise = pd.DataFrame(rows)
    residuals = (
        pd.concat(residual_frames, ignore_index=True)
        if residual_frames else pd.DataFrame()
    )
    return pairwise, residuals


def write_frontier(stats_dir: Path, prefix: str) -> None:
    required = [
        "slider_loo_comparison.csv",
        "slider_ppc_correlation.csv",
        "slider_mcmc_model_summary.csv",
    ]
    if not all((stats_dir / path).exists() for path in required):
        return
    frontier = build_frontier_from_csvs(
        loo_csv=stats_dir / "slider_loo_comparison.csv",
        ppc_csv=stats_dir / "slider_ppc_correlation.csv",
        diagnostics_csv=stats_dir / "slider_mcmc_model_summary.csv",
        ppc_scope="all_cells",
        exclude_diagnostic_fail=True,
    )
    frontier.to_csv(stats_dir / f"{prefix}_pareto_scores.csv", index=False)
    frontier[frontier["posterior_pareto_frontier"]].to_csv(
        stats_dir / f"{prefix}_pareto_frontier.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--prefix", type=str, default="slider_speaker_pilot")
    parser.add_argument("--ppc-scope", type=str, default="all_cells")
    parser.add_argument("--max-r-hat", type=float, default=1.05)
    parser.add_argument("--max-pareto-k", type=float, default=0.7)
    parser.add_argument("--max-frac-pareto-gt-0-7", type=float, default=0.0)
    parser.add_argument("--elpd-gate", type=float, default=0.0)
    parser.add_argument("--ppc-rmse-gate", type=float, default=0.0)
    parser.add_argument("--second-residual-gate", type=float, default=0.05)
    parser.add_argument("--max-new-residual-harm", type=float, default=0.02)
    args = parser.parse_args()

    args.stats_dir.mkdir(parents=True, exist_ok=True)
    model_summary = summarize_model_level(args.stats_dir, args)
    ppc_by_condition = _read_csv(args.stats_dir / "slider_ppc_by_condition.csv")
    pairwise, residuals = pairwise_summary(model_summary, ppc_by_condition, args)

    model_summary.to_csv(args.stats_dir / f"{args.prefix}_model_decision_summary.csv", index=False)
    pairwise.to_csv(args.stats_dir / f"{args.prefix}_pairwise_decisions.csv", index=False)
    if not residuals.empty:
        residuals.to_csv(args.stats_dir / f"{args.prefix}_residual_changes.csv", index=False)
    write_frontier(args.stats_dir, args.prefix)

    print(f"Wrote pilot evaluation CSVs to {args.stats_dir}")
    if not pairwise.empty:
        print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
