"""Compare forward-grid speaker optima with fitted posterior parameters.

This is a compact descriptive audit. It does not read NetCDF artifacts and does
not run inference; it consumes CSV summaries exported by the forward audit and
posterior/heldout pilot analyses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_FORWARD_DIR = THIS_DIR / "results_speaker_variant_forward_audit" / "stats"
DEFAULT_POSTERIOR_DIR = THIS_DIR / "results_speaker_ablation_pilot" / "stats"
DEFAULT_HELDOUT_DIR = THIS_DIR / "results_heldout_pilot" / "stats"
DEFAULT_OUT_DIR = DEFAULT_FORWARD_DIR

VARIANT_TO_POSTERIOR_MODEL = {
    "greedy_incremental_stop": "incremental_recursive",
    "planned_usefulness_order": "planned_usefulness_order",
    "planned_usefulness_signed_order": "planned_usefulness_signed_order",
    "planned_usefulness_mixture": "planned_usefulness_mixture",
}

MODEL_TO_GREEDY_PAIR = {
    "planned_usefulness_order": "planned_vs_greedy_recursive",
    "planned_usefulness_signed_order": "signed_order_vs_greedy_recursive",
    "planned_usefulness_mixture": "mixture_vs_greedy_recursive",
}

PARAMETERS = [
    "alpha",
    "bias",
    "usefulness_order_scale",
    "signed_order_scale",
    "planned_mixture_weight",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def posterior_parameter_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(columns=["model"])
    sub = diagnostics[diagnostics["parameter"].isin(PARAMETERS)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["model"])
    wide = sub.pivot_table(index="model", columns="parameter", values="mean", aggfunc="first")
    wide.columns = [f"posterior_{col}" for col in wide.columns]
    return wide.reset_index()


def pairwise_by_pair(pairwise: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if pairwise.empty or "pair" not in pairwise.columns:
        return pd.DataFrame(columns=["pair"])
    keep = [
        "pair",
        "delta_elpd_candidate_minus_baseline",
        "delta_heldout_elpd_candidate_minus_baseline",
        "ppc_rmse_gain",
        "second_property_abs_residual_reduction",
        "ppc_success",
        "recommended_for_full_run",
    ]
    keep = [col for col in keep if col in pairwise.columns]
    out = pairwise[keep].copy()
    return out.rename(columns={col: f"{prefix}_{col}" for col in keep if col != "pair"})


def build_gap_summary(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    forward_best = read_csv(args.forward_dir / "slider_speaker_variant_grid_best_by_variant.csv")
    diagnostics = read_csv(args.posterior_dir / "slider_mcmc_diagnostics.csv")
    posterior_pairwise = read_csv(args.posterior_dir / "slider_speaker_ablation_eval_pairwise_decisions.csv")
    heldout_pairwise = read_csv(args.heldout_dir / "slider_heldout_eval_pairwise_decisions.csv")

    if forward_best.empty:
        raise FileNotFoundError(args.forward_dir / "slider_speaker_variant_grid_best_by_variant.csv")

    best = forward_best[forward_best["selection_criterion"].eq("condition_rmse")].copy()
    best["posterior_model"] = best["variant"].map(VARIANT_TO_POSTERIOR_MODEL)
    best = best[best["posterior_model"].notna()].copy()
    params = posterior_parameter_table(diagnostics)
    out = best.merge(params, left_on="posterior_model", right_on="model", how="left")

    for param in PARAMETERS:
        if param in out.columns and f"posterior_{param}" in out.columns:
            out[f"{param}_gap_grid_minus_posterior"] = (
                pd.to_numeric(out[param], errors="coerce")
                - pd.to_numeric(out[f"posterior_{param}"], errors="coerce")
            )

    out["greedy_comparison_pair"] = out["posterior_model"].map(MODEL_TO_GREEDY_PAIR)
    out = out.merge(
        pairwise_by_pair(posterior_pairwise, "posterior"),
        left_on="greedy_comparison_pair",
        right_on="pair",
        how="left",
    ).drop(columns=["pair"], errors="ignore")
    out = out.merge(
        pairwise_by_pair(heldout_pairwise, "heldout"),
        left_on="greedy_comparison_pair",
        right_on="pair",
        how="left",
    ).drop(columns=["pair"], errors="ignore")

    keep = [
        "variant",
        "posterior_model",
        "grid_id",
        "condition_rmse",
        "condition_abs_zrdc_residual",
        "alpha",
        "posterior_alpha",
        "alpha_gap_grid_minus_posterior",
        "order_bias",
        "posterior_bias",
        "bias_gap_grid_minus_posterior",
        "usefulness_order_scale",
        "posterior_usefulness_order_scale",
        "usefulness_order_scale_gap_grid_minus_posterior",
        "signed_order_scale",
        "posterior_signed_order_scale",
        "signed_order_scale_gap_grid_minus_posterior",
        "planned_mixture_weight",
        "posterior_planned_mixture_weight",
        "planned_mixture_weight_gap_grid_minus_posterior",
        "greedy_comparison_pair",
        "posterior_second_property_abs_residual_reduction",
        "posterior_ppc_success",
        "posterior_recommended_for_full_run",
        "heldout_delta_heldout_elpd_candidate_minus_baseline",
        "heldout_second_property_abs_residual_reduction",
        "heldout_ppc_success",
        "heldout_recommended_for_full_run",
    ]
    keep = [col for col in keep if col in out.columns]
    out = out[keep].sort_values(["condition_rmse", "condition_abs_zrdc_residual"])

    planned = out[out["variant"].ne("greedy_incremental_stop")]
    best_forward = planned.iloc[0] if not planned.empty else out.iloc[0]
    posterior_second = pd.to_numeric(
        planned.get("posterior_second_property_abs_residual_reduction", pd.Series(dtype=float)),
        errors="coerce",
    )
    heldout_second = pd.to_numeric(
        planned.get("heldout_second_property_abs_residual_reduction", pd.Series(dtype=float)),
        errors="coerce",
    )
    decision = pd.DataFrame(
        [
            {
                "best_forward_variant": best_forward["variant"],
                "best_forward_condition_rmse": best_forward["condition_rmse"],
                "best_forward_zrdc_abs_residual": best_forward["condition_abs_zrdc_residual"],
                "best_posterior_second_property_residual_gain": (
                    float(np.nanmax(posterior_second)) if posterior_second.notna().any() else np.nan
                ),
                "best_heldout_second_property_residual_gain": (
                    float(np.nanmax(heldout_second)) if heldout_second.notna().any() else np.nan
                ),
                "ppc_second_property_gate": args.second_residual_gate,
                "existing_variant_passed_gate": bool(
                    (
                        posterior_second.ge(args.second_residual_gate).any()
                        if posterior_second.notna().any() else False
                    )
                    or (
                        heldout_second.ge(args.second_residual_gate).any()
                        if heldout_second.notna().any() else False
                    )
                ),
                "run_more_existing_variant_mcmc": False,
                "next_action": (
                    "design constrained_or_prior_pulled_planned_mixture before more GPU MCMC"
                ),
            }
        ]
    )
    return out, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-dir", type=Path, default=DEFAULT_FORWARD_DIR)
    parser.add_argument("--posterior-dir", type=Path, default=DEFAULT_POSTERIOR_DIR)
    parser.add_argument("--heldout-dir", type=Path, default=DEFAULT_HELDOUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--second-residual-gate", type=float, default=0.05)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gap, decision = build_gap_summary(args)
    gap_path = args.out_dir / "slider_speaker_parameter_gap_summary.csv"
    decision_path = args.out_dir / "slider_speaker_parameter_gap_decision.csv"
    gap.to_csv(gap_path, index=False)
    decision.to_csv(decision_path, index=False)
    print(f"Wrote {gap_path}")
    print(f"Wrote {decision_path}")
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
