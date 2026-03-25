"""Posterior analysis for the production experiment.

Replaces the Jupyter notebook + 10-writing/export_production_data_for_R.py
+ 10-writing/export_appendix_data_for_R.py.
Uses shared/posterior_utils.py for common operations.

Run from the 05-modelling-production-data/ directory:
    python posterior_analysis.py
    python posterior_analysis.py --export-for-r
    python posterior_analysis.py --models incremental,global --format png
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import sys
import argparse

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

# Add repo root for shared imports
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)

from shared.posterior_utils import (
    load_models,
    run_mcmc_diagnostics,
    plot_participant_intercepts,
    compute_loo_comparison,
    plot_loo_comparison,
    plot_correlation_scatter,
    extract_pp_samples,
    save_summary_text,
)
from helper import import_dataset


# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_TAG = "warmup{nw}_samples{ns}_chains{nc}"

# 2×2 speaker × semantics (all hierarchical)
MAIN_MODELS = {
    "incremental_recursive": "mcmc_results_incremental_speaker_hier_{tag}.nc",
    "incremental_static":    "mcmc_results_incremental_static_speaker_hier_{tag}.nc",
    "global_recursive":      "mcmc_results_global_speaker_hier_{tag}.nc",
    "global_static":         "mcmc_results_global_static_speaker_hier_{tag}.nc",
}

# Ablation models (production only)
ABLATION_MODELS = {
    "incremental_lookahead": "mcmc_results_incremental_lookahead_speaker_hier_{tag}.nc",
    "incremental_lm_only":   "mcmc_results_incremental_lm_only_speaker_hier_{tag}.nc",
    "incremental_rsa_only":  "mcmc_results_incremental_rsa_only_speaker_hier_{tag}.nc",
    "incremental_extended":  "mcmc_results_incremental_extended_speaker_hier_{tag}.nc",
}

GROUP_COLS = ["relevant_property", "sharpness"]

FLAT_TO_CAT = {
    0: "D",   1: "DC",  2: "DCF",  3: "DF",   4: "DFC",
    5: "C",   6: "CD",  7: "CDF",  8: "CF",   9: "CFD",
    10: "F", 11: "FD", 12: "FDC", 13: "FC",  14: "FCD",
}

# Population-level parameters to diagnose
POP_VAR_NAMES = ["alpha", "alpha_D", "alpha_C", "alpha_F",
                 "log_beta", "gamma", "epsilon", "mu_C", "mu_F", "tau"]


# ── Dataset-specific: PPC for categorical production data ───────────────────

def compute_condition_proportions(pred_draws, cond_df, max_draws=500):
    """Compute per-condition utterance-type proportions from posterior draws.

    Parameters
    ----------
    pred_draws : np.ndarray (S, N)
        Posterior predictive draws (integer utterance codes).
    cond_df : pd.DataFrame
        DataFrame with GROUP_COLS columns, aligned with pred_draws axis=1.
    max_draws : int
        Subsample to this many draws for speed.

    Returns
    -------
    pd.DataFrame with columns: relevant_property, sharpness,
        annotation_seq_flat, model_mean, model_lo, model_hi
    """
    n_draws = pred_draws.shape[0]
    draw_idx = np.linspace(0, n_draws - 1, min(max_draws, n_draws), dtype=int)

    records = []
    for d in draw_idx:
        tmp = cond_df.copy()
        tmp["annotation_seq_flat"] = pred_draws[d, :]
        counts = tmp.groupby(GROUP_COLS + ["annotation_seq_flat"]).size().rename("n").reset_index()
        tots = tmp.groupby(GROUP_COLS).size().rename("n_total").reset_index()
        out = counts.merge(tots, on=GROUP_COLS)
        out["p"] = out["n"] / out["n_total"]
        records.append(out[GROUP_COLS + ["annotation_seq_flat", "p"]])

    model_df = pd.concat(records, ignore_index=True)
    model_summary = (
        model_df.groupby(GROUP_COLS + ["annotation_seq_flat"])["p"]
        .agg(
            model_mean="mean",
            model_lo=lambda x: np.percentile(x, 2.5),
            model_hi=lambda x: np.percentile(x, 97.5),
        )
        .reset_index()
    )
    return model_summary


def compute_human_proportions(df, all_codes, n_boot=2000, seed=431):
    """Bootstrap human condition-level utterance proportions.

    Returns
    -------
    pd.DataFrame with columns: relevant_property, sharpness, utterance_code,
        utterance_label, human_mean, human_lo, human_hi, n
    """
    rng = np.random.default_rng(seed)
    human_records = []
    for cond_vals, sub in df.groupby(GROUP_COLS):
        labels = sub["annotation_seq_flat"].to_numpy()
        n = len(labels)
        boot_props = {code: [] for code in all_codes}
        for _ in range(n_boot):
            sample = rng.choice(labels, size=n, replace=True)
            for code in all_codes:
                boot_props[code].append(np.mean(sample == code))
        for code in all_codes:
            human_mean = np.mean(labels == code)
            bp = np.array(boot_props[code])
            human_records.append({
                "relevant_property": cond_vals[0],
                "sharpness": cond_vals[1],
                "utterance_code": int(code),
                "utterance_label": FLAT_TO_CAT[int(code)],
                "human_mean": human_mean,
                "human_lo": np.percentile(bp, 2.5),
                "human_hi": np.percentile(bp, 97.5),
                "n": n,
            })
    return pd.DataFrame(human_records)


def compute_ppc_production(idata_dict, df, out_dir, fmt="pdf"):
    """Compute and plot PPC for production data (categorical).

    Returns dict of {model_name: model_summary_df}.
    """
    os.makedirs(out_dir, exist_ok=True)
    cond_df = df[GROUP_COLS].reset_index(drop=True)
    model_summaries = {}

    for model_name, idata in idata_dict.items():
        pp_flat = extract_pp_samples(idata, max_draws=500)
        model_summary = compute_condition_proportions(pp_flat, cond_df, max_draws=500)
        model_summary["utterance_code"] = model_summary["annotation_seq_flat"].astype(int)
        model_summary["utterance_label"] = model_summary["utterance_code"].map(FLAT_TO_CAT)
        model_summary["model"] = model_name
        model_summaries[model_name] = model_summary

    return model_summaries


# ── Dataset-specific: CSV exports ───────────────────────────────────────────

def export_csvs(df_human, model_summaries, comparison, stats_dir,
                ablation_comparison=None, best_model=None):
    """Export production CSVs: empirical, predictions, correlation, LOO."""

    # 1. Empirical
    out_emp = os.path.join(stats_dir, "production_empirical.csv")
    df_human.to_csv(out_emp, index=False)
    print(f"  [csv] production_empirical.csv ({len(df_human)} rows)")

    # 2. Model predictions (all models stacked)
    if model_summaries:
        all_model_dfs = list(model_summaries.values())
        df_model_all = pd.concat(all_model_dfs, ignore_index=True)
        out_pred = os.path.join(stats_dir, "production_predictions.csv")
        df_model_all.to_csv(out_pred, index=False)
        print(f"  [csv] production_predictions.csv ({len(df_model_all)} rows)")

    # 3. LOO comparison
    out_loo = os.path.join(stats_dir, "production_loo_comparison.csv")
    comparison.to_csv(out_loo)
    print(f"  [csv] production_loo_comparison.csv")

    # 4. Ablation LOO (if available)
    if ablation_comparison is not None:
        out_abl = os.path.join(stats_dir, "production_ablation_loo_comparison.csv")
        ablation_comparison.to_csv(out_abl)
        print(f"  [csv] production_ablation_loo_comparison.csv")

    # 5. Correlation data (best model merged with human)
    if best_model and best_model in model_summaries:
        df_best = model_summaries[best_model]
        df_corr = df_human.merge(
            df_best[GROUP_COLS + ["utterance_code", "model_mean", "model_lo", "model_hi"]],
            on=GROUP_COLS + ["utterance_code"],
            how="outer",
        ).fillna(0)
        df_corr["condition"] = df_corr["relevant_property"] + " | " + df_corr["sharpness"]
        out_corr = os.path.join(stats_dir, "production_correlation.csv")
        df_corr.to_csv(out_corr, index=False)
        print(f"  [csv] production_correlation.csv ({len(df_corr)} rows)")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Production posterior analysis.")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated main model names (default: all 4).")
    parser.add_argument("--ablations", type=str, default=None,
                        help="Comma-separated ablation model names (default: all available).")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--export-for-r", action="store_true",
                        help="Copy CSVs to 10-writing/data/.")
    parser.add_argument("--out-dir", type=str, default="results",
                        help="Output directory for figures and stats.")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"],
                        help="Figure format.")
    args = parser.parse_args()

    fmt = args.format
    out_dir = args.out_dir
    diag_dir = os.path.join(out_dir, "diagnostics")
    ppc_dir = os.path.join(out_dir, "ppc")
    stats_dir = os.path.join(out_dir, "stats")
    for d in [diag_dir, ppc_dir, stats_dir]:
        os.makedirs(d, exist_ok=True)

    tag = DEFAULT_TAG.format(nw=args.warmup, ns=args.samples, nc=args.chains)
    inference_dir = "./inference_data"

    # Resolve model specs
    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        main_specs = {k: v.format(tag=tag) for k, v in MAIN_MODELS.items() if k in selected}
    else:
        main_specs = {k: v.format(tag=tag) for k, v in MAIN_MODELS.items()}

    if args.ablations:
        selected_abl = [m.strip() for m in args.ablations.split(",")]
        abl_specs = {k: v.format(tag=tag) for k, v in ABLATION_MODELS.items() if k in selected_abl}
    else:
        abl_specs = {k: v.format(tag=tag) for k, v in ABLATION_MODELS.items()}

    summary_blocks = []

    # ── 1. Load dataset ──
    print("Loading dataset...")
    data = import_dataset()
    df = data["df"].copy()
    all_codes = sorted(df["annotation_seq_flat"].unique())
    summary_blocks.append(f"Dataset: {len(df)} observations, {len(all_codes)} utterance types")

    # ── 2. Load models ──
    print("Loading main inference data...")
    idata_dict = load_models(main_specs, inference_dir)

    print("Loading ablation inference data...")
    idata_ablation = load_models(abl_specs, inference_dir)

    if not idata_dict:
        print("No main inference data found. Exiting.")
        return

    # ── 3. MCMC diagnostics ──
    print("Running MCMC diagnostics...")
    all_idata = {**idata_dict, **idata_ablation}
    for model_name, idata in all_idata.items():
        # Determine which pop vars exist for this model
        pop_vars = [v for v in POP_VAR_NAMES if v in idata.posterior.data_vars]
        s = run_mcmc_diagnostics(idata, model_name, var_names=pop_vars,
                                 out_dir=diag_dir, fmt=fmt)
        summary_blocks.append(s)

        s_intercept = plot_participant_intercepts(idata, model_name,
                                                  out_dir=diag_dir, fmt=fmt)
        if s_intercept:
            summary_blocks.append(s_intercept)

    # ── 4. Posterior predictive checks ──
    print("Computing posterior predictive checks (main models)...")
    model_summaries = compute_ppc_production(idata_dict, df, ppc_dir, fmt=fmt)

    # ── 5. Human proportions (bootstrap) ──
    print("Computing human bootstrap CIs...")
    df_human = compute_human_proportions(df, all_codes)

    # ── 6. LOO comparison (main models) ──
    print("Computing LOO comparison (main models)...")
    comparison, loo_results = compute_loo_comparison(idata_dict)
    loo_summary = plot_loo_comparison(comparison, loo_results, out_dir=stats_dir, fmt=fmt)
    summary_blocks.append(loo_summary)

    # ── 7. LOO comparison (ablation — all models combined) ──
    ablation_comparison = None
    if idata_ablation:
        print("Computing LOO comparison (ablation + main)...")
        # Exclude rsa_only from LOO if PSIS fails (known issue)
        abl_for_loo = {k: v for k, v in idata_ablation.items()
                       if k != "incremental_rsa_only"}
        combined = {**idata_dict, **abl_for_loo}
        ablation_comparison, abl_loo = compute_loo_comparison(combined)
        abl_loo_summary = plot_loo_comparison(
            ablation_comparison, abl_loo,
            out_dir=os.path.join(stats_dir, "ablation"), fmt=fmt,
        )
        summary_blocks.append(f"Ablation LOO:\n{abl_loo_summary}")

    # ── 8. Correlation scatter (best model) ──
    best = comparison.index[0]
    print(f"Plotting correlation scatter for best model: {best}")
    if best in model_summaries:
        df_best = model_summaries[best]
        merged = df_human.merge(
            df_best[GROUP_COLS + ["utterance_code", "model_mean", "model_lo", "model_hi"]],
            on=GROUP_COLS + ["utterance_code"],
            how="outer",
        ).fillna(0)

        labels = (merged["relevant_property"] + " | " + merged["sharpness"]
                  + " | " + merged["utterance_label"]).tolist()
        corr_summary = plot_correlation_scatter(
            emp_vals=merged["human_mean"].values,
            pred_mean=merged["model_mean"].values,
            pred_lo=merged["model_lo"].values,
            pred_hi=merged["model_hi"].values,
            emp_lo=merged["human_lo"].values,
            emp_hi=merged["human_hi"].values,
            labels=[],  # too many labels for legend
            title=f"Production: {best}",
            out_path=os.path.join(ppc_dir, f"correlation_{best}.{fmt}"),
        )
        summary_blocks.append(corr_summary)

    # ── 9. Export CSVs ──
    print("Exporting CSVs...")
    export_csvs(df_human, model_summaries, comparison, stats_dir,
                ablation_comparison=ablation_comparison, best_model=best)

    # ── 10. Copy to 10-writing/data/ if requested ──
    if args.export_for_r:
        r_data_dir = os.path.join(REPO_ROOT, "10-writing", "data")
        os.makedirs(r_data_dir, exist_ok=True)
        import shutil
        for fname in ["production_empirical.csv", "production_predictions.csv",
                      "production_correlation.csv", "production_loo_comparison.csv",
                      "production_ablation_loo_comparison.csv"]:
            src = os.path.join(stats_dir, fname)
            if not os.path.exists(src):
                # Check ablation subdir
                src = os.path.join(stats_dir, "ablation", fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(r_data_dir, fname))
        print(f"  [export] CSVs copied to {r_data_dir}")

    # ── 11. Summary ──
    save_summary_text(os.path.join(out_dir, "summary.txt"), summary_blocks)
    print("\nDone.")


if __name__ == "__main__":
    main()
