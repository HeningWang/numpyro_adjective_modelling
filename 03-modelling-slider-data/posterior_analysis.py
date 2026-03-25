"""Posterior analysis for the slider experiment.

Replaces the Jupyter notebook + 10-writing/export_slider_data_for_R.py.
Uses shared/posterior_utils.py for common operations.

Run from the 03-modelling-slider-data/ directory:
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
from modelSpecification import import_dataset


# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_TAG = "warmup{nw}_samples{ns}_chains{nc}"

# 2×2 speaker × semantics (all hierarchical)
ALL_MODELS = {
    "incremental_recursive": "mcmc_results_incremental_speaker_hier_{tag}.nc",
    "incremental_static":    "mcmc_results_incremental_static_speaker_hier_{tag}.nc",
    "global_recursive":      "mcmc_results_global_speaker_hier_{tag}.nc",
    "global_static":         "mcmc_results_global_static_speaker_hier_{tag}.nc",
}

GROUP_COLS = ["relevant_property", "sharpness"]
BEST_MODEL = "incremental_recursive"

# Population-level parameters to diagnose per model
POP_VAR_NAMES = ["alpha", "bias", "sigma", "tau"]


# ── Dataset-specific: PPC for continuous slider ─────────────────────────────

def compute_ppc_slider(
    idata_dict, df, out_dir, fmt="pdf",
):
    """Compute and plot posterior predictive checks for slider data.

    For each model: per-observation predictions and condition-level aggregation.
    Returns a dict of {model_name: pred_mean_array} for CSV export.
    """
    os.makedirs(out_dir, exist_ok=True)
    pred_means = {}

    for model_name, idata in idata_dict.items():
        pp_flat = extract_pp_samples(idata, max_draws=None)
        pred_mean = np.clip(pp_flat.mean(axis=0), 0.0, 1.0)
        pred_means[model_name] = pred_mean

        # Per-condition aggregation
        groups = df.groupby(GROUP_COLS).groups
        cond_keys = sorted(groups.keys())
        pp_cond = np.stack(
            [pp_flat[:, list(groups[k])].mean(axis=1) for k in cond_keys],
            axis=1,
        )
        pred_cond_mean = pp_cond.mean(axis=0)
        pred_cond_lo = np.percentile(pp_cond, 2.5, axis=0)
        pred_cond_hi = np.percentile(pp_cond, 97.5, axis=0)

        # Empirical condition means
        emp_cond = {
            k: df.loc[list(idx), "human_slider"].mean()
            for k, idx in groups.items()
        }
        emp_vals = np.array([emp_cond[k] for k in cond_keys])
        labels = [f"{rp} / {sh}" for rp, sh in cond_keys]

        # PPC bar chart: empirical vs predicted
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        cond_df_emp = pd.DataFrame({
            "condition": labels,
            "mean": emp_vals,
        })
        cond_df_pred = pd.DataFrame({
            "condition": labels,
            "mean": pred_cond_mean,
            "lo": pred_cond_lo,
            "hi": pred_cond_hi,
        })
        x = np.arange(len(labels))
        axes[0].bar(x, cond_df_emp["mean"], color="#1f77b4", alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("Mean slider value")
        axes[0].set_title("Empirical")

        axes[1].bar(x, cond_df_pred["mean"], color="#ff7f0e", alpha=0.7)
        axes[1].errorbar(x, cond_df_pred["mean"],
                         yerr=[cond_df_pred["mean"] - cond_df_pred["lo"],
                               cond_df_pred["hi"] - cond_df_pred["mean"]],
                         fmt="none", color="black", capsize=3)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axes[1].set_title(f"Predicted ({model_name})")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"ppc_{model_name}.{fmt}"), bbox_inches="tight")
        plt.close(fig)

    return pred_means


# ── Dataset-specific: CSV exports ───────────────────────────────────────────

def export_csvs(df, idata_dict, pred_means, stats_dir):
    """Export slider CSVs: empirical, predictions, condition summary, LOO."""

    # 1. Empirical
    df_emp = df[["id", "item", "conditions", "sharpness", "relevant_property",
                 "combination", "human_slider"]].copy()
    out_emp = os.path.join(stats_dir, "slider_empirical.csv")
    df_emp.to_csv(out_emp, index=False)
    print(f"  [csv] slider_empirical.csv ({len(df_emp)} rows)")

    # 2. Per-observation predictions
    df_pred = df[["id", "item", "conditions", "sharpness", "relevant_property",
                  "human_slider"]].copy()
    for model_name, pred_mean in pred_means.items():
        df_pred[f"pred_{model_name}"] = pred_mean
    out_pred = os.path.join(stats_dir, "slider_predictions.csv")
    df_pred.to_csv(out_pred, index=False)
    print(f"  [csv] slider_predictions.csv ({len(df_pred)} rows)")

    # 3. Condition-level summary
    cond_emp = df.groupby(GROUP_COLS)["human_slider"].agg(["mean", "count"]).reset_index()
    cond_emp.columns = ["relevant_property", "sharpness", "emp_mean", "n"]

    for model_name, idata in idata_dict.items():
        pp = extract_pp_samples(idata, max_draws=None)
        groups = df.groupby(GROUP_COLS).groups
        rows = []
        for (rp, sh), idx in groups.items():
            idx_list = list(idx)
            cond_samples = pp[:, idx_list].mean(axis=1)
            rows.append({
                "relevant_property": rp,
                "sharpness": sh,
                f"pred_mean_{model_name}": cond_samples.mean(),
                f"pred_lo_{model_name}": np.percentile(cond_samples, 2.5),
                f"pred_hi_{model_name}": np.percentile(cond_samples, 97.5),
            })
        cond_model = pd.DataFrame(rows)
        cond_emp = cond_emp.merge(cond_model, on=GROUP_COLS, how="left")

    out_cond = os.path.join(stats_dir, "slider_condition_summary.csv")
    cond_emp.to_csv(out_cond, index=False)
    print(f"  [csv] slider_condition_summary.csv ({len(cond_emp)} rows)")

    return out_emp, out_pred, out_cond


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Slider posterior analysis.")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all 4).")
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
        model_specs = {k: v.format(tag=tag) for k, v in ALL_MODELS.items() if k in selected}
    else:
        model_specs = {k: v.format(tag=tag) for k, v in ALL_MODELS.items()}

    summary_blocks = []

    # ── 1. Load dataset ──
    print("Loading dataset...")
    states_train, empirical_train, df = import_dataset()
    df["human_slider"] = np.clip(df["prefer_first_1st"].to_numpy(dtype=float), 0.0, 1.0)
    summary_blocks.append(f"Dataset: {len(df)} observations")

    # ── 2. Load models ──
    print("Loading inference data...")
    idata_dict = load_models(model_specs, inference_dir)
    if not idata_dict:
        print("No inference data found. Exiting.")
        return

    # ── 3. MCMC diagnostics ──
    print("Running MCMC diagnostics...")
    for model_name, idata in idata_dict.items():
        s = run_mcmc_diagnostics(idata, model_name, var_names=POP_VAR_NAMES,
                                 out_dir=diag_dir, fmt=fmt)
        summary_blocks.append(s)

        s_intercept = plot_participant_intercepts(idata, model_name,
                                                  out_dir=diag_dir, fmt=fmt)
        if s_intercept:
            summary_blocks.append(s_intercept)

    # ── 4. Posterior predictive checks ──
    print("Computing posterior predictive checks...")
    pred_means = compute_ppc_slider(idata_dict, df, ppc_dir, fmt=fmt)

    # ── 5. LOO comparison ──
    print("Computing LOO comparison...")
    comparison, loo_results = compute_loo_comparison(idata_dict)
    loo_summary = plot_loo_comparison(comparison, loo_results, out_dir=stats_dir, fmt=fmt)
    summary_blocks.append(loo_summary)

    # Save LOO CSV
    comparison.to_csv(os.path.join(stats_dir, "slider_loo_comparison.csv"))
    print(f"  [csv] slider_loo_comparison.csv")

    # ── 6. Correlation scatter (best model) ──
    best = comparison.index[0]
    print(f"Plotting correlation scatter for best model: {best}")
    if best in idata_dict:
        pp_flat = extract_pp_samples(idata_dict[best], max_draws=None)
        groups = df.groupby(GROUP_COLS).groups
        cond_keys = sorted(groups.keys())
        pp_cond = np.stack(
            [pp_flat[:, list(groups[k])].mean(axis=1) for k in cond_keys],
            axis=1,
        )
        pred_cond_mean = pp_cond.mean(axis=0)
        pred_cond_lo = np.percentile(pp_cond, 2.5, axis=0)
        pred_cond_hi = np.percentile(pp_cond, 97.5, axis=0)
        emp_vals = np.array([
            df.loc[list(groups[k]), "human_slider"].mean() for k in cond_keys
        ])
        labels = [f"{rp} / {sh}" for rp, sh in cond_keys]

        corr_summary = plot_correlation_scatter(
            emp_vals, pred_cond_mean, pred_cond_lo, pred_cond_hi,
            labels=labels,
            title=f"Slider: {best}",
            out_path=os.path.join(ppc_dir, f"correlation_{best}.{fmt}"),
        )
        summary_blocks.append(corr_summary)

    # ── 7. Export CSVs ──
    print("Exporting CSVs...")
    export_csvs(df, idata_dict, pred_means, stats_dir)

    # ── 8. Copy to 10-writing/data/ if requested ──
    if args.export_for_r:
        r_data_dir = os.path.join(REPO_ROOT, "10-writing", "data")
        os.makedirs(r_data_dir, exist_ok=True)
        import shutil
        for fname in ["slider_empirical.csv", "slider_predictions.csv",
                      "slider_condition_summary.csv", "slider_loo_comparison.csv"]:
            src = os.path.join(stats_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(r_data_dir, fname))
        print(f"  [export] CSVs copied to {r_data_dir}")

    # ── 9. Summary ──
    save_summary_text(os.path.join(out_dir, "summary.txt"), summary_blocks)
    print("\nDone.")


if __name__ == "__main__":
    main()
