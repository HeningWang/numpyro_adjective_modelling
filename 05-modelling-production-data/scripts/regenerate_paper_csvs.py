"""Regenerate production CSVs from the 2x2best (paper-reported) posteriors.

The paper uses csv=0.85, fsv=0.70 with 5000 warmup / 2000 samples.
NC files: mcmc_results_2x2best_*_warmup5000_samples2000_chains4.nc

Outputs:
  - 10-writing/data/production_predictions.csv
  - 10-writing/data/production_correlation.csv
  - 10-writing/data/production_empirical.csv
  - 10-writing/data/production_loo_comparison.csv

Run from 05-modelling-production-data/:
    python scripts/regenerate_paper_csvs.py
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import sys
import pathlib

import numpy as np
import pandas as pd
import arviz as az

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "05-modelling-production-data"))

from shared.posterior_utils import extract_pp_samples, compute_loo_comparison
import helper
# 2x2best was fit on dc subset only
helper.CONDITIONS_OF_INTEREST = ("erdc", "zrdc", "brdc")
from helper import import_dataset

# ── Config ────────────────────────────────────────────────────────────────────

INFERENCE_DIR = REPO_ROOT / "05-modelling-production-data" / "inference_data"
OUTPUT_DIR = REPO_ROOT / "10-writing" / "data"

MODELS = {
    "incremental_recursive": "mcmc_results_2x2best_incremental_recursive_warmup5000_samples2000_chains4.nc",
    "incremental_static": "mcmc_results_2x2best_incremental_static_warmup5000_samples2000_chains4.nc",
    "global_recursive": "mcmc_results_2x2best_global_recursive_warmup5000_samples2000_chains4.nc",
    "global_static": "mcmc_results_2x2best_global_static_warmup5000_samples2000_chains4.nc",
}

GROUP_COLS = ["relevant_property", "sharpness"]

FLAT_TO_CAT = {
    0: "D",  1: "DC",  2: "DCF",  3: "DF",  4: "DFC",
    5: "C",  6: "CD",  7: "CDF",  8: "CF",  9: "CFD",
    10: "F", 11: "FD", 12: "FDC", 13: "FC", 14: "FCD",
}


# ── Functions ─────────────────────────────────────────────────────────────────

def compute_condition_proportions(pred_draws, cond_df, max_draws=500):
    """Compute per-condition utterance-type proportions from posterior draws."""
    n_draws, n_obs = pred_draws.shape
    use_draws = min(n_draws, max_draws)
    rng = np.random.default_rng(42)
    idx = rng.choice(n_draws, size=use_draws, replace=False)
    pred_sub = pred_draws[idx]

    records = []
    for (rp, sh), grp in cond_df.groupby(GROUP_COLS):
        grp_idx = grp.index.values
        for utt_code in range(15):
            props = (pred_sub[:, grp_idx] == utt_code).mean(axis=1)
            records.append({
                "relevant_property": rp,
                "sharpness": sh,
                "annotation_seq_flat": utt_code,
                "model_mean": float(np.mean(props)),
                "model_lo": float(np.percentile(props, 2.5)),
                "model_hi": float(np.percentile(props, 97.5)),
            })
    return pd.DataFrame(records)


def compute_human_proportions(cond_df, n_boot=2000):
    """Bootstrap human utterance proportions."""
    rng = np.random.default_rng(431)
    records = []
    for (rp, sh), grp in cond_df.groupby(GROUP_COLS):
        obs = grp["annotation_seq_flat"].values
        n = len(obs)
        for utt_code in range(15):
            boot_props = []
            for _ in range(n_boot):
                sample = rng.choice(obs, size=n, replace=True)
                boot_props.append((sample == utt_code).mean())
            boot_props = np.array(boot_props)
            records.append({
                "relevant_property": rp,
                "sharpness": sh,
                "utterance_code": utt_code,
                "utterance_label": FLAT_TO_CAT[utt_code],
                "human_mean": float(np.mean(boot_props)),
                "human_lo": float(np.percentile(boot_props, 2.5)),
                "human_hi": float(np.percentile(boot_props, 97.5)),
                "n": n,
            })
    return pd.DataFrame(records)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    data = import_dataset()
    cond_df = data["df"].reset_index(drop=True)
    print(f"  N = {len(cond_df)} observations")

    # Load models
    print("Loading 2x2best NC files...")
    idata_dict = {}
    for name, fname in MODELS.items():
        path = INFERENCE_DIR / fname
        if not path.exists():
            print(f"  [SKIP] {fname} not found")
            continue
        idata_dict[name] = az.from_netcdf(str(path))
        print(f"  [OK] {name}")

    # Compute LOO comparison
    print("Computing LOO comparison...")
    comparison = az.compare(idata_dict, ic="loo")
    out_loo = OUTPUT_DIR / "production_loo_comparison.csv"
    comparison.to_csv(str(out_loo))
    print(f"  -> {out_loo}")

    # Compute human proportions
    print("Computing human bootstrap proportions...")
    df_human = compute_human_proportions(cond_df)
    out_emp = OUTPUT_DIR / "production_empirical.csv"
    df_human.to_csv(str(out_emp), index=False)
    print(f"  -> {out_emp} ({len(df_human)} rows)")

    # Compute model predictions
    print("Computing model predictions...")
    all_predictions = []
    model_summaries = {}
    for model_name, idata in idata_dict.items():
        print(f"  {model_name}...")
        pp_flat = extract_pp_samples(idata, max_draws=500)
        summary = compute_condition_proportions(pp_flat, cond_df, max_draws=500)
        summary["utterance_code"] = summary["annotation_seq_flat"].astype(int)
        summary["utterance_label"] = summary["utterance_code"].map(FLAT_TO_CAT)
        summary["model"] = model_name
        all_predictions.append(summary)
        model_summaries[model_name] = summary

    df_pred = pd.concat(all_predictions, ignore_index=True)
    out_pred = OUTPUT_DIR / "production_predictions.csv"
    df_pred.to_csv(str(out_pred), index=False)
    print(f"  -> {out_pred} ({len(df_pred)} rows)")

    # Correlation CSV (best model merged with human)
    best_model = "incremental_recursive"
    df_best = model_summaries[best_model]
    df_corr = df_human.merge(
        df_best[GROUP_COLS + ["utterance_code", "model_mean", "model_lo", "model_hi"]],
        on=GROUP_COLS + ["utterance_code"],
        how="outer",
    ).fillna(0)
    df_corr["condition"] = df_corr["relevant_property"] + " | " + df_corr["sharpness"]
    out_corr = OUTPUT_DIR / "production_correlation.csv"
    df_corr.to_csv(str(out_corr), index=False)
    print(f"  -> {out_corr} ({len(df_corr)} rows)")

    # Report R²
    mask = df_corr["human_mean"] > 0
    r = np.corrcoef(df_corr.loc[mask, "human_mean"], df_corr.loc[mask, "model_mean"])[0, 1]
    print(f"\n  R² (non-zero cells) = {r**2:.3f}  (r = {r:.3f})")

    # Report top residuals
    df_corr["residual"] = df_corr["model_mean"] - df_corr["human_mean"]
    df_corr["abs_residual"] = df_corr["residual"].abs()
    top7 = df_corr.nlargest(7, "abs_residual")
    print("\n  Top 7 residuals:")
    print(f"  {'Condition':<20s} {'Sharp':<8s} {'Utt':<5s} {'Emp':>6s} {'Model':>6s} {'Δ':>7s}")
    for _, row in top7.iterrows():
        print(f"  {row['relevant_property']:<20s} {row['sharpness']:<8s} "
              f"{row['utterance_label']:<5s} {row['human_mean']:6.3f} "
              f"{row['model_mean']:6.3f} {row['residual']:+7.3f}")


if __name__ == "__main__":
    main()
