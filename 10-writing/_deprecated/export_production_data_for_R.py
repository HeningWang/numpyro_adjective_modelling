"""
Export production-study posterior data to CSV for R plotting.

Run from the 10-writing/ directory:
    python export_production_data_for_R.py

Outputs (written to 10-writing/data/):
  - production_empirical.csv        : per-condition utterance-type proportions (human)
  - production_predictions.csv      : per-condition utterance-type proportions (4 models)
  - production_correlation.csv      : merged human + inc_hier for correlation plot
  - production_loo_comparison.csv   : LOO model comparison table
"""

import os, sys
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import numpy as np
import pandas as pd
import arviz as az

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
PROD_DIR = os.path.join(ROOT, "..", "05-modelling-production-data")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, PROD_DIR)
from helper import import_dataset

# ── Config ────────────────────────────────────────────────────────────────────
NW, NS, NC = 500, 500, 4
TAG = f"warmup{NW}_samples{NS}_chains{NC}"

# 2×2 speaker × semantics (all hierarchical)
MODEL_SPECS = {
    "incremental_recursive": f"mcmc_results_incremental_speaker_hier_{TAG}.nc",
    "incremental_static":    f"mcmc_results_incremental_static_speaker_hier_{TAG}.nc",
    "global_recursive":      f"mcmc_results_global_speaker_hier_{TAG}.nc",
    "global_static":         f"mcmc_results_global_static_speaker_hier_{TAG}.nc",
}

BEST_MODEL = "incremental_recursive"

FLAT_TO_CAT = {
    0: "D",   1: "DC",  2: "DCF",  3: "DF",   4: "DFC",
    5: "C",   6: "CD",  7: "CDF",  8: "CF",   9: "CFD",
    10: "F", 11: "FD", 12: "FDC", 13: "FC",  14: "FCD",
}

group_cols = ["relevant_property", "sharpness"]

# ── 1. Load empirical data ───────────────────────────────────────────────────
data = import_dataset()
df = data["df"].copy()
all_codes = sorted(df["annotation_seq_flat"].unique())

print(f"N observations: {len(df)}")
print(f"Conditions    : {sorted(df['relevant_property'].unique())}")
print(f"Sharpness     : {sorted(df['sharpness'].unique())}")
print(f"Utterance codes: {all_codes}")

# ── 2. Human condition-level proportions (with bootstrap CIs) ─────────────────
rng = np.random.default_rng(431)
n_boot = 2000

human_records = []
for cond_vals, sub in df.groupby(group_cols):
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

df_human = pd.DataFrame(human_records)
df_human.to_csv(os.path.join(DATA_DIR, "production_empirical.csv"), index=False)
print(f"[✓] production_empirical.csv  ({len(df_human)} rows)")

# ── 3. Model predictions per condition per utterance type ─────────────────────
all_model_dfs = []

for model_name, fname in MODEL_SPECS.items():
    fpath = os.path.join(PROD_DIR, "inference_data", fname)
    if not os.path.exists(fpath):
        print(f"[!] Skipping {model_name}: {fpath} not found")
        continue
    print(f"Loading {model_name}...")
    idata = az.from_netcdf(fpath)

    obs = idata.posterior_predictive["obs"]          # (chains, draws, N)
    pred = np.asarray(obs).reshape(-1, obs.shape[-1])  # (S, N)
    n_draws = pred.shape[0]

    cond_df = df[group_cols].reset_index(drop=True)

    # Subsample draws for speed (use max 500 draws)
    max_draws = min(500, n_draws)
    draw_idx = np.linspace(0, n_draws - 1, max_draws, dtype=int)

    model_records = []
    for d in draw_idx:
        tmp = cond_df.copy()
        tmp["annotation_seq_flat"] = pred[d, :]
        counts = tmp.groupby(group_cols + ["annotation_seq_flat"]).size().rename("n").reset_index()
        tots = tmp.groupby(group_cols).size().rename("n_total").reset_index()
        out = counts.merge(tots, on=group_cols)
        out["p"] = out["n"] / out["n_total"]
        model_records.append(out[group_cols + ["annotation_seq_flat", "p"]])

    model_df = pd.concat(model_records, ignore_index=True)
    model_summary = (
        model_df.groupby(group_cols + ["annotation_seq_flat"])["p"]
        .agg(
            model_mean="mean",
            model_lo=lambda x: np.percentile(x, 2.5),
            model_hi=lambda x: np.percentile(x, 97.5),
        )
        .reset_index()
    )
    model_summary["model"] = model_name
    model_summary["utterance_code"] = model_summary["annotation_seq_flat"].astype(int)
    model_summary["utterance_label"] = model_summary["utterance_code"].map(FLAT_TO_CAT)
    all_model_dfs.append(model_summary)
    print(f"  [✓] {model_name}: {len(model_summary)} rows")

if all_model_dfs:
    df_model_all = pd.concat(all_model_dfs, ignore_index=True)
    df_model_all.to_csv(os.path.join(DATA_DIR, "production_predictions.csv"), index=False)
    print(f"[✓] production_predictions.csv  ({len(df_model_all)} rows)")

# ── 4. Correlation data: human + best model merged ───────────────────────────
inc_hier_pred = [m for m in all_model_dfs if m["model"].iloc[0] == BEST_MODEL]
if inc_hier_pred:
    df_inc_hier = inc_hier_pred[0].copy()
    df_corr = df_human.merge(
        df_inc_hier[group_cols + ["utterance_code", "model_mean", "model_lo", "model_hi"]],
        on=group_cols + ["utterance_code"],
        how="outer",
    ).fillna(0)
    df_corr["condition"] = df_corr["relevant_property"] + " | " + df_corr["sharpness"]
    df_corr.to_csv(os.path.join(DATA_DIR, "production_correlation.csv"), index=False)
    print(f"[✓] production_correlation.csv  ({len(df_corr)} rows)")

# ── 5. LOO comparison table ──────────────────────────────────────────────────
idata_dict = {}
for model_name, fname in MODEL_SPECS.items():
    fpath = os.path.join(PROD_DIR, "inference_data", fname)
    if os.path.exists(fpath):
        idata_dict[model_name] = az.from_netcdf(fpath)

if idata_dict:
    comparison = az.compare(idata_dict, ic="loo", method="stacking", var_name="obs")
    comparison.to_csv(os.path.join(DATA_DIR, "production_loo_comparison.csv"))
    print(f"[✓] production_loo_comparison.csv")
    print(comparison.to_string())

print("\n[Done] All production CSVs exported to", DATA_DIR)
