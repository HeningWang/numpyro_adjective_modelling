"""
Export slider-study posterior data to CSV for R plotting.

Run from the 10-writing/ directory:
    python export_slider_data_for_R.py

Outputs (written to 10-writing/data/):
  - slider_empirical.csv        : raw empirical slider data (dimension_color subset)
  - slider_predictions.csv      : per-observation posterior-predictive means for all 4 models
  - slider_condition_summary.csv: condition-level empirical means + predicted means/CIs per model
  - slider_loo_comparison.csv   : LOO model comparison table
"""

import os, sys
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import numpy as np
import pandas as pd
import arviz as az

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SLIDER_DIR = os.path.join(ROOT, "..", "03-modelling-slider-data")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, SLIDER_DIR)
from modelSpecification import import_dataset

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

# ── 1. Load empirical data ───────────────────────────────────────────────────
states_train, empirical_train, df = import_dataset()
df["human_slider"] = np.clip(df["prefer_first_1st"].to_numpy(dtype=float), 0.0, 1.0)

# Save raw empirical data
df_out = df[["id", "item", "conditions", "sharpness", "relevant_property",
             "combination", "human_slider"]].copy()
df_out.to_csv(os.path.join(DATA_DIR, "slider_empirical.csv"), index=False)
print(f"[✓] slider_empirical.csv  ({len(df_out)} rows)")

# ── 2. Load each model and extract posterior predictive means ─────────────────
pred_cols = {}
loo_results = {}

for model_name, fname in MODEL_SPECS.items():
    fpath = os.path.join(SLIDER_DIR, "inference_data", fname)
    if not os.path.exists(fpath):
        print(f"[!] Skipping {model_name}: {fpath} not found")
        continue
    idata = az.from_netcdf(fpath)

    # Posterior predictive mean per observation
    pp = np.asarray(idata.posterior_predictive["obs"])      # (chains, draws, N)
    pp_flat = pp.reshape(-1, pp.shape[-1])                  # (S, N)
    pred_mean = np.clip(pp_flat.mean(axis=0), 0.0, 1.0)
    pred_cols[f"pred_{model_name}"] = pred_mean

    # LOO
    loo = az.loo(idata, var_name="obs", pointwise=True)
    loo_results[model_name] = {
        "elpd_loo": loo.elpd_loo,
        "se":       loo.se,
        "p_loo":    loo.p_loo,
    }
    print(f"[✓] {model_name}: elpd={loo.elpd_loo:.1f} ± {loo.se:.1f}")

# Save per-observation predictions
df_pred = df[["id", "item", "conditions", "sharpness", "relevant_property",
              "human_slider"]].copy()
for col, vals in pred_cols.items():
    df_pred[col] = vals
df_pred.to_csv(os.path.join(DATA_DIR, "slider_predictions.csv"), index=False)
print(f"[✓] slider_predictions.csv  ({len(df_pred)} rows)")

# ── 3. Condition-level summary ────────────────────────────────────────────────
group_cols = ["relevant_property", "sharpness"]
cond_emp = df.groupby(group_cols)["human_slider"].agg(["mean", "count"]).reset_index()
cond_emp.columns = ["relevant_property", "sharpness", "emp_mean", "n"]

for model_name, fname in MODEL_SPECS.items():
    fpath = os.path.join(SLIDER_DIR, "inference_data", fname)
    if not os.path.exists(fpath):
        continue
    idata = az.from_netcdf(fpath)
    pp = np.asarray(idata.posterior_predictive["obs"]).reshape(-1, len(df))

    groups = df.groupby(group_cols).groups
    rows = []
    for (rp, sh), idx in groups.items():
        idx_list = list(idx)
        cond_samples = pp[:, idx_list].mean(axis=1)
        rows.append({
            "relevant_property": rp,
            "sharpness": sh,
            f"pred_mean_{model_name}": cond_samples.mean(),
            f"pred_lo_{model_name}":   np.percentile(cond_samples, 2.5),
            f"pred_hi_{model_name}":   np.percentile(cond_samples, 97.5),
        })
    cond_model = pd.DataFrame(rows)
    cond_emp = cond_emp.merge(cond_model, on=group_cols, how="left")

cond_emp.to_csv(os.path.join(DATA_DIR, "slider_condition_summary.csv"), index=False)
print(f"[✓] slider_condition_summary.csv  ({len(cond_emp)} rows)")

# ── 4. LOO comparison table ──────────────────────────────────────────────────
idata_dict = {}
for model_name, fname in MODEL_SPECS.items():
    fpath = os.path.join(SLIDER_DIR, "inference_data", fname)
    if os.path.exists(fpath):
        idata_dict[model_name] = az.from_netcdf(fpath)

if idata_dict:
    comparison = az.compare(idata_dict, ic="loo", method="stacking", var_name="obs")
    comparison.to_csv(os.path.join(DATA_DIR, "slider_loo_comparison.csv"))
    print(f"[✓] slider_loo_comparison.csv")
    print(comparison.to_string())

print("\n[Done] All slider CSVs exported to", DATA_DIR)
