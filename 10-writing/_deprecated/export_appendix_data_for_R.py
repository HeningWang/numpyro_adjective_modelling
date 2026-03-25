"""
Export appendix data (production ablation LOO) to CSV for R plotting.

Run from the 10-writing/ directory:
    python export_appendix_data_for_R.py

Outputs (written to 10-writing/data/):
  - production_ablation_loo_comparison.csv : LOO table for 4 main + ablation models
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

# ── Config ────────────────────────────────────────────────────────────────────
NW, NS, NC = 500, 500, 4
TAG = f"warmup{NW}_samples{NS}_chains{NC}"

# 4 main models + 2 usable ablations (rsa_only excluded from LOO due to PSIS failure)
MODEL_SPECS = {
    "incremental_recursive":  f"mcmc_results_incremental_speaker_hier_{TAG}.nc",
    "incremental_static":     f"mcmc_results_incremental_static_speaker_hier_{TAG}.nc",
    "global_recursive":       f"mcmc_results_global_speaker_hier_{TAG}.nc",
    "global_static":          f"mcmc_results_global_static_speaker_hier_{TAG}.nc",
    "incremental_lookahead":  f"mcmc_results_incremental_lookahead_speaker_hier_{TAG}.nc",
    "incremental_lm_only":    f"mcmc_results_incremental_lm_only_speaker_hier_{TAG}.nc",
}

# ── LOO comparison table ─────────────────────────────────────────────────────
idata_dict = {}
for model_name, fname in MODEL_SPECS.items():
    fpath = os.path.join(PROD_DIR, "inference_data", fname)
    if os.path.exists(fpath):
        print(f"Loading {model_name}...")
        idata_dict[model_name] = az.from_netcdf(fpath)
    else:
        print(f"[!] Skipping {model_name}: {fpath} not found")

if idata_dict:
    comparison = az.compare(idata_dict, ic="loo", method="stacking", var_name="obs")
    comparison.to_csv(os.path.join(DATA_DIR, "production_ablation_loo_comparison.csv"))
    print(f"[✓] production_ablation_loo_comparison.csv")
    print(comparison.to_string())

print("\n[Done] Appendix CSVs exported to", DATA_DIR)
