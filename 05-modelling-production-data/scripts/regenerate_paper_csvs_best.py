"""Regenerate production CSVs for the advocated extended model (csv=0.59).

Writes the best model's per-condition predictions and the human/model
correlation table in the same schema the paper's R plotting script expects,
so the PPC bar plot and correlation scatter can be drawn in the shared CSP
ggplot style.

NC file: mcmc_results_contextual_pcalpha_canon_parsimony_no_alphaF_csv059_
         speaker_hier_dc_warmup4000_samples2000_chains4.nc

Outputs:
  - 10-writing/data/production_predictions_best.csv
  - 10-writing/data/production_correlation_best.csv

Run from 05-modelling-production-data/:
    python scripts/regenerate_paper_csvs_best.py
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import pathlib

import numpy as np
import pandas as pd
import arviz as az

# Reuse the exact aggregation logic of the paper-CSV regenerator.
from regenerate_paper_csvs import (
    GROUP_COLS,
    FLAT_TO_CAT,
    compute_condition_proportions,
    compute_human_proportions,
)
from shared.posterior_utils import extract_pp_samples
import helper

helper.CONDITIONS_OF_INTEREST = ("erdc", "zrdc", "brdc")
from helper import import_dataset

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
INFERENCE_DIR = REPO_ROOT / "05-modelling-production-data" / "inference_data"
OUTPUT_DIR = REPO_ROOT / "10-writing" / "data"

NC_NAME = (
    "mcmc_results_contextual_pcalpha_canon_parsimony_no_alphaF_csv059_"
    "speaker_hier_dc_warmup4000_samples2000_chains4.nc"
)
MODEL_NAME = "extended"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    data = import_dataset()
    cond_df = data["df"].reset_index(drop=True)
    print(f"  N = {len(cond_df)} observations")

    nc_path = INFERENCE_DIR / NC_NAME
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)
    print(f"Loading {NC_NAME} ...")
    idata = az.from_netcdf(str(nc_path))

    print("Computing human bootstrap proportions...")
    df_human = compute_human_proportions(cond_df)

    print("Computing model predictions...")
    pp_flat = extract_pp_samples(idata, max_draws=500)
    df_model = compute_condition_proportions(pp_flat, cond_df, max_draws=500)
    df_model["utterance_code"] = df_model["annotation_seq_flat"].astype(int)
    df_model["utterance_label"] = df_model["utterance_code"].map(FLAT_TO_CAT)
    df_model["model"] = MODEL_NAME

    out_pred = OUTPUT_DIR / "production_predictions_best.csv"
    df_model.to_csv(str(out_pred), index=False)
    print(f"  -> {out_pred} ({len(df_model)} rows)")

    df_corr = df_human.merge(
        df_model[GROUP_COLS + ["utterance_code", "model_mean", "model_lo", "model_hi"]],
        on=GROUP_COLS + ["utterance_code"],
        how="outer",
    ).fillna(0)
    df_corr["condition"] = df_corr["relevant_property"] + " | " + df_corr["sharpness"]
    out_corr = OUTPUT_DIR / "production_correlation_best.csv"
    df_corr.to_csv(str(out_corr), index=False)
    print(f"  -> {out_corr} ({len(df_corr)} rows)")

    mask = df_corr["human_mean"] > 0
    r = np.corrcoef(
        df_corr.loc[mask, "human_mean"], df_corr.loc[mask, "model_mean"]
    )[0, 1]
    print(f"\n  R^2 (non-zero cells) = {r**2:.3f}  (r = {r:.3f})")


if __name__ == "__main__":
    main()
