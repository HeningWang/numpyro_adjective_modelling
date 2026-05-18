"""Parsimony-vs-fit frontier (memo §7.8).

Evaluates the iter-18 → parsimony → leave-one-out family head-to-head on the
same PPC R² convention as the ladder (90 cells: relevant_property × sharpness
× utterance; R²(all) and R²(emp ≥ 0.02)), and pairs each with its named
(population) parameter count so the manuscript can report a defensible
parsimony/fit trade-off rather than only the maximal model.

Reads (must be present in inference_data/):
    iter-18 (max)     mcmc_results_contextual_pcalpha_canon_speaker_hier_dc_*.nc
    betafixed         mcmc_results_contextual_pcalpha_canon_betafixed_speaker_hier_dc_*.nc
    parsimony (11)    mcmc_results_contextual_pcalpha_canon_parsimony_speaker_hier_dc_*.nc
    LOO −gamma_sharp  ..._parsimony_no_gammasharp_..._dc_*.nc
    LOO −lambda_suff  ..._parsimony_no_lambdasuff_..._dc_*.nc
    LOO −alpha_F      ..._parsimony_no_alphaF_..._dc_*.nc

Produces (gitignored):
    results/contextual_dc/parsimony_frontier.csv

Self-contained: reuses r2_ladder.compute_r2_row (arviz/numpy/pandas/scipy
only — no JAX/model import). CPU laptop, <60s.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pandas as pd

PROD = Path(__file__).resolve().parent.parent
RES_DIR = PROD / "results" / "contextual_dc"
RES_DIR.mkdir(parents=True, exist_ok=True)
INF = PROD / "inference_data"

sys.path.insert(0, str(PROD / "scripts"))
import r2_ladder as rl
from ppc_lever1_review import (
    load_dataset_dc, compute_human_proportions,
)

# (label, nc filename, n_named_params, what was dropped vs iter-18)
FRONTIER = [
    ("iter18-canon (max)", "mcmc_results_contextual_pcalpha_canon_speaker_hier_dc_warmup4000_samples2000_chains4.nc", 13, "—"),
    ("betafixed", "mcmc_results_contextual_pcalpha_canon_betafixed_speaker_hier_dc_warmup4000_samples2000_chains4.nc", 12, "log_beta_lm"),
    ("parsimony (base)", "mcmc_results_contextual_pcalpha_canon_parsimony_speaker_hier_dc_warmup4000_samples2000_chains4.nc", 11, "log_beta_lm + gamma_len3_erdc"),
    ("parsimony −alpha_F", "mcmc_results_contextual_pcalpha_canon_parsimony_no_alphaF_speaker_hier_dc_warmup4000_samples2000_chains4.nc", 10, "+ alpha_F"),
    ("parsimony −lambda_suff", "mcmc_results_contextual_pcalpha_canon_parsimony_no_lambdasuff_speaker_hier_dc_warmup4000_samples2000_chains4.nc", 10, "+ lambda_suff"),
    ("parsimony −gamma_sharp", "mcmc_results_contextual_pcalpha_canon_parsimony_no_gammasharp_speaker_hier_dc_warmup4000_samples2000_chains4.nc", 10, "+ gamma_sharp"),
]


def main() -> None:
    print("Loading dataset (dc subset)...")
    df_dc, flat_to_cat = load_dataset_dc()
    all_codes = sorted(df_dc["annotation_seq_flat"].unique().tolist())
    print("Computing empirical proportions...")
    emp = compute_human_proportions(df_dc, all_codes, n_boot=1, seed=431)

    rows = []
    print(f"\n{'model':<26} {'#named':<7} {'R²(all)':<9} {'R²(emp≥.02)':<13} "
          f"{'r':<6} {'rhat':<7} {'div':<5}")
    print("-" * 80)
    for label, name, n_named, dropped in FRONTIER:
        row = rl.compute_r2_row(label, INF / name, df_dc, all_codes, flat_to_cat, emp)
        row["n_named"] = n_named
        row["dropped_vs_iter18"] = dropped
        rows.append(row)
        if row.get("status") == "MISSING":
            print(f"{label:<26} MISSING ({name})")
            continue
        print(
            f"{label:<26} {n_named:<7} {row['R2_all']:<9.4f} "
            f"{row['R2_emp_ge_002']:<13.4f} {row['pearson_r']:<6.3f} "
            f"{row['rhat_max']:<7.3f} {row['divergences']:<5}"
        )

    out = pd.DataFrame(rows)[
        ["variant", "n_named", "dropped_vs_iter18", "R2_all", "R2_emp_ge_002",
         "pearson_r", "eps", "rhat_max", "divergences", "chains", "draws", "status"]
    ]
    out_csv = RES_DIR / "parsimony_frontier.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.relative_to(PROD.parent)}")


if __name__ == "__main__":
    main()
