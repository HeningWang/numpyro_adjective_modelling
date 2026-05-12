"""R² ladder — compare PPC correlation R² across all available contextual_dc NCs.

The success criterion for the R²-first loop is PPC correlation R² on the 90
cells (relevant_property × sharpness × utterance), matching the paper's
reporting convention. This script computes R²(all) and R²(emp ≥ 0.02) for
every contextual_dc NC currently on disk, plus mixing health for each, so the
loop can compare a new variant head-to-head against the existing ladder
without recomputing from scratch each iteration.

Reads:
    inference_data/mcmc_results_contextual*hier_dc*.nc

Produces (gitignored):
    results/contextual_dc/r2_ladder.csv

Self-contained: reuses the PPC machinery from ppc_lever1_review.py (which
itself depends only on arviz, numpy, pandas, scipy). Does NOT import any
JAX / model code — works on a CPU laptop in <60s for 7 NCs.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROD = Path(__file__).resolve().parent.parent
RES_DIR = PROD / "results" / "contextual_dc"
RES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROD / "scripts"))
import ppc_lever1_review as plv  # PPC machinery
# Use the canonical (prop × sharp) grouping that matches the paper's R²
plv.GROUP_COLS = ["relevant_property", "sharpness"]
from ppc_lever1_review import (
    load_dataset_dc, compute_model_proportions, compute_human_proportions,
)


def _rhat_max(idata, var_names) -> float:
    """Max r-hat across the named scalars (handles missing vars gracefully)."""
    available = [v for v in var_names if v in idata.posterior.data_vars]
    if not available:
        return float("nan")
    try:
        rhat = az.rhat(idata, var_names=available)
        return float(max(float(rhat[v].values) for v in available))
    except Exception:
        return float("nan")


def _divergences(idata) -> int:
    if "sample_stats" in idata.groups() and "diverging" in idata.sample_stats:
        return int(idata.sample_stats["diverging"].sum().item())
    return 0


def compute_r2_row(label: str, nc_path: Path, df_dc, all_codes, flat_to_cat, emp) -> dict:
    if not nc_path.exists():
        return {"variant": label, "nc": nc_path.name, "status": "MISSING"}
    idata = az.from_netcdf(str(nc_path))
    pp = idata.posterior_predictive["obs"]
    pp_flat = pp.stack(sample=("chain", "draw")).transpose("sample", "item").values
    mod = compute_model_proportions(pp_flat, df_dc, max_draws=200)
    merged = (
        emp[["relevant_property", "sharpness", "utt_code", "human_mean", "n"]]
        .merge(
            mod[["relevant_property", "sharpness", "utt_code", "model_mean"]],
            on=["relevant_property", "sharpness", "utt_code"], how="inner",
        )
    )
    r_all, _ = pearsonr(merged["human_mean"], merged["model_mean"])
    sub = merged[merged["human_mean"] >= 0.02]
    r_sub, _ = pearsonr(sub["human_mean"], sub["model_mean"]) if len(sub) >= 2 else (np.nan, None)

    eps_mean = (
        float(idata.posterior["epsilon"].mean().item())
        if "epsilon" in idata.posterior.data_vars else float("nan")
    )
    rhat_max = _rhat_max(idata, ["epsilon", "alpha_D", "alpha_C", "alpha_F"])
    divs = _divergences(idata)
    n_chains = int(idata.posterior.sizes.get("chain", 0))
    n_draws = int(idata.posterior.sizes.get("draw", 0))

    return {
        "variant": label,
        "nc": nc_path.name,
        "eps": eps_mean,
        "R2_all": r_all ** 2,
        "R2_emp_ge_002": r_sub ** 2 if not np.isnan(r_sub) else np.nan,
        "pearson_r": r_all,
        "rhat_max": rhat_max,
        "divergences": divs,
        "chains": n_chains,
        "draws": n_draws,
        "status": "OK",
    }


def discover_ncs() -> list[tuple[str, Path]]:
    """Find all contextual_dc NCs in inference_data/ and label them by filename."""
    inf = PROD / "inference_data"
    out = []
    # Stable known names (in the order they were produced)
    canonical = [
        ("baseline-6gamma", "mcmc_results_contextual_speaker_hier_dc_warmup4000_samples2000_chains4.nc"),
        ("iter1-lever1-rawLM", "mcmc_results_contextual_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter1.nc"),
        ("iter2-two-comp-LM", "mcmc_results_contextual_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter2.nc"),
        ("iter3-alpha-boost-Normal", "mcmc_results_contextual_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter3.nc"),
        ("iter4-XL-LM", "mcmc_results_contextual_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter4.nc"),
        ("iter5-boost+lambdaSuff", "mcmc_results_contextual_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter5.nc"),
        ("iter6-boost-only", "mcmc_results_contextual_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter6.nc"),
        ("iter7-lambdaunc", "mcmc_results_contextual_lambdaunc_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter7.nc"),
        ("iter8-freewf", "mcmc_results_contextual_freewf_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter8.nc"),
        ("iter9-anchored", "mcmc_results_contextual_anchored_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter9.nc"),
        ("iter10-freewf+anchored", "mcmc_results_contextual_freewf_anchored_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter10.nc"),
        ("iter11-anchored+freewf+2gamma", "mcmc_results_contextual_anchored_gamma_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter11.nc"),
    ]
    for label, name in canonical:
        out.append((label, inf / name))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="R² ladder for contextual_dc NCs.")
    parser.add_argument(
        "--extra-nc", type=str, default=None,
        help="Additional NC path to evaluate (e.g. a new iter not yet in the canonical list).",
    )
    parser.add_argument(
        "--extra-label", type=str, default="extra",
        help="Label for the extra NC.",
    )
    args = parser.parse_args()

    print("Loading dataset (dc subset)...")
    df_dc, flat_to_cat = load_dataset_dc()
    all_codes = sorted(df_dc["annotation_seq_flat"].unique().tolist())
    print("Computing empirical proportions...")
    emp = compute_human_proportions(df_dc, all_codes, n_boot=1, seed=431)

    rows = []
    ncs = discover_ncs()
    if args.extra_nc is not None:
        ncs.append((args.extra_label, Path(args.extra_nc)))

    print(f"\n{'variant':<30} {'eps':<7} {'R²(all)':<9} {'R²(emp≥.02)':<13} "
          f"{'r':<6} {'rhat':<7} {'div':<5}")
    print("-" * 86)
    for label, path in ncs:
        row = compute_r2_row(label, path, df_dc, all_codes, flat_to_cat, emp)
        rows.append(row)
        if row.get("status") == "MISSING":
            print(f"{label:<30} MISSING ({path.name})")
            continue
        print(
            f"{label:<30} {row['eps']:<7.3f} {row['R2_all']:<9.3f} "
            f"{row['R2_emp_ge_002']:<13.3f} {row['pearson_r']:<6.3f} "
            f"{row['rhat_max']:<7.3f} {row['divergences']:<5}"
        )

    out_csv = RES_DIR / "r2_ladder.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.relative_to(PROD.parent)}")


if __name__ == "__main__":
    main()
