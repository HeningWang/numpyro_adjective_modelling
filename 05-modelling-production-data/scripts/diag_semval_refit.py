"""Fixed-constant joint-refit verdict (memo §7.10) — AUTHORITATIVE.

For the recommended model (`…parsimony_no_alphaF`, 10 named, csv/fsv/k/wf
fixed) plus the five free-constant refits (freecsv/freefsv/freek/freewf/
freeall4), reports head-to-head:

  • R²(all) / R²(emp≥.02) and Δ vs the fixed baseline (does freeing help?)
  • the freed constant's posterior mean [94% HDI], and whether the CURRENT
    fixed value lies inside that HDI (if yes → fixing is justified; the
    best fixed value is the posterior mean ≈ current)
  • r̂_max / min-ESS / divergences (is the freed constant identified?)

Decision rule per constant:
  free if  ΔR² materially > 0  AND identified  AND HDI excludes current fix
  else keep fixed; best fixed value = freed posterior mean (≈ current if its
  HDI brackets the current value)

Reads the NCs in inference_data/ (pull from CSP first). Reuses
r2_ladder.compute_r2_row (no JAX/model import). CPU, <90s.
Produces (gitignored): results/contextual_dc/semval_refit_verdict.csv
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import arviz as az
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
INF = HERE / "inference_data"
RES = HERE / "results" / "contextual_dc"
RES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE / "scripts"))
import r2_ladder as rl
import ppc_lever1_review as plv
plv.GROUP_COLS = ["relevant_property", "sharpness"]
from ppc_lever1_review import load_dataset_dc, compute_human_proportions

STEM = "mcmc_results_contextual_pcalpha_canon_parsimony_no_alphaF{tag}_speaker_hier_dc_warmup4000_samples2000_chains4.nc"
BASELINE = ("recommended (all fixed)", "", None, None)
# (label, filename-tag, freed-site, current-fixed-value)
CUR_WF = 0.6856  # WF_FIXED_ITER11_MEDIAN
RUNS = [
    ("free csv", "_freecsv", "color_semval", 0.971),
    ("free fsv", "_freefsv", "form_semval", 0.50),
    ("free k", "_freek", "k", 0.50),
    ("free wf", "_freewf", "wf", CUR_WF),  # via log_wf
]
FREEALL = ("free all 4", "_freeall4",
           [("color_semval", 0.971), ("form_semval", 0.50),
            ("k", 0.50), ("wf", CUR_WF)])


def named_vars(idata):
    base = ["alpha_D", "alpha_C", "lambda_suff", "lambda_form_mod",
            "lambda_noncanon", "gamma_base", "gamma_oneword", "gamma_sharp",
            "epsilon", "tau"]
    extra = [v for v in ("color_semval", "form_semval", "k", "log_wf")
             if v in idata.posterior.data_vars]
    return [v for v in base if v in idata.posterior.data_vars] + extra


def post_summary(idata, site):
    """Return (mean, hdi3, hdi97, rhat, ess) for a site; wf is exp(log_wf)."""
    if site == "wf":
        da = np.exp(idata.posterior["log_wf"])
    else:
        da = idata.posterior[site]
    flat = da.values.reshape(-1)
    hdi = az.hdi(np.asarray(flat), hdi_prob=0.94)
    s = az.summary(idata, var_names=["log_wf" if site == "wf" else site])
    rhat = float(s["r_hat"].iloc[0]) if "r_hat" in s else np.nan
    ess = float(s["ess_bulk"].iloc[0]) if "ess_bulk" in s else np.nan
    return float(flat.mean()), float(hdi[0]), float(hdi[1]), rhat, ess


def main():
    df_dc, flat_to_cat = load_dataset_dc()
    all_codes = sorted(df_dc["annotation_seq_flat"].unique().tolist())
    emp = compute_human_proportions(df_dc, all_codes, n_boot=1, seed=431)

    base_nc = INF / STEM.format(tag="")
    if not base_nc.exists():
        raise SystemExit(f"baseline NC missing: {base_nc.name}")
    b = rl.compute_r2_row("baseline", base_nc, df_dc, all_codes, flat_to_cat, emp)
    base_all, base_sub = b["R2_all"], b["R2_emp_ge_002"]
    print(f"\nRecommended (all 4 fixed): R²(all)={base_all:.4f} "
          f"R²(emp≥.02)={base_sub:.4f}  [10 named]\n")

    rows = []
    hdr = (f"{'variant':<13}{'R²(all)':<9}{'ΔR²':<9}{'R²≥.02':<9}"
           f"{'constant':<13}{'post mean [94% HDI]':<26}{'fixed in HDI?':<14}"
           f"{'r̂':<6}{'ESS':<7}{'div'}")
    print(hdr); print("-" * len(hdr))

    def emit(label, tag, freed):
        nc = INF / STEM.format(tag=tag)
        if not nc.exists():
            print(f"{label:<13}MISSING ({nc.name})")
            return
        r = rl.compute_r2_row(label, nc, df_dc, all_codes, flat_to_cat, emp)
        idata = az.from_netcdf(str(nc))
        for site, cur in freed:
            m, lo, hi, rhat, ess = post_summary(idata, site)
            inside = lo <= cur <= hi
            print(f"{label:<13}{r['R2_all']:<9.4f}{r['R2_all']-base_all:<+9.4f}"
                  f"{r['R2_emp_ge_002']:<9.4f}{site:<13}"
                  f"{m:.4f} [{lo:.3f}, {hi:.3f}]".ljust(26 + 35)[:35]
                  + f"  {('YES '+f'(fix={cur})') if inside else 'NO — '+f'fix={cur} OUT':<14}"
                  f"{rhat:<6.3f}{ess:<7.0f}{r['divergences']}")
            rows.append(dict(variant=label, R2_all=r["R2_all"],
                             dR2_all=r["R2_all"] - base_all,
                             R2_emp_ge_002=r["R2_emp_ge_002"], constant=site,
                             post_mean=m, hdi3=lo, hdi97=hi, fixed_val=cur,
                             fixed_in_hdi=inside, rhat=rhat, ess=ess,
                             divergences=r["divergences"]))

    for label, tag, site, cur in RUNS:
        emit(label, tag, [(site, cur)])
    emit(*FREEALL[:2], FREEALL[2])

    out = RES / "semval_refit_verdict.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out.relative_to(HERE.parent)}")
    print("\nRule: free only if ΔR² materially >0 AND identified AND fixed "
          "value OUTSIDE the freed 94% HDI; else keep fixed (best fixed "
          "value = freed posterior mean, ≈ current if HDI brackets it).")


if __name__ == "__main__":
    main()
