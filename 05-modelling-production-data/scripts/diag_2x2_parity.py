"""2×2 reviewer-defense diagnostic: parameter parity, realized complexity,
and the speaker×semantics interaction (memo §7.12).

Reviewer concern: the 2×2 cells (incremental/global × recursive/static) might
differ in free-parameter count or expressive flexibility, so ELPD/LOO
differences could be a complexity artifact rather than a mechanistic effect.

This script answers it with three checks, for BOTH semantic representations
(csv=0.85 "2x2best" and csv=0.59 "2x2csv059"):

  1. NOMINAL parameter inventory per cell — the sampled population sites and
     the latent count, read from each NC's posterior. Establishes that all
     four cells share an identical inventory (1 α, 1 log_β, 1 τ, N δ): the
     comparison is parameter-matched; differences are purely structural
     (processing order × listener recursion).
  2. REALIZED complexity p_loo per cell — the effective parameters LOO
     actually uses. If global collapses (p_loo ≪ nominal) while incremental
     stays high, that is a *finding* (the global architecture cannot exploit
     the participant hierarchy), not a confound.
  3. The 2×2 ELPD decomposition — speaker main effect, semantics main effect,
     and the **speaker×semantics interaction** (the confound-robust
     inferential target), with pairwise dSE from az.compare.

Also reports global-collapse indicators (α mean, τ mean, δ posterior SD).

Reads inference_data/mcmc_results_{2x2best,2x2csv059}_{cell}_*.nc.
Produces (gitignored): results/contextual_dc/twoby2_parity.csv
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

CELLS = ["incremental_recursive", "incremental_static",
         "global_recursive", "global_static"]
REPRS = {"csv0.85": "mcmc_results_2x2best_{}_warmup5000_samples2000_chains4.nc",
         "csv0.59": "mcmc_results_2x2csv059_{}_warmup5000_samples2000_chains4.nc"}
POP_SITES = ["alpha", "log_beta", "tau"]  # the 3 population params


def inventory(idata):
    """Nominal free-parameter inventory from the posterior group."""
    post = idata.posterior
    pop = [s for s in POP_SITES if s in post.data_vars]
    n_delta = int(post["delta"].sizes.get("delta_dim_0",
                  post["delta"].shape[-1])) if "delta" in post.data_vars else 0
    return pop, n_delta, len(pop) + n_delta


def main() -> None:
    rows = []
    elpd = {}  # (repr, cell) -> (elpd, p_loo)
    for rep, pat in REPRS.items():
        for cell in CELLS:
            p = INF / pat.format(cell)
            if not p.exists():
                print(f"MISSING: {p.name}")
                continue
            idata = az.from_netcdf(str(p))
            pop, n_delta, n_total = inventory(idata)
            loo = az.loo(idata)
            a = float(idata.posterior["alpha"].mean())
            t = float(idata.posterior["tau"].mean())
            dsd = float(idata.posterior["delta"].std()) if "delta" in idata.posterior else np.nan
            elpd[(rep, cell)] = (float(loo.elpd_loo), float(loo.p_loo))
            rows.append({
                "repr": rep, "cell": cell,
                "pop_params": "+".join(pop), "n_pop": len(pop),
                "n_latent_delta": n_delta, "n_nominal_total": n_total,
                "elpd_loo": round(float(loo.elpd_loo), 1),
                "p_loo_realized": round(float(loo.p_loo), 1),
                "alpha_mean": round(a, 3), "tau_mean": round(t, 3),
                "delta_post_sd": round(dsd, 3),
            })

    df = pd.DataFrame(rows)
    print("\n=== 1+2. Inventory (nominal) vs realized complexity (p_loo) ===")
    print(df[["repr", "cell", "pop_params", "n_nominal_total",
              "p_loo_realized", "elpd_loo", "alpha_mean", "tau_mean",
              "delta_post_sd"]].to_string(index=False))

    inv = df["n_nominal_total"].nunique()
    print(f"\nNominal inventory identical across all cells/representations? "
          f"{'YES' if inv == 1 else 'NO'} "
          f"(distinct totals: {sorted(df['n_nominal_total'].unique())})")

    print("\n=== 3. 2×2 ELPD decomposition (per representation) ===")
    for rep in REPRS:
        ir = elpd.get((rep, "incremental_recursive"))
        is_ = elpd.get((rep, "incremental_static"))
        gr = elpd.get((rep, "global_recursive"))
        gs = elpd.get((rep, "global_static"))
        if not all([ir, is_, gr, gs]):
            continue
        spk = ((ir[0] + is_[0]) / 2) - ((gr[0] + gs[0]) / 2)
        sem = ((ir[0] + gr[0]) / 2) - ((is_[0] + gs[0]) / 2)
        inter = (ir[0] - is_[0]) - (gr[0] - gs[0])
        # pairwise dSE for the within-speaker recursive-vs-static contrasts
        def dse(a, b):
            ia = az.from_netcdf(str(INF / REPRS[rep].format(a)))
            ib = az.from_netcdf(str(INF / REPRS[rep].format(b)))
            c = az.compare({a: ia, b: ib}, ic="loo")
            return (float(c.loc[c.index[1], "elpd_diff"]),
                    float(c.loc[c.index[1], "dse"]))
        di, si = dse("incremental_recursive", "incremental_static")
        dg, sg = dse("global_recursive", "global_static")
        print(f"\n[{rep}]")
        print(f"  speaker main effect   (inc−glob): {spk:+8.1f} ELPD")
        print(f"  semantics main effect (rec−stat): {sem:+8.1f} ELPD")
        print(f"  speaker×semantics INTERACTION   : {inter:+8.1f} ELPD")
        print(f"    incremental rec−stat: Δ={di:.2f} dSE={si:.2f} "
              f"|Δ|/dSE={abs(di/si) if si else float('nan'):.2f}")
        print(f"    global      rec−stat: Δ={dg:.2f} dSE={sg:.2f} "
              f"|Δ|/dSE={abs(dg/sg) if sg else float('nan'):.2f}")
        rows.append({"repr": rep, "cell": "_DECOMP_",
                     "pop_params": "speaker_ME=%.1f;sem_ME=%.1f;interaction=%.1f"
                     % (spk, sem, inter),
                     "n_pop": "", "n_latent_delta": "", "n_nominal_total": "",
                     "elpd_loo": "", "p_loo_realized": "",
                     "alpha_mean": "", "tau_mean": "", "delta_post_sd": ""})

    out = RES / "twoby2_parity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
