"""2x2 (speaker × semantics) under the §7.10 representation vs the old grids.

Cross-check: §7.10 found color_semval≈0.59 best for the CONTEXTUAL model.
This compares the 4 simple "reported-style" 2x2 models (incremental/global ×
recursive/static) refit under the §7.10 set (csv=0.59, fsv=0.50, wf=0.6856)
against the historical csv=0.85 and csv=0.971 LOO results.

Reads the csv=0.59 NCs (run_2x2_csv059.py output) for a proper pairwise
az.compare (elpd_diff / dse). The csv=0.85 / csv=0.971 columns are the
recorded historical LOO values (memo §model-comparison; not recomputed).

Produces (gitignored): results/contextual_dc/twoby2_repr_comparison.csv
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import arviz as az
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
INF = HERE / "inference_data"
RES = HERE / "results" / "contextual_dc"
RES.mkdir(parents=True, exist_ok=True)

MODELS = ["incremental_recursive", "incremental_static",
          "global_recursive", "global_static"]
NC = "mcmc_results_2x2csv059_{}_warmup5000_samples2000_chains4.nc"

# Historical LOO (recorded, not recomputed) — memo / 2x2 memory.
ELPD_CSV085 = {"incremental_recursive": -6985.3, "incremental_static": -7000.9,
               "global_recursive": -7728.1, "global_static": -7738.5}
ELPD_CSV097 = {"incremental_recursive": -7180.0, "incremental_static": -7199.0,
               "global_recursive": -7628.0, "global_static": -7626.0}


def main() -> None:
    idata = {}
    for m in MODELS:
        p = INF / NC.format(m)
        if not p.exists():
            raise SystemExit(f"missing csv=0.59 NC: {p.name} "
                             f"(run run_2x2_csv059.py on CSP + pull)")
        idata[m] = az.from_netcdf(str(p))

    cmp = az.compare(idata, ic="loo")
    rows = []
    for m in MODELS:
        c = cmp.loc[m]
        rows.append({
            "model": m,
            "elpd_csv0.971": ELPD_CSV097[m],
            "elpd_csv0.85": ELPD_CSV085[m],
            "elpd_csv0.59": round(float(c["elpd_loo"]), 1),
            "p_loo_csv0.59": round(float(c["p_loo"]), 1),
            "rank_csv0.59": int(c["rank"]),
            "elpd_diff_csv0.59": round(float(c["elpd_diff"]), 2),
            "dse_csv0.59": round(float(c["dse"]), 2),
        })
    df = pd.DataFrame(rows)

    # Key pairwise contrast: recursive vs static within each speaker.
    def contrast(a, b):
        ea, eb = idata[a], idata[b]
        cc = az.compare({a: ea, b: eb}, ic="loo")
        top = cc.index[0]
        d = float(cc.loc[cc.index[1], "elpd_diff"])
        s = float(cc.loc[cc.index[1], "dse"])
        return top, d, s, (abs(d / s) if s > 0 else float("nan"))

    print(df.to_string(index=False))
    print("\nKey contrasts (csv=0.59):")
    for a, b, lbl in [("incremental_recursive", "incremental_static",
                       "incremental: recursive vs static"),
                      ("global_recursive", "global_static",
                       "global: recursive vs static")]:
        top, d, s, z = contrast(a, b)
        verdict = ("TIED (interaction absent)" if z < 2
                   else f"{top} wins")
        print(f"  {lbl}: best={top}, Δ={d:.2f}, dSE={s:.2f}, "
              f"|Δ|/dSE={z:.2f} → {verdict}")

    out = RES / "twoby2_repr_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out.relative_to(HERE.parent)}")
    print("\nNote: §7.10 csv=0.59 (validated on the CONTEXTUAL model) does NOT "
          "transfer to the simple 2x2 — it degrades all four and erases the "
          "recursive>static interaction; csv=0.85 stays the 2x2 representation.")


if __name__ == "__main__":
    main()
