"""Best-model 2×2 (speaker × semantics) — LOO + R² + interaction (memo §7.13).

The paper's main comparison run on the architecture it advocates (the merged
csv=0.59 parsimony model, R²≈0.94) instead of the simple reported-style
models. Four parameter-matched cells (identical 10-named inventory; differ
only in utility accrual × listener recursion):

    inc_rec (= merged best model) | inc_static | glob_rec | glob_static

Reports per cell: PPC R²(all)/R²(emp≥.02) (ladder convention), az.loo
elpd/p_loo, az.compare pairwise dSE, and the 2×2 ELPD decomposition
(speaker ME, semantics ME, speaker×semantics interaction). Asserts inc_rec
reproduces the merged best model (R²(all) ≥ 0.93) — a built-in correctness
check on the cell= refactor.

Reads inference_data/mcmc_results_contextual_pcalpha_canon_parsimony_2x2_{cell}
_speaker_hier_dc_warmup4000_samples2000_chains4.nc. Reuses
r2_ladder.compute_r2_row (no JAX/model import). CPU, <90s.
Produces (gitignored): results/contextual_dc/twoby2_bestmodel.csv
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

sys.path.insert(0, str(HERE / "scripts"))
import r2_ladder as rl
import ppc_lever1_review as plv
plv.GROUP_COLS = ["relevant_property", "sharpness"]
from ppc_lever1_review import load_dataset_dc, compute_human_proportions

CELLS = ["inc_rec", "inc_static", "glob_rec", "glob_static"]
NC = ("mcmc_results_contextual_pcalpha_canon_parsimony_2x2_{}"
      "_speaker_hier_dc_warmup4000_samples2000_chains4.nc")


def main() -> None:
    df_dc, flat_to_cat = load_dataset_dc()
    all_codes = sorted(df_dc["annotation_seq_flat"].unique().tolist())
    emp = compute_human_proportions(df_dc, all_codes, n_boot=1, seed=431)

    rows, idata, elpd = [], {}, {}
    for c in CELLS:
        p = INF / NC.format(c)
        if not p.exists():
            raise SystemExit(f"missing 2x2 NC: {p.name} "
                             f"(run run_2x2_bestmodel.sh on CSP + pull)")
        r = rl.compute_r2_row(c, p, df_dc, all_codes, flat_to_cat, emp)
        id_ = az.from_netcdf(str(p))
        idata[c] = id_
        loo = az.loo(id_)
        elpd[c] = (float(loo.elpd_loo), float(loo.p_loo))
        rows.append({
            "cell": c, "R2_all": round(r["R2_all"], 4),
            "R2_emp_ge_002": round(r["R2_emp_ge_002"], 4),
            "pearson_r": round(r["pearson_r"], 3),
            "elpd_loo": round(float(loo.elpd_loo), 1),
            "p_loo": round(float(loo.p_loo), 1),
            "rhat_max": round(r["rhat_max"], 3),
            "divergences": r["divergences"],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Built-in correctness check: inc_rec == the merged best model.
    ir_r2 = df.loc[df.cell == "inc_rec", "R2_all"].iloc[0]
    assert ir_r2 >= 0.93, (
        f"inc_rec R²(all)={ir_r2} < 0.93 — must reproduce the merged "
        f"csv=0.59 best model; the cell= refactor changed inc_rec")
    print(f"\n[sanity] inc_rec R²(all)={ir_r2} ≥ 0.93 — reproduces the "
          f"merged best model OK")

    cmp = az.compare(idata, ic="loo")
    print("\naz.compare (LOO):")
    print(cmp[["rank", "elpd_loo", "p_loo", "elpd_diff", "dse"]].to_string())

    def contrast(a, b):
        cc = az.compare({a: idata[a], b: idata[b]}, ic="loo")
        d = float(cc.loc[cc.index[1], "elpd_diff"])
        s = float(cc.loc[cc.index[1], "dse"])
        return cc.index[0], d, s, (abs(d / s) if s else float("nan"))

    ir, is_ = elpd["inc_rec"][0], elpd["inc_static"][0]
    gr, gs = elpd["glob_rec"][0], elpd["glob_static"][0]
    spk = (ir + is_) / 2 - (gr + gs) / 2
    sem = (ir + gr) / 2 - (is_ + gs) / 2
    inter = (ir - is_) - (gr - gs)
    print("\n2×2 ELPD decomposition (best model):")
    print(f"  speaker main effect   (inc−glob): {spk:+8.1f}")
    print(f"  semantics main effect (rec−stat): {sem:+8.1f}")
    print(f"  speaker×semantics INTERACTION   : {inter:+8.1f}")
    for a, b, lbl in [("inc_rec", "inc_static", "incremental rec−stat"),
                      ("glob_rec", "glob_static", "global rec−stat")]:
        top, d, s, z = contrast(a, b)
        verdict = "TIED" if z < 2 else f"{top} wins"
        print(f"  {lbl}: best={top} Δ={d:.2f} dSE={s:.2f} |Δ|/dSE={z:.2f} → {verdict}")

    rows.append({"cell": "_DECOMP_",
                 "R2_all": "", "R2_emp_ge_002": "", "pearson_r": "",
                 "elpd_loo": f"spkME={spk:.1f};semME={sem:.1f};inter={inter:.1f}",
                 "p_loo": "", "rhat_max": "", "divergences": ""})
    out = RES / "twoby2_bestmodel.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
