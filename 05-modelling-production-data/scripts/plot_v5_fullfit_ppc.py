"""PPC variants for v5 (F2) fit on full data N=9586.

Two displays:
- dc-only (6-panel): for comparison with slider data and dc-only fit
- all 9 conditions × 2 sharpness (18-panel): complete picture
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from helper import FLAT_TO_CATEGORIES, import_dataset_hier

INT_TO_UTT = {int(k): v for k, v in FLAT_TO_CATEGORIES.items()}
UTT_ORDER = [INT_TO_UTT[i] for i in range(15)]
OUT_DIR = Path("figures/v5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NC = {
    "baseline": "inference_data/mcmc_results_reported_speaker_hier_warmup2000_samples1000_chains4.nc",
    "ext_v1":   "inference_data/mcmc_results_incremental_speaker_hier_warmup2000_samples1000_chains4.nc",
    "v5":       "inference_data/mcmc_results_v5_F2_fulldata_warmup2000_samples1000_chains4.nc",
}

data = import_dataset_hier()
df = data["df"].copy().reset_index(drop=True)
# df now has index 0..n-1 aligned with posterior_predictive's item axis

# Map condition code to family for clarity in 9-panel plot
def condition_family(code):
    dims = code[2:4]
    return {"dc": "D×C", "df": "D×F", "cf": "C×F"}[dims]

df["family"] = df["conditions"].apply(condition_family)


def build_emp(df_sub, group_cols):
    emp = df_sub.groupby(group_cols + ["annotation_seq_flat"]).size().rename("n").reset_index()
    tots = df_sub.groupby(group_cols).size().rename("nt").reset_index()
    emp = emp.merge(tots, on=group_cols)
    emp["emp_prop"] = emp["n"] / emp["nt"]
    emp["utt"] = emp["annotation_seq_flat"].astype(int).map(INT_TO_UTT)
    return emp, tots


def model_props(idata, df_in, group_cols):
    """Compute posterior-predictive proportions grouped by group_cols.
    Uses the FULL posterior predictive samples (fit on all 9586 trials);
    filtering applied here only to the DISPLAY dataframe."""
    pp_full = idata.posterior_predictive["obs"].stack(sample=("chain", "draw")).values
    # keep only rows present in df_in (their original integer index in the fit df)
    kept_idx = df_in.index.to_numpy()
    pp = pp_full[kept_idx, :]
    sub = np.linspace(0, pp.shape[1] - 1, 500, dtype=int)
    rec = []
    for s in sub:
        tmp = df_in.copy()
        tmp["ann"] = pp[:, s]
        c = tmp.groupby(group_cols + ["ann"]).size().rename("n").reset_index()
        t = tmp.groupby(group_cols).size().rename("nt").reset_index()
        c = c.merge(t, on=group_cols)
        c["p"] = c["n"] / c["nt"]
        rec.append(c[group_cols + ["ann", "p"]])
    stacked = pd.concat(rec)
    out = stacked.groupby(group_cols + ["ann"])["p"].agg(
        model_mean="mean",
        model_lo=lambda x: np.percentile(x, 2.5),
        model_hi=lambda x: np.percentile(x, 97.5),
    ).reset_index()
    out["utt"] = out["ann"].astype(int).map(INT_TO_UTT)
    return out


models = {tag: az.from_netcdf(p) for tag, p in NC.items()}


def draw_panels(cells, row_levels, col_levels, row_name, col_name, emp, mod_props_by_tag, tots, fig_size, suptitle, out_name):
    n_r, n_c = len(row_levels), len(col_levels)
    fig, axes = plt.subplots(n_r, n_c, figsize=fig_size, sharey=True)
    if n_r == 1 and n_c == 1:
        axes = np.array([[axes]])
    elif n_r == 1:
        axes = axes[None, :]
    elif n_c == 1:
        axes = axes[:, None]

    bar_w = 0.21
    x = np.arange(len(UTT_ORDER))

    for i, r in enumerate(row_levels):
        for j, c in enumerate(col_levels):
            ax = axes[i, j]
            mask = lambda d: (d[row_name] == r) & (d[col_name] == c)
            e = emp[mask(emp)].set_index("utt").reindex(UTT_ORDER).fillna(0)
            bs = mod_props_by_tag["baseline"][mask(mod_props_by_tag["baseline"])].set_index("utt").reindex(UTT_ORDER).fillna(0)
            v1 = mod_props_by_tag["ext_v1"][mask(mod_props_by_tag["ext_v1"])].set_index("utt").reindex(UTT_ORDER).fillna(0)
            v5 = mod_props_by_tag["v5"][mask(mod_props_by_tag["v5"])].set_index("utt").reindex(UTT_ORDER).fillna(0)

            ax.bar(x - 1.5 * bar_w, e["emp_prop"].values, bar_w, label="empirical", color="steelblue")
            ax.bar(x - 0.5 * bar_w, bs["model_mean"].values, bar_w, label="baseline",
                   yerr=[bs["model_mean"].values - bs["model_lo"].values, bs["model_hi"].values - bs["model_mean"].values],
                   color="#9ecae1", ecolor="#08519c", capsize=2)
            ax.bar(x + 0.5 * bar_w, v1["model_mean"].values, bar_w, label="ext_v1",
                   yerr=[v1["model_mean"].values - v1["model_lo"].values, v1["model_hi"].values - v1["model_mean"].values],
                   color="lightcoral", ecolor="darkred", capsize=2)
            ax.bar(x + 1.5 * bar_w, v5["model_mean"].values, bar_w, label="v5 (F2)",
                   yerr=[v5["model_mean"].values - v5["model_lo"].values, v5["model_hi"].values - v5["model_mean"].values],
                   color="goldenrod", ecolor="darkorange", capsize=2)

            n_trials = int(tots[(tots[row_name] == r) & (tots[col_name] == c)]["nt"].iloc[0]) if ((tots[row_name] == r) & (tots[col_name] == c)).any() else 0
            ax.set_title(f"{r} | {c}  (N={n_trials})", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(UTT_ORDER, rotation=45, fontsize=6)
            ax.set_ylim(0, 1.0)
            if j == 0:
                ax.set_ylabel("proportion", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        out = OUT_DIR / f"{out_name}.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


# ==== (A) Full-data fit, display dc conditions only, 6-panel ====================
df_dc = df[df["conditions"].isin(["erdc", "zrdc", "brdc"])].copy()
emp_dc, tots_dc = build_emp(df_dc, ["relevant_property", "sharpness"])
mod_dc = {tag: model_props(idata, df_dc, ["relevant_property", "sharpness"]) for tag, idata in models.items()}

draw_panels(
    cells=None,
    row_levels=["first", "both", "second"],
    col_levels=["blurred", "sharp"],
    row_name="relevant_property", col_name="sharpness",
    emp=emp_dc, mod_props_by_tag=mod_dc, tots=tots_dc,
    fig_size=(18, 11),
    suptitle="v5 F2 fit on FULL data (N=9586) — display restricted to dc conditions (comparable to slider)",
    out_name="production_ppc_barplot_v5_fullfit_dc_F2",
)

# ==== (B) Full-data fit, all 9 conditions × 2 sharpness = 18 panels =============
emp_all, tots_all = build_emp(df, ["conditions", "sharpness"])
mod_all = {tag: model_props(idata, df, ["conditions", "sharpness"]) for tag, idata in models.items()}

CONDS_ORDERED = ["erdc", "zrdc", "brdc",   # D×C family
                 "erdf", "zrdf", "brdf",   # D×F family
                 "ercf", "zrcf", "brcf"]   # C×F family
draw_panels(
    cells=None,
    row_levels=CONDS_ORDERED,
    col_levels=["blurred", "sharp"],
    row_name="conditions", col_name="sharpness",
    emp=emp_all, mod_props_by_tag=mod_all, tots=tots_all,
    fig_size=(18, 28),
    suptitle="v5 F2 fit on FULL data (N=9586) — all 9 conditions × 2 sharpness (18 panels)",
    out_name="production_ppc_barplot_v5_fullfit_all9_F2",
)

# ==== Stats comparison: dc-display vs full-display ==============================
for tag, emp_tab, mod_tab, label in [
    ("dc", emp_dc, mod_dc, "dc-only (N=3196 display)"),
    ("all", emp_all, mod_all, "all 9 conditions (N=9586 display)"),
]:
    if tag == "dc":
        merged = emp_tab.merge(
            mod_tab["v5"].rename(columns={"model_mean": "p_v5", "ann": "annotation_seq_flat"}),
            on=["relevant_property", "sharpness", "annotation_seq_flat", "utt"], how="outer",
        ).fillna(0)
    else:
        merged = emp_tab.merge(
            mod_tab["v5"].rename(columns={"model_mean": "p_v5", "ann": "annotation_seq_flat"}),
            on=["conditions", "sharpness", "annotation_seq_flat", "utt"], how="outer",
        ).fillna(0)
    r, _ = pearsonr(merged["emp_prop"], merged["p_v5"])
    l1 = (merged["p_v5"] - merged["emp_prop"]).abs().sum()
    print(f"v5 F2 (fit on N=9586) | display: {label}  ->  R² = {r**2:.3f}  L1 = {l1:.2f}  n = {len(merged)}")
