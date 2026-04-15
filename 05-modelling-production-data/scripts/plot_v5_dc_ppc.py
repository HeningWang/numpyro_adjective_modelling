"""PPC barplot + correlation plot for v5 vs ext_v1 on the dc subset."""
import os
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
    "ext_v1": "inference_data/mcmc_results_incremental_speaker_hier_dc_warmup2000_samples1000_chains4.nc",
    "v5":     "inference_data/mcmc_results_v5_speaker_hier_dc_warmup2000_samples1000_chains4.nc",
}

data = import_dataset_hier()
df = data["df"]
df_dc = df[df["conditions"].isin(["erdc", "zrdc", "brdc"])].reset_index(drop=True)

emp = df_dc.groupby(["relevant_property", "sharpness", "annotation_seq_flat"]).size().rename("n").reset_index()
tots = df_dc.groupby(["relevant_property", "sharpness"]).size().rename("nt").reset_index()
emp = emp.merge(tots, on=["relevant_property", "sharpness"])
emp["emp_prop"] = emp["n"] / emp["nt"]
emp["utt"] = emp["annotation_seq_flat"].astype(int).map(INT_TO_UTT)


def model_props(idata):
    pp = idata.posterior_predictive["obs"].stack(sample=("chain", "draw")).values
    sub = np.linspace(0, pp.shape[1] - 1, 500, dtype=int)
    rec = []
    for s in sub:
        tmp = df_dc.copy()
        tmp["ann"] = pp[:, s]
        c = tmp.groupby(["relevant_property", "sharpness", "ann"]).size().rename("n").reset_index()
        t = tmp.groupby(["relevant_property", "sharpness"]).size().rename("nt").reset_index()
        c = c.merge(t, on=["relevant_property", "sharpness"])
        c["p"] = c["n"] / c["nt"]
        rec.append(c[["relevant_property", "sharpness", "ann", "p"]])
    stacked = pd.concat(rec)
    out = stacked.groupby(["relevant_property", "sharpness", "ann"])["p"].agg(
        model_mean="mean",
        model_lo=lambda x: np.percentile(x, 2.5),
        model_hi=lambda x: np.percentile(x, 97.5),
    ).reset_index()
    out["utt"] = out["ann"].astype(int).map(INT_TO_UTT)
    return out


models = {tag: az.from_netcdf(p) for tag, p in NC.items()}
mod_props = {tag: model_props(idata) for tag, idata in models.items()}

# ── PPC barplot ──────────────────────────────────────────────────────────────
PROPS = ["first", "both", "second"]
SHARPS = ["blurred", "sharp"]

fig, axes = plt.subplots(len(PROPS), len(SHARPS), figsize=(16, 10), sharey=True)
bar_w = 0.27
x = np.arange(len(UTT_ORDER))

for i, prop in enumerate(PROPS):
    for j, sharp in enumerate(SHARPS):
        ax = axes[i, j]
        cell_emp = emp[(emp["relevant_property"] == prop) & (emp["sharpness"] == sharp)]
        cell_emp = cell_emp.set_index("utt").reindex(UTT_ORDER).fillna(0)
        cell_v1 = mod_props["ext_v1"][(mod_props["ext_v1"]["relevant_property"] == prop) & (mod_props["ext_v1"]["sharpness"] == sharp)]
        cell_v1 = cell_v1.set_index("utt").reindex(UTT_ORDER).fillna(0)
        cell_v5 = mod_props["v5"][(mod_props["v5"]["relevant_property"] == prop) & (mod_props["v5"]["sharpness"] == sharp)]
        cell_v5 = cell_v5.set_index("utt").reindex(UTT_ORDER).fillna(0)

        ax.bar(x - bar_w, cell_emp["emp_prop"].values, bar_w, label="empirical", color="steelblue")
        ax.bar(x, cell_v1["model_mean"].values, bar_w, label="ext_v1",
               yerr=[cell_v1["model_mean"].values - cell_v1["model_lo"].values,
                     cell_v1["model_hi"].values - cell_v1["model_mean"].values],
               color="lightcoral", ecolor="darkred", capsize=2)
        ax.bar(x + bar_w, cell_v5["model_mean"].values, bar_w, label="v5",
               yerr=[cell_v5["model_mean"].values - cell_v5["model_lo"].values,
                     cell_v5["model_hi"].values - cell_v5["model_mean"].values],
               color="goldenrod", ecolor="darkorange", capsize=2)

        n_trials = int(cell_emp["n"].sum()) if "n" in cell_emp else int(tots[(tots["relevant_property"] == prop) & (tots["sharpness"] == sharp)]["nt"].iloc[0])
        ax.set_title(f"{prop} | {sharp}  (N={n_trials})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(UTT_ORDER, rotation=45, fontsize=7)
        ax.set_ylim(0, 1.0)
        if j == 0:
            ax.set_ylabel("proportion")
        if i == 0 and j == 0:
            ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("Production PPC — dc subset (N=3196): empirical vs ext_v1 vs v5 (F1)", fontsize=12)
fig.tight_layout()
for fmt in ("pdf", "png"):
    out = OUT_DIR / f"production_ppc_barplot_v5_dc.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

# ── Correlation plot ─────────────────────────────────────────────────────────
v1 = mod_props["ext_v1"].rename(columns={"model_mean": "p_v1", "ann": "annotation_seq_flat"})[
    ["relevant_property", "sharpness", "annotation_seq_flat", "utt", "p_v1"]]
v5 = mod_props["v5"].rename(columns={"model_mean": "p_v5", "ann": "annotation_seq_flat"})[
    ["relevant_property", "sharpness", "annotation_seq_flat", "utt", "p_v5"]]
merged = emp.merge(v1, on=["relevant_property", "sharpness", "annotation_seq_flat", "utt"], how="outer").fillna(0)
merged = merged.merge(v5, on=["relevant_property", "sharpness", "annotation_seq_flat", "utt"], how="outer").fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
for ax, col, tag, color in zip(axes, ("p_v1", "p_v5"), ("ext_v1", "v5 (F1)"), ("lightcoral", "goldenrod")):
    ax.scatter(merged["emp_prop"], merged[col], alpha=0.6, color=color, edgecolor="k", s=30)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    r, p = pearsonr(merged["emp_prop"], merged[col])
    r2 = r ** 2
    ax.set_title(f"{tag}\n r = {r:.3f}, R² = {r2:.3f} (n = {len(merged)})")
    ax.set_xlabel("empirical proportion")
    if col == "p_v1":
        ax.set_ylabel("model posterior predictive proportion")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

fig.suptitle("Predicted vs empirical proportions — dc subset", fontsize=12)
fig.tight_layout()
for fmt in ("pdf", "png"):
    out = OUT_DIR / f"production_correlation_v5_dc.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

# Also at thresholds
for thresh in (0.02, 0.05):
    sub = merged[merged["emp_prop"] >= thresh]
    r_v1, _ = pearsonr(sub["emp_prop"], sub["p_v1"])
    r_v5, _ = pearsonr(sub["emp_prop"], sub["p_v5"])
    print(f"Filtered emp >= {thresh}: n = {len(sub)}  R² ext_v1 = {r_v1**2:.3f}  R² v5 = {r_v5**2:.3f}")
