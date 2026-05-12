"""PPC review of the Lever-1 best model (iter1, raw LM, no gammas) — stratified
by experimental condition, with an extra token-set equivalence-class summary
used to decide whether a near-target lapse kernel is worth running.

Reads: inference_data/mcmc_results_contextual_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter1.nc

Produces (all gitignored):
    results/contextual_dc/iter1_residuals_by_condition.csv
    results/contextual_dc/iter1_residuals_by_tokenset.csv
    results/contextual_dc/iter1_top_residual_cells.csv
    results/contextual_dc/iter1_summary.txt
    figures/contextual_dc/iter1_ppc_barplot_by_condition.{pdf,png}
    figures/contextual_dc/iter1_correlation.{pdf,png}

Self-contained: only depends on arviz, numpy, pandas, matplotlib, scipy.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Tuple

warnings.filterwarnings("ignore")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROD_DIR = PROJECT_ROOT / "05-modelling-production-data"
CSV_PATH = PROJECT_ROOT / "01-dataset" / "01-production-data-preprocessed.csv"
NC_PATH = PROD_DIR / "inference_data" / (
    "mcmc_results_contextual_speaker_hier_dc"
    "_warmup4000_samples2000_chains4_vast_iter1.nc"
)
FIG_DIR = PROD_DIR / "figures" / "contextual_dc"
RES_DIR = PROD_DIR / "results" / "contextual_dc"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS_OF_INTEREST = (
    "erdc", "zrdc", "brdc",
    "erdf", "zrdf", "brdf",
    "ercf", "zrcf", "brcf",
)
DC_CONDITIONS = ("erdc", "zrdc", "brdc")
SYMBOL_TO_INDEX = {"D": 0, "C": 1, "F": 2}
MAX_UTT_LEN = 3

GROUP_COLS = ["conditions", "sharpness"]
SHARPS = ["blurred", "sharp"]


def build_annotation_seq_flat(df_all: pd.DataFrame) -> Tuple[pd.Series, dict[int, str]]:
    """Mirror helper.import_dataset()'s annotation_seq_flat construction."""
    utt_strings = df_all["annotation"].astype(str).tolist()
    sequences = np.full((len(utt_strings), MAX_UTT_LEN), -1, dtype=np.int32)
    for row_idx, utt in enumerate(utt_strings):
        for col_idx, symbol in enumerate(utt):
            sequences[row_idx, col_idx] = SYMBOL_TO_INDEX[symbol]
    unique_sequences = np.unique(sequences, axis=0)
    seq_to_idx = {tuple(seq): idx for idx, seq in enumerate(unique_sequences.tolist())}
    flat = np.array([seq_to_idx[tuple(s)] for s in sequences.tolist()], dtype=np.int32)
    inv_symbol = {v: k for k, v in SYMBOL_TO_INDEX.items()}
    flat_to_cat = {
        idx: "".join(inv_symbol[s] for s in seq if s >= 0)
        for seq, idx in seq_to_idx.items()
    }
    return pd.Series(flat, index=df_all.index, name="annotation_seq_flat"), flat_to_cat


def load_dataset_dc() -> Tuple[pd.DataFrame, dict[int, str]]:
    df = pd.read_csv(CSV_PATH).dropna(subset=["annotation"]).copy()
    df = df[df["conditions"].isin(CONDITIONS_OF_INTEREST)].reset_index(drop=True)
    seq_flat, flat_to_cat = build_annotation_seq_flat(df)
    df["annotation_seq_flat"] = seq_flat.to_numpy()
    df_dc = df[df["conditions"].isin(DC_CONDITIONS)].reset_index(drop=True)
    return df_dc, flat_to_cat


def utt_label_to_tokenset_class(utt: str) -> str:
    """Map an utterance label to its token-set equivalence class.

    Two utterances share a class iff their token multisets are equal.
    Singletons: 'D', 'C', 'F'. Length-2: '{D,C}', '{D,F}', '{C,F}'.
    Length-3: '{D,C,F}'.
    """
    tokens = tuple(sorted(set(utt)))
    if len(tokens) == 1:
        return tokens[0]
    return "{" + ",".join(tokens) + "}"


def compute_model_proportions(
    pp_flat: np.ndarray, df_dc: pd.DataFrame, max_draws: int = 500,
) -> pd.DataFrame:
    n_draws = pp_flat.shape[0]
    draws = np.linspace(0, n_draws - 1, min(max_draws, n_draws), dtype=int)
    records = []
    for d in draws:
        tmp = df_dc[GROUP_COLS].copy()
        tmp["utt_code"] = pp_flat[d, :]
        counts = tmp.groupby(GROUP_COLS + ["utt_code"]).size().rename("n").reset_index()
        tot = tmp.groupby(GROUP_COLS).size().rename("n_total").reset_index()
        merged = counts.merge(tot, on=GROUP_COLS)
        merged["p"] = merged["n"] / merged["n_total"]
        records.append(merged[GROUP_COLS + ["utt_code", "p"]])
    stacked = pd.concat(records, ignore_index=True)
    summary = (
        stacked.groupby(GROUP_COLS + ["utt_code"])["p"]
        .agg(
            model_mean="mean",
            model_lo=lambda x: np.percentile(x, 2.5),
            model_hi=lambda x: np.percentile(x, 97.5),
        )
        .reset_index()
    )
    return summary


def compute_human_proportions(
    df_dc: pd.DataFrame, all_codes: list[int], n_boot: int = 2000, seed: int = 431,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for cond_vals, sub in df_dc.groupby(GROUP_COLS):
        labels = sub["annotation_seq_flat"].to_numpy()
        n = len(labels)
        boot_props = {c: np.empty(n_boot) for c in all_codes}
        for b in range(n_boot):
            sample = rng.choice(labels, size=n, replace=True)
            for c in all_codes:
                boot_props[c][b] = np.mean(sample == c)
        for c in all_codes:
            bp = boot_props[c]
            records.append({
                GROUP_COLS[0]: cond_vals[0],
                GROUP_COLS[1]: cond_vals[1],
                "utt_code": int(c),
                "human_mean": float(np.mean(labels == c)),
                "human_lo": float(np.percentile(bp, 2.5)),
                "human_hi": float(np.percentile(bp, 97.5)),
                "n": int(n),
            })
    return pd.DataFrame(records)


def plot_ppc_barplot_by_condition(
    emp: pd.DataFrame, mod: pd.DataFrame, utt_order: list[str], out_stem: Path,
) -> None:
    fig, axes = plt.subplots(
        len(DC_CONDITIONS), len(SHARPS), figsize=(16, 10), sharey=True,
    )
    x = np.arange(len(utt_order))
    bar_w = 0.38

    for i, cond in enumerate(DC_CONDITIONS):
        for j, sharp in enumerate(SHARPS):
            ax = axes[i, j]
            e = (
                emp[(emp["conditions"] == cond) & (emp["sharpness"] == sharp)]
                .set_index("utt").reindex(utt_order).fillna(0)
            )
            m = (
                mod[(mod["conditions"] == cond) & (mod["sharpness"] == sharp)]
                .set_index("utt").reindex(utt_order).fillna(0)
            )
            ax.bar(
                x - bar_w / 2, e["human_mean"].values, bar_w,
                yerr=[
                    e["human_mean"].values - e["human_lo"].values,
                    e["human_hi"].values - e["human_mean"].values,
                ],
                label="empirical", color="#7581B3", ecolor="#414C76", capsize=2,
            )
            ax.bar(
                x + bar_w / 2, m["model_mean"].values, bar_w,
                yerr=[
                    m["model_mean"].values - m["model_lo"].values,
                    m["model_hi"].values - m["model_mean"].values,
                ],
                label="Lever-1 (posterior pred.)",
                color="#C65353", ecolor="#993333", capsize=2,
            )
            n_trials = int(e["n"].iloc[0]) if "n" in e.columns and len(e) else 0
            ax.set_title(f"{cond} | {sharp}  (N={n_trials})", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(utt_order, rotation=45, fontsize=8)
            ax.set_ylim(0, 1.0)
            if j == 0:
                ax.set_ylabel("proportion")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Lever-1 PPC, dc subset — empirical vs posterior predictive by condition",
        fontsize=12,
    )
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        out = out_stem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  [fig] {out.relative_to(PROJECT_ROOT)}")
    plt.close(fig)


def plot_correlation(merged: pd.DataFrame, out_stem: Path) -> dict:
    r_all, _ = pearsonr(merged["human_mean"], merged["model_mean"])
    stats = {"n_cells": len(merged), "r": float(r_all), "r2": float(r_all ** 2)}
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.errorbar(
        merged["human_mean"], merged["model_mean"],
        xerr=[merged["human_mean"] - merged["human_lo"],
              merged["human_hi"] - merged["human_mean"]],
        yerr=[merged["model_mean"] - merged["model_lo"],
              merged["model_hi"] - merged["model_mean"]],
        fmt="o", ms=4, color="#C65353", ecolor="#bbbbbb",
        elinewidth=0.6, alpha=0.7, capsize=0,
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("empirical proportion")
    ax.set_ylabel("posterior-predictive proportion")
    ax.set_title(
        f"Lever-1, dc subset (r = {r_all:.3f}, R² = {r_all ** 2:.3f}, n = {len(merged)})",
        fontsize=11,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        out = out_stem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  [fig] {out.relative_to(PROJECT_ROOT)}")
    plt.close(fig)
    return stats


def main(nc_path: Path | None = None, out_prefix: str = "iter1") -> None:
    nc_path = Path(nc_path) if nc_path is not None else NC_PATH
    if not nc_path.exists():
        raise SystemExit(f".nc file not found: {nc_path}")
    print(f"Loading: {nc_path}")
    idata = az.from_netcdf(str(nc_path))

    n_items_obs = int(idata.observed_data["obs"].sizes["item"])
    pp = idata.posterior_predictive["obs"]
    print(
        f"  posterior_predictive: chains={pp.sizes['chain']} "
        f"draws={pp.sizes['draw']} items={pp.sizes['item']}"
    )
    print(f"  observed items:       {n_items_obs}")

    df_dc, flat_to_cat = load_dataset_dc()
    print(f"  df_dc rows:           {len(df_dc)}")
    if len(df_dc) != n_items_obs:
        raise SystemExit(
            f"Row-count mismatch: dataset filter gave {len(df_dc)} but "
            f".nc has {n_items_obs} items. Aborting (alignment unsafe)."
        )

    obs_codes = idata.observed_data["obs"].values.astype(int)
    df_codes = df_dc["annotation_seq_flat"].to_numpy().astype(int)
    matches = int(np.sum(obs_codes == df_codes))
    pct = matches / len(df_codes)
    print(f"  obs ↔ df_dc match:    {matches}/{len(df_codes)} ({pct:.1%})")
    if pct < 0.99:
        print("  WARNING: order alignment is poor; PPC barplots assume positional match.")

    all_codes = sorted(df_dc["annotation_seq_flat"].unique().tolist())
    utt_order = [flat_to_cat[c] for c in sorted(flat_to_cat.keys())]
    print(f"  utterance types:      {len(utt_order)}: {utt_order}")

    print("Computing model proportions (stratified by conditions × sharpness)...")
    pp_flat = pp.stack(sample=("chain", "draw")).transpose("sample", "item").values
    mod = compute_model_proportions(pp_flat, df_dc, max_draws=500)
    mod["utt"] = mod["utt_code"].map(flat_to_cat)

    print("Computing human bootstrap CIs...")
    emp = compute_human_proportions(df_dc, all_codes, n_boot=2000, seed=431)
    emp["utt"] = emp["utt_code"].map(flat_to_cat)

    # Full cross-product over (conditions × sharpness × utt_code) for residuals
    full_cells = pd.MultiIndex.from_product(
        [DC_CONDITIONS, SHARPS, sorted(flat_to_cat.keys())],
        names=GROUP_COLS + ["utt_code"],
    ).to_frame(index=False)
    merged = (
        full_cells
        .merge(emp, on=GROUP_COLS + ["utt_code"], how="left")
        .merge(mod, on=GROUP_COLS + ["utt_code"], how="left")
    )
    merged["utt"] = merged["utt_code"].map(flat_to_cat)
    for col in ["human_mean", "human_lo", "human_hi",
                "model_mean", "model_lo", "model_hi"]:
        merged[col] = merged[col].fillna(0.0)
    merged["n"] = merged["n"].fillna(0).astype(int)
    merged["residual"] = merged["model_mean"] - merged["human_mean"]
    merged["abs_residual"] = merged["residual"].abs()
    merged["tokenset_class"] = merged["utt"].map(utt_label_to_tokenset_class)

    obs_cells = merged[merged["n"] > 0].copy()
    print(f"  cells with observations: {len(obs_cells)}")

    print("Plotting...")
    plot_ppc_barplot_by_condition(
        emp, mod, utt_order, FIG_DIR / f"{out_prefix}_ppc_barplot_by_condition",
    )
    corr_stats = plot_correlation(obs_cells, FIG_DIR / f"{out_prefix}_correlation")

    # Token-set equivalence-class aggregation — the central decision artifact
    print("Aggregating residuals by token-set class...")
    tokenset_agg = (
        merged.groupby(["conditions", "tokenset_class"])
        .agg(
            signed_residual_sum=("residual", "sum"),
            abs_residual_sum=("abs_residual", "sum"),
            n_cells=("residual", "size"),
        )
        .reset_index()
    )
    tokenset_agg["cancellation_ratio"] = (
        1.0 - tokenset_agg["signed_residual_sum"].abs()
        / tokenset_agg["abs_residual_sum"].replace(0, np.nan)
    ).fillna(0.0)

    print("Saving CSVs...")
    merged.to_csv(RES_DIR / f"{out_prefix}_residuals_by_condition.csv", index=False)
    tokenset_agg.to_csv(RES_DIR / f"{out_prefix}_residuals_by_tokenset.csv", index=False)
    top_resid = (
        obs_cells.sort_values("abs_residual", ascending=False)
        .head(10)[GROUP_COLS + ["utt", "tokenset_class",
                                "human_mean", "model_mean", "residual"]]
    )
    top_resid.to_csv(RES_DIR / f"{out_prefix}_top_residual_cells.csv", index=False)
    for p in [
        RES_DIR / f"{out_prefix}_residuals_by_condition.csv",
        RES_DIR / f"{out_prefix}_residuals_by_tokenset.csv",
        RES_DIR / f"{out_prefix}_top_residual_cells.csv",
    ]:
        print(f"  [csv] {p.relative_to(PROJECT_ROOT)}")

    # Cancellation table: pivot for compact display
    cancel_table = tokenset_agg.pivot(
        index="tokenset_class",
        columns="conditions",
        values="cancellation_ratio",
    ).fillna(0.0)

    abs_table = tokenset_agg.pivot(
        index="tokenset_class",
        columns="conditions",
        values="abs_residual_sum",
    ).fillna(0.0)

    lines = [
        f"PPC review — {out_prefix}",
        f"  .nc file:           {nc_path.name}",
        f"  dc trials:          {len(df_dc)} (113 participants)",
        f"  utterance types:    {len(utt_order)}",
        f"  cells (obs):        {len(obs_cells)} of {len(merged)}",
        "",
        "Fit:",
        f"  Pearson r:          {corr_stats['r']:.3f}",
        f"  R²:                 {corr_stats['r2']:.3f}",
        "",
        "Token-set cancellation ratio per (class, condition):",
        "  (1.0 = signed residuals cancel exactly within class — near-target lapse helps;",
        "   0.0 = no cancellation — class members all biased same direction — won't help)",
        cancel_table.to_string(float_format=lambda x: f"{x:.3f}"),
        "",
        "Absolute residual mass per (class, condition):",
        "  (where the lapse has the biggest fit gap to close)",
        abs_table.to_string(float_format=lambda x: f"{x:.3f}"),
        "",
        "Top 10 residual cells:",
        top_resid.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
    ]
    summary_text = "\n".join(lines)
    (RES_DIR / f"{out_prefix}_summary.txt").write_text(summary_text + "\n")
    print()
    print(summary_text)
    print()
    print(
        f"Outputs in {RES_DIR.relative_to(PROJECT_ROOT)}/  and  "
        f"{FIG_DIR.relative_to(PROJECT_ROOT)}/"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPC review for the contextual_dc speaker.")
    parser.add_argument(
        "--nc-path", type=str, default=None,
        help="Path to .nc file (default: iter1 Lever-1 NC).",
    )
    parser.add_argument(
        "--out-prefix", type=str, default="iter1",
        help="Filename prefix for all output CSVs/figures/summary (default: 'iter1').",
    )
    args = parser.parse_args()
    main(nc_path=args.nc_path, out_prefix=args.out_prefix)
