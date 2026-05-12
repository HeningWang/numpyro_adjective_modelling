"""Diagnostic: where does the contextual model's routing fail vs sufficient_dim?

Joins the iter-1 NC (Lever 1: raw LM, no gammas) with the dataset's per-trial
sufficient_dim / relevant_property / sharpness columns, then cross-tabulates
empirical vs posterior-predictive utterance distributions inside each
(condition, sufficient_dim_label) cell.

Question this answers:
- Is the over/under-prediction at the (condition, utterance) level concentrated
  on trials where sufficient_dim is well-defined? → the lambda_suff boost is
  the failure point (boost not strong enough OR alpha baseline too imbalanced
  to override).
- Or is it concentrated on relevant_property=='both' (no single sufficient dim)?
  → the model's behavior on ambiguous trials is the failure point, not the
  boost mechanism.

Reads:
    inference_data/mcmc_results_contextual_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter1.nc

Produces:
    results/contextual_dc/iter1_suffdim_crosstab.csv
    results/contextual_dc/iter1_suffdim_summary.txt
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Tuple

warnings.filterwarnings("ignore")

import arviz as az
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROD_DIR = PROJECT_ROOT / "05-modelling-production-data"
CSV_PATH = PROJECT_ROOT / "01-dataset" / "01-production-data-preprocessed.csv"
NC_PATH = PROD_DIR / "inference_data" / (
    "mcmc_results_contextual_speaker_hier_dc"
    "_warmup4000_samples2000_chains4_vast_iter1.nc"
)
RES_DIR = PROD_DIR / "results" / "contextual_dc"
RES_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the dc-subset loading + sequence-flat construction from
# helper.import_dataset() so we agree on the same encoding the model saw.
sys.path.insert(0, str(PROD_DIR))
from helper import import_dataset  # noqa: E402

DC_CONDITIONS = ("erdc", "zrdc", "brdc")
SUFFDIM_LABEL = {-1: "both/none", 0: "D", 1: "C", 2: "F"}


def main(nc_path: Path | None = None, out_prefix: str = "iter1") -> None:
    nc_path = Path(nc_path) if nc_path is not None else NC_PATH
    if not nc_path.exists():
        raise SystemExit(f".nc file not found: {nc_path}")
    print(f"Loading: {nc_path}")
    idata = az.from_netcdf(str(nc_path))
    pp = idata.posterior_predictive["obs"]
    n_items_obs = int(idata.observed_data["obs"].sizes["item"])
    print(f"  pp items: {pp.sizes['item']},  observed items: {n_items_obs}")

    # Load dataset (full 9 conditions) and reproduce helper's sufficient_dim
    print("Loading dataset via helper.import_dataset()...")
    data = import_dataset(CSV_PATH)
    df_all = data["df"]
    suff_dim_all = np.asarray(data["sufficient_dim"])

    # Restrict to the dc subset — same filter the inference run used.
    dc_mask = df_all["conditions"].isin(DC_CONDITIONS).to_numpy()
    df_dc = df_all[dc_mask].reset_index(drop=True).copy()
    df_dc["sufficient_dim"] = suff_dim_all[dc_mask]
    print(f"  df_dc rows: {len(df_dc)} (expected to match NC items)")
    if len(df_dc) != n_items_obs:
        raise SystemExit(
            f"Row-count mismatch: dc filter gave {len(df_dc)} but "
            f"NC has {n_items_obs} items."
        )

    # Sanity-check observed alignment
    obs_codes = idata.observed_data["obs"].values.astype(int)
    df_codes = df_dc["annotation_seq_flat"].astype(int).to_numpy()
    pct = float(np.mean(obs_codes == df_codes))
    print(f"  obs ↔ df_dc match: {pct:.1%}")
    if pct < 0.99:
        print("  WARNING: alignment is poor; positional join may be unsafe.")

    # Utterance labels (flat → string)
    df_dc["annotation"] = df_dc["annotation"].astype(str)
    code_to_label = (
        df_dc.drop_duplicates("annotation_seq_flat")
        .set_index("annotation_seq_flat")["annotation"]
        .to_dict()
    )
    utt_labels = [code_to_label[c] for c in sorted(code_to_label)]
    df_dc["suffdim_label"] = df_dc["sufficient_dim"].map(SUFFDIM_LABEL)
    print(f"  utterance labels: {utt_labels}")
    print(f"  suff_dim values in dc: "
          f"{df_dc['suffdim_label'].value_counts().to_dict()}")

    # Cross-tab by (condition, suffdim_label) ----------------------------------
    # 1) Human utterance distribution per cell
    cell_cols = ["conditions", "suffdim_label", "relevant_property"]
    print("\nComputing human distributions...")
    human_dist = (
        df_dc.groupby(cell_cols + ["annotation_seq_flat"])
        .size().rename("n_human").reset_index()
    )
    cell_totals = (
        df_dc.groupby(cell_cols).size().rename("n_cell").reset_index()
    )
    human_dist = human_dist.merge(cell_totals, on=cell_cols)
    human_dist["p_human"] = human_dist["n_human"] / human_dist["n_cell"]

    # 2) Model utterance distribution — average over posterior predictive draws
    print("Computing model distributions...")
    pp_flat = pp.stack(sample=("chain", "draw")).transpose("sample", "item").values
    n_draws_used = min(500, pp_flat.shape[0])
    draw_idx = np.linspace(0, pp_flat.shape[0] - 1, n_draws_used, dtype=int)
    rows = []
    for d in draw_idx:
        tmp = df_dc[cell_cols].copy()
        tmp["utt_code"] = pp_flat[d, :]
        cnt = tmp.groupby(cell_cols + ["utt_code"]).size().rename("n").reset_index()
        tot = tmp.groupby(cell_cols).size().rename("n_total").reset_index()
        m = cnt.merge(tot, on=cell_cols)
        m["p"] = m["n"] / m["n_total"]
        rows.append(m[cell_cols + ["utt_code", "p"]])
    stacked = pd.concat(rows, ignore_index=True)
    model_dist = (
        stacked.groupby(cell_cols + ["utt_code"])["p"]
        .mean().rename("p_model").reset_index()
        .rename(columns={"utt_code": "annotation_seq_flat"})
    )

    # 3) Merge for residual analysis
    full = (
        pd.MultiIndex.from_product(
            [DC_CONDITIONS,
             [SUFFDIM_LABEL[k] for k in (-1, 0, 1)],
             ["first", "second", "both"],
             sorted(code_to_label.keys())],
            names=cell_cols + ["annotation_seq_flat"],
        ).to_frame(index=False)
    )
    merged = (
        full
        .merge(human_dist, on=cell_cols + ["annotation_seq_flat"], how="left")
        .merge(model_dist, on=cell_cols + ["annotation_seq_flat"], how="left")
    )
    merged["p_human"] = merged["p_human"].fillna(0.0)
    merged["p_model"] = merged["p_model"].fillna(0.0)
    merged["n_cell"] = merged["n_cell"].fillna(0).astype(int)
    merged["n_human"] = merged["n_human"].fillna(0).astype(int)
    merged["utt"] = merged["annotation_seq_flat"].map(code_to_label)
    merged["residual"] = merged["p_model"] - merged["p_human"]
    merged["abs_residual"] = merged["residual"].abs()

    # Drop cells with zero trials (impossible suff_dim × condition × relevant_property combos)
    merged = merged[merged["n_cell"] > 0].reset_index(drop=True)

    out_csv = RES_DIR / f"{out_prefix}_suffdim_crosstab.csv"
    merged.to_csv(out_csv, index=False)
    print(f"  [csv] {out_csv.relative_to(PROJECT_ROOT)}")

    # Build summary tables ------------------------------------------------------
    print("\nBuilding summary tables...")

    # (a) Trial counts per (condition, suff_dim)
    counts_tbl = (
        df_dc.groupby(["conditions", "suffdim_label"])
        .size().rename("n_trials").reset_index()
        .pivot(index="suffdim_label", columns="conditions", values="n_trials")
        .fillna(0).astype(int)
    )

    # (b) For each (condition, suff_dim), what's the empirical top utterance and
    # what's the model's top? This is the routing-correctness check.
    def _top(g):
        g_sorted = g.sort_values("p_human", ascending=False).head(3)
        m_sorted = g.sort_values("p_model", ascending=False).head(3)
        return pd.Series({
            "human_top1": f"{g_sorted.iloc[0]['utt']}({g_sorted.iloc[0]['p_human']:.2f})",
            "human_top2": f"{g_sorted.iloc[1]['utt']}({g_sorted.iloc[1]['p_human']:.2f})",
            "model_top1": f"{m_sorted.iloc[0]['utt']}({m_sorted.iloc[0]['p_model']:.2f})",
            "model_top2": f"{m_sorted.iloc[1]['utt']}({m_sorted.iloc[1]['p_model']:.2f})",
            "abs_residual_sum": float(g["abs_residual"].sum()),
        })

    routing_tbl = (
        merged.groupby(["conditions", "suffdim_label"])
        .apply(_top).reset_index()
        .sort_values(["conditions", "suffdim_label"])
    )

    # (c) Absolute residual mass per (suff_dim, condition) — where does the
    # mismatch live? Strong signal: if 'both/none' has a big share, the issue
    # is the unrouted trials; if a specific suff_dim cell does, it's a routing
    # failure to that dim.
    mass_tbl = (
        merged.groupby(["suffdim_label", "conditions"])["abs_residual"]
        .sum().unstack(fill_value=0.0)
    )

    # (d) Largest signed residuals across ALL cells — the worst single mismatches
    top_signed = (
        merged.sort_values("abs_residual", ascending=False)
        .head(15)[cell_cols + ["utt", "p_human", "p_model", "residual", "n_cell"]]
        .reset_index(drop=True)
    )

    lines = [
        "Iter-1 routing diagnostic (Lever 1: raw LM, no gammas)",
        f"  .nc file:  {nc_path.name}",
        f"  dc trials: {len(df_dc)} (113 participants)",
        "",
        "Trial counts per (suff_dim, condition):",
        counts_tbl.to_string(),
        "",
        "Top-2 empirical vs top-2 model utterance per (condition, suff_dim):",
        "  (if these differ → routing failure at this cell)",
        routing_tbl.to_string(index=False),
        "",
        "Absolute residual mass per (suff_dim, condition):",
        "  (rows = suff_dim; cols = condition; mass = Σ |p_model − p_human| over utterances)",
        mass_tbl.to_string(float_format=lambda x: f"{x:.3f}"),
        "",
        "Top 15 signed residuals across all cells:",
        top_signed.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
    ]
    summary_text = "\n".join(lines)
    out_txt = RES_DIR / f"{out_prefix}_suffdim_summary.txt"
    out_txt.write_text(summary_text + "\n")
    print()
    print(summary_text)
    print()
    print(f"All outputs in: {RES_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="suff_dim routing diagnostic.")
    parser.add_argument("--nc-path", type=str, default=None)
    parser.add_argument("--out-prefix", type=str, default="iter1")
    args = parser.parse_args()
    main(nc_path=args.nc_path, out_prefix=args.out_prefix)
