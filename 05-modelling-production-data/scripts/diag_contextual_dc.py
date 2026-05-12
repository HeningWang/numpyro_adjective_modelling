"""Diagnostics for contextual_dc NC files: r-hat, ESS, comparison."""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
INF = HERE.parent / "inference_data"

COLAB = INF / "mcmc_results_contextual_speaker_hier_dc_warmup2000_samples1000_chains4_colab.nc"
PRE = INF / "mcmc_results_contextual_speaker_hier_dc_warmup2000_samples1000_chains4_pre_densemass.nc"


def summarize(nc_path: Path) -> pd.DataFrame:
    idata = az.from_netcdf(str(nc_path))
    summary = az.summary(idata, kind="diagnostics")
    return summary


def show_top_rhat(summary: pd.DataFrame, label: str, n: int = 30) -> None:
    print(f"\n=== {label}: top {n} parameters by r-hat ===")
    cols = [c for c in ("ess_bulk", "ess_tail", "r_hat") if c in summary.columns]
    print(summary.sort_values("r_hat", ascending=False)[cols].head(n).to_string())


def show_targets(summary: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label}: key params ===")
    keys = ["alpha_D", "alpha_C", "alpha_F", "log_beta_lm", "tau", "epsilon"]
    cols = [c for c in ("ess_bulk", "ess_tail", "r_hat") if c in summary.columns]
    available = [k for k in keys if k in summary.index]
    if available:
        print(summary.loc[available, cols].to_string())
    delta_rows = [i for i in summary.index if i.startswith("delta[")]
    delta_raw_rows = [i for i in summary.index if i.startswith("delta_raw[")]
    if delta_rows:
        worst = summary.loc[delta_rows].sort_values("r_hat", ascending=False).head(5)
        print(f"\n  worst-5 delta r-hat (out of {len(delta_rows)}):")
        print(worst[cols].to_string())
        print(f"\n  delta r-hat: max={summary.loc[delta_rows, 'r_hat'].max():.4f}, "
              f"frac>1.01={(summary.loc[delta_rows, 'r_hat'] > 1.01).mean():.2%}")
    if delta_raw_rows:
        print(f"\n  delta_raw r-hat: max={summary.loc[delta_raw_rows, 'r_hat'].max():.4f}, "
              f"min ess_bulk={summary.loc[delta_raw_rows, 'ess_bulk'].min():.0f}")


def compare(pre: pd.DataFrame, post: pd.DataFrame) -> None:
    print("\n=== PRE-densemass vs COLAB (fix): key params ===")
    keys = ["alpha_D", "alpha_C", "alpha_F", "log_beta_lm", "tau", "epsilon"]
    rows = []
    for k in keys:
        if k in pre.index and k in post.index:
            rows.append({
                "param": k,
                "r_hat_pre": pre.loc[k, "r_hat"],
                "r_hat_post": post.loc[k, "r_hat"],
                "ess_bulk_pre": pre.loc[k, "ess_bulk"],
                "ess_bulk_post": post.loc[k, "ess_bulk"],
            })
    print(pd.DataFrame(rows).to_string(index=False))

    # delta summary
    pre_delta = [i for i in pre.index if i.startswith("delta[")]
    post_delta = [i for i in post.index if i.startswith("delta[")]
    if pre_delta and post_delta:
        print("\n=== delta[*] aggregate ===")
        print(pd.DataFrame([
            {"file": "PRE",
             "max_r_hat": pre.loc[pre_delta, "r_hat"].max(),
             "frac_gt_1.01": (pre.loc[pre_delta, "r_hat"] > 1.01).mean(),
             "min_ess_bulk": pre.loc[pre_delta, "ess_bulk"].min()},
            {"file": "COLAB",
             "max_r_hat": post.loc[post_delta, "r_hat"].max(),
             "frac_gt_1.01": (post.loc[post_delta, "r_hat"] > 1.01).mean(),
             "min_ess_bulk": post.loc[post_delta, "ess_bulk"].min()},
        ]).to_string(index=False))


def main() -> int:
    if not COLAB.exists():
        print(f"Missing: {COLAB}", file=sys.stderr)
        return 1
    if not PRE.exists():
        print(f"Missing: {PRE}", file=sys.stderr)
        return 1

    print(f"Loading PRE  : {PRE.name}")
    pre = summarize(PRE)
    print(f"Loading COLAB: {COLAB.name}")
    post = summarize(COLAB)

    show_top_rhat(post, "COLAB (post-fix)", n=30)
    show_targets(post, "COLAB (post-fix)")
    show_targets(pre, "PRE-densemass")
    compare(pre, post)
    return 0


if __name__ == "__main__":
    sys.exit(main())
