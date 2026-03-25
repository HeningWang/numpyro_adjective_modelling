"""Shared posterior analysis utilities for slider and production datasets.

This module provides dataset-agnostic functions for loading ArviZ inference
data, computing MCMC diagnostics, LOO model comparison, participant intercept
plots, correlation scatter plots, and summary text output.

Usage:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from shared.posterior_utils import load_models, run_mcmc_diagnostics, ...
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm, pearsonr


# ── Loading ──────────────────────────────────────────────────────────────────

def load_models(
    model_specs: Dict[str, str],
    inference_dir: str,
) -> Dict[str, az.InferenceData]:
    """Load multiple ArviZ .nc files.

    Parameters
    ----------
    model_specs : dict
        {model_name: filename} mapping.
    inference_dir : str
        Directory containing the .nc files.

    Returns
    -------
    dict of {model_name: az.InferenceData}
    """
    idata_dict = {}
    for name, fname in model_specs.items():
        fpath = os.path.join(inference_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [skip] {fpath} not found")
            continue
        idata_dict[name] = az.from_netcdf(fpath)
        print(f"  [loaded] {name}: {fpath}")
    return idata_dict


# ── MCMC Diagnostics ────────────────────────────────────────────────────────

def run_mcmc_diagnostics(
    idata: az.InferenceData,
    model_name: str,
    var_names: Optional[List[str]] = None,
    out_dir: str = "results/diagnostics",
    fmt: str = "pdf",
) -> str:
    """Generate and save MCMC diagnostic plots: trace, pair, summary.

    Parameters
    ----------
    idata : az.InferenceData
    model_name : str
        Used for filenames and titles.
    var_names : list of str, optional
        Variables to include. If None, uses all posterior variables
        except 'delta' (participant intercepts).
    out_dir : str
    fmt : str
        Figure format ('pdf' or 'png').

    Returns
    -------
    str : plain-text summary string
    """
    os.makedirs(out_dir, exist_ok=True)

    if var_names is None:
        var_names = [v for v in idata.posterior.data_vars if v != "delta"]

    # Summary table
    summary = az.summary(idata, var_names=var_names)
    summary_str = f"=== {model_name} ===\n{summary.to_string()}\n"

    # Trace plot
    axes = az.plot_trace(idata, var_names=var_names, compact=True)
    fig = axes.ravel()[0].get_figure()
    fig.suptitle(model_name, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"trace_{model_name}.{fmt}"), bbox_inches="tight")
    plt.close(fig)

    # Pair plot (only if 2-6 variables to keep readable)
    if 2 <= len(var_names) <= 6:
        axes_pair = az.plot_pair(idata, var_names=var_names, kind="kde", marginals=True)
        fig_pair = axes_pair.ravel()[0].get_figure()
        fig_pair.suptitle(model_name, fontsize=12)
        fig_pair.tight_layout()
        fig_pair.savefig(os.path.join(out_dir, f"pair_{model_name}.{fmt}"), bbox_inches="tight")
        plt.close(fig_pair)

    return summary_str


# ── Participant Intercepts ──────────────────────────────────────────────────

def plot_participant_intercepts(
    idata: az.InferenceData,
    model_name: str,
    out_dir: str = "results/diagnostics",
    fmt: str = "pdf",
) -> Optional[str]:
    """Forest plot + histogram of participant-level random intercepts (delta).

    Returns a summary string, or None if no delta in posterior.
    """
    if "delta" not in idata.posterior.data_vars:
        return None

    os.makedirs(out_dir, exist_ok=True)

    delta_samples = np.asarray(idata.posterior["delta"])     # (chains, draws, n_p)
    delta_flat = delta_samples.reshape(-1, delta_samples.shape[-1])  # (S, n_p)
    delta_mean = delta_flat.mean(axis=0)
    delta_lo = np.percentile(delta_flat, 2.5, axis=0)
    delta_hi = np.percentile(delta_flat, 97.5, axis=0)
    n_p = delta_flat.shape[1]
    order = np.argsort(delta_mean)

    # Tau
    tau_samples = np.asarray(idata.posterior["tau"]).ravel() if "tau" in idata.posterior.data_vars else None
    tau_mean = tau_samples.mean() if tau_samples is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_p * 0.12)))

    # Left: forest plot
    ax = axes[0]
    y_pos = np.arange(n_p)
    ax.barh(y_pos, delta_mean[order], color="#1f77b4", alpha=0.6, height=0.8)
    ax.hlines(y_pos, delta_lo[order], delta_hi[order],
              color="#1f77b4", lw=1.2, alpha=0.7)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_ylabel("Participant (sorted)")
    ax.set_xlabel("δ (intercept)")
    ax.set_title(f"{model_name}: participant intercepts")

    # Right: histogram
    ax2 = axes[1]
    ax2.hist(delta_mean, bins=min(20, n_p // 2 + 1), color="#1f77b4",
             alpha=0.6, density=True, edgecolor="white")
    if tau_mean is not None:
        x_range = np.linspace(delta_mean.min() - 0.05, delta_mean.max() + 0.05, 200)
        ax2.plot(x_range, sp_norm.pdf(x_range, 0, tau_mean),
                 color="#d62728", lw=2, label=f"N(0, τ={tau_mean:.3f})")
        ax2.legend()
    ax2.axvline(0, color="black", lw=1, ls="--")
    ax2.set_xlabel("δ (intercept)")
    ax2.set_title("Distribution of participant intercepts")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"intercepts_{model_name}.{fmt}"), bbox_inches="tight")
    plt.close(fig)

    summary = (
        f"  Participant intercepts ({model_name}): n={n_p}, "
        f"mean(δ)={delta_mean.mean():.4f}, sd(δ)={delta_mean.std():.4f}"
    )
    if tau_mean is not None:
        summary += f", τ={tau_mean:.4f}"
    return summary


# ── LOO Comparison ──────────────────────────────────────────────────────────

def compute_loo_comparison(
    idata_dict: Dict[str, az.InferenceData],
    var_name: str = "obs",
) -> Tuple[pd.DataFrame, Dict[str, az.ELPDData]]:
    """Run az.compare and return the comparison table + per-model LOO objects.

    Returns
    -------
    comparison : pd.DataFrame
        Ranked model comparison table.
    loo_results : dict
        {model_name: az.loo(...)} for downstream Pareto-k analysis.
    """
    loo_results = {}
    for name, idata in idata_dict.items():
        loo_results[name] = az.loo(idata, var_name=var_name, pointwise=True)

    comparison = az.compare(idata_dict, ic="loo", method="stacking", var_name=var_name)
    return comparison, loo_results


def plot_loo_comparison(
    comparison: pd.DataFrame,
    loo_results: Dict[str, az.ELPDData],
    out_dir: str = "results/stats",
    fmt: str = "pdf",
) -> str:
    """ELPD dot plot + Pareto-k diagnostics grid. Returns summary string."""
    os.makedirs(out_dir, exist_ok=True)

    model_names = list(comparison.index)
    n_models = len(model_names)

    # ── ELPD dot plot ──
    fig, ax = plt.subplots(figsize=(7, max(3, n_models * 0.8)))
    elpds = comparison["elpd_loo"].values
    ses = comparison["se"].values
    y_pos = np.arange(n_models)

    for i in range(n_models):
        ax.errorbar(elpds[i], i, xerr=ses[i], fmt="o", color="#1f77b4",
                    ms=8, capsize=6, capthick=1.5, elinewidth=1.5)
        ax.text(elpds[i] + ses[i] + abs(elpds[i]) * 0.005, i,
                f"{elpds[i]:.0f}", va="center", ha="left", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.set_xlabel("ELPD (LOO)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"loo_elpd.{fmt}"), bbox_inches="tight")
    plt.close(fig)

    # ── Pareto-k grid ──
    ncols = min(2, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows),
                             squeeze=False, sharey=True)
    for idx, name in enumerate(model_names):
        ax = axes[idx // ncols, idx % ncols]
        loo = loo_results[name]
        k = loo.pareto_k.values
        ck = np.where(k >= 1.0, "#d62728",
             np.where(k >= 0.7, "#ff7f0e",
             np.where(k >= 0.5, "#bcbd22", "#1f77b4")))
        ax.scatter(np.arange(len(k)), k, c=ck, s=8, alpha=0.6)
        ax.axhline(0.5, color="#bcbd22", ls="--", lw=0.8)
        ax.axhline(0.7, color="#ff7f0e", ls="--", lw=0.8)
        ax.axhline(1.0, color="#d62728", ls="--", lw=0.8)
        n_bad = int(np.sum(k >= 0.7))
        ax.set_title(f"{name}\nelpd={loo.elpd_loo:.1f} ± {loo.se:.1f}  |  k̂≥0.7: {n_bad}/{len(k)}")
        ax.set_xlabel("Observation index")
    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"loo_pareto_k.{fmt}"), bbox_inches="tight")
    plt.close(fig)

    # ── Pointwise LOO differences (best vs second-best) ──
    summary_lines = [f"LOO Model Comparison:\n{comparison.to_string()}\n"]
    if n_models >= 2:
        best, second = model_names[0], model_names[1]
        pw_diff = loo_results[best].loo_i.values - loo_results[second].loo_i.values
        cum = np.cumsum(pw_diff)

        fig, axes_pw = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        axes_pw[0].bar(np.arange(len(pw_diff)), pw_diff,
                       color=np.where(pw_diff > 0, "#1f77b4", "#d62728"),
                       width=1.0, alpha=0.7)
        axes_pw[0].axhline(0, color="black", lw=1)
        axes_pw[0].set_title(
            f"Pointwise Δelpd: {best} − {second}  (total = {pw_diff.sum():.1f})")
        axes_pw[0].set_ylabel("Δelpd")

        axes_pw[1].plot(cum, color="#2ca02c", lw=2)
        axes_pw[1].axhline(0, color="black", lw=1, ls="--")
        axes_pw[1].fill_between(np.arange(len(cum)), cum, 0,
                                where=cum > 0, color="#1f77b4", alpha=0.2)
        axes_pw[1].fill_between(np.arange(len(cum)), cum, 0,
                                where=cum < 0, color="#d62728", alpha=0.2)
        axes_pw[1].set_ylabel("Cumulative Δelpd")
        axes_pw[1].set_xlabel("Observation index")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"loo_pointwise_diff.{fmt}"), bbox_inches="tight")
        plt.close(fig)

        diff_val = comparison["elpd_diff"].iloc[1]
        dse_val = comparison["dse"].iloc[1]
        ratio = abs(diff_val) / dse_val if dse_val > 0 else float("inf")
        summary_lines.append(
            f"Best: {best} vs {second}: Δelpd={diff_val:.2f}, dse={dse_val:.2f}, |Δ|/dse={ratio:.2f}"
        )

    return "\n".join(summary_lines)


# ── Correlation Scatter ─────────────────────────────────────────────────────

def plot_correlation_scatter(
    emp_vals: np.ndarray,
    pred_mean: np.ndarray,
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
    labels: List[str],
    title: str,
    out_path: str,
    emp_lo: Optional[np.ndarray] = None,
    emp_hi: Optional[np.ndarray] = None,
) -> str:
    """Condition-mean correlation scatter with 95% CI error bars.

    Returns a summary string with r and r².
    """
    r, p = pearsonr(emp_vals, pred_mean)
    r_sq = r ** 2

    fig, ax = plt.subplots(figsize=(6, 6))
    lim_lo = min(emp_vals.min(), pred_lo.min()) - 0.02
    lim_hi = max(emp_vals.max(), pred_hi.max()) + 0.02
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, alpha=0.5, label="Identity")

    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for i in range(len(labels)):
        yerr_lo = pred_mean[i] - pred_lo[i]
        yerr_hi = pred_hi[i] - pred_mean[i]
        xerr = None
        if emp_lo is not None and emp_hi is not None:
            xerr = [[emp_vals[i] - emp_lo[i]], [emp_hi[i] - emp_vals[i]]]
        ax.errorbar(emp_vals[i], pred_mean[i],
                    yerr=[[yerr_lo], [yerr_hi]],
                    xerr=xerr,
                    fmt="o", ms=8, color=colors[i],
                    capsize=4, capthick=1.2, elinewidth=1.2,
                    label=labels[i])

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Empirical")
    ax.set_ylabel("Model predicted")
    ax.set_title(f"{title}\nr = {r:.3f}, r² = {r_sq:.3f}")
    ax.set_aspect("equal")
    if len(labels) <= 12:
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return f"  Correlation ({title}): r={r:.4f}, r²={r_sq:.4f}, p={p:.2e}"


# ── Posterior Predictive Extraction ─────────────────────────────────────────

def extract_pp_samples(
    idata: az.InferenceData,
    var_name: str = "obs",
    max_draws: Optional[int] = None,
) -> np.ndarray:
    """Extract posterior predictive samples as a (S, N) array.

    Parameters
    ----------
    idata : az.InferenceData
    var_name : str
    max_draws : int, optional
        If given, subsample to at most this many draws (evenly spaced).

    Returns
    -------
    np.ndarray of shape (S, N)
    """
    obs = idata.posterior_predictive[var_name]
    pp = np.asarray(obs)                         # (chains, draws, N)
    pp_flat = pp.reshape(-1, pp.shape[-1])       # (S, N)
    if max_draws is not None and pp_flat.shape[0] > max_draws:
        idx = np.linspace(0, pp_flat.shape[0] - 1, max_draws, dtype=int)
        pp_flat = pp_flat[idx]
    return pp_flat


# ── Summary Text ────────────────────────────────────────────────────────────

def save_summary_text(path: str, blocks: List[str]) -> None:
    """Write a plain-text summary file from a list of text blocks."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n\n".join(b for b in blocks if b) + "\n")
    print(f"Summary saved: {path}")
