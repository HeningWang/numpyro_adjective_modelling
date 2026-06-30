"""Evaluate heldout ELPD for slider fold artifacts.

The inference artifacts are produced by run_inference.py with
``--hierarchical --heldout_fold``. This script reconstructs the same
condition-balanced folds, scores heldout observations under posterior samples,
and writes compact CSV summaries.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import arviz as az
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.infer.util import log_likelihood
from scipy.special import logsumexp


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

from modelSpecification import import_dataset_hier, slider_is_sharp_vector  # noqa: E402
from run_inference import (  # noqa: E402
    balanced_fold_ids,
    canonicalize_speaker_type,
    get_hier_model,
    PRODUCTION_ANCHOR_ORDERPLAN_SPEAKERS,
    select_listener_precompute,
    slider_sufficiency_vectors,
)


MODEL_TO_SPEAKER = {
    "incremental_recursive": "incremental",
    "incremental_static": "incremental_static",
    "planned_usefulness_order": "planned_usefulness_order",
    "planned_usefulness_order_static": "planned_usefulness_order_static",
    "planned_usefulness_signed_order": "planned_usefulness_signed_order",
    "planned_usefulness_signed_order_static": "planned_usefulness_signed_order_static",
    "planned_usefulness_mixture": "planned_usefulness_mixture",
    "planned_usefulness_mixture_static": "planned_usefulness_mixture_static",
    "planned_usefulness_mixture_anchored": "planned_usefulness_mixture_anchored",
    "planned_usefulness_mixture_anchored_static": "planned_usefulness_mixture_anchored_static",
    "production_anchor_sizesharp_2x2_inc_rec": "production_anchor_sizesharp_2x2_inc_rec",
    "production_anchor_sizesharp_2x2_inc_static": "production_anchor_sizesharp_2x2_inc_static",
    "production_anchor_sizesharp_2x2_glob_rec": "production_anchor_sizesharp_2x2_glob_rec",
    "production_anchor_sizesharp_2x2_glob_static": "production_anchor_sizesharp_2x2_glob_static",
    "production_anchor_reliabilitybackup_2x2_inc_rec": "production_anchor_reliabilitybackup_2x2_inc_rec",
    "production_anchor_reliabilitybackup_2x2_inc_static": "production_anchor_reliabilitybackup_2x2_inc_static",
    "production_anchor_reliabilitybackup_2x2_glob_rec": "production_anchor_reliabilitybackup_2x2_glob_rec",
    "production_anchor_reliabilitybackup_2x2_glob_static": "production_anchor_reliabilitybackup_2x2_glob_static",
    "production_anchor_reliabilitybackup_logalpha_2x2_inc_rec": "production_anchor_reliabilitybackup_logalpha_2x2_inc_rec",
    "production_anchor_reliabilitybackup_logalpha_2x2_inc_static": "production_anchor_reliabilitybackup_logalpha_2x2_inc_static",
    "production_anchor_reliabilitybackup_logalpha_2x2_glob_rec": "production_anchor_reliabilitybackup_logalpha_2x2_glob_rec",
    "production_anchor_reliabilitybackup_logalpha_2x2_glob_static": "production_anchor_reliabilitybackup_logalpha_2x2_glob_static",
    "production_anchor_reliabilitybackup_orderplan_2x2_inc_rec": "production_anchor_reliabilitybackup_orderplan_2x2_inc_rec",
    "production_anchor_reliabilitybackup_orderplan_2x2_inc_static": "production_anchor_reliabilitybackup_orderplan_2x2_inc_static",
    "production_anchor_reliabilitybackup_orderplan_2x2_glob_rec": "production_anchor_reliabilitybackup_orderplan_2x2_glob_rec",
    "production_anchor_reliabilitybackup_orderplan_2x2_glob_static": "production_anchor_reliabilitybackup_orderplan_2x2_glob_static",
    "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_rec": "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_rec",
    "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_static": "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_static",
    "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_glob_rec": "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_glob_rec",
    "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_glob_static": "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_glob_static",
}

PAIR_SPECS = (
    ("planned_usefulness_order", "incremental_recursive", "planned_vs_greedy_recursive"),
    ("planned_usefulness_order_static", "incremental_static", "planned_vs_greedy_static"),
    ("planned_usefulness_signed_order", "planned_usefulness_order", "signed_order_vs_planned_recursive"),
    (
        "planned_usefulness_signed_order_static",
        "planned_usefulness_order_static",
        "signed_order_vs_planned_static",
    ),
    ("planned_usefulness_mixture", "planned_usefulness_order", "mixture_vs_planned_recursive"),
    (
        "planned_usefulness_mixture_static",
        "planned_usefulness_order_static",
        "mixture_vs_planned_static",
    ),
    (
        "planned_usefulness_mixture_anchored",
        "planned_usefulness_mixture",
        "anchored_mixture_vs_mixture_recursive",
    ),
    (
        "planned_usefulness_mixture_anchored_static",
        "planned_usefulness_mixture_static",
        "anchored_mixture_vs_mixture_static",
    ),
    ("planned_usefulness_signed_order", "incremental_recursive", "signed_order_vs_greedy_recursive"),
    (
        "planned_usefulness_signed_order_static",
        "incremental_static",
        "signed_order_vs_greedy_static",
    ),
    ("planned_usefulness_mixture", "incremental_recursive", "mixture_vs_greedy_recursive"),
    (
        "planned_usefulness_mixture_static",
        "incremental_static",
        "mixture_vs_greedy_static",
    ),
    (
        "planned_usefulness_mixture_anchored",
        "incremental_recursive",
        "anchored_mixture_vs_greedy_recursive",
    ),
    (
        "planned_usefulness_mixture_anchored_static",
        "incremental_static",
        "anchored_mixture_vs_greedy_static",
    ),
    (
        "production_anchor_sizesharp_2x2_inc_rec",
        "production_anchor_sizesharp_2x2_glob_rec",
        "production_anchor_architecture_recursive",
    ),
    (
        "production_anchor_sizesharp_2x2_inc_static",
        "production_anchor_sizesharp_2x2_glob_static",
        "production_anchor_architecture_static",
    ),
    (
        "production_anchor_sizesharp_2x2_inc_static",
        "production_anchor_sizesharp_2x2_inc_rec",
        "production_anchor_semantics_incremental",
    ),
    (
        "production_anchor_sizesharp_2x2_glob_static",
        "production_anchor_sizesharp_2x2_glob_rec",
        "production_anchor_semantics_global",
    ),
    (
        "production_anchor_reliabilitybackup_2x2_inc_rec",
        "production_anchor_reliabilitybackup_2x2_glob_rec",
        "production_anchor_reliabilitybackup_architecture_recursive",
    ),
    (
        "production_anchor_reliabilitybackup_2x2_inc_static",
        "production_anchor_reliabilitybackup_2x2_glob_static",
        "production_anchor_reliabilitybackup_architecture_static",
    ),
    (
        "production_anchor_reliabilitybackup_2x2_inc_static",
        "production_anchor_reliabilitybackup_2x2_inc_rec",
        "production_anchor_reliabilitybackup_semantics_incremental",
    ),
    (
        "production_anchor_reliabilitybackup_2x2_glob_static",
        "production_anchor_reliabilitybackup_2x2_glob_rec",
        "production_anchor_reliabilitybackup_semantics_global",
    ),
    (
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_rec",
        "production_anchor_reliabilitybackup_logalpha_2x2_glob_rec",
        "production_anchor_reliabilitybackup_logalpha_architecture_recursive",
    ),
    (
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_static",
        "production_anchor_reliabilitybackup_logalpha_2x2_glob_static",
        "production_anchor_reliabilitybackup_logalpha_architecture_static",
    ),
    (
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_static",
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_rec",
        "production_anchor_reliabilitybackup_logalpha_semantics_incremental",
    ),
    (
        "production_anchor_reliabilitybackup_logalpha_2x2_glob_static",
        "production_anchor_reliabilitybackup_logalpha_2x2_glob_rec",
        "production_anchor_reliabilitybackup_logalpha_semantics_global",
    ),
    (
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_rec",
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_rec",
        "production_anchor_reliabilitybackup_orderplan_logalpha_vs_anchor_recursive",
    ),
    (
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_static",
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_static",
        "production_anchor_reliabilitybackup_orderplan_logalpha_vs_anchor_static",
    ),
    (
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_static",
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_rec",
        "production_anchor_reliabilitybackup_orderplan_logalpha_semantics_incremental",
    ),
    (
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_rec",
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_glob_rec",
        "production_anchor_reliabilitybackup_orderplan_logalpha_architecture_recursive",
    ),
    (
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_static",
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_glob_static",
        "production_anchor_reliabilitybackup_orderplan_logalpha_architecture_static",
    ),
    (
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_glob_static",
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_glob_rec",
        "production_anchor_reliabilitybackup_orderplan_logalpha_semantics_global",
    ),
)

PROPERTY_LABELS = {
    "second": "zrdc_like_second_property",
    "first": "erdc_like_first_property",
    "both": "brdc_like_both_properties",
}


def is_production_anchor_speaker(speaker: str) -> bool:
    return speaker.startswith("production_anchor_sizesharp_2x2_") or speaker.startswith(
        "production_anchor_reliabilitybackup_2x2_"
    ) or speaker.startswith("production_anchor_reliabilitybackup_logalpha_2x2_") or speaker.startswith(
        "production_anchor_reliabilitybackup_orderplan_"
    )


def posterior_samples_from_idata(idata: az.InferenceData) -> dict[str, np.ndarray]:
    samples = {}
    for name, values in idata.posterior.data_vars.items():
        stacked = values.stack(sample=("chain", "draw"))
        dims = ["sample"] + [dim for dim in stacked.dims if dim != "sample"]
        samples[name] = np.asarray(stacked.transpose(*dims).values)
    return samples


def artifact_name(
    speaker: str,
    fold: int,
    num_folds: int,
    warmup: int,
    samples: int,
    chains: int,
    artifact_tag: str = "",
) -> str:
    run_tag = f"_{artifact_tag}" if artifact_tag else ""
    return (
        f"mcmc_results_{speaker}_speaker_hier_fold{fold}of{num_folds}"
        f"{run_tag}_warmup{warmup}_samples{samples}_chains{chains}.nc"
    )


def fold_diagnostics(idata: az.InferenceData, max_r_hat: float) -> dict:
    summary = az.summary(idata, kind="diagnostics", round_to=None)
    r_hat = (
        pd.to_numeric(summary["r_hat"], errors="coerce")
        if "r_hat" in summary else pd.Series(dtype=float)
    )
    ess_bulk = (
        pd.to_numeric(summary["ess_bulk"], errors="coerce")
        if "ess_bulk" in summary else pd.Series(dtype=float)
    )
    ess_tail = (
        pd.to_numeric(summary["ess_tail"], errors="coerce")
        if "ess_tail" in summary else pd.Series(dtype=float)
    )

    diverging = idata.sample_stats.get("diverging")
    n_draws = int(diverging.size) if diverging is not None else 0
    n_divergent = int(np.asarray(diverging).sum()) if diverging is not None else 0
    max_observed_r_hat = float(r_hat.max(skipna=True)) if r_hat.notna().any() else np.nan
    min_ess_bulk = float(ess_bulk.min(skipna=True)) if ess_bulk.notna().any() else np.nan
    min_ess_tail = float(ess_tail.min(skipna=True)) if ess_tail.notna().any() else np.nan
    return {
        "n_parameters": int(summary.shape[0]),
        "max_r_hat": max_observed_r_hat,
        "n_r_hat_gt_1_01": int(r_hat.gt(1.01).sum()) if r_hat.notna().any() else 0,
        "n_r_hat_gt_1_05": int(r_hat.gt(1.05).sum()) if r_hat.notna().any() else 0,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "n_divergent": n_divergent,
        "n_draws": n_draws,
        "divergence_rate": float(n_divergent / n_draws) if n_draws else np.nan,
        "diagnostic_status": "pass" if n_divergent == 0 and (
            np.isnan(max_observed_r_hat) or max_observed_r_hat <= max_r_hat
        ) else "fail",
    }


def heldout_fold_data(fold: int, num_folds: int, fold_seed: int):
    dataset_path = REPO_ROOT / "data" / "01-slider-data-preprocessed.csv"
    states_all, empirical_all, df, participant_idx_all, n_participants = import_dataset_hier(
        str(dataset_path)
    )
    fold_ids = balanced_fold_ids(df, num_folds=num_folds, seed=fold_seed)
    heldout_mask = fold_ids == fold
    train_mask = ~heldout_mask
    train_empirical = np.asarray(empirical_all[train_mask])
    pi0 = float(np.mean(np.isclose(train_empirical, 0.0)))
    pi1 = float(np.mean(np.isclose(train_empirical, 1.0)))
    sufficient_dim_all, has_one_word_solution_all, is_colour_sufficient_all = (
        slider_sufficiency_vectors(df)
    )
    return {
        "states": states_all[heldout_mask],
        "data": empirical_all[heldout_mask],
        "participant_idx": participant_idx_all[heldout_mask],
        "n_participants": n_participants,
        "pi0": pi0,
        "pi1": pi1,
        "df": df.loc[heldout_mask].copy(),
        "is_sharp": jnp.asarray(np.asarray(slider_is_sharp_vector(df))[heldout_mask]),
        "sufficient_dim": sufficient_dim_all[heldout_mask],
        "has_one_word_solution": has_one_word_solution_all[heldout_mask],
        "is_colour_sufficient": is_colour_sufficient_all[heldout_mask],
    }


def score_fold(
    model_name: str,
    fold: int,
    args: argparse.Namespace,
) -> tuple[dict, pd.DataFrame]:
    speaker = canonicalize_speaker_type(MODEL_TO_SPEAKER.get(model_name, model_name))
    path = args.inference_dir / artifact_name(
        speaker,
        fold,
        args.num_folds,
        args.warmup,
        args.samples,
        args.chains,
        args.artifact_tag,
    )
    if not path.exists():
        raise FileNotFoundError(path)

    heldout = heldout_fold_data(fold, args.num_folds, args.fold_seed)
    L1_all, L2_all = select_listener_precompute(speaker, heldout["states"])
    model = get_hier_model(speaker)
    idata = az.from_netcdf(path)
    posterior_samples = posterior_samples_from_idata(idata)
    diagnostics = fold_diagnostics(idata, args.max_r_hat)
    if speaker in PRODUCTION_ANCHOR_ORDERPLAN_SPEAKERS:
        ll = log_likelihood(
            model,
            posterior_samples,
            heldout["states"],
            heldout["data"],
            heldout["pi0"],
            heldout["pi1"],
            heldout["participant_idx"],
            heldout["n_participants"],
            L1_all,
            L2_all,
            heldout["is_sharp"],
            heldout["sufficient_dim"],
            heldout["has_one_word_solution"],
            heldout["is_colour_sufficient"],
            parallel=False,
        )["obs"]
    elif is_production_anchor_speaker(speaker):
        ll = log_likelihood(
            model,
            posterior_samples,
            heldout["states"],
            heldout["data"],
            heldout["pi0"],
            heldout["pi1"],
            heldout["participant_idx"],
            heldout["n_participants"],
            L1_all,
            L2_all,
            heldout["is_sharp"],
            parallel=False,
        )["obs"]
    else:
        ll = log_likelihood(
            model,
            posterior_samples,
            heldout["states"],
            heldout["data"],
            heldout["pi0"],
            heldout["pi1"],
            heldout["participant_idx"],
            heldout["n_participants"],
            L1_all,
            L2_all,
            parallel=False,
        )["obs"]
    ll_np = np.asarray(ll)
    if ll_np.ndim != 2:
        ll_np = ll_np.reshape((ll_np.shape[0], -1))
    pointwise_elpd = logsumexp(ll_np, axis=0) - np.log(ll_np.shape[0])

    detail = heldout["df"][["id", "item", "conditions", "sharpness", "relevant_property"]].copy()
    detail["model"] = model_name
    detail["speaker_type"] = speaker
    detail["fold"] = fold
    detail["heldout_elpd"] = pointwise_elpd
    detail["heldout_log_lik_mean"] = ll_np.mean(axis=0)
    detail["heldout_log_lik_sd"] = ll_np.std(axis=0)

    row = {
        "model": model_name,
        "speaker_type": speaker,
        "fold": fold,
        "artifact": path.name,
        "n_heldout": int(len(pointwise_elpd)),
        "heldout_elpd": float(pointwise_elpd.sum()),
        "heldout_elpd_mean": float(pointwise_elpd.mean()),
        "heldout_log_lik_mean": float(ll_np.mean()),
    }
    row.update(diagnostics)
    return row, detail


def diagnostic_status(max_r_hat: float, n_divergent: int, threshold: float) -> str:
    if n_divergent > 0:
        return "fail"
    if pd.notna(max_r_hat) and max_r_hat > threshold:
        return "fail"
    if pd.notna(max_r_hat) and max_r_hat > 1.01:
        return "warn"
    return "pass"


def summarize_models(model_summary: pd.DataFrame, max_r_hat: float) -> pd.DataFrame:
    total_summary = (
        model_summary.groupby(["model", "speaker_type"], as_index=False)
        .agg(
            total_heldout_elpd=("heldout_elpd", "sum"),
            mean_fold_elpd=("heldout_elpd", "mean"),
            se_fold_elpd=("heldout_elpd", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan),
            n_heldout=("n_heldout", "sum"),
            n_folds=("fold", "nunique"),
            n_parameters=("n_parameters", "max"),
            max_r_hat=("max_r_hat", "max"),
            n_r_hat_gt_1_01=("n_r_hat_gt_1_01", "sum"),
            n_r_hat_gt_1_05=("n_r_hat_gt_1_05", "sum"),
            min_ess_bulk=("min_ess_bulk", "min"),
            min_ess_tail=("min_ess_tail", "min"),
            n_divergent=("n_divergent", "sum"),
            n_draws=("n_draws", "sum"),
        )
    )
    total_summary["divergence_rate"] = total_summary["n_divergent"] / total_summary["n_draws"]
    total_summary["diagnostic_status"] = [
        diagnostic_status(row.max_r_hat, int(row.n_divergent), max_r_hat)
        for row in total_summary.itertuples(index=False)
    ]
    total_summary["diagnostics_ok"] = total_summary["diagnostic_status"].isin(["pass", "warn"]) & total_summary["n_divergent"].eq(0)
    return total_summary.sort_values("total_heldout_elpd", ascending=False)


def summarize_pairs(model_summary: pd.DataFrame) -> pd.DataFrame:
    totals = (
        model_summary.groupby(["model", "speaker_type"], as_index=False)
        .agg(
            total_heldout_elpd=("heldout_elpd", "sum"),
            mean_fold_elpd=("heldout_elpd", "mean"),
            se_fold_elpd=("heldout_elpd", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan),
            n_heldout=("n_heldout", "sum"),
            n_folds=("fold", "nunique"),
        )
    )
    indexed = totals.set_index("model")
    rows = []
    for candidate, baseline, pair in PAIR_SPECS:
        if candidate not in indexed.index or baseline not in indexed.index:
            continue
        delta = indexed.loc[candidate, "total_heldout_elpd"] - indexed.loc[baseline, "total_heldout_elpd"]
        rows.append(
            {
                "pair": pair,
                "candidate": candidate,
                "baseline": baseline,
                "candidate_total_heldout_elpd": indexed.loc[candidate, "total_heldout_elpd"],
                "baseline_total_heldout_elpd": indexed.loc[baseline, "total_heldout_elpd"],
                "delta_heldout_elpd": float(delta),
                "candidate_mean_fold_elpd": indexed.loc[candidate, "mean_fold_elpd"],
                "baseline_mean_fold_elpd": indexed.loc[baseline, "mean_fold_elpd"],
                "candidate_se_fold_elpd": indexed.loc[candidate, "se_fold_elpd"],
                "baseline_se_fold_elpd": indexed.loc[baseline, "se_fold_elpd"],
                "n_folds": int(indexed.loc[candidate, "n_folds"]),
            }
        )
    return pd.DataFrame(rows)


def read_ppc(stats_dir: Path | None, ppc_scope: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if stats_dir is None:
        return None, None
    corr_path = stats_dir / "slider_ppc_correlation.csv"
    cond_path = stats_dir / "slider_ppc_by_condition.csv"
    if not corr_path.exists() or not cond_path.exists():
        return None, None
    ppc = pd.read_csv(corr_path)
    if "scope" in ppc.columns:
        ppc = ppc[ppc["scope"].astype(str).eq(ppc_scope)].copy()
    ppc = ppc.rename(columns={"rmse": "ppc_rmse", "mae": "ppc_mae", "r": "ppc_r", "r2": "ppc_r2"})
    for col in ["ppc_rmse", "ppc_mae", "ppc_r", "ppc_r2"]:
        if col in ppc.columns:
            ppc[col] = pd.to_numeric(ppc[col], errors="coerce")
    return ppc, pd.read_csv(cond_path)


def residual_changes(ppc_by_condition: pd.DataFrame, candidate: str, baseline: str) -> pd.DataFrame:
    cols = ["model", "relevant_property", "sharpness", "human_mean", "model_mean", "signed_residual", "abs_residual"]
    sub = ppc_by_condition[ppc_by_condition["model"].isin([candidate, baseline])][cols].copy()
    wide = sub.pivot_table(
        index=["relevant_property", "sharpness"],
        columns="model",
        values=["model_mean", "signed_residual", "abs_residual"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    wide["candidate"] = candidate
    wide["baseline"] = baseline
    wide["property_label"] = wide["relevant_property"].map(PROPERTY_LABELS).fillna(wide["relevant_property"])
    wide["baseline_model_mean"] = wide[f"model_mean_{baseline}"]
    wide["candidate_model_mean"] = wide[f"model_mean_{candidate}"]
    wide["baseline_signed_residual"] = wide[f"signed_residual_{baseline}"]
    wide["candidate_signed_residual"] = wide[f"signed_residual_{candidate}"]
    wide["baseline_abs_residual"] = wide[f"abs_residual_{baseline}"]
    wide["candidate_abs_residual"] = wide[f"abs_residual_{candidate}"]
    wide["abs_residual_reduction"] = wide["baseline_abs_residual"] - wide["candidate_abs_residual"]
    wide["signed_residual_change"] = wide["candidate_signed_residual"] - wide["baseline_signed_residual"]
    return wide[
        [
            "candidate",
            "baseline",
            "relevant_property",
            "property_label",
            "sharpness",
            "baseline_model_mean",
            "candidate_model_mean",
            "baseline_signed_residual",
            "candidate_signed_residual",
            "baseline_abs_residual",
            "candidate_abs_residual",
            "abs_residual_reduction",
            "signed_residual_change",
        ]
    ]


def mark_heldout_frontier(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required = ["total_heldout_elpd", "ppc_rmse", "n_parameters"]
    for _, candidate in scores.iterrows():
        complete = all(pd.notna(candidate.get(col, np.nan)) for col in required)
        eligible = complete and bool(candidate.get("diagnostics_ok", False))
        dominated = False
        if eligible:
            for _, other in scores.iterrows():
                if other["model"] == candidate["model"]:
                    continue
                other_complete = all(pd.notna(other.get(col, np.nan)) for col in required)
                if not other_complete or not bool(other.get("diagnostics_ok", False)):
                    continue
                no_worse = (
                    other["total_heldout_elpd"] >= candidate["total_heldout_elpd"]
                    and other["ppc_rmse"] <= candidate["ppc_rmse"]
                    and other["n_parameters"] <= candidate["n_parameters"]
                )
                strictly_better = (
                    other["total_heldout_elpd"] > candidate["total_heldout_elpd"]
                    or other["ppc_rmse"] < candidate["ppc_rmse"]
                    or other["n_parameters"] < candidate["n_parameters"]
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
        row = candidate.to_dict()
        row["complete_objectives"] = complete
        row["eligible_for_frontier"] = eligible
        row["heldout_pareto_frontier"] = bool(eligible and not dominated)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["heldout_pareto_frontier", "total_heldout_elpd", "ppc_rmse", "n_parameters"],
        ascending=[False, False, True, True],
    )


def recommend_full_run(
    candidate_diagnostics_ok: bool,
    baseline_diagnostics_ok: bool,
    heldout_success: bool,
    ppc_success: bool,
    candidate_on_frontier: bool,
) -> bool:
    return bool(
        candidate_diagnostics_ok
        and baseline_diagnostics_ok
        and ppc_success
        and (heldout_success or candidate_on_frontier)
    )


def decision_outputs(
    total_summary: pd.DataFrame,
    ppc: pd.DataFrame,
    ppc_by_condition: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_scores = total_summary.merge(ppc, on="model", how="left")
    frontier = mark_heldout_frontier(model_scores)
    frontier_by_model = frontier.set_index("model")
    indexed = model_scores.set_index("model")
    rows = []
    residual_frames = []
    ppc_condition_models = set(ppc_by_condition["model"].astype(str))
    for candidate, baseline, pair in PAIR_SPECS:
        if candidate not in indexed.index or baseline not in indexed.index:
            continue
        cand = indexed.loc[candidate]
        base = indexed.loc[baseline]
        if (
            candidate not in ppc_condition_models
            or baseline not in ppc_condition_models
            or pd.isna(cand.get("ppc_rmse"))
            or pd.isna(base.get("ppc_rmse"))
        ):
            continue
        residual = residual_changes(ppc_by_condition, candidate, baseline)
        residual_frames.append(residual.assign(pair=pair))

        second = residual[residual["relevant_property"].eq("second")]
        first = residual[residual["relevant_property"].eq("first")]
        both = residual[residual["relevant_property"].eq("both")]
        delta_elpd = float(cand["total_heldout_elpd"] - base["total_heldout_elpd"])
        ppc_rmse_gain = float(base["ppc_rmse"] - cand["ppc_rmse"])
        ppc_r_gain = float(cand["ppc_r"] - base["ppc_r"])
        second_gain = float(second["abs_residual_reduction"].mean()) if not second.empty else np.nan
        first_harm = float(-first["abs_residual_reduction"].min()) if not first.empty else np.nan
        both_harm = float(-both["abs_residual_reduction"].min()) if not both.empty else np.nan
        worst_new_harm = np.nanmax([first_harm, both_harm])
        heldout_success = delta_elpd > args.elpd_gate
        ppc_success = bool(
            ppc_rmse_gain >= args.ppc_rmse_gate
            and second_gain >= args.second_residual_gate
            and (not np.isfinite(worst_new_harm) or worst_new_harm <= args.max_new_residual_harm)
        )
        candidate_on_frontier = bool(
            frontier_by_model.loc[candidate, "heldout_pareto_frontier"]
        ) if candidate in frontier_by_model.index else False
        rows.append(
            {
                "pair": pair,
                "candidate": candidate,
                "baseline": baseline,
                "candidate_diagnostics_ok": bool(cand["diagnostics_ok"]),
                "baseline_diagnostics_ok": bool(base["diagnostics_ok"]),
                "delta_heldout_elpd_candidate_minus_baseline": delta_elpd,
                "heldout_elpd_success": bool(heldout_success),
                "baseline_ppc_rmse": base["ppc_rmse"],
                "candidate_ppc_rmse": cand["ppc_rmse"],
                "ppc_rmse_gain": ppc_rmse_gain,
                "baseline_ppc_r": base["ppc_r"],
                "candidate_ppc_r": cand["ppc_r"],
                "ppc_r_gain": ppc_r_gain,
                "second_property_abs_residual_reduction": second_gain,
                "worst_first_or_both_abs_residual_harm": worst_new_harm,
                "ppc_success": ppc_success,
                "candidate_on_heldout_frontier": candidate_on_frontier,
                "recommended_for_full_run": recommend_full_run(
                    bool(cand["diagnostics_ok"]),
                    bool(base["diagnostics_ok"]),
                    bool(heldout_success),
                    bool(ppc_success),
                    bool(candidate_on_frontier),
                ),
                "decision_basis": "heldout_elpd_and_ppc",
            }
        )
    pairwise = pd.DataFrame(rows)
    residuals = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    return model_scores, pairwise, residuals, frontier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--inference-dir", type=Path, default=THIS_DIR / "inference_data")
    parser.add_argument("--out-dir", type=Path, default=THIS_DIR / "results_heldout_pilot" / "stats")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=13)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument(
        "--artifact-tag", type=str, default="",
        help="Optional artifact tag inserted before warmup/sample/chains in filenames.",
    )
    parser.add_argument(
        "--posterior-stats-dir",
        type=Path,
        default=None,
        help=(
            "Optional stats directory from posterior_analysis.py containing "
            "slider_ppc_correlation.csv and slider_ppc_by_condition.csv. "
            "When present, heldout decision and frontier CSVs are exported."
        ),
    )
    parser.add_argument("--decision-prefix", type=str, default="slider_heldout_eval")
    parser.add_argument("--ppc-scope", type=str, default="all_cells")
    parser.add_argument("--max-r-hat", type=float, default=1.05)
    parser.add_argument("--elpd-gate", type=float, default=0.0)
    parser.add_argument("--ppc-rmse-gate", type=float, default=0.0)
    parser.add_argument("--second-residual-gate", type=float, default=0.05)
    parser.add_argument("--max-new-residual-harm", type=float, default=0.02)
    parser.add_argument("--write-pointwise", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    rows = []
    detail_frames = []
    for model_name in models:
        for fold in range(args.num_folds):
            row, detail = score_fold(model_name, fold, args)
            rows.append(row)
            if args.write_pointwise:
                detail_frames.append(detail)

    model_summary = pd.DataFrame(rows)
    total_summary = summarize_models(model_summary, args.max_r_hat)
    pairwise = summarize_pairs(model_summary)

    model_summary.to_csv(args.out_dir / "slider_heldout_elpd_by_fold.csv", index=False)
    total_summary.to_csv(args.out_dir / "slider_heldout_elpd_model_summary.csv", index=False)
    pairwise.to_csv(args.out_dir / "slider_heldout_elpd_pairwise.csv", index=False)
    if args.write_pointwise and detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(
            args.out_dir / "slider_heldout_elpd_pointwise.csv",
            index=False,
        )

    ppc, ppc_by_condition = read_ppc(args.posterior_stats_dir, args.ppc_scope)
    if ppc is not None and ppc_by_condition is not None:
        model_scores, decisions, residuals, frontier = decision_outputs(
            total_summary,
            ppc,
            ppc_by_condition,
            args,
        )
        model_scores.to_csv(
            args.out_dir / f"{args.decision_prefix}_model_decision_summary.csv",
            index=False,
        )
        decisions.to_csv(
            args.out_dir / f"{args.decision_prefix}_pairwise_decisions.csv",
            index=False,
        )
        if not residuals.empty:
            residuals.to_csv(
                args.out_dir / f"{args.decision_prefix}_residual_changes.csv",
                index=False,
            )
        frontier.to_csv(
            args.out_dir / f"{args.decision_prefix}_pareto_scores.csv",
            index=False,
        )
        frontier[frontier["heldout_pareto_frontier"]].to_csv(
            args.out_dir / f"{args.decision_prefix}_pareto_frontier.csv",
            index=False,
        )

    print(f"Wrote heldout ELPD CSVs to {args.out_dir}")
    print(total_summary.to_string(index=False))
    if not pairwise.empty:
        print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
