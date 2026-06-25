"""Forward audit for bounded-form response-policy production variants.

This script does not run inference. It starts from the fitted production
response-policy pilot means and applies two bounded-form pressures:

* sufficient-form pair boost: when a one-word adjective is already sufficient,
  prefer the two-word description that adds form as a bounded grounding cue
  (DF or CF), rather than all available adjectives;
* three-word penalty: penalize three-adjective piles.

The audit writes compact CSV summaries before any plotting or inference
decision. A gate pass means the variant is worth piloting on GPU; it is not
posterior evidence.
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import modelSpecification as ms  # noqa: E402
from architecture_contrast_audit import (  # noqa: E402
    classify_utterance,
    overinformativeness_class,
    safe_kl,
)
from helper import import_dataset_hier  # noqa: E402


GROUP_COLS = ["relevant_property", "sharpness"]
UTTERANCE_LABELS = ms.UTTERANCE_LABELS

BASELINE_STATIC_MODEL = (
    "principled_salience_stop_regularized_responsepolicy_2x2_inc_static_fixedeps"
)
BASELINE_RECURSIVE_MODEL = (
    "principled_salience_stop_regularized_responsepolicy_2x2_inc_rec_fixedeps"
)
BOUNDED_STATIC_MODEL = (
    "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static_fixedeps"
)
BOUNDED_RECURSIVE_MODEL = (
    "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec_fixedeps"
)


def _filter_conditions(data: dict, condition_subset: str) -> dict:
    if not condition_subset:
        return data
    codes = tuple(c.strip() for c in condition_subset.split(",") if c.strip())
    if not codes:
        return data
    df = data["df"].reset_index(drop=True)
    keep = df["conditions"].isin(codes).to_numpy()
    if not keep.any():
        raise ValueError(f"--condition-subset matched zero rows: {codes}")
    keep_idx = jnp.asarray(np.where(keep)[0])
    out = dict(data)
    for key in (
        "states_train",
        "empirical_seq_flat",
        "empirical_flat",
        "empirical_seq",
        "seq_mask",
        "sharpness_idx",
        "is_colour_sufficient",
        "sufficient_dim",
        "has_one_word_solution",
        "participant_idx",
        "condition_idx",
    ):
        if key in out and out[key] is not None:
            out[key] = out[key][keep_idx]
    out["df"] = df.loc[keep].reset_index(drop=True)
    return out


def _float_grid(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _read_params(path: Path, model: str) -> dict[str, float]:
    params = {
        "alpha": 0.70,
        "beta_order": float(np.exp(0.67)),
        "lambda_salience": 1.31,
        "rho_salience_stop": 0.10,
        "lambda_sufficient_single": 2.02,
        "lambda_reliability_form": 1.42,
        "epsilon": 0.003,
    }
    if not path.exists():
        return params
    diag = pd.read_csv(path)
    if "model" not in diag.columns or "parameter" not in diag.columns:
        return params
    sub = diag[diag["model"].eq(model)].copy()
    if sub.empty:
        return params
    means = sub.set_index("parameter")["mean"].to_dict()
    for key in (
        "alpha",
        "lambda_salience",
        "rho_salience_stop",
        "lambda_sufficient_single",
        "lambda_reliability_form",
        "epsilon",
    ):
        if key in means and np.isfinite(means[key]):
            params[key] = float(means[key])
    if "log_beta_order" in means and np.isfinite(means["log_beta_order"]):
        params["beta_order"] = float(np.exp(means["log_beta_order"]))
    return params


def _speaker_probabilities(
    data: dict,
    params: dict[str, float],
    *,
    recursive: bool,
) -> np.ndarray:
    n = len(data["df"])
    alpha = jnp.full((n,), params["alpha"], dtype=jnp.float32)
    probs = ms.jitted_speaker_principled_response_policy_hier(
        data["states_train"],
        data["sufficient_dim"],
        data["has_one_word_solution"],
        data["sharpness_idx"],
        data["is_colour_sufficient"],
        alpha,
        jnp.float32(params["beta_order"]),
        jnp.float32(params["lambda_salience"]),
        jnp.float32(params["rho_salience_stop"]),
        jnp.float32(params["lambda_sufficient_single"]),
        jnp.float32(params["lambda_reliability_form"]),
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(0.59),
        jnp.float32(0.50),
        jnp.float32(0.50),
        jnp.float32(0.6856),
        jnp.float32(params["epsilon"]),
        ms.LOG_LM_ORDER_ONLY_15,
        ms.BASE_VISUAL_SALIENCE,
        recursive=recursive,
        size_context_mode="posterior",
    )
    return np.asarray(probs)


def _bounded_masks(data: dict) -> tuple[np.ndarray, np.ndarray]:
    sufficient_dim = np.asarray(data["sufficient_dim"], dtype=np.int32)
    has_one_word_solution = np.asarray(data["has_one_word_solution"], dtype=np.float32)
    full_present = np.asarray(ms.FULL_PRESENT_15, dtype=np.float32)
    n_words = np.asarray(ms.N_WORDS, dtype=np.float32)

    sufficient_present = np.zeros((len(sufficient_dim), len(UTTERANCE_LABELS)))
    for i, dim in enumerate(sufficient_dim):
        if dim >= 0:
            sufficient_present[i] = full_present[:, dim]
    sufficient_form_pair = (
        (has_one_word_solution[:, None] > 0.5)
        & (sufficient_dim[:, None] >= 0)
        & (sufficient_dim[:, None] != 2)
        & (n_words[None, :] == 2.0)
        & (full_present[None, :, 2] > 0.5)
        & (sufficient_present > 0.5)
    )
    three_word = n_words == 3.0
    return sufficient_form_pair.astype(float), three_word.astype(float)


def _apply_bounded_form(
    probs: np.ndarray,
    masks: tuple[np.ndarray, np.ndarray],
    *,
    sufficient_form_pair_boost: float,
    three_word_penalty: float,
) -> np.ndarray:
    sufficient_form_pair, three_word = masks
    logits = (
        np.log(np.clip(probs, 1e-12, None))
        + sufficient_form_pair_boost * sufficient_form_pair
        - three_word_penalty * three_word[None, :]
    )
    logits = logits - logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def _empirical_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    conditions = df[GROUP_COLS].drop_duplicates().sort_values(GROUP_COLS)
    grid = conditions.merge(
        pd.DataFrame(
            {
                "utterance_code": list(range(len(UTTERANCE_LABELS))),
                "utterance_label": UTTERANCE_LABELS,
            }
        ),
        how="cross",
    )
    counts = (
        df.groupby(GROUP_COLS + ["annotation_seq_flat"])
        .size()
        .rename("n")
        .reset_index()
        .rename(columns={"annotation_seq_flat": "utterance_code"})
    )
    totals = df.groupby(GROUP_COLS).size().rename("condition_n").reset_index()
    out = grid.merge(counts, on=GROUP_COLS + ["utterance_code"], how="left")
    out = out.merge(totals, on=GROUP_COLS, how="left")
    out["n"] = out["n"].fillna(0).astype(int)
    out["empirical_proportion"] = out["n"] / out["condition_n"]
    return out


def _model_by_condition(
    data: dict,
    probs: np.ndarray,
    *,
    model: str,
    semantics: str,
    sufficient_form_pair_boost: float,
    three_word_penalty: float,
) -> pd.DataFrame:
    df = data["df"].reset_index(drop=True)
    tmp = df[GROUP_COLS].copy()
    for code, label in enumerate(UTTERANCE_LABELS):
        tmp[label] = probs[:, code]
    long = tmp.melt(
        id_vars=GROUP_COLS,
        var_name="utterance_label",
        value_name="model_proportion",
    )
    out = (
        long.groupby(GROUP_COLS + ["utterance_label"], as_index=False)[
            "model_proportion"
        ].mean()
    )
    out["utterance_code"] = out["utterance_label"].map(
        {label: code for code, label in enumerate(UTTERANCE_LABELS)}
    )
    out["model"] = model
    out["semantics"] = semantics
    out["sufficient_form_pair_boost"] = sufficient_form_pair_boost
    out["three_word_penalty"] = three_word_penalty
    return out


def _condition_residuals(
    empirical: pd.DataFrame,
    model_rows: pd.DataFrame,
) -> pd.DataFrame:
    merged = model_rows.merge(
        empirical[
            GROUP_COLS
            + [
                "utterance_code",
                "utterance_label",
                "empirical_proportion",
                "condition_n",
            ]
        ],
        on=GROUP_COLS + ["utterance_code", "utterance_label"],
        how="left",
    )
    merged["signed_residual"] = (
        merged["model_proportion"] - merged["empirical_proportion"]
    )
    merged["abs_residual"] = merged["signed_residual"].abs()
    return merged


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(
        [classify_utterance(label) for label in df["utterance_label"]],
        index=df.index,
    )
    out = pd.concat([df.copy(), features], axis=1)
    out["overinformativeness_class"] = [
        overinformativeness_class(label, rel)
        for label, rel in zip(out["utterance_label"], out["relevant_property"])
    ]
    return out


def _category_residuals(condition_residuals: pd.DataFrame) -> pd.DataFrame:
    base = _add_features(condition_residuals)
    records = []
    specs = {
        "length": "length_class",
        "selection": "selection_class",
        "order": "order_class",
        "overinformativeness": "overinformativeness_class",
    }
    group_base = [
        "model",
        "semantics",
        "sufficient_form_pair_boost",
        "three_word_penalty",
    ] + GROUP_COLS
    for summary_type, category_col in specs.items():
        grouped = (
            base.groupby(group_base + [category_col], as_index=False)
            .agg(
                empirical_proportion=("empirical_proportion", "sum"),
                model_proportion=("model_proportion", "sum"),
                condition_n=("condition_n", "max"),
            )
            .rename(columns={category_col: "category"})
        )
        grouped["summary_type"] = summary_type
        grouped["signed_residual"] = (
            grouped["model_proportion"] - grouped["empirical_proportion"]
        )
        grouped["abs_residual"] = grouped["signed_residual"].abs()
        records.append(grouped)
    return pd.concat(records, ignore_index=True)


def _metric_rows(
    condition_residuals: pd.DataFrame,
    category_residuals: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        condition_residuals.assign(summary_type="utterance_cells", category="utterance"),
        category_residuals,
    ]
    group_cols = [
        "model",
        "semantics",
        "sufficient_form_pair_boost",
        "three_word_penalty",
        "summary_type",
    ]
    rows = []
    for frame in frames:
        for keys, sub in frame.groupby(group_cols, dropna=False):
            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "scope": "overall",
                    "n_cells": int(len(sub)),
                    "mae": float(sub["abs_residual"].mean()),
                    "rmse": float(np.sqrt(np.mean(sub["signed_residual"] ** 2))),
                    "kl_empirical_to_model": safe_kl(
                        sub["empirical_proportion"],
                        sub["model_proportion"],
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _target_residual_deltas(
    condition_residuals: pd.DataFrame,
    *,
    n_target_cells: int,
) -> pd.DataFrame:
    rows = []
    baseline_by_semantics = {
        "context_fixed": BASELINE_STATIC_MODEL,
        "context_updating": BASELINE_RECURSIVE_MODEL,
    }
    candidate_by_semantics = {
        "context_fixed": BOUNDED_STATIC_MODEL,
        "context_updating": BOUNDED_RECURSIVE_MODEL,
    }
    index_cols = [
        "model",
        "semantics",
        "sufficient_form_pair_boost",
        "three_word_penalty",
    ] + GROUP_COLS + ["utterance_label"]
    indexed = condition_residuals.set_index(index_cols)

    for semantics, baseline in baseline_by_semantics.items():
        base_rows = condition_residuals[
            condition_residuals["model"].eq(baseline)
            & condition_residuals["semantics"].eq(semantics)
            & condition_residuals["sufficient_form_pair_boost"].eq(0.0)
            & condition_residuals["three_word_penalty"].eq(0.0)
        ].sort_values("abs_residual", ascending=False)
        target_cells = base_rows.head(n_target_cells)
        candidates = condition_residuals[
            condition_residuals["model"].eq(candidate_by_semantics[semantics])
            & condition_residuals["semantics"].eq(semantics)
            & (
                condition_residuals["sufficient_form_pair_boost"].gt(0.0)
                | condition_residuals["three_word_penalty"].gt(0.0)
            )
        ][
            [
                "sufficient_form_pair_boost",
                "three_word_penalty",
            ]
        ].drop_duplicates()
        for _, candidate in candidates.iterrows():
            param = (
                float(candidate["sufficient_form_pair_boost"]),
                float(candidate["three_word_penalty"]),
            )
            for _, base in target_cells.iterrows():
                key_base = (
                    baseline,
                    semantics,
                    0.0,
                    0.0,
                    base["relevant_property"],
                    base["sharpness"],
                    base["utterance_label"],
                )
                key_candidate = (
                    candidate_by_semantics[semantics],
                    semantics,
                    param[0],
                    param[1],
                    base["relevant_property"],
                    base["sharpness"],
                    base["utterance_label"],
                )
                if key_base not in indexed.index or key_candidate not in indexed.index:
                    continue
                cand = indexed.loc[key_candidate]
                rows.append(
                    {
                        "candidate": candidate_by_semantics[semantics],
                        "baseline": baseline,
                        "semantics": semantics,
                        "sufficient_form_pair_boost": param[0],
                        "three_word_penalty": param[1],
                        "relevant_property": base["relevant_property"],
                        "sharpness": base["sharpness"],
                        "utterance_label": base["utterance_label"],
                        "baseline_signed_residual": float(base["signed_residual"]),
                        "candidate_signed_residual": float(cand["signed_residual"]),
                        "baseline_abs_residual": float(base["abs_residual"]),
                        "candidate_abs_residual": float(cand["abs_residual"]),
                        "abs_residual_reduction": float(
                            base["abs_residual"] - cand["abs_residual"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _gate_decision(
    metrics: pd.DataFrame,
    target_deltas: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    pivot = metrics.pivot_table(
        index=[
            "model",
            "semantics",
            "sufficient_form_pair_boost",
            "three_word_penalty",
        ],
        columns="summary_type",
        values="rmse",
        aggfunc="first",
    ).reset_index()
    baseline = pivot[
        pivot["model"].isin([BASELINE_STATIC_MODEL, BASELINE_RECURSIVE_MODEL])
    ].set_index("semantics")
    candidates = pivot[
        pivot["model"].isin([BOUNDED_STATIC_MODEL, BOUNDED_RECURSIVE_MODEL])
        & (
            pivot["sufficient_form_pair_boost"].gt(0.0)
            | pivot["three_word_penalty"].gt(0.0)
        )
    ].copy()
    records = []
    for _, cand in candidates.iterrows():
        base = baseline.loc[cand["semantics"]]
        target = target_deltas[
            target_deltas["candidate"].eq(cand["model"])
            & target_deltas["sufficient_form_pair_boost"].eq(
                cand["sufficient_form_pair_boost"]
            )
            & target_deltas["three_word_penalty"].eq(cand["three_word_penalty"])
        ]
        target_mean_gain = (
            float(target["abs_residual_reduction"].mean()) if not target.empty else np.nan
        )
        target_worst_harm = (
            float(np.maximum(-target["abs_residual_reduction"], 0).max())
            if not target.empty else np.nan
        )
        category_cols = [
            col for col in ("length", "selection", "order", "overinformativeness")
            if col in cand.index and col in base.index
        ]
        records.append(
            {
                "candidate": cand["model"],
                "baseline": base["model"],
                "semantics": cand["semantics"],
                "sufficient_form_pair_boost": cand["sufficient_form_pair_boost"],
                "three_word_penalty": cand["three_word_penalty"],
                "utterance_rmse_gain": float(
                    base["utterance_cells"] - cand["utterance_cells"]
                ),
                "category_rmse_gain": float(
                    np.mean([base[col] - cand[col] for col in category_cols])
                ),
                "target_abs_residual_reduction_mean": target_mean_gain,
                "target_worst_abs_residual_harm": target_worst_harm,
            }
        )
    scored = pd.DataFrame(records)
    if scored.empty:
        return scored
    scored["target_harm_ok"] = (
        scored["target_worst_abs_residual_harm"].isna()
        | (scored["target_worst_abs_residual_harm"] <= args.max_target_harm)
    )
    scored["forward_gate_pass"] = scored["target_harm_ok"] & (
        (scored["utterance_rmse_gain"] >= args.utterance_rmse_gate)
        | (scored["category_rmse_gain"] >= args.category_rmse_gate)
        | (scored["target_abs_residual_reduction_mean"] >= args.target_residual_gate)
    )
    scored["rank_score"] = (
        scored["utterance_rmse_gain"]
        + scored["category_rmse_gain"]
        + scored["target_abs_residual_reduction_mean"].fillna(0.0)
        - np.maximum(scored["target_worst_abs_residual_harm"].fillna(0.0) - args.max_target_harm, 0.0)
    )
    scored = scored.sort_values("rank_score", ascending=False).reset_index(drop=True)
    scored["recommended_for_gpu_pilot"] = False
    pass_rows = scored.index[scored["forward_gate_pass"]].tolist()
    if pass_rows:
        scored.loc[pass_rows[0], "recommended_for_gpu_pilot"] = True
    scored["utterance_rmse_gate"] = args.utterance_rmse_gate
    scored["category_rmse_gate"] = args.category_rmse_gate
    scored["target_residual_gate"] = args.target_residual_gate
    scored["max_target_harm"] = args.max_target_harm
    return scored


def _success_standard(args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "forward_audit",
                "criterion": "run_gpu_pilot_if_any_gate_passes",
                "utterance_rmse_gain": args.utterance_rmse_gate,
                "category_rmse_gain": args.category_rmse_gate,
                "target_abs_residual_reduction_mean": args.target_residual_gate,
                "max_target_worst_abs_residual_harm": args.max_target_harm,
            },
            {
                "stage": "posterior_pilot",
                "criterion": "advance_if_diagnostic_clean_and_elpd_or_pareto_ppc",
                "max_r_hat": args.max_r_hat,
                "max_divergences": 0,
                "positive_delta_elpd_over_matched_anchor": 0.0,
                "ppc_rmse_gain": args.posterior_ppc_rmse_gate,
                "target_abs_residual_reduction_mean": args.posterior_target_residual_gate,
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results_bounded_form_forward_audit/stats")
    parser.add_argument(
        "--anchor-diagnostics",
        default=(
            "results_response_policy_fixedeps_pilot/stats/"
            "production_simplified_mcmc_diagnostics.csv"
        ),
    )
    parser.add_argument("--condition-subset", default="erdc,zrdc,brdc")
    parser.add_argument("--pair-boost-grid", default="0,0.25,0.5,0.75,1,1.25")
    parser.add_argument("--three-word-penalty-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--n-target-cells", type=int, default=12)
    parser.add_argument("--utterance-rmse-gate", type=float, default=0.005)
    parser.add_argument("--category-rmse-gate", type=float, default=0.02)
    parser.add_argument("--target-residual-gate", type=float, default=0.03)
    parser.add_argument("--max-target-harm", type=float, default=0.05)
    parser.add_argument("--max-r-hat", type=float, default=1.05)
    parser.add_argument("--posterior-ppc-rmse-gate", type=float, default=0.005)
    parser.add_argument("--posterior-target-residual-gate", type=float, default=0.03)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = SCRIPT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_path = Path(args.anchor_diagnostics)
    if not anchor_path.is_absolute():
        anchor_path = SCRIPT_DIR / anchor_path

    data = _filter_conditions(
        import_dataset_hier(state_encoding="target_match"),
        args.condition_subset,
    )
    empirical = _empirical_by_condition(data["df"])
    masks = _bounded_masks(data)
    grids = list(
        product(
            _float_grid(args.pair_boost_grid),
            _float_grid(args.three_word_penalty_grid),
        )
    )

    model_frames = []
    for recursive, semantics, baseline_model, bounded_model in (
        (False, "context_fixed", BASELINE_STATIC_MODEL, BOUNDED_STATIC_MODEL),
        (True, "context_updating", BASELINE_RECURSIVE_MODEL, BOUNDED_RECURSIVE_MODEL),
    ):
        params = _read_params(anchor_path, baseline_model)
        base_probs = _speaker_probabilities(data, params, recursive=recursive)
        model_frames.append(
            _model_by_condition(
                data,
                base_probs,
                model=baseline_model,
                semantics=semantics,
                sufficient_form_pair_boost=0.0,
                three_word_penalty=0.0,
            )
        )
        for pair_boost, three_penalty in grids:
            probs = _apply_bounded_form(
                base_probs,
                masks,
                sufficient_form_pair_boost=pair_boost,
                three_word_penalty=three_penalty,
            )
            model_frames.append(
                _model_by_condition(
                    data,
                    probs,
                    model=bounded_model,
                    semantics=semantics,
                    sufficient_form_pair_boost=pair_boost,
                    three_word_penalty=three_penalty,
                )
            )

    model_rows = pd.concat(model_frames, ignore_index=True)
    condition_residuals = _condition_residuals(empirical, model_rows)
    category_residuals = _category_residuals(condition_residuals)
    metrics = _metric_rows(condition_residuals, category_residuals)
    target_deltas = _target_residual_deltas(
        condition_residuals,
        n_target_cells=args.n_target_cells,
    )
    gate = _gate_decision(metrics, target_deltas, args)

    empirical.to_csv(out_dir / "production_bounded_form_empirical.csv", index=False)
    condition_residuals.to_csv(
        out_dir / "production_bounded_form_by_condition.csv",
        index=False,
    )
    category_residuals.to_csv(
        out_dir / "production_bounded_form_category_residuals.csv",
        index=False,
    )
    metrics.to_csv(out_dir / "production_bounded_form_summary_metrics.csv", index=False)
    target_deltas.to_csv(
        out_dir / "production_bounded_form_target_residual_deltas.csv",
        index=False,
    )
    gate.to_csv(out_dir / "production_bounded_form_gate_decision.csv", index=False)
    _success_standard(args).to_csv(
        out_dir / "production_bounded_form_success_standard.csv",
        index=False,
    )

    print(f"Wrote bounded-form forward audit CSVs to {out_dir}")
    if not gate.empty:
        print(gate.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
