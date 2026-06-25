"""Forward audit for production response-policy speaker pressures.

This script does not run inference. It starts from the current production
incremental 2x2 anchor predictions and applies small, interpretable
utterance-level policy pressures:

* sufficient-single boost: prefer the one-word adjective that is already
  sufficient on one-word-solution displays;
* reliability-form boost: prefer utterances containing form when colour is not
  sufficient, treating form as a reliability safeguard rather than a greedy
  local informativeness token;
* non-sufficient-colour penalty: penalize bare colour when colour alone is not
  sufficient.

The audit writes compact CSV summaries before any plotting or inference
decision. A gate pass means the policy is worth promoting into a registered
NumPyro model and piloting on GPU; it is not itself posterior evidence.
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
from planned_prefix_forward_audit import _read_anchor_params  # noqa: E402


GROUP_COLS = ["conditions", "relevant_property", "sharpness"]
UTTERANCE_LABELS = ms.UTTERANCE_LABELS

ANCHOR_MODEL = "principled_salience_stop_regularized_2x2_inc_static"
RECURSIVE_ANCHOR_MODEL = "principled_salience_stop_regularized_2x2_inc_rec"
POLICY_STATIC_MODEL = "principled_salience_stop_regularized_responsepolicy_2x2_inc_static"
POLICY_RECURSIVE_MODEL = "principled_salience_stop_regularized_responsepolicy_2x2_inc_rec"

TARGET_RESIDUAL_SPECS = (
    # Strongest production failures under the current context-fixed anchor.
    ("selection", "ALL", "both", "ALL", "redundant_form"),
    ("selection", "ALL", "both", "ALL", "colour_only"),
    ("selection", "zrdc", "second", "ALL", "colour_only"),
    ("overinformativeness", "zrdc", "second", "ALL", "over_minimal_length"),
    ("selection", "erdc", "first", "ALL", "size_only"),
    ("selection", "erdc", "first", "ALL", "size_colour"),
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


def _speaker_probabilities(
    data: dict,
    params: dict[str, float],
    *,
    recursive: bool,
) -> np.ndarray:
    n = len(data["df"])
    alpha = jnp.full((n,), params["alpha"], dtype=jnp.float32)
    probs = ms.jitted_speaker_principled_hier(
        data["states_train"],
        data["sufficient_dim"],
        data["has_one_word_solution"],
        data["sharpness_idx"],
        alpha,
        jnp.float32(params["beta_order"]),
        jnp.float32(params["lambda_salience"]),
        jnp.float32(params["rho_salience_stop"]),
        jnp.float32(params["gamma_uncertainty_len"]),
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


def _policy_masks(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sufficient_dim = np.asarray(data["sufficient_dim"], dtype=np.int32)
    has_one_word_solution = np.asarray(data["has_one_word_solution"], dtype=np.float32)
    is_colour_sufficient = np.asarray(data["is_colour_sufficient"], dtype=np.float32)
    full_present = np.asarray(ms.FULL_PRESENT_15, dtype=np.float32)
    n_words = np.asarray(ms.N_WORDS, dtype=np.float32)

    one_dim = (full_present.sum(axis=1) == 1.0)
    only_sufficient = (
        one_dim[None, :]
        & (full_present[None, :, :] == 1.0).any(axis=2)
        & (
            np.argmax(full_present, axis=1)[None, :]
            == sufficient_dim[:, None]
        )
        & (has_one_word_solution[:, None] > 0.5)
        & (n_words[None, :] == 1.0)
    )

    colour_not_sufficient = is_colour_sufficient[:, None] < 0.5
    form_present = np.asarray(ms.F_PRESENT_15, dtype=np.float32)[None, :] > 0.5
    reliability_form = colour_not_sufficient & form_present

    colour_only = (
        colour_not_sufficient
        & (full_present[None, :, 1] > 0.5)
        & (full_present[None, :, 0] < 0.5)
        & (full_present[None, :, 2] < 0.5)
    )
    return (
        only_sufficient.astype(np.float32),
        reliability_form.astype(np.float32),
        colour_only.astype(np.float32),
    )


def _apply_policy(
    probs: np.ndarray,
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    sufficient_single_boost: float,
    reliability_form_boost: float,
    nonsufficient_colour_penalty: float,
) -> np.ndarray:
    single_mask, form_mask, colour_penalty_mask = masks
    logp = np.log(np.clip(probs, 1e-12, None))
    adjusted = (
        logp
        + sufficient_single_boost * single_mask
        + reliability_form_boost * form_mask
        - nonsufficient_colour_penalty * colour_penalty_mask
    )
    adjusted = adjusted - adjusted.max(axis=1, keepdims=True)
    out = np.exp(adjusted)
    return out / out.sum(axis=1, keepdims=True)


def _model_by_condition(
    data: dict,
    probs: np.ndarray,
    *,
    model: str,
    semantics: str,
    sufficient_single_boost: float,
    reliability_form_boost: float,
    nonsufficient_colour_penalty: float,
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
    out["sufficient_single_boost"] = sufficient_single_boost
    out["reliability_form_boost"] = reliability_form_boost
    out["nonsufficient_colour_penalty"] = nonsufficient_colour_penalty
    return out


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


def _condition_residuals(empirical: pd.DataFrame, model_rows: pd.DataFrame) -> pd.DataFrame:
    merged = model_rows.merge(
        empirical[
            GROUP_COLS
            + ["utterance_code", "utterance_label", "empirical_proportion", "condition_n"]
        ],
        on=GROUP_COLS + ["utterance_code", "utterance_label"],
        how="left",
    )
    merged["signed_residual"] = merged["model_proportion"] - merged["empirical_proportion"]
    merged["abs_residual"] = merged["signed_residual"].abs()
    return merged


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
        "sufficient_single_boost",
        "reliability_form_boost",
        "nonsufficient_colour_penalty",
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
    param_cols = [
        "model",
        "semantics",
        "sufficient_single_boost",
        "reliability_form_boost",
        "nonsufficient_colour_penalty",
        "summary_type",
    ]
    rows = []
    for frame in frames:
        for keys, sub in frame.groupby(param_cols, dropna=False):
            row = dict(zip(param_cols, keys))
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


def _target_residual_deltas(category_residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    index_cols = [
        "model",
        "semantics",
        "sufficient_single_boost",
        "reliability_form_boost",
        "nonsufficient_colour_penalty",
        "summary_type",
        "conditions",
        "relevant_property",
        "sharpness",
        "category",
    ]
    indexed = category_residuals.set_index(index_cols)
    candidates = category_residuals[
        category_residuals["model"].isin([POLICY_STATIC_MODEL, POLICY_RECURSIVE_MODEL])
        & (
            category_residuals["sufficient_single_boost"].gt(0)
            | category_residuals["reliability_form_boost"].gt(0)
            | category_residuals["nonsufficient_colour_penalty"].gt(0)
        )
    ][
        [
            "model",
            "semantics",
            "sufficient_single_boost",
            "reliability_form_boost",
            "nonsufficient_colour_penalty",
        ]
    ].drop_duplicates()
    baseline_by_semantics = {
        "context_fixed": ANCHOR_MODEL,
        "context_updating": RECURSIVE_ANCHOR_MODEL,
    }
    available_conditions = sorted(category_residuals["conditions"].unique())
    available_sharpness = sorted(category_residuals["sharpness"].unique())
    for _, candidate in candidates.iterrows():
        baseline = baseline_by_semantics[candidate["semantics"]]
        param = (
            float(candidate["sufficient_single_boost"]),
            float(candidate["reliability_form_boost"]),
            float(candidate["nonsufficient_colour_penalty"]),
        )
        for summary_type, condition, rel, sharpness, category in TARGET_RESIDUAL_SPECS:
            cond_values = available_conditions if condition == "ALL" else [condition]
            sharp_values = available_sharpness if sharpness == "ALL" else [sharpness]
            for cond, sharp in product(cond_values, sharp_values):
                key_base = (
                    baseline,
                    candidate["semantics"],
                    0.0,
                    0.0,
                    0.0,
                    summary_type,
                    cond,
                    rel,
                    sharp,
                    category,
                )
                key_cand = (
                    candidate["model"],
                    candidate["semantics"],
                    *param,
                    summary_type,
                    cond,
                    rel,
                    sharp,
                    category,
                )
                if key_base not in indexed.index or key_cand not in indexed.index:
                    continue
                base = indexed.loc[key_base]
                cand = indexed.loc[key_cand]
                rows.append(
                    {
                        "candidate": candidate["model"],
                        "baseline": baseline,
                        "semantics": candidate["semantics"],
                        "sufficient_single_boost": param[0],
                        "reliability_form_boost": param[1],
                        "nonsufficient_colour_penalty": param[2],
                        "summary_type": summary_type,
                        "conditions": cond,
                        "relevant_property": rel,
                        "sharpness": sharp,
                        "category": category,
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
            "sufficient_single_boost",
            "reliability_form_boost",
            "nonsufficient_colour_penalty",
        ],
        columns="summary_type",
        values="rmse",
        aggfunc="first",
    ).reset_index()
    baseline = pivot[pivot["model"].isin([ANCHOR_MODEL, RECURSIVE_ANCHOR_MODEL])].copy()
    baseline = baseline.set_index("semantics")
    candidates = pivot[
        pivot["model"].isin([POLICY_STATIC_MODEL, POLICY_RECURSIVE_MODEL])
        & (
            pivot["sufficient_single_boost"].gt(0)
            | pivot["reliability_form_boost"].gt(0)
            | pivot["nonsufficient_colour_penalty"].gt(0)
        )
    ].copy()
    records = []
    for _, cand in candidates.iterrows():
        base = baseline.loc[cand["semantics"]]
        target = target_deltas[
            target_deltas["candidate"].eq(cand["model"])
            & target_deltas["sufficient_single_boost"].eq(cand["sufficient_single_boost"])
            & target_deltas["reliability_form_boost"].eq(cand["reliability_form_boost"])
            & target_deltas["nonsufficient_colour_penalty"].eq(
                cand["nonsufficient_colour_penalty"]
            )
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
                "sufficient_single_boost": cand["sufficient_single_boost"],
                "reliability_form_boost": cand["reliability_form_boost"],
                "nonsufficient_colour_penalty": cand["nonsufficient_colour_penalty"],
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
        return pd.DataFrame()
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
        + scored["target_abs_residual_reduction_mean"].fillna(0)
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


def _float_grid(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results_response_policy_forward_audit/stats")
    parser.add_argument(
        "--anchor-diagnostics",
        default=(
            "results_principled_2x2_regularized_local/stats/"
            "production_simplified_mcmc_diagnostics.csv"
        ),
    )
    parser.add_argument("--anchor-model", default=ANCHOR_MODEL)
    parser.add_argument("--condition-subset", default="")
    parser.add_argument("--single-grid", default="0,0.5,1,1.5,2,3")
    parser.add_argument("--form-grid", default="0,0.5,1,1.5,2,3")
    parser.add_argument("--colour-penalty-grid", default="0,0.5,1,1.5,2,3")
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
    params = _read_anchor_params(anchor_path, args.anchor_model)

    data = _filter_conditions(
        import_dataset_hier(state_encoding="target_match"),
        args.condition_subset,
    )
    empirical = _empirical_by_condition(data["df"])
    masks = _policy_masks(data)

    grids = list(
        product(
            _float_grid(args.single_grid),
            _float_grid(args.form_grid),
            _float_grid(args.colour_penalty_grid),
        )
    )

    model_frames = []
    for recursive, semantics, base_model, policy_model in (
        (False, "context_fixed", ANCHOR_MODEL, POLICY_STATIC_MODEL),
        (True, "context_updating", RECURSIVE_ANCHOR_MODEL, POLICY_RECURSIVE_MODEL),
    ):
        base_probs = _speaker_probabilities(data, params, recursive=recursive)
        model_frames.append(
            _model_by_condition(
                data,
                base_probs,
                model=base_model,
                semantics=semantics,
                sufficient_single_boost=0.0,
                reliability_form_boost=0.0,
                nonsufficient_colour_penalty=0.0,
            )
        )
        for single_boost, form_boost, colour_penalty in grids:
            if single_boost == form_boost == colour_penalty == 0.0:
                continue
            policy_probs = _apply_policy(
                base_probs,
                masks,
                sufficient_single_boost=single_boost,
                reliability_form_boost=form_boost,
                nonsufficient_colour_penalty=colour_penalty,
            )
            model_frames.append(
                _model_by_condition(
                    data,
                    policy_probs,
                    model=policy_model,
                    semantics=semantics,
                    sufficient_single_boost=single_boost,
                    reliability_form_boost=form_boost,
                    nonsufficient_colour_penalty=colour_penalty,
                )
            )

    model_rows = pd.concat(model_frames, ignore_index=True)
    condition_residuals = _condition_residuals(empirical, model_rows)
    category_residuals = _category_residuals(condition_residuals)
    metrics = _metric_rows(condition_residuals, category_residuals)
    target_deltas = _target_residual_deltas(category_residuals)
    gate = _gate_decision(metrics, target_deltas, args)

    empirical.to_csv(out_dir / "production_response_policy_empirical.csv", index=False)
    condition_residuals.to_csv(
        out_dir / "production_response_policy_by_condition.csv",
        index=False,
    )
    category_residuals.to_csv(
        out_dir / "production_response_policy_category_residuals.csv",
        index=False,
    )
    metrics.to_csv(out_dir / "production_response_policy_summary_metrics.csv", index=False)
    target_deltas.to_csv(
        out_dir / "production_response_policy_target_residual_deltas.csv",
        index=False,
    )
    gate.to_csv(out_dir / "production_response_policy_gate_decision.csv", index=False)
    pd.DataFrame(
        [
            {
                **params,
                "condition_subset": args.condition_subset,
                "single_grid": args.single_grid,
                "form_grid": args.form_grid,
                "colour_penalty_grid": args.colour_penalty_grid,
            }
        ]
    ).to_csv(out_dir / "production_response_policy_anchor_params.csv", index=False)
    _success_standard(args).to_csv(
        out_dir / "production_response_policy_success_standard.csv",
        index=False,
    )

    print(f"Wrote production response-policy forward audit CSVs to {out_dir}")
    if not gate.empty:
        print(gate.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
