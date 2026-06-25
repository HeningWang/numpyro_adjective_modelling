"""Forward audit for size-sharp production response-policy variants.

This script does not run inference. It starts from the fitted bounded-form
production pilot means and applies two narrow utterance-level pressures:

* boost exact bare size (D) when size alone is sufficient in a sharp display;
* penalize exact two-word size+form pairs (DF/FD) under the same gate.

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
from helper import import_dataset_hier  # noqa: E402


GROUP_COLS = ["relevant_property", "sharpness"]
UTTERANCE_LABELS = ms.UTTERANCE_LABELS

BASELINE_STATIC_MODEL = (
    "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static_fixedeps"
)
BASELINE_RECURSIVE_MODEL = (
    "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec_fixedeps"
)
SIZE_SHARP_STATIC_MODEL = (
    "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_static_fixedeps"
)
SIZE_SHARP_RECURSIVE_MODEL = (
    "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_rec_fixedeps"
)


def _float_grid(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


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


def _read_params(path: Path, model: str) -> dict[str, float]:
    params = {
        "alpha": 1.66,
        "beta_order": float(np.exp(0.61)),
        "lambda_salience": 1.34,
        "rho_salience_stop": 0.01,
        "lambda_sufficient_single": 1.90,
        "lambda_reliability_form": 1.90,
        "lambda_sufficient_form_pair": 0.71,
        "lambda_three_word_penalty": 0.88,
        "lambda_sharp_form_suppression": 0.0,
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
        "lambda_sufficient_form_pair",
        "lambda_three_word_penalty",
        "lambda_sharp_form_suppression",
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
        jnp.float32(params["lambda_sufficient_form_pair"]),
        jnp.float32(params["lambda_three_word_penalty"]),
        jnp.float32(params["lambda_sharp_form_suppression"]),
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


def _size_sharp_masks(data: dict) -> tuple[np.ndarray, np.ndarray]:
    sufficient_dim = np.asarray(data["sufficient_dim"], dtype=np.int32)
    has_one_word_solution = np.asarray(data["has_one_word_solution"], dtype=np.float32)
    is_sharp = np.asarray(data["sharpness_idx"], dtype=np.float32)
    full_present = np.asarray(ms.FULL_PRESENT_15, dtype=np.float32)
    n_words = np.asarray(ms.N_WORDS, dtype=np.float32)

    trial_gate = (
        (is_sharp > 0.5)
        & (has_one_word_solution > 0.5)
        & (sufficient_dim == 0)
    )
    exact_d = (n_words == 1.0) & (full_present[:, 0] > 0.5)
    exact_size_form_pair = (
        (n_words == 2.0)
        & (full_present[:, 0] > 0.5)
        & (full_present[:, 1] < 0.5)
        & (full_present[:, 2] > 0.5)
    )
    return (
        (trial_gate[:, None] & exact_d[None, :]).astype(float),
        (trial_gate[:, None] & exact_size_form_pair[None, :]).astype(float),
    )


def _apply_size_sharp(
    probs: np.ndarray,
    masks: tuple[np.ndarray, np.ndarray],
    *,
    single_bonus: float,
    form_pair_penalty: float,
) -> np.ndarray:
    exact_d, exact_size_form_pair = masks
    logits = (
        np.log(np.clip(probs, 1e-12, None))
        + single_bonus * exact_d
        - form_pair_penalty * exact_size_form_pair
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
    single_bonus: float,
    form_pair_penalty: float,
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
    out["lambda_size_sharp_single_bonus"] = single_bonus
    out["lambda_size_sharp_form_pair_penalty"] = form_pair_penalty
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


def _summary_metrics(condition_residuals: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model",
        "semantics",
        "lambda_size_sharp_single_bonus",
        "lambda_size_sharp_form_pair_penalty",
    ]
    rows = []
    for keys, sub in condition_residuals.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_cells": int(len(sub)),
                "mae": float(sub["abs_residual"].mean()),
                "rmse": float(np.sqrt(np.mean(sub["signed_residual"] ** 2))),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _cell_abs(
    indexed: pd.DataFrame,
    model: str,
    semantics: str,
    single_bonus: float,
    form_pair_penalty: float,
    relevant_property: str,
    sharpness: str,
    utterance_label: str,
) -> float:
    key = (
        model,
        semantics,
        single_bonus,
        form_pair_penalty,
        relevant_property,
        sharpness,
        utterance_label,
    )
    return float(indexed.loc[key]["abs_residual"])


def _candidate_rows(
    condition_residuals: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    baseline_by_semantics = {
        "context_fixed": BASELINE_STATIC_MODEL,
        "context_updating": BASELINE_RECURSIVE_MODEL,
    }
    candidate_by_semantics = {
        "context_fixed": SIZE_SHARP_STATIC_MODEL,
        "context_updating": SIZE_SHARP_RECURSIVE_MODEL,
    }
    index_cols = [
        "model",
        "semantics",
        "lambda_size_sharp_single_bonus",
        "lambda_size_sharp_form_pair_penalty",
    ] + GROUP_COLS + ["utterance_label"]
    indexed = condition_residuals.set_index(index_cols)
    metric_index = metrics.set_index(
        [
            "model",
            "semantics",
            "lambda_size_sharp_single_bonus",
            "lambda_size_sharp_form_pair_penalty",
        ]
    )

    rows = []
    for semantics, baseline in baseline_by_semantics.items():
        candidate_model = candidate_by_semantics[semantics]
        candidates = condition_residuals[
            condition_residuals["model"].eq(candidate_model)
            & (
                condition_residuals["lambda_size_sharp_single_bonus"].gt(0.0)
                | condition_residuals[
                    "lambda_size_sharp_form_pair_penalty"
                ].gt(0.0)
            )
        ][
            [
                "lambda_size_sharp_single_bonus",
                "lambda_size_sharp_form_pair_penalty",
            ]
        ].drop_duplicates()
        base_metric = metric_index.loc[(baseline, semantics, 0.0, 0.0)]
        base_df = _cell_abs(indexed, baseline, semantics, 0.0, 0.0, "first", "sharp", "DF")
        base_d = _cell_abs(indexed, baseline, semantics, 0.0, 0.0, "first", "sharp", "D")
        base_cf = _cell_abs(indexed, baseline, semantics, 0.0, 0.0, "second", "sharp", "CF")
        for _, candidate in candidates.iterrows():
            single_bonus = float(candidate["lambda_size_sharp_single_bonus"])
            form_pair_penalty = float(
                candidate["lambda_size_sharp_form_pair_penalty"]
            )
            cand_metric = metric_index.loc[
                (candidate_model, semantics, single_bonus, form_pair_penalty)
            ]
            cand_df = _cell_abs(
                indexed,
                candidate_model,
                semantics,
                single_bonus,
                form_pair_penalty,
                "first",
                "sharp",
                "DF",
            )
            cand_d = _cell_abs(
                indexed,
                candidate_model,
                semantics,
                single_bonus,
                form_pair_penalty,
                "first",
                "sharp",
                "D",
            )
            cand_cf = _cell_abs(
                indexed,
                candidate_model,
                semantics,
                single_bonus,
                form_pair_penalty,
                "second",
                "sharp",
                "CF",
            )
            rows.append(
                {
                    "variant": candidate_model,
                    "baseline": baseline,
                    "semantics": semantics,
                    "lambda_size_sharp_single_bonus": single_bonus,
                    "lambda_size_sharp_form_pair_penalty": form_pair_penalty,
                    "condition_rmse": float(cand_metric["rmse"]),
                    "baseline_condition_rmse": float(base_metric["rmse"]),
                    "condition_rmse_delta": float(
                        cand_metric["rmse"] - base_metric["rmse"]
                    ),
                    "first_sharp_DF_abs_residual": cand_df,
                    "first_sharp_DF_baseline_abs_residual": base_df,
                    "first_sharp_DF_abs_residual_reduction": base_df - cand_df,
                    "first_sharp_D_abs_residual": cand_d,
                    "first_sharp_D_baseline_abs_residual": base_d,
                    "first_sharp_D_abs_residual_reduction": base_d - cand_d,
                    "second_sharp_CF_abs_residual": cand_cf,
                    "second_sharp_CF_baseline_abs_residual": base_cf,
                    "second_sharp_CF_abs_residual_worsening": max(cand_cf - base_cf, 0.0),
                }
            )
    return pd.DataFrame(rows)


def build_gate_decision(
    rows: pd.DataFrame,
    *,
    first_sharp_df_gate: float = 0.05,
    first_sharp_d_gate: float = 0.04,
    second_sharp_cf_max_worsening: float = 0.015,
    max_condition_rmse_delta: float = 0.001,
) -> pd.DataFrame:
    scored = rows.copy()
    pass_mask = (
        scored["first_sharp_DF_abs_residual_reduction"].ge(first_sharp_df_gate)
        & scored["first_sharp_D_abs_residual_reduction"].ge(first_sharp_d_gate)
        & scored["second_sharp_CF_abs_residual_worsening"].le(
            second_sharp_cf_max_worsening
        )
        & scored["condition_rmse_delta"].le(max_condition_rmse_delta)
    )
    reasons = []
    for _, row in scored.iterrows():
        failures = []
        if row["first_sharp_DF_abs_residual_reduction"] < first_sharp_df_gate:
            failures.append("first/sharp DF residual")
        if row["first_sharp_D_abs_residual_reduction"] < first_sharp_d_gate:
            failures.append("first/sharp D residual")
        if row["second_sharp_CF_abs_residual_worsening"] > second_sharp_cf_max_worsening:
            failures.append("second/sharp CF residual")
        if row["condition_rmse_delta"] > max_condition_rmse_delta:
            failures.append("condition RMSE")
        if failures:
            reasons.append("failed gate: " + ", ".join(failures))
        else:
            reasons.append(
                "target residuals pass without unacceptable second/sharp CF or RMSE harm"
            )
    scored["full_inference_gate"] = np.where(pass_mask, "pass", "fail")
    scored["gate_reason"] = reasons
    scored["rank_score"] = (
        scored["first_sharp_DF_abs_residual_reduction"]
        + scored["first_sharp_D_abs_residual_reduction"]
        - np.maximum(
            scored["second_sharp_CF_abs_residual_worsening"]
            - second_sharp_cf_max_worsening,
            0.0,
        )
        - np.maximum(scored["condition_rmse_delta"] - max_condition_rmse_delta, 0.0)
    )
    scored = scored.sort_values(
        ["full_inference_gate", "rank_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    scored["recommended_for_gpu_pilot"] = False
    pass_rows = scored.index[scored["full_inference_gate"].eq("pass")].tolist()
    if pass_rows:
        scored.loc[pass_rows[0], "recommended_for_gpu_pilot"] = True
    scored["first_sharp_df_gate"] = first_sharp_df_gate
    scored["first_sharp_d_gate"] = first_sharp_d_gate
    scored["second_sharp_cf_max_worsening"] = second_sharp_cf_max_worsening
    scored["max_condition_rmse_delta"] = max_condition_rmse_delta
    return scored


def _success_standard(args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "forward_audit",
                "criterion": "first_sharp_DF_abs_residual_reduction",
                "threshold": args.first_sharp_df_gate,
                "direction": ">=",
            },
            {
                "stage": "forward_audit",
                "criterion": "first_sharp_D_abs_residual_reduction",
                "threshold": args.first_sharp_d_gate,
                "direction": ">=",
            },
            {
                "stage": "forward_audit",
                "criterion": "second_sharp_CF_abs_residual_worsening",
                "threshold": args.second_sharp_cf_max_worsening,
                "direction": "<=",
            },
            {
                "stage": "forward_audit",
                "criterion": "condition_rmse_delta",
                "threshold": args.max_condition_rmse_delta,
                "direction": "<=",
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results_size_sharp_forward_audit/stats")
    parser.add_argument(
        "--anchor-diagnostics",
        default=(
            "results_bounded_form_2x2_pilot/stats/"
            "production_simplified_mcmc_diagnostics.csv"
        ),
    )
    parser.add_argument("--condition-subset", default="erdc,zrdc,brdc")
    parser.add_argument("--single-bonus-grid", default="0,0.5,1,1.5,2,2.5")
    parser.add_argument("--form-pair-penalty-grid", default="0,0.5,1,1.5,2,2.5")
    parser.add_argument("--first-sharp-df-gate", type=float, default=0.05)
    parser.add_argument("--first-sharp-d-gate", type=float, default=0.04)
    parser.add_argument("--second-sharp-cf-max-worsening", type=float, default=0.015)
    parser.add_argument("--max-condition-rmse-delta", type=float, default=0.001)
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
    masks = _size_sharp_masks(data)
    grids = list(
        product(
            _float_grid(args.single_bonus_grid),
            _float_grid(args.form_pair_penalty_grid),
        )
    )

    model_frames = []
    anchor_rows = []
    for recursive, semantics, baseline_model, candidate_model in (
        (False, "context_fixed", BASELINE_STATIC_MODEL, SIZE_SHARP_STATIC_MODEL),
        (True, "context_updating", BASELINE_RECURSIVE_MODEL, SIZE_SHARP_RECURSIVE_MODEL),
    ):
        params = _read_params(anchor_path, baseline_model)
        anchor_rows.append({"model": baseline_model, "semantics": semantics, **params})
        base_probs = _speaker_probabilities(data, params, recursive=recursive)
        model_frames.append(
            _model_by_condition(
                data,
                base_probs,
                model=baseline_model,
                semantics=semantics,
                single_bonus=0.0,
                form_pair_penalty=0.0,
            )
        )
        for single_bonus, form_pair_penalty in grids:
            probs = _apply_size_sharp(
                base_probs,
                masks,
                single_bonus=single_bonus,
                form_pair_penalty=form_pair_penalty,
            )
            model_frames.append(
                _model_by_condition(
                    data,
                    probs,
                    model=candidate_model,
                    semantics=semantics,
                    single_bonus=single_bonus,
                    form_pair_penalty=form_pair_penalty,
                )
            )

    model_rows = pd.concat(model_frames, ignore_index=True)
    condition_residuals = _condition_residuals(empirical, model_rows)
    metrics = _summary_metrics(condition_residuals)
    candidate_rows = _candidate_rows(condition_residuals, metrics)
    gate = build_gate_decision(
        candidate_rows,
        first_sharp_df_gate=args.first_sharp_df_gate,
        first_sharp_d_gate=args.first_sharp_d_gate,
        second_sharp_cf_max_worsening=args.second_sharp_cf_max_worsening,
        max_condition_rmse_delta=args.max_condition_rmse_delta,
    )

    empirical.to_csv(out_dir / "production_size_sharp_empirical.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(
        out_dir / "production_size_sharp_anchor_params.csv",
        index=False,
    )
    condition_residuals.to_csv(
        out_dir / "production_size_sharp_by_condition.csv",
        index=False,
    )
    metrics.to_csv(
        out_dir / "production_size_sharp_summary_metrics.csv",
        index=False,
    )
    candidate_rows.to_csv(
        out_dir / "production_size_sharp_target_residual_deltas.csv",
        index=False,
    )
    gate.to_csv(
        out_dir / "production_size_sharp_gate_decision.csv",
        index=False,
    )
    _success_standard(args).to_csv(
        out_dir / "production_size_sharp_success_standard.csv",
        index=False,
    )

    print(f"Wrote size-sharp forward audit CSVs to {out_dir}")
    if not gate.empty:
        print(gate.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
