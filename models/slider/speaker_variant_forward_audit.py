"""Forward screen for theoretically motivated slider speaker variants.

This script does not run inference. It uses target-relative semantics and
evaluates whether planned incremental speakers improve the PPC patterns that
the current greedy speaker misses before any server MCMC is launched.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from semantic_variant_forward_audit import (
    DEFAULT_DATASET,
    GROUP_COLS,
    build_states,
    listener_values,
    load_slider_dataset,
    incremental_prediction,
    global_prediction,
)


DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results_speaker_variant_forward_audit" / "stats"
EPS = 1e-20


VARIANT_COMPLEXITY = {
    "greedy_incremental_stop": {
        "mechanism_count": 2,
        "free_parameter_count": 2,
        "description": "current local first-word choice with continuation/stop competition",
    },
    "planned_incremental_stop": {
        "mechanism_count": 3,
        "free_parameter_count": 3,
        "description": "first-word choice uses soft value of stopping or continuing",
    },
    "planned_usefulness_order": {
        "mechanism_count": 4,
        "free_parameter_count": 4,
        "description": "planned stop speaker with order prior moderated by referential usefulness",
    },
    "planned_usefulness_signed_order": {
        "mechanism_count": 4,
        "free_parameter_count": 4,
        "description": "planned stop speaker whose usefulness adjustment can reward colour-initial order",
    },
    "planned_usefulness_mixture": {
        "mechanism_count": 5,
        "free_parameter_count": 5,
        "description": "convex mixture of greedy local and planned usefulness-order speakers",
    },
    "global_full_utterance": {
        "mechanism_count": 2,
        "free_parameter_count": 2,
        "description": "full-utterance RSA comparator",
    },
}

NEW_ABLATION_VARIANTS = {
    "planned_usefulness_signed_order",
    "planned_usefulness_mixture",
}
REFERENCE_PLANNED_VARIANT = "planned_usefulness_order"


def softmax2(a: float, b: float) -> tuple[float, float]:
    x = np.asarray([a, b], dtype=float)
    p = np.exp(x - np.max(x))
    p = p / p.sum()
    return float(p[0]), float(p[1])


def final_utilities(listener: dict[str, dict[str, float | bool]],
                    alpha: float, order_bias: float) -> dict[str, float]:
    return {
        "D": alpha * np.log(np.clip(float(listener["D"]["target_listener_prob"]), EPS, 1.0)),
        "C": alpha * np.log(np.clip(float(listener["C"]["target_listener_prob"]), EPS, 1.0)),
        "DC": alpha * np.log(np.clip(float(listener["DC"]["target_listener_prob"]), EPS, 1.0)),
        "CD": alpha * np.log(np.clip(float(listener["CD"]["target_listener_prob"]), EPS, 1.0)) - order_bias,
    }


def logsumexp(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    m = float(np.max(arr))
    return float(m + np.log(np.sum(np.exp(arr - m))))


def planned_stop_prediction(listener: dict[str, dict[str, float | bool]],
                            alpha: float, order_bias: float) -> dict[str, float]:
    util = final_utilities(listener, alpha, order_bias)

    value_size_prefix = logsumexp([util["D"], util["DC"]])
    value_colour_prefix = logsumexp([util["C"], util["CD"]])
    first_size, first_colour = softmax2(value_size_prefix, value_colour_prefix)

    continue_after_size, _ = softmax2(util["DC"], util["D"])
    continue_after_colour, _ = softmax2(util["CD"], util["C"])

    chain_dc = first_size * continue_after_size
    chain_cd = first_colour * continue_after_colour
    pred_slider, _ = softmax2(np.log(np.clip(chain_dc, EPS, 1.0)),
                              np.log(np.clip(chain_cd, EPS, 1.0)))

    return {
        "pred_slider": pred_slider,
        "first_step_size_prob": first_size,
        "continue_after_size": continue_after_size,
        "continue_after_colour": continue_after_colour,
        "chain_dc": chain_dc,
        "chain_cd": chain_cd,
        "effective_order_bias": order_bias,
    }


def usefulness_moderated_order_bias(listener: dict[str, dict[str, float | bool]],
                                    base_order_bias: float,
                                    usefulness_scale: float) -> float:
    size_usefulness = float(listener["D"]["target_listener_prob"])
    colour_usefulness = float(listener["C"]["target_listener_prob"])
    colour_advantage = colour_usefulness - size_usefulness
    return float(np.maximum(base_order_bias - usefulness_scale * colour_advantage, 0.0))


def signed_usefulness_moderated_order_bias(listener: dict[str, dict[str, float | bool]],
                                           base_order_bias: float,
                                           signed_scale: float) -> float:
    size_usefulness = float(listener["D"]["target_listener_prob"])
    colour_usefulness = float(listener["C"]["target_listener_prob"])
    colour_advantage = colour_usefulness - size_usefulness
    return float(base_order_bias - signed_scale * colour_advantage)


def mixture_prediction(listener: dict[str, dict[str, float | bool]],
                       alpha: float,
                       order_bias: float,
                       usefulness_scale: float,
                       planned_weight: float) -> dict[str, float]:
    greedy = incremental_prediction(listener, alpha, order_bias)
    effective_order_bias = usefulness_moderated_order_bias(
        listener,
        order_bias,
        usefulness_scale,
    )
    planned = planned_stop_prediction(listener, alpha, effective_order_bias)
    weight = float(np.clip(planned_weight, 0.0, 1.0))
    out = {}
    for key in ("pred_slider", "first_step_size_prob",
                "continue_after_size", "continue_after_colour",
                "chain_dc", "chain_cd"):
        out[key] = weight * float(planned[key]) + (1.0 - weight) * float(greedy[key])
    out["effective_order_bias"] = effective_order_bias
    out["planned_component"] = float(planned["pred_slider"])
    out["greedy_component"] = float(greedy["pred_slider"])
    out["planned_mixture_weight"] = weight
    return out


def predict_variant(variant: str,
                    listener: dict[str, dict[str, float | bool]],
                    args: argparse.Namespace) -> dict[str, float]:
    if variant == "greedy_incremental_stop":
        out = incremental_prediction(listener, args.alpha, args.order_bias)
        out["effective_order_bias"] = args.order_bias
        return out
    if variant == "planned_incremental_stop":
        return planned_stop_prediction(listener, args.alpha, args.order_bias)
    if variant == "planned_usefulness_order":
        order_bias = usefulness_moderated_order_bias(
            listener,
            args.order_bias,
            args.usefulness_order_scale,
        )
        return planned_stop_prediction(listener, args.alpha, order_bias)
    if variant == "planned_usefulness_signed_order":
        order_bias = signed_usefulness_moderated_order_bias(
            listener,
            args.order_bias,
            args.signed_order_scale,
        )
        return planned_stop_prediction(listener, args.alpha, order_bias)
    if variant == "planned_usefulness_mixture":
        return mixture_prediction(
            listener,
            args.alpha,
            args.order_bias,
            args.usefulness_order_scale,
            args.planned_mixture_weight,
        )
    if variant == "global_full_utterance":
        out = global_prediction(listener, args.alpha, args.order_bias)
        out["effective_order_bias"] = args.order_bias
        return out
    raise ValueError(f"Unknown variant: {variant}")


def parse_float_grid(value: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("Grid argument must contain at least one numeric value.")
    return out


def build_listener_records(df: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    records = []
    for obs_idx, row in df.reset_index(drop=True).iterrows():
        states = build_states(row, "target_match")
        listener = listener_values(
            states,
            size_context_mode=args.size_context_mode,
            color_sem=args.color_sem,
            k=args.k,
            wf=args.wf,
        )
        records.append(
            {
                "obs_idx": obs_idx,
                "listener": listener,
                "human_slider": float(row["human_slider"]),
                "group": tuple(row[col] for col in GROUP_COLS),
            }
        )
    return records


def listener_record_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    listeners = [record["listener"] for record in records]
    groups = [record["group"] for record in records]
    group_codes, group_uniques = pd.factorize(pd.Series(groups), sort=True)
    group_frame = pd.DataFrame(list(group_uniques), columns=GROUP_COLS)
    return {
        "D": np.asarray([item["D"]["target_listener_prob"] for item in listeners], dtype=float),
        "C": np.asarray([item["C"]["target_listener_prob"] for item in listeners], dtype=float),
        "DC": np.asarray([item["DC"]["target_listener_prob"] for item in listeners], dtype=float),
        "CD": np.asarray([item["CD"]["target_listener_prob"] for item in listeners], dtype=float),
        "human_slider": np.asarray([record["human_slider"] for record in records], dtype=float),
        "group_codes": group_codes.astype(int),
        "group_counts": np.bincount(group_codes),
        "group_conditions": group_frame["conditions"].astype(str).to_numpy(),
    }


def sigmoid_delta(a: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(np.asarray(b) - np.asarray(a), -60.0, 60.0)))


def planned_stop_array(
    arrays: dict[str, np.ndarray],
    alpha: float,
    order_bias: np.ndarray | float,
) -> np.ndarray:
    util_d = alpha * np.log(np.clip(arrays["D"], EPS, 1.0))
    util_c = alpha * np.log(np.clip(arrays["C"], EPS, 1.0))
    util_dc = alpha * np.log(np.clip(arrays["DC"], EPS, 1.0))
    util_cd = alpha * np.log(np.clip(arrays["CD"], EPS, 1.0)) - order_bias

    value_size = np.logaddexp(util_d, util_dc)
    value_colour = np.logaddexp(util_c, util_cd)
    first_size = sigmoid_delta(value_size, value_colour)
    first_colour = 1.0 - first_size

    continue_after_size = sigmoid_delta(util_dc, util_d)
    continue_after_colour = sigmoid_delta(util_cd, util_c)

    chain_dc = first_size * continue_after_size
    chain_cd = first_colour * continue_after_colour
    return sigmoid_delta(
        np.log(np.clip(chain_dc, EPS, 1.0)),
        np.log(np.clip(chain_cd, EPS, 1.0)),
    )


def incremental_array(arrays: dict[str, np.ndarray], alpha: float, bias: float) -> np.ndarray:
    d = np.power(np.clip(arrays["D"], EPS, 1.0), alpha)
    c = np.power(np.clip(arrays["C"], EPS, 1.0), alpha)
    first_size = d / np.clip(d + c, EPS, None)
    first_colour = 1.0 - first_size

    cont_dc = np.power(np.clip(arrays["DC"], EPS, 1.0), alpha)
    stop_d = d
    continue_after_size = cont_dc / np.clip(cont_dc + stop_d, EPS, None)

    cont_cd = np.power(np.clip(arrays["CD"], EPS, 1.0), alpha)
    stop_c = c
    continue_after_colour = cont_cd / np.clip(cont_cd + stop_c, EPS, None)

    chain_dc = first_size * continue_after_size
    chain_cd = first_colour * continue_after_colour
    return sigmoid_delta(
        np.log(np.clip(chain_dc, EPS, 1.0)),
        np.log(np.clip(chain_cd, EPS, 1.0)) - bias,
    )


def global_array(arrays: dict[str, np.ndarray], alpha: float, bias: float) -> np.ndarray:
    util_dc = alpha * np.log(np.clip(arrays["DC"], EPS, 1.0))
    util_cd = alpha * (np.log(np.clip(arrays["CD"], EPS, 1.0)) - bias)
    return sigmoid_delta(util_dc, util_cd)


def predict_variant_array(
    variant: str,
    arrays: dict[str, np.ndarray],
    params: dict[str, float],
) -> np.ndarray:
    alpha = float(params["alpha"])
    order_bias = float(params["order_bias"])
    if variant == "greedy_incremental_stop":
        return incremental_array(arrays, alpha, order_bias)
    if variant == "planned_incremental_stop":
        return planned_stop_array(arrays, alpha, order_bias)
    if variant == "planned_usefulness_order":
        color_advantage = arrays["C"] - arrays["D"]
        effective_bias = np.maximum(
            order_bias - float(params["usefulness_order_scale"]) * color_advantage,
            0.0,
        )
        return planned_stop_array(arrays, alpha, effective_bias)
    if variant == "planned_usefulness_signed_order":
        color_advantage = arrays["C"] - arrays["D"]
        effective_bias = order_bias - float(params["signed_order_scale"]) * color_advantage
        return planned_stop_array(arrays, alpha, effective_bias)
    if variant == "planned_usefulness_mixture":
        color_advantage = arrays["C"] - arrays["D"]
        effective_bias = np.maximum(
            order_bias - float(params["usefulness_order_scale"]) * color_advantage,
            0.0,
        )
        planned = planned_stop_array(arrays, alpha, effective_bias)
        greedy = incremental_array(arrays, alpha, order_bias)
        weight = float(np.clip(params["planned_mixture_weight"], 0.0, 1.0))
        return weight * planned + (1.0 - weight) * greedy
    if variant == "global_full_utterance":
        return global_array(arrays, alpha, order_bias)
    raise ValueError(f"Unknown variant: {variant}")


def iter_parameter_grid(args: argparse.Namespace):
    alphas = parse_float_grid(args.grid_alpha)
    order_biases = parse_float_grid(args.grid_order_bias)
    usefulness_scales = parse_float_grid(args.grid_usefulness_order_scale)
    signed_scales = parse_float_grid(args.grid_signed_order_scale)
    mixture_weights = parse_float_grid(args.grid_planned_mixture_weight)

    for alpha, order_bias in product(alphas, order_biases):
        yield "greedy_incremental_stop", {
            "alpha": alpha,
            "order_bias": order_bias,
            "usefulness_order_scale": np.nan,
            "signed_order_scale": np.nan,
            "planned_mixture_weight": np.nan,
        }
        yield "planned_incremental_stop", {
            "alpha": alpha,
            "order_bias": order_bias,
            "usefulness_order_scale": np.nan,
            "signed_order_scale": np.nan,
            "planned_mixture_weight": np.nan,
        }
        yield "global_full_utterance", {
            "alpha": alpha,
            "order_bias": order_bias,
            "usefulness_order_scale": np.nan,
            "signed_order_scale": np.nan,
            "planned_mixture_weight": np.nan,
        }
        for usefulness_scale in usefulness_scales:
            yield "planned_usefulness_order", {
                "alpha": alpha,
                "order_bias": order_bias,
                "usefulness_order_scale": usefulness_scale,
                "signed_order_scale": np.nan,
                "planned_mixture_weight": np.nan,
            }
            for mixture_weight in mixture_weights:
                yield "planned_usefulness_mixture", {
                    "alpha": alpha,
                    "order_bias": order_bias,
                    "usefulness_order_scale": usefulness_scale,
                    "signed_order_scale": np.nan,
                    "planned_mixture_weight": mixture_weight,
                }
        for signed_scale in signed_scales:
            yield "planned_usefulness_signed_order", {
                "alpha": alpha,
                "order_bias": order_bias,
                "usefulness_order_scale": np.nan,
                "signed_order_scale": signed_scale,
                "planned_mixture_weight": np.nan,
            }


def summarize_grid_cell(
    arrays: dict[str, np.ndarray],
    variant: str,
    params: dict[str, float],
) -> dict[str, float | str | bool]:
    pred = predict_variant_array(variant, arrays, params)
    human = arrays["human_slider"]
    residual = pred - human
    n = len(human)

    sum_pred = float(pred.sum())
    sum_human = float(human.sum())
    sum_pred_sq = float(np.square(pred).sum())
    sum_human_sq = float(np.square(human).sum())
    sum_pred_human = float((pred * human).sum())
    sum_resid_sq = float(np.square(residual).sum())
    sum_abs_resid = float(np.abs(residual).sum())

    cov = sum_pred_human - (sum_pred * sum_human / n)
    var_pred = sum_pred_sq - (sum_pred * sum_pred / n)
    var_human = sum_human_sq - (sum_human * sum_human / n)
    corr = cov / np.sqrt(var_pred * var_human) if var_pred > 0.0 and var_human > 0.0 else np.nan

    group_residual = np.bincount(
        arrays["group_codes"],
        weights=residual,
        minlength=len(arrays["group_counts"]),
    ) / arrays["group_counts"]
    cond_residual = group_residual.astype(float)
    group_conditions = arrays["group_conditions"]

    out = {
        "variant": variant,
        **params,
        "observation_rmse": float(np.sqrt(sum_resid_sq / n)),
        "observation_mae": float(sum_abs_resid / n),
        "observation_correlation": float(corr),
        "observation_r2": float(corr * corr) if np.isfinite(corr) else np.nan,
        "condition_rmse": float(np.sqrt(np.mean(np.square(cond_residual)))),
        "condition_mae": float(np.mean(np.abs(cond_residual))),
        "condition_abs_zrdc_residual": float(
            np.mean(np.abs(group_residual[group_conditions == "zrdc"]))
        ),
        "condition_abs_erdc_residual": float(
            np.mean(np.abs(group_residual[group_conditions == "erdc"]))
        ),
        "condition_abs_brdc_residual": float(
            np.mean(np.abs(group_residual[group_conditions == "brdc"]))
        ),
        **VARIANT_COMPLEXITY[variant],
    }
    return out


def parameter_grid_summary(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    records = build_listener_records(df, args)
    arrays = listener_record_arrays(records)
    rows = []
    for grid_id, (variant, params) in enumerate(iter_parameter_grid(args)):
        row = summarize_grid_cell(arrays, variant, params)
        row["grid_id"] = grid_id
        row["elpd_loo"] = np.nan
        row["elpd_loo_se"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["condition_rmse", "condition_abs_zrdc_residual", "mechanism_count"]
    ).reset_index(drop=True)


def parameter_grid_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "condition_rmse",
        "condition_abs_zrdc_residual",
        "condition_abs_erdc_residual",
        "condition_abs_brdc_residual",
        "mechanism_count",
    ]
    rows = []
    for i, candidate in summary.iterrows():
        dominated = False
        for j, other in summary.iterrows():
            if i == j:
                continue
            no_worse = all(other[metric] <= candidate[metric] for metric in metrics)
            strictly_better = any(other[metric] < candidate[metric] for metric in metrics)
            if no_worse and strictly_better:
                dominated = True
                break
        row = candidate.to_dict()
        row["grid_pareto_frontier"] = not dominated
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["grid_pareto_frontier", "condition_rmse"],
        ascending=[False, True],
    ).reset_index(drop=True)


def parameter_grid_best_by_variant(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, sub in summary.groupby("variant", sort=False):
        best_rmse = sub.sort_values(["condition_rmse", "condition_abs_zrdc_residual"]).iloc[0]
        best_zrdc = sub.sort_values(["condition_abs_zrdc_residual", "condition_rmse"]).iloc[0]
        for criterion, row in [
            ("condition_rmse", best_rmse),
            ("condition_abs_zrdc_residual", best_zrdc),
        ]:
            out = row.to_dict()
            out["selection_criterion"] = criterion
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["selection_criterion", "condition_rmse"])


def parameter_grid_gate_decision(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    best_greedy = (
        summary[summary["variant"].eq("greedy_incremental_stop")]
        .sort_values(["condition_rmse", "condition_abs_zrdc_residual"])
        .iloc[0]
    )
    best_planned = (
        summary[summary["variant"].eq(REFERENCE_PLANNED_VARIANT)]
        .sort_values(["condition_rmse", "condition_abs_zrdc_residual"])
        .iloc[0]
    )
    new_summary = summary[summary["variant"].isin(NEW_ABLATION_VARIANTS)]
    best_new_rmse = new_summary.sort_values(
        ["condition_rmse", "condition_abs_zrdc_residual"]
    ).iloc[0]
    best_new_zrdc = new_summary.sort_values(
        ["condition_abs_zrdc_residual", "condition_rmse"]
    ).iloc[0]
    frontier = parameter_grid_frontier(summary)
    new_frontier = frontier[
        frontier["variant"].isin(NEW_ABLATION_VARIANTS)
        & frontier["grid_pareto_frontier"]
    ].copy()
    if new_frontier.empty:
        best_new_frontier = best_new_rmse
    else:
        best_new_frontier = new_frontier.sort_values(
            ["condition_rmse", "condition_abs_zrdc_residual", "mechanism_count"]
        ).iloc[0]

    planned_rmse_gain_vs_greedy = float(best_greedy["condition_rmse"] - best_planned["condition_rmse"])
    new_rmse_gain_vs_planned = float(best_planned["condition_rmse"] - best_new_rmse["condition_rmse"])
    new_zrdc_gain_vs_planned = float(
        best_planned["condition_abs_zrdc_residual"]
        - best_new_zrdc["condition_abs_zrdc_residual"]
    )
    return pd.DataFrame(
        [
            {
                "best_greedy_variant": best_greedy["variant"],
                "best_greedy_grid_id": best_greedy["grid_id"],
                "best_planned_variant": best_planned["variant"],
                "best_planned_grid_id": best_planned["grid_id"],
                "best_new_rmse_variant": best_new_rmse["variant"],
                "best_new_rmse_grid_id": best_new_rmse["grid_id"],
                "best_new_zrdc_variant": best_new_zrdc["variant"],
                "best_new_zrdc_grid_id": best_new_zrdc["grid_id"],
                "best_new_frontier_variant": best_new_frontier["variant"],
                "best_new_frontier_grid_id": best_new_frontier["grid_id"],
                "best_greedy_condition_rmse": best_greedy["condition_rmse"],
                "best_planned_condition_rmse": best_planned["condition_rmse"],
                "best_new_condition_rmse": best_new_rmse["condition_rmse"],
                "planned_condition_rmse_gain_vs_greedy": planned_rmse_gain_vs_greedy,
                "new_condition_rmse_gain_vs_planned": new_rmse_gain_vs_planned,
                "best_planned_condition_zrdc_abs_residual": best_planned["condition_abs_zrdc_residual"],
                "best_new_condition_zrdc_abs_residual": best_new_zrdc["condition_abs_zrdc_residual"],
                "new_condition_zrdc_abs_residual_gain_vs_planned": new_zrdc_gain_vs_planned,
                "new_variant_on_grid_frontier": bool(not new_frontier.empty),
                "rmse_gate": args.rmse_gate,
                "zrdc_gate": args.zrdc_gate,
                "run_new_ablation_pilot": bool(
                    new_rmse_gain_vs_planned >= args.rmse_gate
                    or new_zrdc_gain_vs_planned >= args.zrdc_gate
                    or not new_frontier.empty
                ),
            }
        ]
    )


def build_predictions(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    variants = list(VARIANT_COMPLEXITY)
    for obs_idx, row in df.reset_index(drop=True).iterrows():
        states = build_states(row, "target_match")
        listener = listener_values(
            states,
            size_context_mode=args.size_context_mode,
            color_sem=args.color_sem,
            k=args.k,
            wf=args.wf,
        )
        for variant in variants:
            pred = predict_variant(variant, listener, args)
            rows.append(
                {
                    "obs_idx": obs_idx,
                    "id": row["id"],
                    "item": row["item"],
                    "conditions": row["conditions"],
                    "list": row["list"],
                    "trials": row["trials"],
                    "relevant_property": row["relevant_property"],
                    "sharpness": row["sharpness"],
                    "human_slider": row["human_slider"],
                    "variant": variant,
                    **pred,
                    "signed_residual": pred["pred_slider"] - row["human_slider"],
                    "abs_residual": abs(pred["pred_slider"] - row["human_slider"]),
                }
            )
    return pd.DataFrame(rows)


def condition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        predictions.groupby(["variant"] + GROUP_COLS, as_index=False)
        .agg(
            human_mean=("human_slider", "mean"),
            pred_mean=("pred_slider", "mean"),
            pred_min=("pred_slider", "min"),
            pred_max=("pred_slider", "max"),
            signed_residual_mean=("signed_residual", "mean"),
            abs_residual_mean=("abs_residual", "mean"),
            first_step_size_mean=("first_step_size_prob", "mean"),
            continue_after_size_mean=("continue_after_size", "mean"),
            continue_after_colour_mean=("continue_after_colour", "mean"),
            effective_order_bias_mean=("effective_order_bias", "mean"),
            n=("pred_slider", "size"),
        )
    )
    rmse = (
        predictions.groupby(["variant"] + GROUP_COLS)["signed_residual"]
        .apply(lambda x: float(np.sqrt(np.mean(np.square(x)))))
        .rename("rmse")
        .reset_index()
    )
    return grouped.merge(rmse, on=["variant"] + GROUP_COLS, how="left")


def model_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    cond = condition_summary(predictions)
    rows = []
    for variant, sub in predictions.groupby("variant"):
        residual = sub["signed_residual"].to_numpy(dtype=float)
        human = sub["human_slider"].to_numpy(dtype=float)
        pred = sub["pred_slider"].to_numpy(dtype=float)
        corr = float(np.corrcoef(human, pred)[0, 1])
        cond_sub = cond[cond["variant"].eq(variant)]
        cond_residual = cond_sub["signed_residual_mean"].to_numpy(dtype=float)
        rows.append(
            {
                "variant": variant,
                "observation_rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "observation_mae": float(np.mean(np.abs(residual))),
                "observation_correlation": corr,
                "observation_r2": corr * corr,
                "condition_rmse": float(np.sqrt(np.mean(np.square(cond_residual)))),
                "condition_mae": float(np.mean(np.abs(cond_residual))),
                "condition_abs_zrdc_residual": float(
                    cond_sub[cond_sub["conditions"].eq("zrdc")]["signed_residual_mean"].abs().mean()
                ),
                "condition_abs_erdc_residual": float(
                    cond_sub[cond_sub["conditions"].eq("erdc")]["signed_residual_mean"].abs().mean()
                ),
                "condition_abs_brdc_residual": float(
                    cond_sub[cond_sub["conditions"].eq("brdc")]["signed_residual_mean"].abs().mean()
                ),
                "observation_abs_zrdc_residual": float(
                    sub[sub["conditions"].eq("zrdc")]["abs_residual"].mean()
                ),
                "observation_abs_erdc_residual": float(
                    sub[sub["conditions"].eq("erdc")]["abs_residual"].mean()
                ),
                "observation_abs_brdc_residual": float(
                    sub[sub["conditions"].eq("brdc")]["abs_residual"].mean()
                ),
                **VARIANT_COMPLEXITY[variant],
                "elpd_loo": np.nan,
                "elpd_loo_se": np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["condition_rmse", "mechanism_count"]).reset_index(drop=True)


def residual_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cond = condition_summary(predictions)
    worst = cond.reindex(cond["signed_residual_mean"].abs().sort_values(ascending=False).index)
    return cond, worst.head(30).reset_index(drop=True)


def pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, candidate in summary.iterrows():
        dominated = False
        for _, other in summary.iterrows():
            if other["variant"] == candidate["variant"]:
                continue
            no_worse = (
                other["condition_rmse"] <= candidate["condition_rmse"]
                and other["condition_abs_zrdc_residual"] <= candidate["condition_abs_zrdc_residual"]
                and other["mechanism_count"] <= candidate["mechanism_count"]
            )
            strictly_better = (
                other["condition_rmse"] < candidate["condition_rmse"]
                or other["condition_abs_zrdc_residual"] < candidate["condition_abs_zrdc_residual"]
                or other["mechanism_count"] < candidate["mechanism_count"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        row = candidate.to_dict()
        row["forward_pareto_frontier"] = not dominated
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["forward_pareto_frontier", "condition_rmse"],
        ascending=[False, True],
    )


def gate_decision(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    baseline = summary[summary["variant"].eq("greedy_incremental_stop")].iloc[0]
    best = summary.sort_values("condition_rmse").iloc[0]
    zrdc_best = summary.sort_values("condition_abs_zrdc_residual").iloc[0]

    rmse_gain = float(baseline["condition_rmse"] - best["condition_rmse"])
    zrdc_gain = float(baseline["condition_abs_zrdc_residual"] - zrdc_best["condition_abs_zrdc_residual"])
    return pd.DataFrame(
        [
            {
                "baseline_variant": baseline["variant"],
                "best_rmse_variant": best["variant"],
                "best_zrdc_variant": zrdc_best["variant"],
                "baseline_condition_rmse": baseline["condition_rmse"],
                "best_condition_rmse": best["condition_rmse"],
                "condition_rmse_gain": rmse_gain,
                "baseline_condition_zrdc_abs_residual": baseline["condition_abs_zrdc_residual"],
                "best_condition_zrdc_abs_residual": zrdc_best["condition_abs_zrdc_residual"],
                "condition_zrdc_abs_residual_gain": zrdc_gain,
                "rmse_gate": args.rmse_gate,
                "zrdc_gate": args.zrdc_gate,
                "run_pilot_inference": bool(rmse_gain >= args.rmse_gate or zrdc_gain >= args.zrdc_gate),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--order-bias", type=float, default=2.0)
    parser.add_argument("--usefulness-order-scale", type=float, default=8.0)
    parser.add_argument("--signed-order-scale", type=float, default=8.0)
    parser.add_argument("--planned-mixture-weight", type=float, default=0.5)
    parser.add_argument("--color-sem", type=float, default=0.8)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--wf", type=float, default=1.0)
    parser.add_argument(
        "--size-context-mode",
        choices=["static", "posterior", "comparison_class"],
        default="static",
    )
    parser.add_argument("--rmse-gate", type=float, default=0.01)
    parser.add_argument("--zrdc-gate", type=float, default=0.02)
    parser.add_argument(
        "--run-parameter-grid",
        action="store_true",
        help="Export compact parameter-grid summaries for speaker ablation screening.",
    )
    parser.add_argument("--grid-alpha", type=str, default="1.0,1.5,2.0,3.0,4.0")
    parser.add_argument("--grid-order-bias", type=str, default="0.0,0.25,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--grid-usefulness-order-scale", type=str, default="0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,8.0")
    parser.add_argument("--grid-signed-order-scale", type=str, default="0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,8.0")
    parser.add_argument("--grid-planned-mixture-weight", type=str, default="0.0,0.25,0.5,0.75,1.0")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_slider_dataset(args.dataset)

    predictions = build_predictions(df, args)
    cond_summary, worst_residuals = residual_tables(predictions)
    summary = model_summary(predictions)
    frontier = pareto_frontier(summary)
    gate = gate_decision(summary, args)

    predictions.to_csv(args.out_dir / "slider_speaker_variant_predictions.csv", index=False)
    cond_summary.to_csv(args.out_dir / "slider_speaker_variant_condition_summary.csv", index=False)
    worst_residuals.to_csv(args.out_dir / "slider_speaker_variant_worst_residuals.csv", index=False)
    summary.to_csv(args.out_dir / "slider_speaker_variant_model_summary.csv", index=False)
    frontier.to_csv(args.out_dir / "slider_speaker_variant_forward_frontier.csv", index=False)
    gate.to_csv(args.out_dir / "slider_speaker_variant_gate_decision.csv", index=False)

    if args.run_parameter_grid:
        grid_summary = parameter_grid_summary(df, args)
        grid_frontier = parameter_grid_frontier(grid_summary)
        grid_best = parameter_grid_best_by_variant(grid_summary)
        grid_gate = parameter_grid_gate_decision(grid_summary, args)

        grid_summary.to_csv(args.out_dir / "slider_speaker_variant_grid_summary.csv", index=False)
        grid_frontier.to_csv(args.out_dir / "slider_speaker_variant_grid_frontier.csv", index=False)
        grid_best.to_csv(args.out_dir / "slider_speaker_variant_grid_best_by_variant.csv", index=False)
        grid_gate.to_csv(args.out_dir / "slider_speaker_variant_grid_gate_decision.csv", index=False)

    print(f"Wrote slider speaker-variant forward audit CSVs to {args.out_dir}")
    print(gate.to_string(index=False))
    if args.run_parameter_grid:
        print(grid_gate.to_string(index=False))


if __name__ == "__main__":
    main()
