"""Forward screen for theoretically motivated slider speaker variants.

This script does not run inference. It uses target-relative semantics and
evaluates whether planned incremental speakers improve the PPC patterns that
the current greedy speaker misses before any server MCMC is launched.
"""

from __future__ import annotations

import argparse
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
    "global_full_utterance": {
        "mechanism_count": 2,
        "free_parameter_count": 2,
        "description": "full-utterance RSA comparator",
    },
}


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
    if variant == "global_full_utterance":
        out = global_prediction(listener, args.alpha, args.order_bias)
        out["effective_order_bias"] = args.order_bias
        return out
    raise ValueError(f"Unknown variant: {variant}")


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

    print(f"Wrote slider speaker-variant forward audit CSVs to {args.out_dir}")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
