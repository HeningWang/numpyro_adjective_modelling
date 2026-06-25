"""Forward audit for semantic variants on the slider dataset.

This script does not run inference. It compares the current canonical colour
encoding and posterior-updating size semantics against target-match encoding
and comparison-class size semantics on the existing dimension-colour slider
data, then exports CSV summaries used to decide whether new slider inference is
warranted.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "01-slider-data-preprocessed.csv"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results_semantic_variant_forward_audit" / "stats"

OBJECT_LABELS = tuple("ABCDEF")
DISPLAY_KEYS = ["item", "conditions", "list", "trials"]
GROUP_COLS = ["conditions", "relevant_property", "sharpness"]
UTTERANCE_LABELS = ("D", "C", "DC", "CD")
ARCHITECTURES = ("incremental", "global")

VARIANTS = (
    {
        "variant": "current_context_fixed",
        "semantic_family": "current",
        "semantic_regime": "context_fixed",
        "state_encoding": "canonical",
        "size_context_mode": "static",
    },
    {
        "variant": "current_context_updating",
        "semantic_family": "current",
        "semantic_regime": "context_updating",
        "state_encoding": "canonical",
        "size_context_mode": "posterior",
    },
    {
        "variant": "target_match_context_fixed",
        "semantic_family": "target_match",
        "semantic_regime": "context_fixed",
        "state_encoding": "target_match",
        "size_context_mode": "static",
    },
    {
        "variant": "target_match_context_updating",
        "semantic_family": "target_match",
        "semantic_regime": "context_updating",
        "state_encoding": "target_match",
        "size_context_mode": "posterior",
    },
    {
        "variant": "tmcc_context_updating",
        "semantic_family": "tmcc",
        "semantic_regime": "context_updating",
        "state_encoding": "target_match",
        "size_context_mode": "comparison_class",
    },
)


def normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF using only the Python stdlib and NumPy."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def normalize(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0.0 or not np.isfinite(total):
        if fallback is None:
            return np.ones_like(values, dtype=float) / len(values)
        return normalize(fallback)
    return values / total


def build_states(row: pd.Series, state_encoding: str) -> np.ndarray:
    sizes = row[[f"size_{label}" for label in OBJECT_LABELS]].to_numpy(dtype=float)
    colors_raw = row[[f"color_{label}" for label in OBJECT_LABELS]].astype(str).to_numpy()
    forms_raw = row[[f"form_{label}" for label in OBJECT_LABELS]].astype(str).to_numpy()

    if state_encoding == "canonical":
        colors = (colors_raw == "blue").astype(float)
        forms = (forms_raw == "circle").astype(float)
    elif state_encoding == "target_match":
        colors = (colors_raw == str(row["color_A"])).astype(float)
        forms = (forms_raw == str(row["form_A"])).astype(float)
    else:
        raise ValueError(f"Unknown state_encoding: {state_encoding}")

    return np.stack([sizes, colors, forms], axis=1)


def midrange_threshold(
    states: np.ndarray,
    state_prior: np.ndarray,
    k: float,
    q_low: float = 0.2,
    q_high: float = 0.8,
) -> float:
    sizes = states[:, 0]
    prior = normalize(np.asarray(state_prior, dtype=float))
    sort_idx = np.argsort(sizes)
    sizes_sorted = sizes[sort_idx]
    cdf = np.cumsum(prior[sort_idx])
    idx_low = min(int(np.searchsorted(cdf, q_low, side="left")), len(sizes_sorted) - 1)
    idx_high = min(int(np.searchsorted(cdf, q_high, side="left")), len(sizes_sorted) - 1)
    x_min_mid = sizes_sorted[idx_low]
    x_max_mid = sizes_sorted[idx_high]
    return float(x_max_mid - k * (x_max_mid - x_min_mid))


def size_meaning(states: np.ndarray, state_prior: np.ndarray, k: float, wf: float) -> np.ndarray:
    theta = midrange_threshold(states, state_prior, k)
    sizes = states[:, 0]
    denom = wf * np.sqrt(np.square(sizes) + theta**2 + 1e-8)
    return normal_cdf((sizes - theta) / denom)


def color_meaning(states: np.ndarray, color_sem: float) -> np.ndarray:
    colors = states[:, 1]
    return np.where(colors == 1.0, color_sem, 1.0 - color_sem)


def comparison_class_prior(states: np.ndarray, class_mask: np.ndarray) -> tuple[np.ndarray, bool]:
    mask = class_mask.astype(float)
    if float(mask.sum()) <= 0.0:
        return np.ones(states.shape[0], dtype=float) / states.shape[0], True
    return mask / mask.sum(), False


def interpret_utterance(
    states: np.ndarray,
    utterance: str,
    size_context_mode: str,
    color_sem: float,
    k: float,
    wf: float,
) -> tuple[np.ndarray, bool]:
    n_obj = states.shape[0]
    uniform = np.ones(n_obj, dtype=float) / n_obj
    posterior = uniform.copy()
    class_mask = np.ones(n_obj, dtype=bool)
    used_fallback = False

    for token in reversed(utterance):
        if token == "C":
            meaning = color_meaning(states, color_sem)
            posterior = normalize(posterior * meaning, fallback=uniform)
            class_mask = class_mask & (states[:, 1] == 1.0)
        elif token == "D":
            if size_context_mode == "static":
                size_prior = uniform
            elif size_context_mode == "posterior":
                size_prior = posterior
            elif size_context_mode == "comparison_class":
                size_prior, fallback = comparison_class_prior(states, class_mask)
                used_fallback = used_fallback or fallback
            else:
                raise ValueError(f"Unknown size_context_mode: {size_context_mode}")
            meaning = size_meaning(states, size_prior, k, wf)
            posterior = normalize(posterior * meaning, fallback=uniform)
        else:
            raise ValueError(f"Unknown token {token!r} in utterance {utterance!r}")

    return posterior, used_fallback


def listener_values(
    states: np.ndarray,
    size_context_mode: str,
    color_sem: float,
    k: float,
    wf: float,
) -> dict[str, dict[str, float | bool]]:
    out = {}
    for label in UTTERANCE_LABELS:
        posterior, fallback = interpret_utterance(
            states,
            label,
            size_context_mode=size_context_mode,
            color_sem=color_sem,
            k=k,
            wf=wf,
        )
        out[label] = {
            "target_listener_prob": float(posterior[0]),
            "comparison_class_fallback": bool(fallback),
        }
    return out


def incremental_prediction(
    listener: dict[str, dict[str, float | bool]],
    alpha: float,
    bias: float,
) -> dict[str, float]:
    eps = 1e-20
    l1_d = float(listener["D"]["target_listener_prob"])
    l1_c = float(listener["C"]["target_listener_prob"])
    l2_dc = float(listener["DC"]["target_listener_prob"])
    l2_cd = float(listener["CD"]["target_listener_prob"])

    num_d = np.power(np.clip(l1_d, eps, 1.0), alpha)
    num_c = np.power(np.clip(l1_c, eps, 1.0), alpha)
    first_size = num_d / np.clip(num_d + num_c, eps, None)
    first_colour = num_c / np.clip(num_d + num_c, eps, None)

    cont_dc = np.power(np.clip(l2_dc, eps, 1.0), alpha)
    stop_d = np.power(np.clip(l1_d, eps, 1.0), alpha)
    continue_after_size = cont_dc / np.clip(cont_dc + stop_d, eps, None)

    cont_cd = np.power(np.clip(l2_cd, eps, 1.0), alpha)
    stop_c = np.power(np.clip(l1_c, eps, 1.0), alpha)
    continue_after_colour = cont_cd / np.clip(cont_cd + stop_c, eps, None)

    chain_dc = first_size * continue_after_size
    chain_cd = first_colour * continue_after_colour
    logits = np.log(np.clip([chain_dc, chain_cd], eps, 1.0)) - np.array([0.0, bias])
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()

    return {
        "pred_slider": float(probs[0]),
        "first_step_size_prob": float(first_size),
        "continue_after_size": float(continue_after_size),
        "continue_after_colour": float(continue_after_colour),
        "chain_dc": float(chain_dc),
        "chain_cd": float(chain_cd),
    }


def global_prediction(
    listener: dict[str, dict[str, float | bool]],
    alpha: float,
    bias: float,
) -> dict[str, float]:
    eps = 1e-20
    logits = (
        np.log(
            np.clip(
                [
                    float(listener["DC"]["target_listener_prob"]),
                    float(listener["CD"]["target_listener_prob"]),
                ],
                eps,
                1.0,
            )
        )
        - np.array([0.0, bias])
    )
    logits = alpha * logits
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    return {
        "pred_slider": float(probs[0]),
        "first_step_size_prob": np.nan,
        "continue_after_size": np.nan,
        "continue_after_colour": np.nan,
        "chain_dc": np.nan,
        "chain_cd": np.nan,
    }


def load_slider_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["combination"].eq("dimension_color")].reset_index(drop=True)
    df["human_slider"] = np.clip(df["prefer_first_1st"].to_numpy(dtype=float), 0, 100) / 100
    return df


def build_listener_by_display(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    unique = (
        df.drop_duplicates(DISPLAY_KEYS)
        .sort_values(DISPLAY_KEYS)
        .reset_index(drop=True)
    )
    rows = []
    for _, row in unique.iterrows():
        for variant in VARIANTS:
            states = build_states(row, variant["state_encoding"])
            listener = listener_values(
                states,
                size_context_mode=variant["size_context_mode"],
                color_sem=args.color_sem,
                k=args.k,
                wf=args.wf,
            )
            for label, values in listener.items():
                rows.append(
                    {
                        **{key: row[key] for key in DISPLAY_KEYS + GROUP_COLS},
                        **variant,
                        "utterance_label": label,
                        **values,
                    }
                )
    return pd.DataFrame(rows)


def build_speaker_by_observation(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for obs_idx, row in df.reset_index(drop=True).iterrows():
        for variant in VARIANTS:
            states = build_states(row, variant["state_encoding"])
            listener = listener_values(
                states,
                size_context_mode=variant["size_context_mode"],
                color_sem=args.color_sem,
                k=args.k,
                wf=args.wf,
            )
            for architecture in ARCHITECTURES:
                if architecture == "incremental":
                    pred = incremental_prediction(listener, args.alpha, args.bias)
                else:
                    pred = global_prediction(listener, args.alpha, args.bias)
                rows.append(
                    {
                        "obs_idx": obs_idx,
                        "id": row["id"],
                        **{key: row[key] for key in DISPLAY_KEYS + GROUP_COLS},
                        "human_slider": row["human_slider"],
                        **variant,
                        "architecture": architecture,
                        **pred,
                        "signed_residual": pred["pred_slider"] - row["human_slider"],
                        "abs_residual": abs(pred["pred_slider"] - row["human_slider"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_listener_conditions(listener_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        listener_df.groupby(
            ["variant", "semantic_family", "semantic_regime", "state_encoding", "size_context_mode"]
            + GROUP_COLS
            + ["utterance_label"],
            as_index=False,
        )
        .agg(
            target_listener_mean=("target_listener_prob", "mean"),
            target_listener_min=("target_listener_prob", "min"),
            target_listener_max=("target_listener_prob", "max"),
            fallback_rate=("comparison_class_fallback", "mean"),
            n_displays=("target_listener_prob", "size"),
        )
    )

    one_word = summary[summary["utterance_label"].isin(["D", "C"])]
    wide = one_word.pivot_table(
        index=["variant"] + GROUP_COLS,
        columns="utterance_label",
        values="target_listener_mean",
    ).reset_index()
    wide["size_minus_colour_listener"] = wide["D"] - wide["C"]
    return summary.merge(
        wide[["variant"] + GROUP_COLS + ["size_minus_colour_listener"]],
        on=["variant"] + GROUP_COLS,
        how="left",
    )


def summarize_speaker_conditions(speaker_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        speaker_df.groupby(
            ["variant", "semantic_family", "semantic_regime", "state_encoding", "size_context_mode", "architecture"]
            + GROUP_COLS,
            as_index=False,
        )
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
            chain_dc_mean=("chain_dc", "mean"),
            chain_cd_mean=("chain_cd", "mean"),
            n=("pred_slider", "size"),
        )
    )
    agg["rmse"] = (
        speaker_df.groupby(
            ["variant", "semantic_family", "semantic_regime", "state_encoding", "size_context_mode", "architecture"]
            + GROUP_COLS
        )["signed_residual"]
        .apply(lambda x: float(np.sqrt(np.mean(np.square(x)))))
        .to_numpy()
    )
    return agg


def delta_rows(
    speaker_df: pd.DataFrame,
    comparison_name: str,
    left_variant: str,
    right_variant: str,
) -> pd.DataFrame:
    left = speaker_df[speaker_df["variant"].eq(left_variant)]
    right = speaker_df[speaker_df["variant"].eq(right_variant)]
    merged = left.merge(
        right[
            ["obs_idx", "architecture", "pred_slider"]
        ].rename(columns={"pred_slider": "right_pred_slider"}),
        on=["obs_idx", "architecture"],
        how="inner",
    )
    merged["delta_right_minus_left"] = merged["right_pred_slider"] - merged["pred_slider"]
    out = (
        merged.groupby(["architecture"] + GROUP_COLS, as_index=False)
        .agg(
            mean_signed_delta=("delta_right_minus_left", "mean"),
            mean_abs_delta=("delta_right_minus_left", lambda x: float(np.mean(np.abs(x)))),
            max_abs_delta=("delta_right_minus_left", lambda x: float(np.max(np.abs(x)))),
            n=("delta_right_minus_left", "size"),
        )
    )
    out["comparison"] = comparison_name
    out["left_variant"] = left_variant
    out["right_variant"] = right_variant
    return out[
        ["comparison", "left_variant", "right_variant", "architecture"]
        + GROUP_COLS
        + ["mean_signed_delta", "mean_abs_delta", "max_abs_delta", "n"]
    ]


def build_delta_summary(speaker_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("current_context_effect", "current_context_fixed", "current_context_updating"),
        ("target_match_encoding_effect", "current_context_fixed", "target_match_context_fixed"),
        ("target_match_posterior_context_effect", "target_match_context_fixed", "target_match_context_updating"),
        ("tmcc_context_effect", "target_match_context_fixed", "tmcc_context_updating"),
        ("full_tmcc_vs_current_updating", "current_context_updating", "tmcc_context_updating"),
    ]
    return pd.concat(
        [delta_rows(speaker_df, name, left, right) for name, left, right in comparisons],
        ignore_index=True,
    )


def gate_decision(
    listener_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    encoding_delta = delta_summary[delta_summary["comparison"].eq("target_match_encoding_effect")]
    tmcc_delta = delta_summary[delta_summary["comparison"].eq("tmcc_context_effect")]
    current_context_delta = delta_summary[delta_summary["comparison"].eq("current_context_effect")]

    encoding_value = float(encoding_delta["mean_abs_delta"].max())
    tmcc_context_value = float(tmcc_delta["mean_abs_delta"].max())
    current_context_value = float(current_context_delta["mean_abs_delta"].max())

    fixed = listener_summary[
        listener_summary["variant"].isin(["current_context_fixed", "target_match_context_fixed"])
        & listener_summary["utterance_label"].eq("D")
    ][["variant"] + GROUP_COLS + ["size_minus_colour_listener"]]
    wide = fixed.pivot_table(
        index=GROUP_COLS,
        columns="variant",
        values="size_minus_colour_listener",
    ).reset_index()
    wide["encoding_listener_contrast_delta"] = (
        wide["target_match_context_fixed"] - wide["current_context_fixed"]
    )
    listener_contrast_value = float(wide["encoding_listener_contrast_delta"].abs().mean())

    encoding_gate = encoding_value >= args.slider_prediction_gate
    context_gate = tmcc_context_value >= args.slider_prediction_gate
    listener_gate = listener_contrast_value >= args.listener_contrast_gate

    return pd.DataFrame(
        [
            {
                "current_context_mean_abs_delta_max": current_context_value,
                "target_match_encoding_mean_abs_delta_max": encoding_value,
                "tmcc_context_mean_abs_delta_max": tmcc_context_value,
                "listener_size_colour_contrast_delta": listener_contrast_value,
                "slider_prediction_threshold": args.slider_prediction_gate,
                "listener_contrast_threshold": args.listener_contrast_gate,
                "encoding_gate_pass": bool(encoding_gate),
                "context_gate_pass": bool(context_gate),
                "listener_gate_pass": bool(listener_gate),
                "run_slider_pilot_inference": bool(encoding_gate or context_gate or listener_gate),
            }
        ]
    )


def write_params(args: argparse.Namespace, out_dir: Path, n_observations: int, n_displays: int) -> None:
    pd.DataFrame(
        [
            {
                "n_observations": n_observations,
                "n_unique_displays": n_displays,
                "alpha": args.alpha,
                "bias": args.bias,
                "color_sem": args.color_sem,
                "k": args.k,
                "wf": args.wf,
                "slider_prediction_gate": args.slider_prediction_gate,
                "listener_contrast_gate": args.listener_contrast_gate,
                "variants": ",".join(v["variant"] for v in VARIANTS),
            }
        ]
    ).to_csv(out_dir / "slider_semantic_variant_audit_params.csv", index=False)


def assert_finite(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        values = df[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {column}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--bias", type=float, default=2.0)
    parser.add_argument("--color-sem", type=float, default=0.8)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--wf", type=float, default=1.0)
    parser.add_argument("--slider-prediction-gate", type=float, default=0.02)
    parser.add_argument("--listener-contrast-gate", type=float, default=0.05)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_slider_dataset(args.dataset)
    n_displays = len(df.drop_duplicates(DISPLAY_KEYS))

    listener_by_display = build_listener_by_display(df, args)
    listener_condition_summary = summarize_listener_conditions(listener_by_display)
    speaker_by_observation = build_speaker_by_observation(df, args)
    speaker_condition_summary = summarize_speaker_conditions(speaker_by_observation)
    delta_summary = build_delta_summary(speaker_by_observation)
    gate = gate_decision(listener_condition_summary, delta_summary, args)

    assert_finite(listener_by_display, ["target_listener_prob"])
    assert_finite(speaker_by_observation, ["pred_slider", "signed_residual", "abs_residual"])
    assert_finite(delta_summary, ["mean_signed_delta", "mean_abs_delta", "max_abs_delta"])

    listener_by_display.to_csv(args.out_dir / "slider_semantic_variant_listener_by_display.csv", index=False)
    listener_condition_summary.to_csv(args.out_dir / "slider_semantic_variant_listener_condition_summary.csv", index=False)
    speaker_by_observation.to_csv(args.out_dir / "slider_semantic_variant_speaker_by_observation.csv", index=False)
    speaker_condition_summary.to_csv(args.out_dir / "slider_semantic_variant_speaker_condition_summary.csv", index=False)
    delta_summary.to_csv(args.out_dir / "slider_semantic_variant_delta_summary.csv", index=False)
    gate.to_csv(args.out_dir / "slider_semantic_variant_gate_decision.csv", index=False)
    write_params(args, args.out_dir, n_observations=len(df), n_displays=n_displays)

    print(f"Wrote slider semantic forward audit CSVs to {args.out_dir}")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
