"""Forward audit for the target-match comparison-class semantic variant.

This script does not run inference.  It deterministically compares the current
canonical/posterior-updating semantics with the target-match comparison-class
variant over the existing size-colour production displays, then writes the
gate statistics used to decide whether server MCMC is warranted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import modelSpecification as ms  # noqa: E402
from helper import import_dataset  # noqa: E402


SIZE_COLOUR_CONDITIONS = ("erdc", "zrdc", "brdc")
DISPLAY_KEYS = ["item", "conditions", "list"]
GROUP_COLS = ["conditions", "relevant_property", "sharpness"]
LISTENER_UTTERANCES = ("D", "DC", "CD")
GATE_UTTERANCES = ("DC", "CD")


def utterance_indices(labels: tuple[str, ...]) -> dict[str, int]:
    return {label: ms.UTTERANCE_LABELS.index(label) for label in labels}


def comparison_class_fallback_for_label(states: np.ndarray, label: str) -> bool:
    """Return True if a size adjective sees an empty right comparison class."""
    class_mask = np.ones(states.shape[0], dtype=bool)
    saw_size = False
    fallback = False
    for token in reversed(label):
        if token == "D":
            saw_size = True
            if not class_mask.any():
                fallback = True
        elif token == "C":
            class_mask &= states[:, 1] == 1
        elif token == "F":
            class_mask &= states[:, 2] == 1
    return bool(saw_size and fallback)


def listener_matrix(
    states: np.ndarray,
    utterances: jnp.ndarray,
    mode: str,
    args: argparse.Namespace,
) -> np.ndarray:
    states_jnp = jnp.asarray(states, dtype=jnp.float32)

    if mode == "static":
        fn = lambda s: ms.incremental_semantics_jax_frozen(
            s,
            color_sem=args.color_sem,
            form_sem=args.form_sem,
            k=args.k,
            wf=args.wf,
            utterances=utterances,
        )
    elif mode == "posterior":
        fn = lambda s: ms.incremental_semantics_jax(
            s,
            color_sem=args.color_sem,
            form_sem=args.form_sem,
            k=args.k,
            wf=args.wf,
            utterances=utterances,
        )
    elif mode == "comparison_class":
        fn = lambda s: ms.incremental_semantics_jax_comparison_class(
            s,
            color_sem=args.color_sem,
            form_sem=args.form_sem,
            k=args.k,
            wf=args.wf,
            utterances=utterances,
        )
    else:
        raise ValueError(f"Unknown listener mode: {mode}")

    return np.asarray(jax.vmap(fn)(states_jnp))


def build_listener_tables(
    canonical_data: dict,
    target_match_data: dict,
    unique_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx_by_label = utterance_indices(LISTENER_UTTERANCES)
    selected_utterances = jnp.asarray(
        [np.asarray(ms.utterance_list[idx_by_label[label]]) for label in LISTENER_UTTERANCES],
        dtype=jnp.int32,
    )

    df = canonical_data["df"].iloc[unique_idx].reset_index(drop=True)
    canonical_states = np.asarray(canonical_data["states_train"])[unique_idx]
    target_states = np.asarray(target_match_data["states_train"])[unique_idx]

    variants = [
        ("current_context_fixed", "current", "context_fixed", canonical_states, "static"),
        ("current_context_updating", "current", "context_updating", canonical_states, "posterior"),
        ("tmcc_context_fixed", "tmcc", "context_fixed", target_states, "static"),
        ("tmcc_context_updating", "tmcc", "context_updating", target_states, "comparison_class"),
    ]

    rows = []
    for variant, semantic_family, semantic_regime, states, mode in variants:
        listeners = listener_matrix(states, selected_utterances, mode, args)
        for display_i, row in df.iterrows():
            for utt_i, label in enumerate(LISTENER_UTTERANCES):
                rows.append(
                    {
                        "variant": variant,
                        "semantic_family": semantic_family,
                        "semantic_regime": semantic_regime,
                        "item": row["item"],
                        "conditions": row["conditions"],
                        "list": row["list"],
                        "relevant_property": row["relevant_property"],
                        "sharpness": row["sharpness"],
                        "utterance_label": label,
                        "target_listener_prob": float(listeners[display_i, utt_i, 0]),
                        "comparison_class_fallback": (
                            comparison_class_fallback_for_label(target_states[display_i], label)
                            if semantic_family == "tmcc" and semantic_regime == "context_updating"
                            else False
                        ),
                    }
                )

    listener_df = pd.DataFrame(rows)
    summary_rows = []
    for family in ("current", "tmcc"):
        sub = listener_df[listener_df["semantic_family"].eq(family)]
        wide = sub.pivot_table(
            index=DISPLAY_KEYS + ["utterance_label"],
            columns="semantic_regime",
            values="target_listener_prob",
        ).reset_index()
        wide["delta_updating_minus_fixed"] = (
            wide["context_updating"] - wide["context_fixed"]
        )
        for label, label_df in wide.groupby("utterance_label"):
            summary_rows.append(
                {
                    "semantic_family": family,
                    "utterance_label": label,
                    "n_displays": int(len(label_df)),
                    "mean_abs_delta_target_listener": float(
                        label_df["delta_updating_minus_fixed"].abs().mean()
                    ),
                    "mean_signed_delta_target_listener": float(
                        label_df["delta_updating_minus_fixed"].mean()
                    ),
                    "max_abs_delta_target_listener": float(
                        label_df["delta_updating_minus_fixed"].abs().max()
                    ),
                }
            )
    return listener_df, pd.DataFrame(summary_rows)


def speaker_probabilities(
    states: np.ndarray,
    data: dict,
    unique_idx: np.ndarray,
    recursive: bool,
    size_context_mode: str,
    args: argparse.Namespace,
) -> np.ndarray:
    n = len(unique_idx)
    alpha = jnp.full((n,), args.alpha, dtype=jnp.float32)
    return np.asarray(
        ms.jitted_speaker_principled_hier(
            jnp.asarray(states[unique_idx], dtype=jnp.float32),
            data["sufficient_dim"][unique_idx],
            data["has_one_word_solution"][unique_idx],
            data["sharpness_idx"][unique_idx],
            alpha,
            args.beta_order,
            args.lambda_salience,
            args.rho_salience_stop,
            args.gamma_uncertainty_len,
            args.color_sem,
            args.form_sem,
            args.k,
            args.wf,
            args.epsilon,
            ms.LOG_LM_ORDER_ONLY_15,
            ms.BASE_VISUAL_SALIENCE,
            recursive=recursive,
            size_context_mode=size_context_mode,
        )
    )


def build_speaker_tables(
    canonical_data: dict,
    target_match_data: dict,
    unique_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = canonical_data["df"].iloc[unique_idx].reset_index(drop=True)
    canonical_states = np.asarray(canonical_data["states_train"])
    target_states = np.asarray(target_match_data["states_train"])

    variants = [
        (
            "current_context_fixed",
            "current",
            "context_fixed",
            canonical_states,
            canonical_data,
            False,
            "posterior",
        ),
        (
            "current_context_updating",
            "current",
            "context_updating",
            canonical_states,
            canonical_data,
            True,
            "posterior",
        ),
        (
            "tmcc_context_fixed",
            "tmcc",
            "context_fixed",
            target_states,
            target_match_data,
            False,
            "comparison_class",
        ),
        (
            "tmcc_context_updating",
            "tmcc",
            "context_updating",
            target_states,
            target_match_data,
            True,
            "comparison_class",
        ),
    ]

    records = []
    for variant, family, regime, states, data, recursive, mode in variants:
        probs = speaker_probabilities(
            states,
            data,
            unique_idx,
            recursive=recursive,
            size_context_mode=mode,
            args=args,
        )
        tmp = df[GROUP_COLS].copy()
        for code, label in enumerate(ms.UTTERANCE_LABELS):
            tmp[label] = probs[:, code]
        long = tmp.melt(
            id_vars=GROUP_COLS,
            var_name="utterance_label",
            value_name="speaker_prob",
        )
        grouped = (
            long.groupby(GROUP_COLS + ["utterance_label"], as_index=False)["speaker_prob"]
            .mean()
        )
        grouped["variant"] = variant
        grouped["semantic_family"] = family
        grouped["semantic_regime"] = regime
        records.append(grouped)

    by_condition = pd.concat(records, ignore_index=True)
    summary_rows = []
    for family in ("current", "tmcc"):
        sub = by_condition[by_condition["semantic_family"].eq(family)]
        wide = sub.pivot_table(
            index=GROUP_COLS + ["utterance_label"],
            columns="semantic_regime",
            values="speaker_prob",
        ).reset_index()
        wide["delta_updating_minus_fixed"] = (
            wide["context_updating"] - wide["context_fixed"]
        )
        wide["abs_delta_updating_minus_fixed"] = wide["delta_updating_minus_fixed"].abs()
        wide["semantic_family"] = family
        summary_rows.append(wide)
    return by_condition, pd.concat(summary_rows, ignore_index=True)


def gate_decision(
    listener_summary: pd.DataFrame,
    speaker_delta_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    tmcc_dc_listener = listener_summary[
        listener_summary["semantic_family"].eq("tmcc")
        & listener_summary["utterance_label"].eq("DC")
    ]
    listener_delta = float(tmcc_dc_listener["mean_abs_delta_target_listener"].iloc[0])

    speaker_gate_rows = speaker_delta_summary[
        speaker_delta_summary["semantic_family"].eq("tmcc")
        & speaker_delta_summary["relevant_property"].eq("both")
        & speaker_delta_summary["utterance_label"].isin(GATE_UTTERANCES)
    ]
    max_speaker_delta = float(speaker_gate_rows["abs_delta_updating_minus_fixed"].max())

    listener_gate_pass = listener_delta >= args.listener_gate
    speaker_gate_pass = max_speaker_delta >= args.speaker_gate
    return pd.DataFrame(
        [
            {
                "listener_metric": "tmcc_DC_mean_abs_delta_target_listener",
                "listener_value": listener_delta,
                "listener_threshold": args.listener_gate,
                "listener_gate_pass": listener_gate_pass,
                "speaker_metric": "tmcc_both_DC_CD_max_abs_condition_delta",
                "speaker_value": max_speaker_delta,
                "speaker_threshold": args.speaker_gate,
                "speaker_gate_pass": speaker_gate_pass,
                "run_pilot_inference": bool(listener_gate_pass or speaker_gate_pass),
            }
        ]
    )


def write_params(args: argparse.Namespace, out_dir: Path, n_displays: int) -> None:
    params = {
        "n_unique_displays": n_displays,
        "conditions": list(SIZE_COLOUR_CONDITIONS),
        "display_keys": DISPLAY_KEYS,
        "listener_utterances": list(LISTENER_UTTERANCES),
        "gate_utterances": list(GATE_UTTERANCES),
        "listener_gate": args.listener_gate,
        "speaker_gate": args.speaker_gate,
        "alpha": args.alpha,
        "beta_order": args.beta_order,
        "lambda_salience": args.lambda_salience,
        "rho_salience_stop": args.rho_salience_stop,
        "gamma_uncertainty_len": args.gamma_uncertainty_len,
        "color_sem": args.color_sem,
        "form_sem": args.form_sem,
        "k": args.k,
        "wf": args.wf,
        "epsilon": args.epsilon,
    }
    pd.DataFrame([params]).to_csv(out_dir / "semantic_variant_audit_params.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="results_semantic_variant_forward_audit/stats",
        help="Directory for audit CSV outputs.",
    )
    parser.add_argument("--listener-gate", type=float, default=0.01)
    parser.add_argument("--speaker-gate", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--beta-order", type=float, default=1.0)
    parser.add_argument("--lambda-salience", type=float, default=0.8)
    parser.add_argument("--rho-salience-stop", type=float, default=0.75)
    parser.add_argument("--gamma-uncertainty-len", type=float, default=0.0)
    parser.add_argument("--color-sem", type=float, default=0.59)
    parser.add_argument("--form-sem", type=float, default=0.50)
    parser.add_argument("--k", type=float, default=0.50)
    parser.add_argument("--wf", type=float, default=0.6856)
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_data = import_dataset(state_encoding="canonical")
    target_match_data = import_dataset(state_encoding="target_match")

    df = canonical_data["df"].reset_index(drop=True)
    dc_df = df[df["conditions"].isin(SIZE_COLOUR_CONDITIONS)]
    unique_idx = (
        dc_df.drop_duplicates(DISPLAY_KEYS)
        .sort_values(DISPLAY_KEYS)
        .index.to_numpy(dtype=np.int64)
    )

    listener_df, listener_summary = build_listener_tables(
        canonical_data,
        target_match_data,
        unique_idx,
        args,
    )
    speaker_by_condition, speaker_delta_summary = build_speaker_tables(
        canonical_data,
        target_match_data,
        unique_idx,
        args,
    )
    gate_df = gate_decision(listener_summary, speaker_delta_summary, args)

    listener_df.to_csv(out_dir / "semantic_variant_listener_by_display.csv", index=False)
    listener_summary.to_csv(out_dir / "semantic_variant_listener_summary.csv", index=False)
    speaker_by_condition.to_csv(out_dir / "semantic_variant_speaker_by_condition.csv", index=False)
    speaker_delta_summary.to_csv(out_dir / "semantic_variant_speaker_delta_summary.csv", index=False)
    gate_df.to_csv(out_dir / "semantic_variant_gate_decision.csv", index=False)
    write_params(args, out_dir, n_displays=len(unique_idx))

    print(f"Wrote semantic forward audit CSVs to {out_dir}")
    print(gate_df.to_string(index=False))


if __name__ == "__main__":
    main()
