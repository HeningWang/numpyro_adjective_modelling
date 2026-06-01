"""Plug-in sweep for fixed salience constants in the principled production model.

This script does not refit MCMC. It uses posterior mean parameters from an
existing principled_salience_stop fit, varies fixed constants/multipliers, and
scores expected condition-level predictions against empirical proportions.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import modelSpecification as ms
from helper import import_dataset_hier


GROUP_COLS = ["relevant_property", "sharpness"]
UTTERANCE_LABELS = ms.UTTERANCE_LABELS


def filter_condition_subset(data, condition_subset: str):
    if not condition_subset:
        return data

    subset_codes = tuple(c.strip() for c in condition_subset.split(",") if c.strip())
    df = data["df"]
    keep_mask = df["conditions"].isin(subset_codes).to_numpy()
    keep_idx = np.where(keep_mask)[0]
    keep_idx_jnp = jnp.asarray(keep_idx)
    if len(keep_idx) == 0:
        raise ValueError(f"No rows matched condition subset {subset_codes}")

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
    ):
        if key in data and data[key] is not None:
            data[key] = data[key][keep_idx_jnp]

    data["df"] = df.iloc[keep_idx].reset_index(drop=True)

    old_pid = np.asarray(data["participant_idx"])[keep_idx]
    unique_p = sorted(set(old_pid.tolist()))
    remap = {p: i for i, p in enumerate(unique_p)}
    data["participant_idx"] = jnp.asarray(
        np.array([remap[p] for p in old_pid], dtype=np.int32),
        dtype=jnp.int32,
    )
    data["n_participants"] = len(unique_p)
    return data


def posterior_mean_params(idata_path: Path):
    idata = az.from_netcdf(idata_path)
    posterior = idata.posterior

    def mean(name: str) -> float:
        return float(posterior[name].mean().values)

    delta = posterior["delta"].mean(dim=("chain", "draw")).values
    return {
        "alpha": mean("alpha"),
        "beta_order": float(np.exp(mean("log_beta_order"))),
        "lambda_salience": mean("lambda_salience"),
        "rho_salience_stop": mean("rho_salience_stop"),
        "epsilon": mean("epsilon"),
        "delta": np.asarray(delta),
    }


def empirical_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond_vals, sub in df.groupby(GROUP_COLS):
        labels = sub["annotation_seq_flat"].to_numpy()
        for code, label in enumerate(UTTERANCE_LABELS):
            rows.append(
                {
                    "relevant_property": cond_vals[0],
                    "sharpness": cond_vals[1],
                    "utterance_code": code,
                    "utterance_label": label,
                    "human_mean": float(np.mean(labels == code)),
                }
            )
    return pd.DataFrame(rows)


def expected_condition_table(probs: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[GROUP_COLS].reset_index(drop=True).copy()
    rows = []
    for cond_vals, idx in tmp.groupby(GROUP_COLS).groups.items():
        p = probs[np.asarray(list(idx)), :].mean(axis=0)
        for code, value in enumerate(p):
            rows.append(
                {
                    "relevant_property": cond_vals[0],
                    "sharpness": cond_vals[1],
                    "utterance_code": code,
                    "utterance_label": UTTERANCE_LABELS[code],
                    "model_mean": float(value),
                }
            )
    return pd.DataFrame(rows)


def score_predictions(model_df: pd.DataFrame, human_df: pd.DataFrame) -> dict[str, float]:
    merged = human_df.merge(
        model_df,
        on=GROUP_COLS + ["utterance_code", "utterance_label"],
        how="inner",
    )
    residual = merged["model_mean"] - merged["human_mean"]
    r = float(np.corrcoef(merged["human_mean"], merged["model_mean"])[0, 1])
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))

    c_rows = merged[merged["utterance_label"].eq("C")]
    second_c = c_rows[c_rows["relevant_property"].eq("second")]
    both_c = c_rows[c_rows["relevant_property"].eq("both")]
    first_c = c_rows[c_rows["relevant_property"].eq("first")]

    second_c_abs = float(np.mean(np.abs(second_c["model_mean"] - second_c["human_mean"])))
    both_c_abs = float(np.mean(np.abs(both_c["model_mean"] - both_c["human_mean"])))
    first_c_abs = float(np.mean(np.abs(first_c["model_mean"] - first_c["human_mean"])))
    second_c_signed = float(np.mean(second_c["model_mean"] - second_c["human_mean"]))
    both_c_signed = float(np.mean(both_c["model_mean"] - both_c["human_mean"]))

    return {
        "r": r,
        "r2": r * r,
        "mae": mae,
        "rmse": rmse,
        "second_c_abs": second_c_abs,
        "second_c_signed": second_c_signed,
        "both_c_abs": both_c_abs,
        "both_c_signed": both_c_signed,
        "first_c_abs": first_c_abs,
        "balanced_loss": rmse + second_c_abs + 0.5 * both_c_abs,
    }


def predict_probs(data, params, base, lambda_scale, rho_scale, beta_scale, color_sem):
    participant_idx = np.asarray(data["participant_idx"])
    alpha_per_trial = np.maximum(
        params["alpha"] + params["delta"][participant_idx],
        0.0,
    ).astype(np.float32)
    return np.asarray(
        ms.jitted_speaker_principled_hier(
            data["states_train"],
            data["sufficient_dim"],
            data["has_one_word_solution"],
            data["sharpness_idx"],
            jnp.asarray(alpha_per_trial),
            params["beta_order"] * beta_scale,
            params["lambda_salience"] * lambda_scale,
            params["rho_salience_stop"] * rho_scale,
            0.0,
            color_sem,
            0.50,
            0.50,
            0.6856,
            params["epsilon"],
            ms.LOG_LM_ORDER_ONLY_15,
            jnp.asarray(base, dtype=jnp.float32),
        )
    )


def evaluate(data, human_df, params, config):
    probs = predict_probs(
        data,
        params,
        base=np.array([config["base_d"], config["base_c"], config["base_f"]], dtype=np.float32),
        lambda_scale=config["lambda_scale"],
        rho_scale=config["rho_scale"],
        beta_scale=config["beta_scale"],
        color_sem=config["color_sem"],
    )
    model_df = expected_condition_table(probs, data["df"])
    return {**config, **score_predictions(model_df, human_df)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--idata",
        default="inference_data/mcmc_results_principled_salience_stop_speaker_hier_dc_warmup2000_samples1000_chains4.nc",
    )
    parser.add_argument("--condition-subset", default="erdc,zrdc,brdc")
    parser.add_argument("--out-dir", default="results_principled_salience_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = filter_condition_subset(import_dataset_hier(), args.condition_subset)
    human_df = empirical_table(data["df"])
    params = posterior_mean_params(Path(args.idata))

    baseline_config = {
        "base_d": 0.0,
        "base_c": 1.0,
        "base_f": 0.25,
        "lambda_scale": 1.0,
        "rho_scale": 1.0,
        "beta_scale": 1.0,
        "color_sem": 0.59,
        "stage": "baseline",
    }

    stage1_configs = []
    for base_d in [-0.3, 0.0, 0.3]:
        for base_c in [0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2]:
            for base_f in [0.0, 0.25, 0.5]:
                stage1_configs.append(
                    {
                        **baseline_config,
                        "base_d": base_d,
                        "base_c": base_c,
                        "base_f": base_f,
                        "stage": "base_grid",
                    }
                )

    rows = [evaluate(data, human_df, params, baseline_config)]
    rows.extend(evaluate(data, human_df, params, cfg) for cfg in stage1_configs)
    stage1_df = pd.DataFrame(rows).drop_duplicates(
        subset=["base_d", "base_c", "base_f", "lambda_scale", "rho_scale", "beta_scale", "color_sem"]
    )

    top_base = (
        stage1_df.sort_values(["balanced_loss", "rmse"])
        .head(8)[["base_d", "base_c", "base_f"]]
        .drop_duplicates()
        .to_dict("records")
    )
    baseline_row = stage1_df[stage1_df["stage"].eq("baseline")].iloc[0]
    promising_base = (
        stage1_df[
            (stage1_df["r"] >= 0.72)
            & (stage1_df["rmse"] <= baseline_row["rmse"])
            & (stage1_df["second_c_abs"] < baseline_row["second_c_abs"])
        ]
        .sort_values(["second_c_abs", "rmse"])
        .head(8)[["base_d", "base_c", "base_f"]]
        .drop_duplicates()
        .to_dict("records")
    )

    stage2_configs = []
    for base in top_base:
        for lambda_scale in [0.75, 1.0, 1.25]:
            for rho_scale in [0.75, 1.0, 1.25]:
                for beta_scale in [0.85, 1.0, 1.15]:
                    for color_sem in [0.59, 0.70, 0.80]:
                        stage2_configs.append(
                            {
                                **base,
                                "lambda_scale": lambda_scale,
                                "rho_scale": rho_scale,
                                "beta_scale": beta_scale,
                                "color_sem": color_sem,
                                "stage": "top_base_param_grid",
                            }
                        )
    for base in promising_base:
        for lambda_scale in [0.85, 1.0, 1.15]:
            for rho_scale in [0.85, 1.0, 1.15]:
                for beta_scale in [0.90, 1.0, 1.10]:
                    for color_sem in [0.59, 0.65, 0.70]:
                        stage2_configs.append(
                            {
                                **base,
                                "lambda_scale": lambda_scale,
                                "rho_scale": rho_scale,
                                "beta_scale": beta_scale,
                                "color_sem": color_sem,
                                "stage": "promising_param_grid",
                            }
                        )

    rows.extend(evaluate(data, human_df, params, cfg) for cfg in stage2_configs)
    sweep_df = pd.DataFrame(rows).drop_duplicates(
        subset=["base_d", "base_c", "base_f", "lambda_scale", "rho_scale", "beta_scale", "color_sem"]
    )
    sweep_df = sweep_df.sort_values(["balanced_loss", "rmse"]).reset_index(drop=True)

    sweep_df.to_csv(out_dir / "salience_sweep_all.csv", index=False)
    sweep_df.head(25).to_csv(out_dir / "salience_sweep_top_balanced.csv", index=False)
    sweep_df.sort_values(["second_c_abs", "both_c_abs", "rmse"]).head(25).to_csv(
        out_dir / "salience_sweep_top_second_c.csv",
        index=False,
    )
    sweep_df.sort_values(["rmse", "second_c_abs"]).head(25).to_csv(
        out_dir / "salience_sweep_top_rmse.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(
        sweep_df["both_c_abs"],
        sweep_df["second_c_abs"],
        c=sweep_df["rmse"],
        cmap="viridis",
        s=18,
        alpha=0.75,
    )
    baseline = sweep_df[
        (sweep_df["base_d"].eq(0.0))
        & (sweep_df["base_c"].eq(1.0))
        & (sweep_df["base_f"].eq(0.25))
        & (sweep_df["lambda_scale"].eq(1.0))
        & (sweep_df["rho_scale"].eq(1.0))
        & (sweep_df["beta_scale"].eq(1.0))
        & (sweep_df["color_sem"].eq(0.59))
    ].iloc[0]
    ax.scatter(
        [baseline["both_c_abs"]],
        [baseline["second_c_abs"]],
        marker="x",
        c="red",
        s=80,
        label="current fixed constants",
    )
    ax.set_xlabel("both-context |C residual|")
    ax.set_ylabel("second-context |C residual|")
    ax.set_title("Salience constant sweep tradeoff")
    ax.legend(frameon=False)
    fig.colorbar(ax.collections[0], ax=ax, label="overall RMSE")
    fig.tight_layout()
    fig.savefig(out_dir / "salience_sweep_tradeoff.png", dpi=220)

    print("Baseline:")
    print(baseline.to_string())
    print("\nTop balanced:")
    print(sweep_df.head(10).to_string(index=False))
    print(f"\nSaved sweep outputs to {out_dir}")


if __name__ == "__main__":
    main()
