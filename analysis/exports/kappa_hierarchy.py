"""Localize the predictive gain from participant-specific successive-choice weights.

The analysis compares the participant-weight model with its shared-weight counterpart.
All summaries use existing pointwise PSIS-LOO values and participant posteriors;
no model is refit here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/production_model_input.csv"
SHARED = None
JOINT = None
PARAMETERS = ROOT / "paper/data/production_participant_parameter_intervals.csv"
PARTICIPANT_OUTPUT = ROOT / "paper/data/production_kappa_hierarchy_participant_gain.csv"
OUTCOME_OUTPUT = ROOT / "paper/data/production_kappa_hierarchy_outcome_gain.csv"
CORRELATION_OUTPUT = ROOT / "paper/data/production_kappa_hierarchy_gain_correlation.csv"

SHARED_KAPPA = 0.4476953309272129
SHARED_MODEL = "K-UPD-HO"
JOINT_MODEL = "K-UPD-HKO"
SEED = 20260813
N_DRAWS = 200_000
BATCH = 5_000


def weighted_correlation_draws(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    draws: list[np.ndarray] = []
    for start in range(0, N_DRAWS, BATCH):
        size = min(BATCH, N_DRAWS - start)
        weights = rng.exponential(size=(size, len(x)))
        weights /= weights.sum(axis=1, keepdims=True)
        mean_x = weights @ x
        mean_y = weights @ y
        covariance = weights @ (x * y) - mean_x * mean_y
        variance_x = weights @ (x * x) - mean_x * mean_x
        variance_y = weights @ (y * y) - mean_y * mean_y
        draws.append(covariance / np.sqrt(variance_x * variance_y))
    return np.concatenate(draws)


def bootstrap_group_totals(participant_cells: pd.DataFrame) -> pd.DataFrame:
    grouping = ["grouping", "response_length", "initial_adjective"]
    participant_ids = np.sort(participant_cells["id"].unique())
    matrix = (
        participant_cells.pivot_table(
            index="id",
            columns=grouping,
            values="elpd_gain",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(participant_ids, fill_value=0.0)
    )
    groups = pd.DataFrame(list(matrix.columns), columns=grouping)
    observed = matrix.sum(axis=0).to_numpy()
    rng = np.random.default_rng(SEED + 1)
    posterior_chunks: list[np.ndarray] = []
    for start in range(0, N_DRAWS, BATCH):
        size = min(BATCH, N_DRAWS - start)
        weights = rng.exponential(size=(size, len(participant_ids)))
        weights /= weights.sum(axis=1, keepdims=True)
        posterior_chunks.append(len(participant_ids) * weights @ matrix.to_numpy())
    posterior = np.vstack(posterior_chunks)
    output = groups.copy()
    output["estimate_elpd"] = observed
    output["posterior_mean"] = posterior.mean(axis=0)
    output["credible_lower_95"] = np.quantile(posterior, 0.025, axis=0)
    output["credible_upper_95"] = np.quantile(posterior, 0.975, axis=0)
    output["probability_positive"] = np.mean(posterior > 0, axis=0)
    output["participant_count"] = len(participant_ids)
    output["posterior_draws"] = N_DRAWS
    output["posterior_method"] = "participant-level Dirichlet(1,...,1) Bayesian bootstrap"
    return output


def main() -> None:
    observed = pd.read_csv(DATA)
    shared = pd.read_csv(SHARED)[["canonical_row_position", SHARED_MODEL]]
    joint = pd.read_csv(JOINT)[["canonical_row_position", JOINT_MODEL]]
    kappa = (
        pd.read_csv(PARAMETERS)
        .loc[lambda data: data["parameter"].eq("Successive-choice contribution")]
        .rename(columns={"participant_id": "id", "posterior_mean": "kappa_mean"})
        [["id", "kappa_mean", "posterior_q025", "posterior_q975"]]
    )

    rows = (
        observed[["canonical_row_position", "id", "annotation"]]
        .merge(shared, on="canonical_row_position", validate="one_to_one")
        .merge(joint, on="canonical_row_position", validate="one_to_one")
    )
    rows["elpd_gain"] = rows[JOINT_MODEL] - rows[SHARED_MODEL]
    rows["response_length"] = rows["annotation"].str.len()
    rows["initial_adjective"] = rows["annotation"].str[0]

    participants = (
        rows.groupby("id", as_index=False)["elpd_gain"].sum()
        .merge(kappa, on="id", validate="one_to_one")
    )
    participants["shared_kappa"] = SHARED_KAPPA
    participants["distance_from_shared_kappa"] = np.abs(
        participants["kappa_mean"] - SHARED_KAPPA
    )

    x = participants["distance_from_shared_kappa"].to_numpy()
    y = participants["elpd_gain"].to_numpy()
    correlation_draws = weighted_correlation_draws(x, y)
    correlation = pd.DataFrame([{
        "estimand": "Correlation between participant ELPD gain and distance from shared kappa",
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "posterior_mean": float(correlation_draws.mean()),
        "posterior_median": float(np.median(correlation_draws)),
        "credible_lower_95": float(np.quantile(correlation_draws, 0.025)),
        "credible_upper_95": float(np.quantile(correlation_draws, 0.975)),
        "probability_positive": float(np.mean(correlation_draws > 0)),
        "participant_count": len(participants),
        "posterior_draws": N_DRAWS,
        "posterior_method": "participant-level Dirichlet(1,...,1) Bayesian bootstrap",
    }])

    length_cells = (
        rows.groupby(["id", "response_length"], as_index=False)["elpd_gain"].sum()
        .assign(grouping="Response length", initial_adjective="All")
    )
    initial_cells = (
        rows.groupby(["id", "initial_adjective"], as_index=False)["elpd_gain"].sum()
        .assign(grouping="Initial adjective", response_length=0)
    )
    joint_cells = (
        rows.groupby(["id", "response_length", "initial_adjective"], as_index=False)["elpd_gain"].sum()
        .assign(grouping="Length by initial adjective")
    )
    outcome = bootstrap_group_totals(
        pd.concat([length_cells, initial_cells, joint_cells], ignore_index=True)
    )

    PARTICIPANT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    participants.to_csv(PARTICIPANT_OUTPUT, index=False)
    outcome.to_csv(OUTCOME_OUTPUT, index=False)
    correlation.to_csv(CORRELATION_OUTPUT, index=False)
    print(correlation.to_string(index=False))
    print(outcome.loc[outcome["grouping"].eq("Response length")].to_string(index=False))


if __name__ == "__main__":
    raise SystemExit("Use analysis/reconcile_primary_exports.py with a completed run directory.")
