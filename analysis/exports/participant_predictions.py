"""Export joint participant predictions from the selected production posterior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/production_model_input.csv"
ARTIFACT = None
PARAMETER_OUTPUT = ROOT / "paper/data/production_participant_parameter_intervals.csv"
PREDICTION_OUTPUT = ROOT / "paper/data/production_participant_prediction_summary.csv"
GAP_OUTPUT = ROOT / "paper/data/production_participant_prediction_gaps.csv"
POPULATION_OUTPUT = ROOT / "paper/data/production_joint_model_population_parameters.csv"
DEPENDENCE_OUTPUT = ROOT / "paper/data/production_joint_model_parameter_dependence.csv"
UTTERANCES = np.array(["D", "DC", "DCF", "DF", "DFC", "C", "CD", "CDF", "CF", "CFD", "F", "FD", "FDC", "FC", "FCD"])
PARAMETERS = {
    "Pragmatic optimality": "alpha_by_participant",
    "Successive-choice contribution": "kappa_by_participant",
    "Stable-order weighting": "beta_order_by_participant",
}


def participant_values(dataset: xr.Dataset, variable: str) -> np.ndarray:
    posterior = dataset[variable]
    participant_dimension = next(dim for dim in posterior.dims if dim not in {"chain", "draw"})
    return posterior.stack(sample=("chain", "draw")).transpose(participant_dimension, "sample").values


def parameter_summary(dataset: xr.Dataset, participant_ids: np.ndarray) -> pd.DataFrame:
    rows = []
    for label, variable in PARAMETERS.items():
        values = participant_values(dataset, variable)
        if values.shape[0] != len(participant_ids):
            raise ValueError(f"{variable}: participant count mismatch")
        rows.append(pd.DataFrame({
            "parameter": label,
            "participant_id": participant_ids,
            "posterior_mean": values.mean(axis=1),
            "posterior_q025": np.quantile(values, 0.025, axis=1),
            "posterior_q975": np.quantile(values, 0.975, axis=1),
        }))
    return pd.concat(rows, ignore_index=True)


def prediction_summary(
    observed: pd.DataFrame,
    posterior_predictive: np.ndarray,
    observed_codes: np.ndarray,
    parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    conditions = ["combination", "relevant_property", "sharpness"]
    for parameter in PARAMETERS:
        estimates = parameters.loc[parameters["parameter"].eq(parameter)].set_index("participant_id")["posterior_mean"]
        lower, upper = estimates.quantile([0.15, 0.85])
        groups = pd.Series("Middle 70%", index=estimates.index)
        groups.loc[estimates <= lower] = "Lower 15%"
        groups.loc[estimates >= upper] = "Upper 15%"
        for group_label in ("Lower 15%", "Middle 70%", "Upper 15%"):
            ids = groups.index[groups.eq(group_label)]
            group_rows = observed["id"].isin(ids).to_numpy()
            for condition in observed.loc[group_rows, conditions].drop_duplicates().itertuples(index=False):
                mask = group_rows.copy()
                for column, value in zip(conditions, condition):
                    mask &= observed[column].to_numpy() == value
                indices = np.flatnonzero(mask)
                for code, category in enumerate(UTTERANCES):
                    draws = (posterior_predictive[:, indices] == code).mean(axis=1)
                    rows.append({
                        "parameter": parameter,
                        "parameter_group": group_label,
                        **dict(zip(conditions, condition)),
                        "category": category,
                        "predicted_mean": float(draws.mean()),
                        "predicted_q025": float(np.quantile(draws, 0.025)),
                        "predicted_q975": float(np.quantile(draws, 0.975)),
                        "observed_proportion": float((observed_codes[indices] == code).mean()),
                        "participants": len(ids),
                        "trials": len(indices),
                    })
    summary = pd.DataFrame(rows)
    tails = summary.loc[summary["parameter_group"].isin(["Lower 15%", "Upper 15%"])]
    index = ["parameter", *conditions, "category"]
    gaps = tails.pivot(index=index, columns="parameter_group", values="predicted_mean").reset_index().rename_axis(columns=None)
    gaps["upper_minus_lower"] = gaps["Upper 15%"] - gaps["Lower 15%"]
    gaps["absolute_gap"] = gaps["upper_minus_lower"].abs()
    empirical = tails.pivot(index=index, columns="parameter_group", values="observed_proportion").reset_index().rename_axis(columns=None)
    empirical["observed_upper_minus_lower"] = empirical["Upper 15%"] - empirical["Lower 15%"]
    gaps = gaps.merge(empirical[index + ["observed_upper_minus_lower"]], on=index, validate="one_to_one")
    gaps["tail_difference_residual"] = gaps["observed_upper_minus_lower"] - gaps["upper_minus_lower"]
    gaps["absolute_tail_difference_residual"] = gaps["tail_difference_residual"].abs()
    return summary, gaps


def main() -> None:
    observed = pd.read_csv(DATA).reset_index(drop=True)
    participant_ids = np.sort(observed["id"].unique())
    posterior = xr.open_dataset(ARTIFACT, group="posterior", engine="h5netcdf")
    predictive = np.asarray(
        xr.open_dataset(ARTIFACT, group="posterior_predictive", engine="h5netcdf")["obs"]
    ).reshape(-1, len(observed))
    observed_codes = np.asarray(
        xr.open_dataset(ARTIFACT, group="observed_data", engine="h5netcdf")["obs"]
    ).reshape(-1)
    parameters = parameter_summary(posterior, participant_ids)
    predictions, gaps = prediction_summary(observed, predictive, observed_codes, parameters)

    population_rows = []
    for parameter, variable in (
        ("Pragmatic optimality location", "alpha"),
        ("Successive-choice population location", "kappa_mean"),
        ("Stable-order log-weight location", "log_beta_order"),
        ("Pragmatic optimality participant SD", "tau_log_alpha"),
        ("Successive-choice participant SD", "tau_kappa"),
        ("Stable-order participant SD", "tau_order"),
    ):
        values = np.asarray(posterior[variable]).reshape(-1)
        population_rows.append({
            "parameter": parameter,
            "posterior_mean": float(values.mean()),
            "posterior_median": float(np.median(values)),
            "posterior_q025": float(np.quantile(values, 0.025)),
            "posterior_q975": float(np.quantile(values, 0.975)),
        })

    participant_draws = {
        label: participant_values(posterior, variable)
        for label, variable in PARAMETERS.items()
    }
    dependence_rows = []
    labels = list(participant_draws)
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1:]:
            left = participant_draws[left_label]
            right = participant_draws[right_label]
            draw_correlations = np.array([
                np.corrcoef(left[:, draw], right[:, draw])[0, 1]
                for draw in range(left.shape[1])
            ])
            dependence_rows.append({
                "parameter_1": left_label,
                "parameter_2": right_label,
                "posterior_mean_correlation": float(draw_correlations.mean()),
                "posterior_median_correlation": float(np.median(draw_correlations)),
                "posterior_q025": float(np.quantile(draw_correlations, 0.025)),
                "posterior_q975": float(np.quantile(draw_correlations, 0.975)),
            })

    PARAMETER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    parameters.to_csv(PARAMETER_OUTPUT, index=False)
    predictions.to_csv(PREDICTION_OUTPUT, index=False)
    gaps.to_csv(GAP_OUTPUT, index=False)
    pd.DataFrame(population_rows).to_csv(POPULATION_OUTPUT, index=False)
    pd.DataFrame(dependence_rows).to_csv(DEPENDENCE_OUTPUT, index=False)
    print(parameters.groupby("parameter").size())
    print(gaps.groupby("parameter")["absolute_gap"].max())


if __name__ == "__main__":
    raise SystemExit("Use analysis/reconcile_primary_exports.py with a completed run directory.")
