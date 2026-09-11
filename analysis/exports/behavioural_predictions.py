"""Export recoded Figure 7 PPC intervals from the frozen plan-guided model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def participant_bootstrap(
    participant: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float, float]:
    aggregates = (
        pd.DataFrame(
            {
                "participant": participant,
                "numerator": numerator,
                "denominator": denominator,
            }
        )
        .groupby("participant", sort=False)[["numerator", "denominator"]]
        .sum()
        .to_numpy(dtype=float)
    )
    indices = rng.integers(0, len(aggregates), size=(draws, len(aggregates)))
    sampled = aggregates[indices].sum(axis=1)
    rates = np.divide(
        sampled[:, 0],
        sampled[:, 1],
        out=np.full(len(sampled), np.nan),
        where=sampled[:, 1] > 0,
    )
    rates = rates[np.isfinite(rates)]
    mean = numerator.sum() / denominator.sum()
    return mean, *np.quantile(rates, [0.025, 0.975])


def context_label(value: str) -> str:
    return {
        "first": "Size sufficient",
        "both": "Both necessary",
        "second": "Colour sufficient",
    }[value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    data = pd.read_csv(args.data).reset_index(drop=True)
    observed_ds = xr.open_dataset(args.inference, group="observed_data", engine="h5netcdf")
    predictive_ds = xr.open_dataset(
        args.inference, group="posterior_predictive", engine="h5netcdf"
    )
    observed_codes = np.asarray(observed_ds["obs"]).reshape(-1)
    predicted_codes = np.asarray(predictive_ds["obs"]).reshape(-1, len(data))
    if len(data) != len(observed_codes):
        raise ValueError("Observed data and model artifact have different row counts")

    response_key = (
        pd.DataFrame({"response": data["annotation"], "code": observed_codes})
        .drop_duplicates()
        .sort_values("code")
    )
    if len(response_key) != 15 or response_key["code"].nunique() != 15:
        raise ValueError("Response annotations do not map one-to-one to 15 codes")
    code_for = dict(zip(response_key["response"], response_key["code"]))
    responses = list(code_for)
    size_initial_codes = [code_for[u] for u in responses if u.startswith("D")]

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    group_columns = ["relevant_property", "sharpness"]
    size_colour = data[data["combination"] == "dimension_color"]
    for (relevant_property, sharpness), cell in size_colour.groupby(
        group_columns, sort=True
    ):
        idx = cell.index.to_numpy(dtype=int)
        context = context_label(relevant_property)
        discriminability = "Low" if sharpness == "blurred" else "High"

        observed_size_initial = np.isin(observed_codes[idx], size_initial_codes)
        observed_mean, observed_lower, observed_upper = participant_bootstrap(
            cell["id"].to_numpy(),
            observed_size_initial.astype(int),
            np.ones(len(idx), dtype=int),
            rng,
            args.bootstrap_draws,
        )
        predicted_rates = np.isin(
            predicted_codes[:, idx], size_initial_codes
        ).mean(axis=1)
        rows.extend(
            [
                {
                    "outcome": "Size-initial responses",
                    "context": context,
                    "discriminability": discriminability,
                    "source": "Observed",
                    "proportion": observed_mean,
                    "lower": observed_lower,
                    "upper": observed_upper,
                },
                {
                    "outcome": "Size-initial responses",
                    "context": context,
                    "discriminability": discriminability,
                    "source": "Plan-guided",
                    "proportion": predicted_rates.mean(),
                    "lower": np.quantile(predicted_rates, 0.025),
                    "upper": np.quantile(predicted_rates, 0.975),
                },
            ]
        )

        if relevant_property == "both":
            redundant_codes = [code_for[u] for u in responses if len(u) == 3]
        else:
            redundant_codes = [code_for[u] for u in responses if len(u) > 1]
        observed_redundant = np.isin(observed_codes[idx], redundant_codes)
        observed_mean, observed_lower, observed_upper = participant_bootstrap(
            cell["id"].to_numpy(),
            observed_redundant.astype(int),
            np.ones(len(idx), dtype=int),
            rng,
            args.bootstrap_draws,
        )
        predicted_rates = np.isin(predicted_codes[:, idx], redundant_codes).mean(axis=1)
        rows.extend(
            [
                {
                    "outcome": "Redundant adjective use",
                    "context": context,
                    "discriminability": discriminability,
                    "source": "Observed",
                    "proportion": observed_mean,
                    "lower": observed_lower,
                    "upper": observed_upper,
                },
                {
                    "outcome": "Redundant adjective use",
                    "context": context,
                    "discriminability": discriminability,
                    "source": "Plan-guided",
                    "proportion": predicted_rates.mean(),
                    "lower": np.quantile(predicted_rates, 0.025),
                    "upper": np.quantile(predicted_rates, 0.975),
                },
            ]
        )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} recoded PPC rows to {args.output}")


if __name__ == "__main__":
    main()
