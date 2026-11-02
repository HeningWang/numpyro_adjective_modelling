"""Summarise the population-level posteriors of the eight primary production fits.

Reads the primary NetCDF artifacts (one per model) and writes the posterior mean
and 95% interval of every scalar parameter. Participant-level parameters are
summarised elsewhere (participant_predictions.py). No model is refit here.

Usage: python posterior_parameters.py <artifact_dir> <output_csv>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

MODELS = ("G-HO", "I-HO", "K-HO", "G-UPD-HO", "I-UPD-HO", "K-UPD-HO", "K-HKO", "K-UPD-HKO")


def summarise(artifact: Path, model: str) -> list[dict]:
    posterior = xr.open_dataset(artifact, group="posterior", engine="h5netcdf")
    rows = []
    for name, values in posterior.data_vars.items():
        if set(values.dims) != {"chain", "draw"}:
            continue
        draws = values.values.ravel()
        if np.ptp(draws) == 0:
            continue  # fixed quantities (e.g. kappa at an endpoint, Laplace epsilon)
        rows.append(dict(
            model=model, parameter=name, posterior_mean=draws.mean(),
            posterior_q025=np.quantile(draws, .025), posterior_q975=np.quantile(draws, .975),
        ))
    return rows


def main(artifact_dir: Path, output: Path) -> None:
    rows = []
    for model in MODELS:
        artifact = artifact_dir / f"{model}.nc"
        if not artifact.exists():
            raise FileNotFoundError(artifact)
        rows.extend(summarise(artifact, model))
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
