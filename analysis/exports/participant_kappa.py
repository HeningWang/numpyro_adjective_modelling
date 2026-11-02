"""Export the participant-specific degree of planning from the best-predicting model.

Reads the K-HKO NetCDF artifact and writes each participant's posterior mean and
95% interval of kappa_i. No model is refit here.

Usage: python participant_kappa.py <K-HKO.nc> <output_csv>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def main(artifact: Path, output: Path) -> None:
    posterior = xr.open_dataset(artifact, group="posterior", engine="h5netcdf")
    kappa = posterior["kappa_by_participant"].stack(sample=("chain", "draw")).transpose("participants", "sample")
    values = kappa.values
    pd.DataFrame(dict(
        participant=kappa["participants"].values,
        posterior_mean=values.mean(axis=1),
        posterior_q025=np.quantile(values, .025, axis=1),
        posterior_q975=np.quantile(values, .975, axis=1),
    )).to_csv(output, index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
