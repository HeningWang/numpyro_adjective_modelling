"""Posterior means + diagnostics for global params in a contextual_dc NC.

By default reads the canonical baseline (warmup4000). Pass a path or filename
as the first CLI argument to inspect any other .nc file in inference_data/.
"""
import sys
from pathlib import Path

import arviz as az

DEFAULT_NC = (
    Path(__file__).resolve().parent.parent
    / "inference_data"
    / "mcmc_results_contextual_speaker_hier_dc_warmup4000_samples2000_chains4.nc"
)

ALL_KEYS = [
    "alpha_D", "alpha_C", "alpha_F",
    "log_beta_lm", "beta_lm",
    "lambda_suff",
    "gamma_1", "gamma_2",
    "gamma_oneword_1", "gamma_oneword_2",
    "gamma_sharp_1", "gamma_sharp_2",
    "beta_nat", "beta_len",
    "epsilon", "tau",
]


def _resolve(arg: str | None) -> Path:
    if arg is None:
        return DEFAULT_NC
    p = Path(arg)
    if p.is_file():
        return p
    candidate = DEFAULT_NC.parent / arg
    if candidate.is_file():
        return candidate
    raise SystemExit(f"NC not found: tried {p!s} and {candidate!s}")


def main() -> None:
    nc = _resolve(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"NC: {nc}")
    idata = az.from_netcdf(str(nc))
    available = list(idata.posterior.data_vars.keys())
    keys = [k for k in ALL_KEYS if k in available]
    missing = [k for k in ALL_KEYS if k not in available]
    if missing:
        print(f"(skipped — not in posterior: {missing})")
    summary = az.summary(idata, var_names=keys, kind="all")
    cols = ["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]
    print(summary[cols].to_string())

    # Divergences (sample_stats may or may not include them)
    if "sample_stats" in idata.groups() and "diverging" in idata.sample_stats:
        n_div = int(idata.sample_stats["diverging"].sum().item())
        print(f"\nDivergences: {n_div}")


if __name__ == "__main__":
    main()
