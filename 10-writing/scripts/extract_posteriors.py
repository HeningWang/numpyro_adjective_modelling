"""Extract posterior summaries from NetCDF inference files for draft reporting.

Prints mean, SD, and 94% HDI for key parameters of all 2x2 models
(slider and production).
"""
import arviz as az
import numpy as np

import pathlib
_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SLIDER_DIR = str(_ROOT / "03-modelling-slider-data" / "inference_data")
PROD_DIR = str(_ROOT / "05-modelling-production-data" / "inference_data")

SLIDER_MODELS = {
    "incremental_recursive": f"{SLIDER_DIR}/mcmc_results_incremental_speaker_hier_warmup500_samples500_chains4.nc",
    "incremental_static": f"{SLIDER_DIR}/mcmc_results_incremental_static_speaker_hier_warmup500_samples500_chains4.nc",
    "global_recursive": f"{SLIDER_DIR}/mcmc_results_global_speaker_hier_warmup500_samples500_chains4.nc",
    "global_static": f"{SLIDER_DIR}/mcmc_results_global_static_speaker_hier_warmup500_samples500_chains4.nc",
}

PROD_MODELS = {
    "incremental_recursive": f"{PROD_DIR}/mcmc_results_2x2best_incremental_recursive_warmup5000_samples2000_chains4.nc",
    "incremental_static": f"{PROD_DIR}/mcmc_results_2x2best_incremental_static_warmup5000_samples2000_chains4.nc",
    "global_recursive": f"{PROD_DIR}/mcmc_results_2x2best_global_recursive_warmup5000_samples2000_chains4.nc",
    "global_static": f"{PROD_DIR}/mcmc_results_2x2best_global_static_warmup5000_samples2000_chains4.nc",
}

# Parameters to report for each dataset
SLIDER_PARAMS = ["alpha", "bias", "sigma", "tau"]
PROD_PARAMS = ["alpha", "log_beta", "tau"]


def summarise_idata(idata, params):
    """Return summary dict: {param: (mean, sd, hdi_low, hdi_high, rhat_max, ess_min)}."""
    results = {}
    posterior = idata.posterior
    for p in params:
        if p not in posterior:
            continue
        vals = posterior[p].values  # shape (chains, draws) or (chains, draws, ...)
        flat = vals.flatten()
        hdi = az.hdi(np.array(flat), hdi_prob=0.94)
        # Compute r-hat and ESS
        rhat = az.rhat(idata, var_names=[p])[p].values
        ess = az.ess(idata, var_names=[p])[p].values
        rhat_val = float(np.max(rhat)) if np.ndim(rhat) > 0 else float(rhat)
        ess_val = float(np.min(ess)) if np.ndim(ess) > 0 else float(ess)
        results[p] = {
            "mean": float(np.mean(flat)),
            "sd": float(np.std(flat)),
            "hdi_low": float(hdi[0]),
            "hdi_high": float(hdi[1]),
            "rhat_max": rhat_val,
            "ess_min": ess_val,
        }
    return results


def print_summary(dataset_name, models, params):
    print(f"\n{'='*70}")
    print(f"  {dataset_name}")
    print(f"{'='*70}")
    for model_name, path in models.items():
        print(f"\n--- {model_name} ---")
        try:
            idata = az.from_netcdf(path)
        except FileNotFoundError:
            print(f"  [FILE NOT FOUND: {path}]")
            continue

        # Global diagnostics
        all_rhat = az.rhat(idata)
        all_ess = az.ess(idata)
        rhat_vals = []
        ess_vals = []
        for var in all_rhat:
            v = all_rhat[var].values
            rhat_vals.append(float(np.max(v)))
        for var in all_ess:
            v = all_ess[var].values
            ess_vals.append(float(np.min(v)))
        print(f"  Global: max R-hat = {max(rhat_vals):.4f}, min ESS = {min(ess_vals):.0f}")

        summary = summarise_idata(idata, params)
        for p, s in summary.items():
            print(f"  {p:>12s}: mean={s['mean']:.3f}, SD={s['sd']:.3f}, "
                  f"94% HDI=[{s['hdi_low']:.3f}, {s['hdi_high']:.3f}], "
                  f"R-hat={s['rhat_max']:.4f}, ESS={s['ess_min']:.0f}")


if __name__ == "__main__":
    print_summary("SLIDER MODELS", SLIDER_MODELS, SLIDER_PARAMS)
    print_summary("PRODUCTION MODELS", PROD_MODELS, PROD_PARAMS)
