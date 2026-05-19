"""az.compare on the best-model 2x2 cells -> paper LOO-comparison CSV.

Produces 10-writing/data/production_loo_comparison_best.csv in the same
schema as production_loo_comparison.csv (az.compare index = cell name,
renamed to the canonical model keys the R plotting script expects), so the
ELPD plot can be drawn in the shared CSP ggplot style.

Run from 05-modelling-production-data/:
    python scripts/loo_compare_2x2_best.py
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import pathlib

import arviz as az

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
INFERENCE_DIR = REPO_ROOT / "05-modelling-production-data" / "inference_data"
OUTPUT_DIR = REPO_ROOT / "10-writing" / "data"

# cell -> (NC file, canonical model key used by the R model_labels map)
CELLS = {
    "inc_rec": (
        "mcmc_results_contextual_pcalpha_canon_parsimony_2x2_inc_rec_"
        "speaker_hier_dc_warmup4000_samples2000_chains4.nc",
        "incremental_recursive",
    ),
    "inc_static": (
        "mcmc_results_contextual_pcalpha_canon_parsimony_2x2_inc_static_"
        "speaker_hier_dc_warmup4000_samples2000_chains4.nc",
        "incremental_static",
    ),
    "glob_rec": (
        "mcmc_results_contextual_pcalpha_canon_parsimony_2x2_glob_rec_"
        "speaker_hier_dc_warmup4000_samples2000_chains4.nc",
        "global_recursive",
    ),
    "glob_static": (
        "mcmc_results_contextual_pcalpha_canon_parsimony_2x2_glob_static_"
        "speaker_hier_dc_warmup4000_samples2000_chains4.nc",
        "global_static",
    ),
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    idata_dict = {}
    for _, (fname, key) in CELLS.items():
        path = INFERENCE_DIR / fname
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"  [load] {key}")
        idata_dict[key] = az.from_netcdf(str(path))

    print("Computing az.compare (LOO)...")
    comparison = az.compare(idata_dict, ic="loo")
    out = OUTPUT_DIR / "production_loo_comparison_best.csv"
    comparison.to_csv(str(out))
    print(f"  -> {out}")
    print(comparison[["rank", "elpd_loo", "p_loo", "elpd_diff", "dse"]])


if __name__ == "__main__":
    main()
