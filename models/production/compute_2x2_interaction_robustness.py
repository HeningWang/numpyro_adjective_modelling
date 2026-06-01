"""Compute pointwise-LOO 2x2 interaction robustness summaries."""
import json
import math
import os

import arviz as az
import numpy as np
import pandas as pd


SETS = {
    "old_contextual_pcalpha": {
        "inc_rec": "inference_data/mcmc_results_contextual_pcalpha_canon_parsimony_2x2_inc_rec_speaker_hier_dc_warmup4000_samples2000_chains4.nc",
        "inc_static": "inference_data/mcmc_results_contextual_pcalpha_canon_parsimony_2x2_inc_static_speaker_hier_dc_warmup4000_samples2000_chains4.nc",
        "glob_rec": "inference_data/mcmc_results_contextual_pcalpha_canon_parsimony_2x2_glob_rec_speaker_hier_dc_warmup4000_samples2000_chains4.nc",
        "glob_static": "inference_data/mcmc_results_contextual_pcalpha_canon_parsimony_2x2_glob_static_speaker_hier_dc_warmup4000_samples2000_chains4.nc",
    },
    "new_principled_regularized": {
        "inc_rec": "inference_data/mcmc_results_principled_salience_stop_regularized_2x2_inc_rec_speaker_hier_dc_warmup4000_samples2000_chains4.nc",
        "inc_static": "inference_data/mcmc_results_principled_salience_stop_regularized_2x2_inc_static_speaker_hier_dc_warmup4000_samples2000_chains4.nc",
        "glob_rec": "inference_data/mcmc_results_principled_salience_stop_regularized_2x2_glob_rec_speaker_hier_dc_warmup4000_samples2000_chains4.nc",
        "glob_static": "inference_data/mcmc_results_principled_salience_stop_regularized_2x2_glob_static_speaker_hier_dc_warmup4000_samples2000_chains4.nc",
    },
    "new_principled_global_fixedeps": {
        "inc_rec": "inference_data/mcmc_results_principled_salience_stop_regularized_2x2_inc_rec_speaker_hier_dc_warmup4000_samples2000_chains4.nc",
        "inc_static": "inference_data/mcmc_results_principled_salience_stop_regularized_2x2_inc_static_speaker_hier_dc_warmup4000_samples2000_chains4.nc",
        "glob_rec": "inference_data/mcmc_results_principled_salience_stop_regularized_2x2_glob_rec_fixedeps_speaker_hier_dc_warmup4000_samples2000_chains4.nc",
        "glob_static": "inference_data/mcmc_results_principled_salience_stop_regularized_2x2_glob_static_fixedeps_speaker_hier_dc_warmup4000_samples2000_chains4.nc",
    },
}


def compute_loo(idata):
    loo = az.loo(idata, var_name="obs", pointwise=True)
    return loo, np.asarray(loo.loo_i.values, dtype=float)


def max_rhat(idata):
    candidates = (
        "alpha", "alpha_D", "alpha_C", "alpha_F", "log_beta_order",
        "log_beta", "lambda_salience", "rho_salience_stop", "epsilon", "tau",
    )
    var_names = []
    for name in candidates:
        if name not in idata.posterior:
            continue
        values = np.asarray(idata.posterior[name].values, dtype=float)
        if np.nanstd(values) <= 1e-12:
            continue
        var_names.append(name)
    if not var_names:
        return float("nan")
    return float(az.summary(idata, var_names=var_names)["r_hat"].max())


def main():
    rows = []
    for set_name, files in SETS.items():
        print(f"Loading {set_name}...")
        idatas = {key: az.from_netcdf(path) for key, path in files.items()}
        loos = {}
        pointwise = {}
        rhats = {}
        for key, idata in idatas.items():
            loos[key], pointwise[key] = compute_loo(idata)
            rhats[key] = max_rhat(idata)

        inc_gain = float(loos["inc_rec"].elpd_loo - loos["inc_static"].elpd_loo)
        glob_gain = float(loos["glob_rec"].elpd_loo - loos["glob_static"].elpd_loo)
        interaction = inc_gain - glob_gain
        pointwise_interaction = (
            pointwise["inc_rec"] - pointwise["inc_static"]
            - pointwise["glob_rec"] + pointwise["glob_static"]
        )
        interaction_se = float(
            math.sqrt(len(pointwise_interaction) * np.var(pointwise_interaction, ddof=1))
        )

        rows.append({
            "set": set_name,
            "inc_rec_elpd": float(loos["inc_rec"].elpd_loo),
            "inc_static_elpd": float(loos["inc_static"].elpd_loo),
            "glob_rec_elpd": float(loos["glob_rec"].elpd_loo),
            "glob_static_elpd": float(loos["glob_static"].elpd_loo),
            "inc_recursive_gain": inc_gain,
            "global_recursive_gain": glob_gain,
            "interaction_did": interaction,
            "interaction_se": interaction_se,
            "interaction_z": interaction / interaction_se,
            "warnings": ";".join(f"{key}:{bool(loos[key].warning)}" for key in files),
            "max_rhat": max(rhats.values()),
            "rhats": json.dumps(rhats, sort_keys=True),
            "bad_pareto_k_ge_0.7": json.dumps({
                key: int(np.sum(np.asarray(loos[key].pareto_k.values) >= 0.7))
                for key in files
            }, sort_keys=True),
        })

    outdir = "results_principled_2x2_regularized/robustness"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "loo_interaction_robustness.csv")
    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    print(df.to_string(index=False))
    print(f"saved {outpath}")


if __name__ == "__main__":
    main()
