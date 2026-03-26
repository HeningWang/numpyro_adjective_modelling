"""Run full 2x2 (speaker x semantics) with best representation settings.

csv=0.85, fsv=0.70, wf=1.0, k=0.5
All models: single alpha, log_beta, tau, delta (reported model structure)

Run on server with GPU:
    JAX_PLATFORMS='' XLA_FLAGS='' python run_2x2_best_repr.py
"""
import os, time, json
os.environ.setdefault("JAX_PLATFORMS", "")
os.environ.setdefault("XLA_FLAGS", "")

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az

from modelSpecification import (
    jitted_speaker_hier,
    jitted_speaker_frozen_hier,
    jitted_global_speaker_hier,
    jitted_global_speaker_static_hier,
)

print("Devices:", jax.devices())

CSV, FSV, WF, K = 0.85, 0.70, 1.0, 0.5
NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 5000, 2000, 4


def make_model(jitted_fn, per_dim_alpha=False):
    """Create reported-style model with given speaker function."""
    def model_fn(states=None, empirical=None,
                 participant_idx=None, n_participants=None):
        alpha    = numpyro.sample("alpha", dist.HalfNormal(5.0))
        log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
        beta     = jnp.exp(log_beta)
        tau      = numpyro.sample("tau", dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_pt = jnp.maximum(alpha + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            if per_dim_alpha:
                probs = jitted_fn(
                    states, alpha_pt, alpha_pt, alpha_pt,
                    CSV, FSV, K, WF, beta, 0.0, 0.0,
                )
            else:
                probs = jitted_fn(
                    states, alpha_pt, CSV, FSV, K, WF, beta, 0.0, 0.0,
                )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model_fn


MODELS = [
    ("incremental_recursive", jitted_speaker_hier, True),
    ("incremental_static", jitted_speaker_frozen_hier, True),
    ("global_recursive", jitted_global_speaker_hier, False),
    ("global_static", jitted_global_speaker_static_hier, False),
]

# Load 3-condition subset
import helper
helper.CONDITIONS_OF_INTEREST = ("erdc", "zrdc", "brdc")
from helper import import_dataset_hier
data = import_dataset_hier()
print("Data: {} participants, {} obs".format(
    data["n_participants"], len(data["states_train"])))

states = data["states_train"]
emp    = data["empirical_seq_flat"]
pidx   = data["participant_idx"]
npart  = data["n_participants"]

results = []
for name, jitted_fn, per_dim in MODELS:
    sep = "=" * 60
    print("\n" + sep)
    print("  Running: " + name)
    print(sep)

    model_fn = make_model(jitted_fn, per_dim_alpha=per_dim)
    kernel = NUTS(model_fn, target_accept_prob=0.85, max_tree_depth=7)
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method="vectorized",
                progress_bar=True)

    t0 = time.time()
    mcmc.run(random.PRNGKey(4711), states, emp, pidx, npart)
    elapsed = time.time() - t0
    print("  Done in {:.1f}s".format(elapsed))

    mcmc.print_summary(exclude_deterministic=False)

    # Save idata with posterior predictive
    predictive = Predictive(model_fn, mcmc.get_samples())
    pred = predictive(random.PRNGKey(42), states, None, pidx, npart)
    idata = az.from_numpyro(
        mcmc,
        posterior_predictive={"obs": np.array(pred["obs"])},
    )

    out_nc = ("./inference_data/mcmc_results_2x2best_{}"
              "_warmup{}_samples{}_chains{}.nc".format(
                  name, NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS))
    idata.to_netcdf(out_nc)
    print("  Saved: " + out_nc)

    # LOO
    loo = az.loo(idata, pointwise=True)

    post = mcmc.get_samples()
    r = {
        "model": name,
        "elpd_loo": float(loo.elpd_loo),
        "p_loo": float(loo.p_loo),
        "alpha_mean": float(np.mean(np.array(post["alpha"]))),
        "log_beta_mean": float(np.mean(np.array(post["log_beta"]))),
        "tau_mean": float(np.mean(np.array(post["tau"]))),
        "elapsed_s": elapsed,
    }
    results.append(r)
    print("  ELPD={:.1f}, p_loo={:.1f}, alpha={:.3f}, log_beta={:.3f}".format(
        r["elpd_loo"], r["p_loo"], r["alpha_mean"], r["log_beta_mean"]))

# Summary
sep = "=" * 80
print("\n" + sep)
print("2x2 RESULTS (best repr: csv=0.85, fsv=0.70, wf=1.0)")
print(sep)
results.sort(key=lambda x: -x["elpd_loo"])
for r in results:
    print("  {:>25s}: ELPD={:9.1f}  p_loo={:6.1f}  "
          "alpha={:.3f}  log_beta={:.3f}  tau={:.3f}  ({:.0f}s)".format(
              r["model"], r["elpd_loo"], r["p_loo"],
              r["alpha_mean"], r["log_beta_mean"], r["tau_mean"],
              r["elapsed_s"]))

with open("2x2_best_repr_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: 2x2_best_repr_results.json")
