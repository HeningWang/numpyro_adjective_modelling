"""Grid search over representation parameters (color_semval, form_semval, wf).

Uses the extended v1 model structure (per-dim alpha + gamma + epsilon)
with different fixed semantic values. Short runs for quick comparison.

Run on server with GPU:
    JAX_PLATFORMS='' XLA_FLAGS='' python run_grid_search_representations.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "")
os.environ.setdefault("XLA_FLAGS", "")

import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, log_likelihood
import arviz as az

from helper import import_dataset_hier
from modelSpecification import (
    incremental_speaker,
    jitted_speaker_hier,
    n_utt,
)

print(f"Devices: {jax.devices()}")

# ── Grid ─────────────────────────────────────────────────────────────────────
GRID = [
    # label,               color_semval, form_semval, wf
    ("baseline",           0.971,        0.50,        1.0),
    ("sharp_size",         0.971,        0.50,        0.3),
    ("lower_color",        0.85,         0.50,        1.0),
    ("active_form",        0.971,        0.70,        1.0),
    ("sharp_size_low_col", 0.85,         0.50,        0.3),
    ("all_three",          0.85,         0.70,        0.3),
    ("moderate",           0.90,         0.65,        0.5),
]

# ── MCMC config ──────────────────────────────────────────────────────────────
NUM_WARMUP  = 1000
NUM_SAMPLES = 500
NUM_CHAINS  = 4
K_FIXED     = 0.5


def make_model(color_semval, form_semval, wf):
    """Factory: return a likelihood function with fixed semantics."""

    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None):
        k = K_FIXED

        log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
        beta     = jnp.exp(log_beta)

        alpha_D  = numpyro.sample("alpha_D", dist.HalfNormal(5.0))
        alpha_C  = numpyro.sample("alpha_C", dist.HalfNormal(5.0))
        alpha_F  = numpyro.sample("alpha_F", dist.HalfNormal(5.0))
        gamma    = numpyro.sample("gamma", dist.Normal(0.0, 1.0))
        epsilon  = numpyro.sample("epsilon", dist.Beta(1.0, 50.0))
        tau      = numpyro.sample("tau", dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_pt = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_pt = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_pt = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_hier(
                states, alpha_D_pt, alpha_C_pt, alpha_F_pt,
                color_semval, form_semval, k, wf, beta, gamma, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)

    return model


def run_one(label, color_semval, form_semval, wf, data):
    """Run inference for one grid point and return summary dict."""
    states   = data["states_train"]
    emp      = data["empirical_seq_flat"]
    pidx     = data["participant_idx"]
    npart    = data["n_participants"]

    model = make_model(color_semval, form_semval, wf)

    kernel = NUTS(model, target_accept_prob=0.85, max_tree_depth=5)
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method="vectorized",
                progress_bar=True)

    t0 = time.time()
    mcmc.run(random.PRNGKey(4711), states, emp, pidx, npart)
    elapsed = time.time() - t0

    # Save idata
    predictive = Predictive(model, mcmc.get_samples())
    pred = predictive(random.PRNGKey(42), states, None, pidx, npart)
    idata = az.from_numpyro(
        mcmc,
        posterior_predictive={"obs": np.array(pred["obs"])},
    )

    out_nc = (
        f"./inference_data/grid_repr_{label}"
        f"_warmup{NUM_WARMUP}_samples{NUM_SAMPLES}_chains{NUM_CHAINS}.nc"
    )
    idata.to_netcdf(out_nc)

    # LOO
    try:
        loo = az.loo(idata, scale="log")
        elpd = float(loo.elpd_loo)
        p_loo = float(loo.p_loo)
    except Exception as e:
        elpd = float("nan")
        p_loo = float("nan")
        print(f"  LOO failed: {e}")

    # Extract population params
    post = mcmc.get_samples()
    summary = {
        "label": label,
        "color_semval": color_semval,
        "form_semval": form_semval,
        "wf": wf,
        "elpd_loo": elpd,
        "p_loo": p_loo,
        "elapsed_s": elapsed,
    }
    for pname in ["alpha_D", "alpha_C", "alpha_F", "gamma", "epsilon",
                   "log_beta", "tau"]:
        if pname in post:
            vals = np.array(post[pname]).flatten()
            summary[f"{pname}_mean"] = float(np.mean(vals))
            summary[f"{pname}_median"] = float(np.median(vals))

    return summary


def main():
    # Load subset data
    import helper
    helper.CONDITIONS_OF_INTEREST = ("erdc", "zrdc", "brdc")
    data = import_dataset_hier()
    print(f"Data: {data['n_participants']} participants, "
          f"{len(data['states_train'])} obs")

    results = []
    for label, csv, fsv, wf in GRID:
        print(f"\n{'='*60}")
        print(f"  {label}: csv={csv}, fsv={fsv}, wf={wf}")
        print(f"{'='*60}")
        r = run_one(label, csv, fsv, wf, data)
        results.append(r)
        print(f"  ELPD={r['elpd_loo']:.1f}, p_loo={r['p_loo']:.1f}, "
              f"time={r['elapsed_s']:.0f}s")
        print(f"  alpha_D={r.get('alpha_D_mean','?'):.2f}, "
              f"alpha_C={r.get('alpha_C_mean','?'):.2f}, "
              f"alpha_F={r.get('alpha_F_mean','?'):.2f}")
        print(f"  gamma={r.get('gamma_mean','?'):.2f}, "
              f"epsilon={r.get('epsilon_mean','?'):.3f}")

    # Summary table
    print(f"\n{'='*80}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*80}")
    print(f"{'label':>25s}  csv   fsv    wf   ELPD     p_loo  eps    gamma  aD    aC    aF")
    print("-" * 100)
    results.sort(key=lambda x: -x["elpd_loo"])
    for r in results:
        print(f"{r['label']:>25s}  {r['color_semval']:.3f} {r['form_semval']:.2f} "
              f"{r['wf']:.1f}  {r['elpd_loo']:8.1f} {r['p_loo']:6.1f}  "
              f"{r.get('epsilon_mean',0):.3f}  {r.get('gamma_mean',0):5.2f}  "
              f"{r.get('alpha_D_mean',0):5.2f} {r.get('alpha_C_mean',0):5.2f} "
              f"{r.get('alpha_F_mean',0):5.2f}")

    # Save to JSON
    with open("grid_repr_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: grid_repr_results.json")


if __name__ == "__main__":
    main()
