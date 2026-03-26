"""Grid search over representation parameters for the REPORTED model.

Uses the reported model structure (single alpha + log_beta + tau + delta).
No gamma, no epsilon, no per-dim alpha.

Run on server with GPU:
    JAX_PLATFORMS='' XLA_FLAGS='' python run_grid_search_reported.py
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
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az

from helper import import_dataset_hier
from modelSpecification import jitted_speaker_hier

print(f"Devices: {jax.devices()}")

# ── Grid ─────────────────────────────────────────────────────────────────────
GRID = [
    # label,                    csv,   fsv,  wf
    ("baseline",                0.971, 0.50, 1.0),
    ("lowcol",                  0.85,  0.50, 1.0),
    ("lowcol_wf07",             0.85,  0.50, 0.7),
    ("lowcol_wf05",             0.85,  0.50, 0.5),
    ("lowcol_wf03",             0.85,  0.50, 0.3),
    ("wf07",                    0.971, 0.50, 0.7),
    ("wf05",                    0.971, 0.50, 0.5),
    ("wf03",                    0.971, 0.50, 0.3),
    ("lowcol_actform",          0.85,  0.70, 1.0),
    ("lowcol_actform_wf05",     0.85,  0.70, 0.5),
    ("moderate",                0.90,  0.65, 0.5),
    ("very_low_col",            0.75,  0.50, 1.0),
]

# ── MCMC config ──────────────────────────────────────────────────────────────
NUM_WARMUP  = 1000
NUM_SAMPLES = 500
NUM_CHAINS  = 4
K_FIXED     = 0.5


def make_reported_model(color_semval, form_semval, wf):
    """Factory: reported model (single alpha) with fixed semantics."""

    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None):
        alpha    = numpyro.sample("alpha", dist.HalfNormal(5.0))
        log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
        beta     = jnp.exp(log_beta)
        tau      = numpyro.sample("tau", dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_pt = jnp.maximum(alpha + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_hier(
                states, alpha_pt, alpha_pt, alpha_pt,
                color_semval, form_semval, K_FIXED, wf, beta, 0.0, 0.0,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)

    return model


def run_one(label, color_semval, form_semval, wf, data):
    """Run inference for one grid point and return summary dict."""
    states = data["states_train"]
    emp    = data["empirical_seq_flat"]
    pidx   = data["participant_idx"]
    npart  = data["n_participants"]

    model_fn = make_reported_model(color_semval, form_semval, wf)

    kernel = NUTS(model_fn, target_accept_prob=0.85, max_tree_depth=5)
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method="vectorized",
                progress_bar=True)

    t0 = time.time()
    mcmc.run(random.PRNGKey(4711), states, emp, pidx, npart)
    elapsed = time.time() - t0

    # Save idata
    predictive = Predictive(model_fn, mcmc.get_samples())
    pred = predictive(random.PRNGKey(42), states, None, pidx, npart)
    idata = az.from_numpyro(
        mcmc,
        posterior_predictive={"obs": np.array(pred["obs"])},
    )

    out_nc = (
        f"./inference_data/grid_reported_{label}"
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

    # Extract params
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
    for pname in ["alpha", "log_beta", "tau"]:
        if pname in post:
            vals = np.array(post[pname]).flatten()
            summary[f"{pname}_mean"] = float(np.mean(vals))
            summary[f"{pname}_median"] = float(np.median(vals))

    return summary


def main():
    # Load subset data (original 3 conditions)
    import helper
    helper.CONDITIONS_OF_INTEREST = ("erdc", "zrdc", "brdc")
    data = import_dataset_hier()
    print(f"Data: {data['n_participants']} participants, "
          f"{len(data['states_train'])} obs")

    results = []
    for label, csv_val, fsv, wf in GRID:
        print(f"\n{'='*60}")
        print(f"  {label}: csv={csv_val}, fsv={fsv}, wf={wf}")
        print(f"{'='*60}")
        r = run_one(label, csv_val, fsv, wf, data)
        results.append(r)
        print(f"  ELPD={r['elpd_loo']:.1f}, p_loo={r['p_loo']:.1f}, "
              f"time={r['elapsed_s']:.0f}s")
        print(f"  alpha={r.get('alpha_mean','?'):.3f}, "
              f"log_beta={r.get('log_beta_mean','?'):.3f}, "
              f"tau={r.get('tau_mean','?'):.3f}")

    # Summary table
    print(f"\n{'='*80}")
    print("GRID SEARCH RESULTS (reported model)")
    print(f"{'='*80}")
    print(f"{'label':>25s}  csv   fsv    wf   ELPD      p_loo  alpha  log_beta  tau")
    print("-" * 95)
    results.sort(key=lambda x: -x["elpd_loo"])
    for r in results:
        print(f"{r['label']:>25s}  {r['color_semval']:.3f} {r['form_semval']:.2f} "
              f"{r['wf']:.1f}  {r['elpd_loo']:9.1f} {r['p_loo']:6.1f}  "
              f"{r.get('alpha_mean',0):5.3f}  {r.get('log_beta_mean',0):7.3f}  "
              f"{r.get('tau_mean',0):5.3f}")

    # Save
    with open("grid_reported_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: grid_reported_results.json")


if __name__ == "__main__":
    main()
