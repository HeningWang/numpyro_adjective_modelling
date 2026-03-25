"""CLI entry point for running MCMC inference on the slider dataset.

Usage:
    python run_inference.py --speaker_type incremental --hierarchical \
        --num_warmup 500 --num_samples 500 --num_chains 4
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import argparse
import numpy as np
import arviz as az
import jax
from jax import random
from jax.random import PRNGKey
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive

from modelSpecification import (
    import_dataset,
    import_dataset_hier,
    precompute_listeners,
    precompute_listeners_frozen,
    canonicalize_speaker_type,
    likelihood_gb_speaker,
    likelihood_inc_speaker,
    likelihood_gb_speaker_static,
    likelihood_inc_speaker_frozen,
    likelihood_gb_speaker_hier,
    likelihood_inc_speaker_hier,
    likelihood_gb_speaker_static_hier,
    likelihood_inc_speaker_frozen_hier,
    ZOIB,
)


def run_inference(
    speaker_type: str = "global",
    num_samples: int = 1000,
    num_warmup: int = 1000,
    num_chains: int = 4,
):
    canonical_speaker_type = canonicalize_speaker_type(speaker_type)
    states_train, empirical_train, df = import_dataset()

    empirical_train_np = np.asarray(empirical_train)
    pi0 = float(np.mean(np.isclose(empirical_train_np, 0.0)))
    pi1 = float(np.mean(np.isclose(empirical_train_np, 1.0)))
    if (pi0 + pi1) >= 0.95:
        raise ValueError(f"Boundary masses too large for ZOIB: pi0+pi1={pi0+pi1:.3f}")

    if canonical_speaker_type in ("global", "incremental"):
        L1_all, L2_all = precompute_listeners(states_train)
    else:
        L1_all, L2_all = precompute_listeners_frozen(states_train)
    print("Listeners precomputed.")

    output_file_name = (
        f"./inference_data/mcmc_results_{canonical_speaker_type}_speaker"
        f"_warmup{num_warmup}_samples{num_samples}_chains{num_chains}.nc"
    )
    print(f"Output file: {output_file_name}")
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
        print(f"Removed existing file: {output_file_name}")

    rng_key = random.PRNGKey(11)
    rng_key, rng_key_ = random.split(rng_key)

    if canonical_speaker_type == "global":
        model = likelihood_gb_speaker
    elif canonical_speaker_type == "incremental":
        model = likelihood_inc_speaker
    elif canonical_speaker_type == "global_static":
        model = likelihood_gb_speaker_static
    elif canonical_speaker_type == "incremental_static":
        model = likelihood_inc_speaker_frozen
    else:
        raise ValueError(
            "Invalid speaker type. Choose 'global', 'incremental', "
            "'global_static', or 'incremental_static' "
            "(legacy alias: 'incremental_frozen')."
        )

    kernel = NUTS(model, dense_mass=True, max_tree_depth=8, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="vectorized")
    mcmc.run(rng_key_, states_train, empirical_train, pi0, pi1, L1_all, L2_all)
    mcmc.print_summary()

    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model, posterior_samples)(
        PRNGKey(1), states_train, None, pi0, pi1, L1_all, L2_all
    )
    prior = Predictive(model, num_samples=1000)(
        PRNGKey(2), states_train, None, pi0, pi1, L1_all, L2_all
    )

    N = states_train.shape[0]
    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords={"states": np.arange(N)},
        dims={"obs": ["states"]},
    )
    az.to_netcdf(numpyro_data, output_file_name)
    assert os.path.exists(output_file_name), f"Save failed: {output_file_name} not found"
    size_mb = os.path.getsize(output_file_name) / 1024 / 1024
    print(f"Saved: {output_file_name}  ({size_mb:.1f} MB)")


def run_inference_hier(
    speaker_type: str = "global",
    num_samples: int = 1000,
    num_warmup: int = 1000,
    num_chains: int = 4,
):
    """Run MCMC for the hierarchical (random participant intercept) speaker model."""
    canonical_speaker_type = canonicalize_speaker_type(speaker_type)
    states_train, empirical_train, df, participant_idx, n_participants = import_dataset_hier()
    print(f"Hierarchical model: {n_participants} participants, {len(states_train)} observations")

    empirical_train_np = np.asarray(empirical_train)
    pi0 = float(np.mean(np.isclose(empirical_train_np, 0.0)))
    pi1 = float(np.mean(np.isclose(empirical_train_np, 1.0)))
    if (pi0 + pi1) >= 0.95:
        raise ValueError(f"Boundary masses too large for ZOIB: pi0+pi1={pi0+pi1:.3f}")

    if canonical_speaker_type in ("global", "incremental"):
        L1_all, L2_all = precompute_listeners(states_train)
    else:
        L1_all, L2_all = precompute_listeners_frozen(states_train)
    print("Listeners precomputed.")

    output_file_name = (
        f"./inference_data/mcmc_results_{canonical_speaker_type}_speaker_hier"
        f"_warmup{num_warmup}_samples{num_samples}_chains{num_chains}.nc"
    )
    print(f"Output file: {output_file_name}")
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
        print(f"Removed existing file: {output_file_name}")

    rng_key = random.PRNGKey(11)
    rng_key, rng_key_ = random.split(rng_key)

    if canonical_speaker_type == "global":
        model = likelihood_gb_speaker_hier
    elif canonical_speaker_type == "incremental":
        model = likelihood_inc_speaker_hier
    elif canonical_speaker_type == "global_static":
        model = likelihood_gb_speaker_static_hier
    elif canonical_speaker_type == "incremental_static":
        model = likelihood_inc_speaker_frozen_hier
    else:
        raise ValueError(
            "Invalid speaker type. Choose 'global', 'incremental', "
            "'global_static', or 'incremental_static' "
            "(legacy alias: 'incremental_frozen')."
        )

    kernel = NUTS(model, dense_mass=False, max_tree_depth=8, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
    )
    mcmc.run(
        rng_key_, states_train, empirical_train,
        pi0, pi1, participant_idx, n_participants, L1_all, L2_all,
    )
    mcmc.print_summary(exclude_deterministic=False)

    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model, posterior_samples)(
        PRNGKey(1), states_train, None,
        pi0, pi1, participant_idx, n_participants, L1_all, L2_all,
    )
    prior = Predictive(model, num_samples=1000)(
        PRNGKey(2), states_train, None,
        pi0, pi1, participant_idx, n_participants, L1_all, L2_all,
    )

    N = states_train.shape[0]
    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords={
            "states": np.arange(N),
            "participants": np.arange(n_participants),
        },
        dims={"obs": ["states"], "delta": ["participants"]},
    )
    az.to_netcdf(numpyro_data, output_file_name)
    assert os.path.exists(output_file_name), f"Save failed: {output_file_name} not found"
    size_mb = os.path.getsize(output_file_name) / 1024 / 1024
    print(f"Saved: {output_file_name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speaker inference with NumPyro.")
    parser.add_argument("--speaker_type", type=str,
                        choices=["global", "incremental", "global_static",
                                 "incremental_static", "incremental_frozen"],
                        default="global",
                        help="Choose the speaker model type.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of posterior samples.")
    parser.add_argument("--num_warmup", type=int, default=500, help="Number of warm-up iterations.")
    parser.add_argument("--num_chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--test", action="store_true", help="Run test function and exit.")
    parser.add_argument(
        "--hierarchical", action="store_true",
        help="Run hierarchical model with random participant intercepts."
    )

    args = parser.parse_args()

    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(args.num_chains)
    jax.local_device_count()

    if args.test:
        pass
    elif args.hierarchical:
        run_inference_hier(
            speaker_type=args.speaker_type,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
        )
    else:
        run_inference(
            speaker_type=args.speaker_type,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
        )
