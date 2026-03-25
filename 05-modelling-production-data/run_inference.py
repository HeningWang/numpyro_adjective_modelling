"""CLI entry point for running MCMC inference on the production dataset.

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

from helper import import_dataset, import_dataset_hier
from modelSpecification import (
    canonicalize_speaker_type,
    likelihood_function_global_speaker,
    likelihood_function_incremental_speaker,
    likelihood_function_global_speaker_static,
    likelihood_function_incremental_speaker_frozen,
    likelihood_function_global_speaker_hier,
    likelihood_function_incremental_speaker_hier,
    likelihood_function_global_speaker_static_hier,
    likelihood_function_incremental_speaker_frozen_hier,
    likelihood_function_incremental_lm_only_hier,
    likelihood_function_incremental_rsa_only_hier,
    likelihood_function_incremental_speaker_lookahead_hier,
    likelihood_function_incremental_speaker_extended_hier,
)


# Model dispatch tables
FLAT_MODELS = {
    "global": (likelihood_function_global_speaker, 0.9, 5),
    "incremental": (likelihood_function_incremental_speaker, 0.85, 5),
    "global_static": (likelihood_function_global_speaker_static, 0.9, 5),
    "incremental_static": (likelihood_function_incremental_speaker_frozen, 0.85, 5),
}

HIER_MODELS = {
    "global": (likelihood_function_global_speaker_hier, 0.9, 5),
    "incremental": (likelihood_function_incremental_speaker_hier, 0.85, 5),
    "global_static": (likelihood_function_global_speaker_static_hier, 0.9, 5),
    "incremental_static": (likelihood_function_incremental_speaker_frozen_hier, 0.85, 5),
    "incremental_lm_only": (likelihood_function_incremental_lm_only_hier, 0.85, 5),
    "incremental_rsa_only": (likelihood_function_incremental_rsa_only_hier, 0.85, 5),
    "incremental_lookahead": (likelihood_function_incremental_speaker_lookahead_hier, 0.85, 5),
    "incremental_extended": (likelihood_function_incremental_speaker_extended_hier, 0.85, 5),
}


def run_inference(
    speaker_type: str = "global",
    num_warmup: int = 100,
    num_samples: int = 250,
    num_chains: int = 4,
):
    canonical_speaker_type = canonicalize_speaker_type(speaker_type)

    output_file_name = (
        f"./inference_data/mcmc_results_{canonical_speaker_type}_speaker"
        f"_warmup{num_warmup}_samples{num_samples}_chains{num_chains}.nc"
    )
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
        print(f"Deleted existing file: {output_file_name}")

    data = import_dataset()
    states_train = data["states_train"]
    empirical_train_seq_flat = data["empirical_seq_flat"]

    print("States train shape:", states_train.shape)
    print("Empirical train flat shape:", empirical_train_seq_flat.shape)
    print("Output file name:", output_file_name)

    rng_key = random.PRNGKey(4711)
    rng_key, rng_key_ = random.split(rng_key)

    if canonical_speaker_type not in FLAT_MODELS:
        raise ValueError(
            f"Unknown speaker_type '{canonical_speaker_type}'. "
            f"Choose from: {list(FLAT_MODELS.keys())}"
        )
    model, target_accept_prob, max_tree_depth = FLAT_MODELS[canonical_speaker_type]

    kernel = NUTS(model, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="vectorized")
    mcmc.run(rng_key_, states_train, empirical_train_seq_flat)
    mcmc.print_summary()

    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model, posterior_samples)(
        PRNGKey(1), states_train, None
    )
    prior = Predictive(model, num_samples=500)(
        PRNGKey(2), states_train, None
    )

    N = states_train.shape[0]
    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords={"item": np.arange(N)},
        dims={"obs": ["item"]},
    )
    az.to_netcdf(numpyro_data, output_file_name)
    print(f"Saved: {output_file_name}")


def run_inference_hier(
    speaker_type: str = "global",
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    min_proportion: float = 0.0,
):
    """Run MCMC for the hierarchical (random participant alpha) speaker model."""
    canonical_speaker_type = canonicalize_speaker_type(speaker_type)

    tag = f"_top" if min_proportion > 0 else ""
    output_file_name = (
        f"./inference_data/mcmc_results_{canonical_speaker_type}_speaker_hier{tag}"
        f"_warmup{num_warmup}_samples{num_samples}_chains{num_chains}.nc"
    )
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
        print(f"Deleted existing file: {output_file_name}")

    data = import_dataset_hier(min_proportion=min_proportion)
    states_train       = data["states_train"]
    empirical_seq_flat = data["empirical_seq_flat"]
    participant_idx    = data["participant_idx"]
    n_participants     = data["n_participants"]

    print(f"Hierarchical model: {n_participants} participants, {len(states_train)} observations")
    print(f"Output file: {output_file_name}")

    rng_key = random.PRNGKey(4711)
    rng_key, rng_key_ = random.split(rng_key)

    if canonical_speaker_type not in HIER_MODELS:
        raise ValueError(
            f"Unknown speaker_type '{canonical_speaker_type}'. "
            f"Choose from: {list(HIER_MODELS.keys())}"
        )
    model, target_accept_prob, max_tree_depth = HIER_MODELS[canonical_speaker_type]

    kernel = NUTS(model, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
    )
    mcmc.run(
        rng_key_,
        states_train, empirical_seq_flat,
        participant_idx, n_participants,
    )
    mcmc.print_summary(exclude_deterministic=False)

    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model, posterior_samples)(
        PRNGKey(1), states_train, None, participant_idx, n_participants
    )
    prior = Predictive(model, num_samples=500)(
        PRNGKey(2), states_train, None, participant_idx, n_participants
    )

    N = states_train.shape[0]
    coords = {"item": np.arange(N)}
    dims   = {"obs": ["item"]}
    if "delta" in posterior_samples:
        coords["participants"] = np.arange(n_participants)
        dims["delta"] = ["participants"]

    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords=coords,
        dims=dims,
    )
    az.to_netcdf(numpyro_data, output_file_name)
    assert os.path.exists(output_file_name), f"Save failed: {output_file_name} not found"
    print(f"Saved: {output_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speaker inference with NumPyro.")
    parser.add_argument("--speaker_type", type=str,
                        choices=["global", "incremental", "global_static", "incremental_static",
                                 "incremental_frozen", "incremental_lm_only",
                                 "incremental_rsa_only", "incremental_lookahead",
                                 "incremental_extended"],
                        default="incremental",
                        help="Choose the speaker model type.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of posterior samples.")
    parser.add_argument("--num_warmup", type=int, default=500, help="Number of warm-up iterations.")
    parser.add_argument("--num_chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--test", action="store_true", help="Run test function and exit.")
    parser.add_argument(
        "--hierarchical", action="store_true",
        help="Run hierarchical model with random per-participant alpha intercepts.",
    )
    parser.add_argument(
        "--min-proportion", type=float, default=0.0,
        help="Filter training data to utterance types with max proportion >= this in any condition.",
    )

    args = parser.parse_args()

    if args.test:
        from modelSpecification import test
        test()
    elif args.hierarchical:
        run_inference_hier(
            speaker_type=args.speaker_type,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            min_proportion=args.min_proportion,
        )
    else:
        run_inference(
            speaker_type=args.speaker_type,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
        )
