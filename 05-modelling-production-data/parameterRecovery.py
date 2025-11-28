import os
import argparse
import jax
import numpy as np
import numpyro
from numpyro.infer import Predictive, MCMC, NUTS
import jax.numpy as jnp
from helper import import_dataset, normalize, build_utterance_prior_jax
from modelSpecification import likelihood_function_global_speaker


def draw_samples_from_prior(num_samples: int = 100_000) -> dict:
    """
    Draw parameter samples from the specified priors using NumPy and return them
    as JAX arrays for downstream use.
    """
    rng = np.random.default_rng()
    n = int(num_samples)

    samples = {
        "alpha": np.abs(rng.normal(loc=0.0, scale=2.0, size=n)),
        "color_semval": rng.uniform(0.0, 1.0, size=n),
        "k": rng.uniform(0.0, 1.0, size=n),
        "bias_subjectivity": rng.normal(loc=0.0, scale=2.0, size=n),
        "bias_length": rng.normal(loc=0.0, scale=2.0, size=n),
    }

    return {name: jnp.asarray(values) for name, values in samples.items()}

def generate_predictive_data(params, num_samples, key):
    pass

def run_inference(synthetic_data, num_warmup, num_samples, key):
    pass

def posterior_samples_analysis(posterior_samples, true_params):
    pass

def posterior_predictive_analysis(posterior_samples, synthetic_data, key):
    pass
