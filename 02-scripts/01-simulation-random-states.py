# Import libraries
from dataclasses import dataclass, field
import itertools
import jax
import jaxlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpyro
import numpyro.distributions as dist
from scipy.stats import truncnorm
from jax import random
import numpy as np
import scipy
import core_rsa
import argparse
import os

# Default wf: fixed cognitive perceptual blur parameter, constant across conditions
WF_DEFAULT = 0.5

@dataclass
class Object:
    size_distribution: str = "normal"
    upper: int = 30
    lower: int = 1
    p: float = 0.5
    sd_spread: float = 7.75  # controls size spread:
                              #   narrow (~2)  = blurred context (sizes clustered)
                              #   default (7.75) = current baseline
                              #   wide (~15)   = sharp context (sizes spread out)

    def sample_size(self):
        mean = (self.upper + self.lower) / 2
        sd = self.sd_spread

        if self.size_distribution == "normal":
            a = (self.lower - mean) / sd
            b = (self.upper - mean) / sd
            return scipy.stats.truncnorm.rvs(loc=mean, scale=sd, a=a, b=b)

        if self.size_distribution == "left-skewed":
            mean = mean / 2 * 3
            a = (self.lower - mean) / sd
            b = (self.upper - mean) / sd
            return scipy.stats.truncnorm.rvs(loc=mean, scale=sd, a=a, b=b)

        if self.size_distribution == "right-skewed":
            mean = mean / 2
            a = (self.lower - mean) / sd
            b = (self.upper - mean) / sd
            return scipy.stats.truncnorm.rvs(loc=mean, scale=sd, a=a, b=b)

        if self.size_distribution == "flat":
            return np.random.uniform(low=self.lower, high=self.upper)

        raise ValueError(f"Size distribution '{self.size_distribution}' not implemented.")

    def generate(self):
        size  = self.sample_size()
        color = np.random.binomial(n=1, p=self.p)
        form  = np.random.binomial(n=1, p=self.p)
        return jnp.array([size, color, form])


@dataclass
class Context:
    nobj: int = 6
    size_distribution: str = "normal"
    upper: int = 30
    lower: int = 1
    p: float = 0.5
    sd_spread: float = 7.75

    def generate(self):
        return jnp.array([
            Object(size_distribution=self.size_distribution,
                   upper=self.upper,
                   lower=self.lower,
                   p=self.p,
                   sd_spread=self.sd_spread).generate()
            for _ in range(self.nobj)
        ])


@dataclass
class AllContext:
    sample_size: int = 10
    nobj: int = 6
    size_distribution: str = "normal"
    upper: int = 30
    lower: int = 1
    p: float = 0.5
    sd_spread: float = 7.75

    def generate(self):
        return jnp.array([
            Context(nobj=self.nobj,
                    size_distribution=self.size_distribution,
                    upper=self.upper,
                    lower=self.lower,
                    p=self.p,
                    sd_spread=self.sd_spread).generate()
            for _ in range(self.sample_size)
        ])


def modify_referent(context):
    """
    Place the biggest object at index 0 and set its color=1, form=1.
    This defines the referent as the biggest, most-featured object.

    Input:  (n_obj, 3)
    Output: (n_obj, 3) with referent at row 0
    """
    max_index = jnp.argmax(context[:, 0])
    modified = context.at[0, 0].set(context[max_index, 0])
    modified = modified.at[max_index, 0].set(context[0, 0])
    modified = modified.at[0, 1].set(1)  # color = blue
    modified = modified.at[0, 2].set(1)  # form  = circle
    return modified


def record_communicative_success(pragmatic_listener_matrix):
    """
    Extract communicative success (Option B): L1(referent | utterance).

    pragmatic_listener_matrix shape: (2, n_obj)
      row 0: L1(s | "big blue")
      row 1: L1(s | "blue big")
    Referent is always at column 0 (after modify_referent).

    Returns:
      probs_big_blue: L1(referent | "big blue")
      probs_blue_big: L1(referent | "blue big")
    """
    probs_big_blue = pragmatic_listener_matrix[0, 0]
    probs_blue_big = pragmatic_listener_matrix[1, 0]
    return probs_big_blue, probs_blue_big


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run RSA simulation with random states.'
    )
    parser.add_argument('--nobj',             type=int,   default=6,
                        help='Number of objects in the context.')
    parser.add_argument('--sample_size',      type=int,   default=10,
                        help='Number of contexts to generate.')
    parser.add_argument('--color_semvalue',   type=float, default=0.95,
                        help='Semantic value for color.')
    parser.add_argument('--wf',               type=float, default=WF_DEFAULT,
                        help='Perceptual blur (cognitive parameter, fixed across conditions).')
    parser.add_argument('--k',                type=float, default=0.5,
                        help='k-percentage for determining mid-range threshold.')
    parser.add_argument('--speaker',          type=str,   default="incremental_speaker",
                        help='Speaker model: incremental_speaker or global_speaker.')
    parser.add_argument('--alpha',            type=float, default=1.0,
                        help='Rationality parameter for speaker.')
    parser.add_argument('--bias',             type=float, default=0.0,
                        help='Utterance cost bias (against "blue big").')
    parser.add_argument('--world_length',     type=int,   default=2,
                        help='Utterance length for RSA.')
    parser.add_argument('--size_distribution', type=str,  default="normal",
                        help='Shape of size distribution: normal, left-skewed, right-skewed.')
    parser.add_argument('--sd_spread',        type=float, default=7.75,
                        help='Std dev of size distribution. '
                             'Small (~2) = blurred/clustered; '
                             'large (~15) = sharp/spread-out.')
    args = parser.parse_args()

    # Generate contexts
    states_train = AllContext(
        sample_size=args.sample_size,
        nobj=args.nobj,
        size_distribution=args.size_distribution,
        sd_spread=args.sd_spread,
    ).generate()  # (sample_size, nobj, 3)

    # Vectorize over contexts
    pragmatic_listener_vmap = jax.vmap(
        core_rsa.pragmatic_listener,
        in_axes=(0,    # states
                 None, # alpha
                 None, # bias
                 None, # color_semvalue
                 None, # form_semvalue (ignored)
                 None, # wf
                 None, # k
                 None, # speaker
                 None) # world_length
    )
    modify_referent_vmap           = jax.vmap(modify_referent,           in_axes=0)
    record_communicative_success_vmap = jax.vmap(record_communicative_success, in_axes=0)

    # Place referent at index 0 in every context
    modified_states = modify_referent_vmap(states_train)

    # Compute pragmatic listener: L1(referent | utterance)
    results = pragmatic_listener_vmap(
        modified_states,
        args.alpha,
        args.bias,
        args.color_semvalue,
        None,            # form_semvalue unused
        args.wf,
        args.k,
        args.speaker,
        args.world_length,
    )

    # Extract communicative success probabilities
    probs_big_blue, probs_blue_big = record_communicative_success_vmap(results)

    probs_big_blue = probs_big_blue.tolist()
    probs_blue_big = probs_blue_big.tolist()

    # Build output DataFrame
    df = pd.DataFrame({
        "probs_big_blue":    probs_big_blue,
        "probs_blue_big":    probs_blue_big,
        "alpha":             [args.alpha]             * args.sample_size,
        "bias":              [args.bias]              * args.sample_size,
        "nobj":              [args.nobj]              * args.sample_size,
        "color_semvalue":    [args.color_semvalue]    * args.sample_size,
        "wf":                [args.wf]                * args.sample_size,
        "k":                 [args.k]                 * args.sample_size,
        "speaker":           [args.speaker]           * args.sample_size,
        "size_distribution": [args.size_distribution] * args.sample_size,
        "sd_spread":         [args.sd_spread]         * args.sample_size,
        "sample_size":       [args.sample_size]       * args.sample_size,
        "world_length":      [args.world_length]      * args.sample_size,
    })

    output_filename = "../04-simulation-w-randomstates/simulation_full_run_4.csv"

    if not os.path.isfile(output_filename):
        df.to_csv(output_filename, index=False)
    else:
        df.to_csv(output_filename, mode='a', header=False, index=False)
