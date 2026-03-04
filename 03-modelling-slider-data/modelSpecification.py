import os
os.environ["JAX_PLATFORMS"] = "cpu"        # ← force CPU before JAX init  
os.environ["JAX_TRACEBACK_FILTERING"] = "off"  # optional: full tracebacks  
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"  

import argparse
import functools
import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import scipy
import arviz as az
import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive
from jax import lax
from sklearn.model_selection import train_test_split
from helper import link_function
from jax.random import PRNGKey
numpyro.set_platform("cpu")

print(jax.__version__)
jax.devices()

def get_midrange_extrema_from_context(
    states: jnp.ndarray,
    state_prior: jnp.ndarray,
    q_low: float = 0.2,
    q_high: float = 0.8,
) -> tuple[float, float]:
    """
    Compute mid-range extrema x_min^mid(C), x_max^mid(C) from the context C.

    Args:
        states:      (n_obj, 3) array; size is states[:, 0].
        state_prior: (n_obj,) array; contextual distribution C(s).
        q_low:       lower quantile for mid-range band (0 < q_low < q_high < 1).
        q_high:      upper quantile for mid-range band.

    Returns:
        x_min_mid, x_max_mid: floats defining the mid-range band.
    """
    sizes = states[:, 0]  # shape (n_obj,)

    # Sort by size
    idx = jnp.argsort(sizes)
    sizes_sorted = sizes[idx]
    prior_sorted = state_prior[idx]

    # CDF of the contextual prior over sorted sizes
    cdf = jnp.cumsum(prior_sorted)

    def quantile_index(q):
        # First index where CDF >= q
        # (works because cdf[-1] = 1, so mask is True somewhere)
        mask = cdf >= q
        return jnp.argmax(mask)

    i_low = quantile_index(q_low)
    i_high = quantile_index(q_high)

    x_min_mid = sizes_sorted[i_low]
    x_max_mid = sizes_sorted[i_high]

    return x_min_mid, x_max_mid

def get_threshold_k_midrange_jax(
    states: jnp.ndarray,
    state_prior: jnp.ndarray,
    k: float,
    q_low: float = 0.2,
    q_high: float = 0.8,
) -> float:
    """
    Compute the context-dependent mid-range max-min threshold θ_k(C).

    θ_k(C) = x_max^mid(C) - k * (x_max^mid(C) - x_min^mid(C))

    Args:
        states:      (n_obj, 3)
        state_prior: (n_obj,)
        k:           interpolation parameter in [0, 1]
        q_low:       lower quantile for mid-range band
        q_high:      upper quantile for mid-range band

    Returns:
        theta_k: scalar threshold.
    """
    x_min_mid, x_max_mid = get_midrange_extrema_from_context(
        states, state_prior, q_low=q_low, q_high=q_high
    )
    theta_k = x_max_mid - k * (x_max_mid - x_min_mid)
    return theta_k

def size_semantic_value_midrange(
    size: jnp.ndarray,      # scalar or array
    theta_k: float,
    w_f: float,
    gamma: float = 1.0,     # size gain / discriminability (higher = sharper semantics)
) -> jnp.ndarray:
    """
    Graded size semantics with mid-range threshold θ_k(C), noise w_f, and gain gamma.

    ⟦size⟧_C(s) = Φ( gamma * (x - θ_k) / (w_f * sqrt(x^2 + θ_k^2)) )
    """
    eps = 1e-8
    denom = w_f * jnp.sqrt(size**2 + theta_k**2 + eps)
    z = gamma * (size - theta_k) / denom
    return dist.Normal(0.0, 1.0).cdf(z)

def _size_meaning_vector(
    states: jnp.ndarray,   # (n_obj, 3)
    prior: jnp.ndarray,    # (n_obj,)
    k: float,
    wf: float,
    gamma: float = 1.0,
    q_low: float = 0.2,
    q_high: float = 0.8,
) -> jnp.ndarray:
    """Context-dependent graded size semantics ⟦size⟧_C(s) for all objects."""
    sizes = states[:, 0]  # (n_obj,)

    x_min_mid, x_max_mid = get_midrange_extrema_from_context(
        states, prior, q_low=q_low, q_high=q_high
    )
    theta_k = x_max_mid - k * (x_max_mid - x_min_mid)

    eps = 1e-8
    denom = wf * jnp.sqrt(sizes**2 + theta_k**2 + eps)
    z = gamma * (sizes - theta_k) / denom
    return dist.Normal(0.0, 1.0).cdf(z)  # (n_obj,)

# --- atomic adjective meaning (no sampling, just semantic values) ---

def adjMeaning(
    word: int,                 # 0 = size (“big”), 1 = color (“blue”)
    states: jnp.ndarray,       # (n_obj, 3)
    current_state_prior: jnp.ndarray,  # (n_obj,)
    color_semvalue: float = 0.8,
    wf: float = 0.6,
    k: float = 0.5,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """
    Deterministic lexical meaning ⟦w⟧_C(s) for one adjective w in context C.

    Returns a vector of shape (n_obj,) with semantic values in [0,1].
    Uses JAX control flow (lax.cond), so it is vmap/jit-safe.
    """

    def color_branch(_):
        # Bernoulli-style color semantics, context independent
        x_color = states[:, 1]  # 0/1 feature
        return x_color * color_semvalue + (1.0 - x_color) * (1.0 - color_semvalue)

    def size_branch(_):
        # Context-dependent graded size semantics
        return size_semantic_value_midrange(
            size=states[:, 0],
            theta_k=get_threshold_k_midrange_jax(
                states=states,
                state_prior=current_state_prior,
                k=k,
            ),
            w_f=wf,
            gamma=gamma,
        )

    # word == 1 → color, else → size (we only ever use 0 or 1 here)
    return lax.cond(word == 1, color_branch, size_branch, operand=None)

def literal_listener_one_word(
    states: jnp.ndarray,              # (n_obj, 3)
    color_semvalue: float = 0.90,
    wf: float = 1,
    k: float = 0.5,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """
    Incremental literal listener for single-word utterances:
      row 0: L(s | 'big')
      row 1: L(s | 'blue')
    Returns array of shape (2, n_obj).
    """
    n_obj = states.shape[0]
    prior0 = jnp.ones(n_obj) / n_obj  # uniform prior over states

    # tokens: 0 = size ("big"), 1 = color ("blue")
    tokens = jnp.array([0, 1], dtype=jnp.int32)

    def update_token(token: int) -> jnp.ndarray:
        m = adjMeaning(
            word=token,
            states=states,
            current_state_prior=prior0,
            color_semvalue=color_semvalue,
            wf=wf,
            k=k,
            gamma=gamma,
        )
        unnorm = prior0 * m
        norm = jnp.clip(unnorm.sum(), 1e-20)
        return unnorm / norm  # (n_obj,)

    # shape: (2, n_obj)
    return jax.vmap(update_token)(tokens)

def literal_listener_recursive(
    word_length: int,
    states: jnp.ndarray,              # (n_obj, 3)
    color_semvalue: float = 0.90,
    wf: float = 1,
    k: float = 0.5,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """
    Incremental literal listener using backward functional semantics.

    We only need two utterances:
      - row 0: "big blue"  (size → color)   encoded as [0, 1]
      - row 1: "blue big"  (color → size)   encoded as [1, 0]

    Returns:
        probs: jnp.ndarray of shape (2, n_obj)
               row 0: P(s | "big blue")
               row 1: P(s | "blue big")
    """
    assert word_length == 2, "This implementation is tailored to 2-word utterances."

    n_obj = states.shape[0]
    # Prior over states P(s): uniform
    prior0 = jnp.ones(n_obj) / n_obj

    # encode the two utterances: 0=size, 1=color
    utterances = jnp.array([
        [0, 1],  # "big blue"
        [1, 0],  # "blue big"
    ], dtype=jnp.int32)

    def update_one_token(prior: jnp.ndarray, token: int) -> jnp.ndarray:
        """
        Single backward update:
          P_{t-1}(s) ∝ P_t(s) * ⟦w_t⟧_{C_t}(s),
        where C_t is encoded by the current prior.
        """
        m = adjMeaning(
            word=token,
            states=states,
            current_state_prior=prior,
            color_semvalue=color_semvalue,
            wf=wf,
            k=k,
            gamma=gamma,
        )
        unnorm = prior * m
        norm = jnp.clip(unnorm.sum(), 1e-20)
        return unnorm / norm

    def interpret_utterance(tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Backward composition for a single utterance u = (w1, w2):
          start with P_2(s) = P_prior(s),
          then apply w_2, then w_1.
        """
        tokens_rev = tokens[::-1]  # right-to-left
        def step(p, tok):
            new_p = update_one_token(p, tok)
            return new_p, None

        final_belief, _ = lax.scan(step, prior0, tokens_rev)
        return final_belief  # (n_obj,)

    # vmap over the 2 utterances → (2, n_obj)
    probs = jax.vmap(interpret_utterance)(utterances)
    return probs

def speaker_recursive(
    word_length,
    states: jnp.ndarray,
    alpha: float = 1.0,
    bias: float = 2.0,
    color_semvalue: float = 0.98,
    form_semvalue: float = 0.98,  # kept for API compatibility, unused here
    wf: float = 0.6,
    k: float = 0.5,
    gamma: float = 1.0,
):
    """
    Truly token-wise incremental RSA speaker for the 2-utterance slider setup.

    Utterances:
      u_0 = "big blue"  (size -> color)
      u_1 = "blue big"  (color -> size)

    Returns:
        S_inc: jnp.ndarray of shape (n_obj, 2)
               S_inc[j, 0] = P_inc("big blue" | s_j)
               S_inc[j, 1] = P_inc("blue big" | s_j)
    """
    assert word_length == 2, "speaker_recursive is currently implemented for 2-word utterances."

    n_obj = states.shape[0]

    # --- 1. Literal listeners for 1-word and 2-word utterances ---

    # L1: shape (2, n_obj)
    #   row 0: L(s | 'big')
    #   row 1: L(s | 'blue')
    L1 = literal_listener_one_word(
        states,
        color_semvalue=color_semvalue,
        wf=wf,
        k=k,
        gamma=gamma,
    )  # (2, n_obj)

    # L2: shape (2, n_obj)
    #   row 0: L(s | 'big blue')
    #   row 1: L(s | 'blue big')
    L2 = literal_listener_recursive(
        word_length,
        states,
        color_semvalue=color_semvalue,
        wf=wf,
        k=k,
        gamma=gamma,
    )  # (2, n_obj)

    # transpose to (n_obj, 2) for easier per-state handling
    L1_T = L1.T  # (n_obj, 2)
    L2_T = L2.T  # (n_obj, 2)

    # unpack for readability
    L1_big  = L1_T[:, 0]   # L(s_j | 'big')
    L1_blue = L1_T[:, 1]   # L(s_j | 'blue')

    L2_bigblue = L2_T[:, 0]  # L(s_j | 'big blue')
    L2_bluebig = L2_T[:, 1]  # L(s_j | 'blue big')

    eps = 1e-20

    # --- 2. Step 1: choose first token (big vs blue) ---

    num_big  = jnp.power(jnp.clip(L1_big,  eps, 1.0), alpha)
    num_blue = jnp.power(jnp.clip(L1_blue, eps, 1.0), alpha)
    Z1 = jnp.clip(num_big + num_blue, eps)

    S1_big  = num_big  / Z1   # P_1(w1='big'  | s_j)
    S1_blue = num_blue / Z1   # P_1(w1='blue' | s_j)

    # --- 3. Step 2: given prefix, choose STOP or the second adjective ---

    # Prefix 'big': next is STOP (-> 'big') or 'blue' (-> 'big blue')
    num_bigblue = jnp.power(jnp.clip(L2_bigblue, eps, 1.0), alpha)  # candidate second word 'blue'
    num_stop_big = jnp.power(jnp.clip(L1_big, eps, 1.0), alpha)     # stopping after 'big'
    Z2_big = jnp.clip(num_bigblue + num_stop_big, eps)

    S2_blue_given_big = num_bigblue / Z2_big   # P_2(w2='blue' | prefix='big', s_j)

    # Prefix 'blue': next is STOP (-> 'blue') or 'big' (-> 'blue big')
    num_bluebig = jnp.power(jnp.clip(L2_bluebig, eps, 1.0), alpha)  # candidate second word 'big'
    num_stop_blue = jnp.power(jnp.clip(L1_blue, eps, 1.0), alpha)   # stopping after 'blue'
    Z2_blue = jnp.clip(num_bluebig + num_stop_blue, eps)

    S2_big_given_blue = num_bluebig / Z2_blue  # P_2(w2='big' | prefix='blue', s_j)

    # --- 4. Path probabilities for length-2 utterances (chain rule) ---

    # P_chain("big blue" | s_j) ∝ S1(big|s_j) * S2(blue | prefix='big', s_j)
    P_chain_bigblue = S1_big * S2_blue_given_big

    # P_chain("blue big" | s_j) ∝ S1(blue|s_j) * S2(big | prefix='blue', s_j)
    P_chain_bluebig = S1_blue * S2_big_given_blue

    # stack: (n_obj, 2)
    P_chain = jnp.stack([P_chain_bigblue, P_chain_bluebig], axis=-1)
    P_chain = jnp.clip(P_chain, eps, 1.0)

    # --- 5. Add utterance-level cost (bias against "blue big") ---

    # cost[0] = 0 for "big blue"
    # cost[1] = bias for "blue big"
    utt_cost = jnp.array([0.0, bias])  # (2,)

    # Utilities: log P_chain(u | s_j) - cost(u)
    logP_chain = jnp.log(P_chain)
    util = logP_chain - utt_cost  # broadcast over states

    # --- 6. Final incremental speaker distribution over the 2 utterances ---

    S_inc = jax.nn.softmax(util, axis=-1)  # (n_obj, 2)

    # Extract only the probs for "big blue" and target referent (index 0)
    probs_bigblue_referent = S_inc[0, 0]  # (n_obj,)
    return probs_bigblue_referent

def global_speaker(
    states,
    alpha: float = 1.0,
    bias: float = 2.0,
    color_semvalue: float = 0.98,
    k: float = 0.5,
    wf: float = 0.8,
    gamma: float = 1.0,
):
    listener = literal_listener_recursive(
        2,
        states,
        color_semvalue=color_semvalue,
        wf=wf,
        k=k,
        gamma=gamma,
    )  # (2, n_obj)

    eps = 1e-20
    utt_cost = jnp.array([0.0, bias])          # (2,)
    util_speaker = jnp.log(jnp.clip(listener.T, eps, 1.0)) - utt_cost  # (n_obj, 2)
    softmax_result = jax.nn.softmax(alpha * util_speaker, axis=-1)

    # Extract only the probs for "big blue" and target referent (index 0)
    probs_bigblue_referent = softmax_result[0, 0]  # (n_obj,)
    return probs_bigblue_referent

vectorized_gb_speaker = jax.vmap(global_speaker, in_axes=(0,    # states
                                                          None, # alpha
                                                          None, # bias
                                                          None, # color_semvalue
                                                          None, # k
                                                          None, # wf
                                                          0,    # gamma (per-trial)
                                                          ))

# Define a function to encode the states of the objects
def encode_states(line):
      states = []
      for i in range(6):
        color = 1 if line.iloc[11 + i] == "blue" else 0
        form = 1 if line.iloc[17 + i] == "circle" else 0
        new_obj = (line.iloc[5 + i], color, form) # size, color, form
        states.append(new_obj)
      return jnp.array(states)

def import_dataset(file_path = "../01-dataset/01-slider-data-preprocessed.csv"):
   # Import the data
    df = pd.read_csv(file_path)

    # Subset data to only include combination dimension_color
    df = df[df['combination'] == 'dimension_color']
    df.reset_index(inplace=True, drop=True)

    # Mutate the dataset to include the states of the objects
    df["states"] = df.apply(lambda row: encode_states(row), axis=1)

    # Transform/rescale slider value from range 0 to 100 to 0 to 1
    df.prefer_first_1st = jnp.clip(df.prefer_first_1st.to_numpy(), 0, 100)
    df.prefer_first_1st = df.prefer_first_1st/100

        
    # split the dataset into training and test sets
    #train, test = train_test_split(df, test_size=0.99, random_state=42)

    # Use the whole dataset as training set
    train = df

    # Extract the states and empirical data and store them in JAX arrays
    states_train = jnp.stack([cell for cell in train.states])
    empirical_train = jnp.array(train.prefer_first_1st.to_numpy())

    return states_train, empirical_train, df
    

def test_core_rsa():
    # Import the data
    file_path = "../01-dataset/01-slider-data-preprocessed.csv"
    df = pd.read_csv(file_path)

    # subset data to only include combination dimension_color
    df = df[df['combination'] == 'dimension_color']
    df.reset_index(inplace=True, drop=True)

    # Mutate the dataset to include the states of the objects
    df_experiment = df.copy()
    df_experiment["states"] = df_experiment.apply(lambda row: encode_states(row), axis=1)

    # Transform/rescale slider value from range 0 to 100 to 0 to 1
    df_experiment.prefer_first_1st = jnp.clip(df_experiment.prefer_first_1st.to_numpy(), 0, 100)
    df_experiment.prefer_first_1st = df_experiment.prefer_first_1st/100
    print(df_experiment.prefer_first_1st.describe())

    # 118 brdc sharp, target not the biggest
    # 12 brdc blurred

    # 119 erdc sharp
    # 15 erdc blurred

    index = 119
    states_manuell = jnp.array([[10., 1., 1.],
                    [3., 1., 1.],
                    [3., 1., 1.],
                    [3., 1., 0.],
                    [3., 1., 0.],
                    [1., 0., 1.]], dtype=jnp.float32)

    #states_example = states_manuell
    states_example = df_experiment.iloc[index, df_experiment.columns.get_loc("states")]
    condition = df_experiment.iloc[index, df_experiment.columns.get_loc("conditions")]
    distribution = df_experiment.iloc[index, df_experiment.columns.get_loc("sharpness")]
    preference = df_experiment.iloc[index, df_experiment.columns.get_loc("prefer_first_1st")]
    print(states_example)
    print(condition + " " + distribution)
    print(preference)
    print(f"literal listener one word: {literal_listener_one_word(states_example, color_semvalue=0.8)}")
    print(f"literal listener two words: {literal_listener_recursive(2,states_example, color_semvalue=0.8)}")
    print(f"speaker one word: {global_speaker(states_example, color_semvalue=0.8)}")
    print("________________________________________")
    print(f"speaker two words incremental: {speaker_recursive(2,states_example, color_semvalue=0.8)}")


def link_logit(p,s):
    x0 = 1 / (jnp.exp(1 / (-2 * s)) + 1)
    xtrans = p * (x0 - (1 - x0)) + (1 - x0)
    return s * -jnp.log((1 / xtrans) - 1) + 0.5

vectorized_inc_speaker = jax.vmap(speaker_recursive, in_axes=(None,0,None,None,None,None,None,None,0))

# Fixed parameters
FIXED_BIAS = 2.0
FIXED_COLOR_SEMVALUE = 0.8
FIXED_K = 0.5
FIXED_WF = 1.0
# Sharpness-conditioned size gain (same convention as production model)
GAMMA_BLURRED = 0.9
GAMMA_SHARP   = 2.0

@jax.jit
def jitted_global_speaker(states, alpha, bias, gamma):
    return vectorized_gb_speaker(
        states,
        alpha,
        bias,
        FIXED_COLOR_SEMVALUE,
        FIXED_K,
        FIXED_WF,
        gamma,
    )

@jax.jit
def jitted_incremental_speaker(states, alpha, bias, gamma):
    return vectorized_inc_speaker(
        2,
        states,
        alpha,
        bias,
        FIXED_COLOR_SEMVALUE,
        FIXED_COLOR_SEMVALUE,
        FIXED_WF,
        FIXED_K,
        gamma,
    )


# Fast incremental path: precompute listeners outside MCMC loop.
# literal_listener_one_word / literal_listener_recursive depend only on
# (states, gamma), NOT on alpha or bias -- so they can be evaluated once
# before MCMC and passed as static data to every likelihood call.

_vmap_L1 = jax.vmap(
    lambda s, g: literal_listener_one_word(
        s, FIXED_COLOR_SEMVALUE, FIXED_WF, FIXED_K, g
    ),
    in_axes=(0, 0),
)
_vmap_L2 = jax.vmap(
    lambda s, g: literal_listener_recursive(
        2, s, FIXED_COLOR_SEMVALUE, FIXED_WF, FIXED_K, g
    ),
    in_axes=(0, 0),
)


@jax.jit
def precompute_listeners_all(states, gamma):
    """Precompute L1 and L2 listener arrays for all N trials (run once before MCMC).

    Returns
    -------
    L1_all : (N, 2, n_obj) -- one-word listener
    L2_all : (N, 2, n_obj) -- two-word listener
    """
    return _vmap_L1(states, gamma), _vmap_L2(states, gamma)


def _inc_speaker_from_listeners(L1, L2, alpha, bias):
    """Incremental speaker for one trial given precomputed listener arrays.

    Parameters
    ----------
    L1   : (2, n_obj)  -- [L(big), L(blue)]
    L2   : (2, n_obj)  -- [L(big blue), L(blue big)]
    alpha: float
    bias : float

    Returns
    -------
    float : P_inc(big_blue | referent 0)
    """
    eps = 1e-20
    L1_big     = L1[0]
    L1_blue    = L1[1]
    L2_bigblue = L2[0]
    L2_bluebig = L2[1]

    num_big  = jnp.power(jnp.clip(L1_big,  eps, 1.0), alpha)
    num_blue = jnp.power(jnp.clip(L1_blue, eps, 1.0), alpha)
    Z1       = jnp.clip(num_big + num_blue, eps)
    S1_big   = num_big  / Z1
    S1_blue  = num_blue / Z1

    num_bigblue  = jnp.power(jnp.clip(L2_bigblue, eps, 1.0), alpha)
    num_stop_big = jnp.power(jnp.clip(L1_big,     eps, 1.0), alpha)
    Z2_big            = jnp.clip(num_bigblue + num_stop_big, eps)
    S2_blue_given_big = num_bigblue / Z2_big

    num_bluebig   = jnp.power(jnp.clip(L2_bluebig, eps, 1.0), alpha)
    num_stop_blue = jnp.power(jnp.clip(L1_blue,    eps, 1.0), alpha)
    Z2_blue           = jnp.clip(num_bluebig + num_stop_blue, eps)
    S2_big_given_blue = num_bluebig / Z2_blue

    P_bigblue = S1_big  * S2_blue_given_big
    P_bluebig = S1_blue * S2_big_given_blue
    P_chain   = jnp.clip(jnp.stack([P_bigblue, P_bluebig], axis=-1), eps, 1.0)

    utt_cost = jnp.array([0.0, bias])
    util     = jnp.log(P_chain) - utt_cost
    S_inc    = jax.nn.softmax(util, axis=-1)
    return S_inc[0, 0]


# Vectorise over observations; alpha and bias remain scalar
jitted_inc_speaker_fast = jax.jit(
    jax.vmap(_inc_speaker_from_listeners, in_axes=(0, 0, None, None))
)


# Fast incremental path: precompute listeners outside MCMC loop.
# literal_listener_one_word / literal_listener_recursive depend only on
# (states, gamma), NOT on alpha or bias -- so they can be evaluated once
# before MCMC and passed as static data to every likelihood call.

_vmap_L1 = jax.vmap(
    lambda s, g: literal_listener_one_word(
        s, FIXED_COLOR_SEMVALUE, FIXED_WF, FIXED_K, g
    ),
    in_axes=(0, 0),
)
_vmap_L2 = jax.vmap(
    lambda s, g: literal_listener_recursive(
        2, s, FIXED_COLOR_SEMVALUE, FIXED_WF, FIXED_K, g
    ),
    in_axes=(0, 0),
)


@jax.jit
def precompute_listeners_all(states, gamma):
    """Precompute L1 and L2 listener arrays for all N trials (run once before MCMC).

    Returns
    -------
    L1_all : (N, 2, n_obj) -- one-word listener
    L2_all : (N, 2, n_obj) -- two-word listener
    """
    return _vmap_L1(states, gamma), _vmap_L2(states, gamma)


def _inc_speaker_from_listeners(L1, L2, alpha, bias):
    """Incremental speaker for one trial given precomputed listener arrays.

    Parameters
    ----------
    L1   : (2, n_obj)  -- [L(big), L(blue)]
    L2   : (2, n_obj)  -- [L(big blue), L(blue big)]
    alpha: float
    bias : float

    Returns
    -------
    float : P_inc(big_blue | referent 0)
    """
    eps = 1e-20
    L1_big     = L1[0]
    L1_blue    = L1[1]
    L2_bigblue = L2[0]
    L2_bluebig = L2[1]

    num_big  = jnp.power(jnp.clip(L1_big,  eps, 1.0), alpha)
    num_blue = jnp.power(jnp.clip(L1_blue, eps, 1.0), alpha)
    Z1       = jnp.clip(num_big + num_blue, eps)
    S1_big   = num_big  / Z1
    S1_blue  = num_blue / Z1

    num_bigblue  = jnp.power(jnp.clip(L2_bigblue, eps, 1.0), alpha)
    num_stop_big = jnp.power(jnp.clip(L1_big,     eps, 1.0), alpha)
    Z2_big            = jnp.clip(num_bigblue + num_stop_big, eps)
    S2_blue_given_big = num_bigblue / Z2_big

    num_bluebig   = jnp.power(jnp.clip(L2_bluebig, eps, 1.0), alpha)
    num_stop_blue = jnp.power(jnp.clip(L1_blue,    eps, 1.0), alpha)
    Z2_blue           = jnp.clip(num_bluebig + num_stop_blue, eps)
    S2_big_given_blue = num_bluebig / Z2_blue

    P_bigblue = S1_big  * S2_blue_given_big
    P_bluebig = S1_blue * S2_big_given_blue
    P_chain   = jnp.clip(jnp.stack([P_bigblue, P_bluebig], axis=-1), eps, 1.0)

    utt_cost = jnp.array([0.0, bias])
    util     = jnp.log(P_chain) - utt_cost
    S_inc    = jax.nn.softmax(util, axis=-1)
    return S_inc[0, 0]


# Vectorise over observations; alpha and bias remain scalar
jitted_inc_speaker_fast = jax.jit(
    jax.vmap(_inc_speaker_from_listeners, in_axes=(0, 0, None, None))
)


# Fast incremental path: precompute listeners outside MCMC loop.
# literal_listener_one_word / literal_listener_recursive depend only on
# (states, gamma), NOT on alpha or bias -- so they can be evaluated once
# before MCMC and passed as static data to every likelihood call.

_vmap_L1 = jax.vmap(
    lambda s, g: literal_listener_one_word(
        s, FIXED_COLOR_SEMVALUE, FIXED_WF, FIXED_K, g
    ),
    in_axes=(0, 0),
)
_vmap_L2 = jax.vmap(
    lambda s, g: literal_listener_recursive(
        2, s, FIXED_COLOR_SEMVALUE, FIXED_WF, FIXED_K, g
    ),
    in_axes=(0, 0),
)


@jax.jit
def precompute_listeners_all(states, gamma):
    """Precompute L1 and L2 listener arrays for all N trials (run once before MCMC).

    Returns
    -------
    L1_all : (N, 2, n_obj) -- one-word listener
    L2_all : (N, 2, n_obj) -- two-word listener
    """
    return _vmap_L1(states, gamma), _vmap_L2(states, gamma)


def _inc_speaker_from_listeners(L1, L2, alpha, bias):
    """Incremental speaker for one trial given precomputed listener arrays.

    Parameters
    ----------
    L1   : (2, n_obj)  -- [L(big), L(blue)]
    L2   : (2, n_obj)  -- [L(big blue), L(blue big)]
    alpha: float
    bias : float

    Returns
    -------
    float : P_inc(big_blue | referent 0)
    """
    eps = 1e-20
    L1_big     = L1[0]
    L1_blue    = L1[1]
    L2_bigblue = L2[0]
    L2_bluebig = L2[1]

    num_big  = jnp.power(jnp.clip(L1_big,  eps, 1.0), alpha)
    num_blue = jnp.power(jnp.clip(L1_blue, eps, 1.0), alpha)
    Z1       = jnp.clip(num_big + num_blue, eps)
    S1_big   = num_big  / Z1
    S1_blue  = num_blue / Z1

    num_bigblue  = jnp.power(jnp.clip(L2_bigblue, eps, 1.0), alpha)
    num_stop_big = jnp.power(jnp.clip(L1_big,     eps, 1.0), alpha)
    Z2_big            = jnp.clip(num_bigblue + num_stop_big, eps)
    S2_blue_given_big = num_bigblue / Z2_big

    num_bluebig   = jnp.power(jnp.clip(L2_bluebig, eps, 1.0), alpha)
    num_stop_blue = jnp.power(jnp.clip(L1_blue,    eps, 1.0), alpha)
    Z2_blue           = jnp.clip(num_bluebig + num_stop_blue, eps)
    S2_big_given_blue = num_bluebig / Z2_blue

    P_bigblue = S1_big  * S2_blue_given_big
    P_bluebig = S1_blue * S2_big_given_blue
    P_chain   = jnp.clip(jnp.stack([P_bigblue, P_bluebig], axis=-1), eps, 1.0)

    utt_cost = jnp.array([0.0, bias])
    util     = jnp.log(P_chain) - utt_cost
    S_inc    = jax.nn.softmax(util, axis=-1)
    return S_inc[0, 0]


# Vectorise over observations; alpha and bias remain scalar
jitted_inc_speaker_fast = jax.jit(
    jax.vmap(_inc_speaker_from_listeners, in_axes=(0, 0, None, None))
)

class ZOIB(dist.Distribution):
    arg_constraints = {
        "mu": constraints.unit_interval,
        "sigma": constraints.positive,
        "pi0": constraints.unit_interval,
        "pi1": constraints.unit_interval,
    }
    support = constraints.unit_interval

    def __init__(self, mu: jnp.ndarray, sigma: float, pi0: float, pi1: float, validate_args=None):
        self.mu = jnp.clip(mu, 1e-6, 1.0 - 1e-6)
        self.sigma = jnp.clip(sigma, 1e-6)
        self.pi0 = jnp.clip(pi0, 0.0, 0.49)
        self.pi1 = jnp.clip(pi1, 0.0, 0.49)
        self.cont_weight = jnp.clip(1.0 - self.pi0 - self.pi1, 1e-6, 1.0)
        concentration = jnp.clip(1.0 / (self.sigma**2 + 1e-6), 1e-3, 1e6)
        self.beta_a = jnp.clip(self.mu * concentration, 1e-3, 1e6)
        self.beta_b = jnp.clip((1.0 - self.mu) * concentration, 1e-3, 1e6)
        batch_shape = jnp.shape(self.mu)
        super().__init__(batch_shape=batch_shape, event_shape=(), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        key_comp, key_beta = random.split(key)
        mix_probs = jnp.stack(
            [
                jnp.full(self.batch_shape, self.pi0),
                jnp.full(self.batch_shape, self.pi1),
                jnp.full(self.batch_shape, self.cont_weight),
            ],
            axis=-1,
        )
        comp = dist.Categorical(probs=mix_probs).sample(key_comp, sample_shape=sample_shape)
        beta_sample = dist.Beta(self.beta_a, self.beta_b).sample(key_beta, sample_shape=sample_shape)
        return jnp.where(comp == 0, 0.0, jnp.where(comp == 1, 1.0, beta_sample))

    def log_prob(self, value):
        value = jnp.clip(value, 0.0, 1.0)
        eps = 1e-6  # must be > float32 machine epsilon (~1.2e-7) so 1.0 - eps != 1.0 in float32
        is_zero = jnp.isclose(value, 0.0)
        is_one = jnp.isclose(value, 1.0)
        beta_lp = dist.Beta(self.beta_a, self.beta_b).log_prob(jnp.clip(value, eps, 1.0 - eps))
        logp_cont = jnp.log(self.cont_weight + eps) + beta_lp
        logp = jnp.where(is_zero, jnp.log(self.pi0 + eps), logp_cont)
        logp = jnp.where(is_one, jnp.log(self.pi1 + eps), logp)
        return logp

def likelihood_gb_speaker(states=None, data=None, pi0: float=0.01, pi1: float=0.01, sharpness_idx=None, infer_gamma: bool = True):
    alpha = numpyro.sample("alpha", dist.HalfNormal(5.0))
    bias  = numpyro.sample("bias",  dist.HalfNormal(2.0))  # ordering cost: >0 prefers "big blue"
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))
    if infer_gamma:
        gamma_blurred = numpyro.sample("gamma_blurred", dist.HalfNormal(3.0))
        gamma_sharp   = numpyro.sample("gamma_sharp",   dist.HalfNormal(3.0))
    else:
        gamma_blurred = GAMMA_BLURRED
        gamma_sharp   = GAMMA_SHARP

    if sharpness_idx is None:
        sharpness_idx = jnp.zeros(len(states))
    gamma = jnp.where(sharpness_idx > 0.5, gamma_sharp, gamma_blurred)

    with numpyro.plate("data", len(states)):
        model_prob = jitted_global_speaker(states, alpha, bias, gamma)
        model_prob = jnp.clip(model_prob, 1e-6, 1 - 1e-6)
        if data is not None:
            data = jnp.clip(data, 0.0, 1.0)
        numpyro.sample("obs", ZOIB(model_prob, sigma, pi0, pi1), obs=data)

def likelihood_inc_speaker(states=None, data=None, pi0: float=0.01, pi1: float=0.01, sharpness_idx=None, infer_gamma: bool = True):
    alpha = numpyro.sample("alpha", dist.HalfNormal(5.0))
    bias  = numpyro.sample("bias",  dist.HalfNormal(2.0))  # ordering cost: >0 prefers "big blue"
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))
    if infer_gamma:
        gamma_blurred = numpyro.sample("gamma_blurred", dist.HalfNormal(3.0))
        gamma_sharp   = numpyro.sample("gamma_sharp",   dist.HalfNormal(3.0))
    else:
        gamma_blurred = GAMMA_BLURRED
        gamma_sharp   = GAMMA_SHARP

    if sharpness_idx is None:
        sharpness_idx = jnp.zeros(len(states))
    gamma = jnp.where(sharpness_idx > 0.5, gamma_sharp, gamma_blurred)

    with numpyro.plate("data", len(states)):
        model_prob = jitted_incremental_speaker(states, alpha, bias, gamma)
        model_prob = jnp.clip(model_prob, 1e-6, 1 - 1e-6)
        if data is not None:
            data = jnp.clip(data, 0.0, 1.0)
        numpyro.sample("obs", ZOIB(model_prob, sigma, pi0, pi1), obs=data)

# ══════════════════════════════════════════════════════════════════════════════
# Hierarchical (random participant intercepts) — Option A
# New functions only; all functions above are left intact.
# ══════════════════════════════════════════════════════════════════════════════




def likelihood_inc_speaker_fast(
    states=None,
    data=None,
    pi0=0.01,
    pi1=0.01,
    sharpness_idx=None,
    L1_all=None,
    L2_all=None,
):
    """Like likelihood_inc_speaker but uses precomputed L1/L2 arrays."""
    alpha = numpyro.sample("alpha", dist.HalfNormal(5.0))
    bias  = numpyro.sample("bias",  dist.HalfNormal(2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))

    with numpyro.plate("data", L1_all.shape[0]):
        model_prob = jitted_inc_speaker_fast(L1_all, L2_all, alpha, bias)
        model_prob = jnp.clip(model_prob, 1e-6, 1 - 1e-6)
        if data is not None:
            data = jnp.clip(data, 0.0, 1.0)
        numpyro.sample("obs", ZOIB(model_prob, sigma, pi0, pi1), obs=data)

def import_dataset_hier(file_path="../01-dataset/01-slider-data-preprocessed.csv"):
    """Extends import_dataset() with participant indices for hierarchical models.

    Returns
    -------
    states_train    : jnp.ndarray  (N, n_obj, 3)
    empirical_train : jnp.ndarray  (N,)  slider values in [0, 1]
    df              : pd.DataFrame full preprocessed data frame
    participant_idx : jnp.ndarray  int32 (N,) — 0-based participant index per obs
    n_participants  : int          number of unique participants
    """
    df = pd.read_csv(file_path)
    df = df[df["combination"] == "dimension_color"]
    df.reset_index(inplace=True, drop=True)
    df["states"] = df.apply(lambda row: encode_states(row), axis=1)
    df.prefer_first_1st = jnp.clip(df.prefer_first_1st.to_numpy(), 0, 100)
    df.prefer_first_1st = df.prefer_first_1st / 100
    train = df
    states_train    = jnp.stack([cell for cell in train.states])
    empirical_train = jnp.array(train.prefer_first_1st.to_numpy())
    # Deterministic 0-based participant index sorted by participant id
    participant_ids_np = train["id"].to_numpy()
    unique_ids         = np.unique(participant_ids_np)
    id_to_idx          = {int(pid): i for i, pid in enumerate(unique_ids)}
    participant_idx    = jnp.array(
        [id_to_idx[int(pid)] for pid in participant_ids_np], dtype=jnp.int32
    )
    n_participants = int(len(unique_ids))
    return states_train, empirical_train, df, participant_idx, n_participants


def likelihood_gb_speaker_hier(
    states=None, data=None,
    pi0: float = 0.01, pi1: float = 0.01,
    sharpness_idx=None,
    participant_idx=None, n_participants: int = 1,
    infer_gamma: bool = True,
):
    """Global speaker with per-participant additive intercepts (Option A).

    Population parameters (identical priors to likelihood_gb_speaker):
        alpha ~ HalfNormal(5)
        bias  ~ HalfNormal(2)
        sigma ~ HalfNormal(0.3)
    Participant-level random intercepts:
        tau     ~ HalfNormal(0.2)        # SD of baseline shifts
        delta_i ~ Normal(0, tau)         # intercept for participant i
    Observation model:
        mu_j = clip(RSA(alpha, bias, s_j) + delta[participant_idx[j]], 1e-6, 1-1e-6)
        y_j  ~ ZOIB(mu_j, sigma, pi0, pi1)
    """
    alpha = numpyro.sample("alpha", dist.HalfNormal(5.0))
    bias  = numpyro.sample("bias",  dist.HalfNormal(2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))
    tau   = numpyro.sample("tau",   dist.HalfNormal(0.2))
    if infer_gamma:
        gamma_blurred = numpyro.sample("gamma_blurred", dist.HalfNormal(3.0))
        gamma_sharp   = numpyro.sample("gamma_sharp",   dist.HalfNormal(3.0))
    else:
        gamma_blurred = GAMMA_BLURRED
        gamma_sharp   = GAMMA_SHARP

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    if sharpness_idx is None:
        sharpness_idx = jnp.zeros(len(states))
    gamma = jnp.where(sharpness_idx > 0.5, gamma_sharp, gamma_blurred)

    with numpyro.plate("data", len(states)):
        rsa_prob    = jitted_global_speaker(states, alpha, bias, gamma)
        model_prob  = jnp.clip(rsa_prob + delta[participant_idx], 1e-6, 1 - 1e-6)
        if data is not None:
            data = jnp.clip(data, 0.0, 1.0)
        numpyro.sample("obs", ZOIB(model_prob, sigma, pi0, pi1), obs=data)


def likelihood_inc_speaker_hier(
    states=None, data=None,
    pi0: float = 0.01, pi1: float = 0.01,
    sharpness_idx=None,
    participant_idx=None, n_participants: int = 1,
    infer_gamma: bool = True,
):
    """Incremental speaker with per-participant additive intercepts (Option A).

    Same structure as likelihood_gb_speaker_hier but uses the incremental RSA.
    """
    alpha = numpyro.sample("alpha", dist.HalfNormal(5.0))
    bias  = numpyro.sample("bias",  dist.HalfNormal(2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))
    tau   = numpyro.sample("tau",   dist.HalfNormal(0.2))
    if infer_gamma:
        gamma_blurred = numpyro.sample("gamma_blurred", dist.HalfNormal(3.0))
        gamma_sharp   = numpyro.sample("gamma_sharp",   dist.HalfNormal(3.0))
    else:
        gamma_blurred = GAMMA_BLURRED
        gamma_sharp   = GAMMA_SHARP

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    if sharpness_idx is None:
        sharpness_idx = jnp.zeros(len(states))
    gamma = jnp.where(sharpness_idx > 0.5, gamma_sharp, gamma_blurred)

    with numpyro.plate("data", len(states)):
        rsa_prob    = jitted_incremental_speaker(states, alpha, bias, gamma)
        model_prob  = jnp.clip(rsa_prob + delta[participant_idx], 1e-6, 1 - 1e-6)
        if data is not None:
            data = jnp.clip(data, 0.0, 1.0)
        numpyro.sample("obs", ZOIB(model_prob, sigma, pi0, pi1), obs=data)





def likelihood_inc_speaker_hier_fast(
    states=None,
    data=None,
    pi0=0.01,
    pi1=0.01,
    sharpness_idx=None,
    participant_idx=None,
    n_participants=1,
    L1_all=None,
    L2_all=None,
):
    """Hierarchical incremental speaker using precomputed L1/L2 arrays."""
    alpha = numpyro.sample("alpha", dist.HalfNormal(5.0))
    bias  = numpyro.sample("bias",  dist.HalfNormal(2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))
    tau   = numpyro.sample("tau",   dist.HalfNormal(0.2))

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    with numpyro.plate("data", L1_all.shape[0]):
        rsa_prob   = jitted_inc_speaker_fast(L1_all, L2_all, alpha, bias)
        model_prob = jnp.clip(rsa_prob + delta[participant_idx], 1e-6, 1 - 1e-6)
        if data is not None:
            data = jnp.clip(data, 0.0, 1.0)
        numpyro.sample("obs", ZOIB(model_prob, sigma, pi0, pi1), obs=data)

def run_inference_hier(
    speaker_type: str = "global",
    num_samples: int = 1000,
    num_warmup: int = 1000,
    num_chains: int = 4,
    infer_gamma: bool = True,
):
    """Run MCMC for the hierarchical (random participant intercept) speaker model.

    Saves results to
        ./inference_data/mcmc_results_{speaker_type}_speaker_hier_warmup{W}_samples{S}_chains{C}.nc
    """
    states_train, empirical_train, df, participant_idx, n_participants = import_dataset_hier()
    print(f"Hierarchical model: {n_participants} participants, {len(states_train)} observations")

    empirical_train_np = np.asarray(empirical_train)
    pi0 = float(np.mean(np.isclose(empirical_train_np, 0.0)))
    pi1 = float(np.mean(np.isclose(empirical_train_np, 1.0)))
    if (pi0 + pi1) >= 0.95:
        raise ValueError(f"Boundary masses too large for ZOIB: pi0+pi1={pi0+pi1:.3f}")

    sharpness_idx = jnp.array((df["sharpness"] == "sharp").astype(float).to_numpy())

    # Precompile JIT paths (gamma is now inferred; compile with dummy=1.0)
    gamma_dummy = jnp.ones(states_train.shape[0])
    _ = jitted_global_speaker(states_train, 2.0, 2.0, gamma_dummy).block_until_ready()
    _ = jitted_incremental_speaker(states_train, 2.0, 2.0, gamma_dummy).block_until_ready()

    gamma_tag = "infer_gamma" if infer_gamma else "fix_gamma"
    output_file_name = (
        f"./inference_data/mcmc_results_{speaker_type}_speaker_hier_{gamma_tag}"
        f"_warmup{num_warmup}_samples{num_samples}_chains{num_chains}.nc"
    )
    print(f"Output file: {output_file_name}")
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
        print(f"Removed existing file: {output_file_name}")

    rng_key = random.PRNGKey(11)
    rng_key, rng_key_ = random.split(rng_key)

    extra_args_hier: tuple = ()
    if speaker_type == "global":
        _base_model = likelihood_gb_speaker_hier
    elif speaker_type == "incremental":
        _base_model = likelihood_inc_speaker_hier
    else:
        raise ValueError("Invalid speaker type. Choose 'global' or 'incremental'.")
    model = functools.partial(_base_model, infer_gamma=infer_gamma)

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
        pi0, pi1, sharpness_idx, participant_idx, n_participants,
        *extra_args_hier,
    )
    mcmc.print_summary(exclude_deterministic=False)

    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model, posterior_samples)(
        PRNGKey(1), states_train, None,
        pi0, pi1, sharpness_idx, participant_idx, n_participants,
        *extra_args_hier,
    )
    prior = Predictive(model, num_samples=1000)(
        PRNGKey(2), states_train, None,
        pi0, pi1, sharpness_idx, participant_idx, n_participants,
        *extra_args_hier,
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


def run_inference(
    speaker_type: str = "global",
    num_samples: int = 1000,
    num_warmup: int = 1000,
    num_chains: int = 4,
    infer_gamma: bool = True,
):
    # Import the dataset
    states_train, empirical_train, df = import_dataset()

    # Precompute boundary masses for ZOIB outside MCMC
    empirical_train_np = np.asarray(empirical_train)
    pi0 = float(np.mean(np.isclose(empirical_train_np, 0.0)))
    pi1 = float(np.mean(np.isclose(empirical_train_np, 1.0)))
    if (pi0 + pi1) >= 0.95:
        raise ValueError(f"Boundary masses too large for ZOIB: pi0+pi1={pi0+pi1:.3f}")

    sharpness_idx = jnp.array((df["sharpness"] == "sharp").astype(float).to_numpy())

    # Precompile JIT paths before MCMC (gamma is now inferred; compile with dummy=1.0)
    gamma_dummy = jnp.ones(states_train.shape[0])
    _ = jitted_global_speaker(states_train, 2.0, 2.0, gamma_dummy).block_until_ready()
    _ = jitted_incremental_speaker(states_train, 2.0, 2.0, gamma_dummy).block_until_ready()

    gamma_tag = "infer_gamma" if infer_gamma else "fix_gamma"
    output_file_name = (
        f"./inference_data/mcmc_results_{speaker_type}_speaker_{gamma_tag}"
        f"_warmup{num_warmup}_samples{num_samples}_chains{num_chains}.nc"
    )
    print(f"Output file: {output_file_name}")

    # Remove existing file so we never silently overwrite a stale result
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
        print(f"Removed existing file: {output_file_name}")

    # Define a random key
    rng_key = random.PRNGKey(11)
    rng_key, rng_key_ = random.split(rng_key)
    extra_args: tuple = ()
    if speaker_type == "global":
        _base_model = likelihood_gb_speaker
    elif speaker_type == "incremental":
        _base_model = likelihood_inc_speaker
    else:
        raise ValueError("Invalid speaker type. Choose 'global' or 'incremental'.")
    model = functools.partial(_base_model, infer_gamma=infer_gamma)

    # Define and run MCMC
    kernel = NUTS(model, dense_mass=True, max_tree_depth=8, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method="vectorized")
    mcmc.run(rng_key_, states_train, empirical_train, pi0, pi1, sharpness_idx, *extra_args)

    # Print the summary of the posterior distribution
    mcmc.print_summary()

    # Get the MCMC samples and convert to a numpyro ArviZ InferenceData object
    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model, posterior_samples)(
        PRNGKey(1), states_train, None, pi0, pi1, sharpness_idx, *extra_args
    )
    prior = Predictive(model, num_samples=1000)(
        PRNGKey(2), states_train, None, pi0, pi1, sharpness_idx, *extra_args
    )

    N = states_train.shape[0]  # 3196

    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords={"states": np.arange(N)},
        dims={"obs": ["states"]},
    )

    # Write the inference data to a netcdf file
    az.to_netcdf(numpyro_data, output_file_name)
    assert os.path.exists(output_file_name), f"Save failed: {output_file_name} not found"
    size_mb = os.path.getsize(output_file_name) / 1024 / 1024
    print(f"Saved: {output_file_name}  ({size_mb:.1f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speaker inference with NumPyro.")
    parser.add_argument("--speaker_type", type=str, choices=["global", "incremental"], default="global",
                        help="Choose the speaker model type.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of posterior samples.")
    parser.add_argument("--num_warmup", type=int, default=500, help="Number of warm-up iterations.")
    parser.add_argument("--num_chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--test", action="store_true", help="Run test function and exit.")
    parser.add_argument(
        "--hierarchical", action="store_true",
        help="Run hierarchical model with random participant intercepts (Option A)."
    )
    parser.add_argument(
        "--infer_gamma", action=argparse.BooleanOptionalAction, default=True,
        help="Infer gamma_blurred/gamma_sharp (default). Use --no_infer_gamma to fix them."
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
            infer_gamma=args.infer_gamma,
        )
    else:
        run_inference(
            speaker_type=args.speaker_type,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            infer_gamma=args.infer_gamma,
        )