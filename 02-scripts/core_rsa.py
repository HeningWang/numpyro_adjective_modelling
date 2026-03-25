import os

import jax
import jax.numpy as jnp
from jax import random, vmap, lax
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import scipy

import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from sklearn.model_selection import train_test_split
numpyro.set_platform("cpu")

print(jax.__version__)
jax.devices()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalize(arr, axis=1):
    return arr / jnp.sum(arr, axis=axis, keepdims=True)

def link_function(x, param=1):
    return 1 / (1 + jnp.exp(param * -(x - 0.5)))

def link_logit(p, s):
    x0 = 1 / (jnp.exp(1 / (-2 * s)) + 1)
    xtrans = p * (x0 - (1 - x0)) + (1 - x0)
    return s * -jnp.log((1 / xtrans) - 1) + 0.5


# ---------------------------------------------------------------------------
# Threshold: mid-range max-min threshold  θ_k(C)
# Replaces the old sample-based threshold functions.
# ---------------------------------------------------------------------------

def get_midrange_extrema_from_context(states, state_prior, q_low=0.2, q_high=0.8):
    """
    Compute mid-range extrema x_min^mid(C), x_max^mid(C).

    Args:
        states:      (n_obj, 3)
        state_prior: (n_obj,)  — contextual distribution C(s), must sum to 1
        q_low, q_high: quantile bounds for the mid-range band

    Returns:
        x_min_mid, x_max_mid: scalars
    """
    sizes = states[:, 0]
    idx = jnp.argsort(sizes)
    sizes_sorted = sizes[idx]
    prior_sorted = state_prior[idx]
    cdf = jnp.cumsum(prior_sorted)

    i_low  = jnp.argmax(cdf >= q_low)
    i_high = jnp.argmax(cdf >= q_high)

    return sizes_sorted[i_low], sizes_sorted[i_high]


def get_threshold_k_midrange_jax(states, state_prior, k=0.5, q_low=0.2, q_high=0.8):
    """
    θ_k(C) = x_max^mid(C) - k * (x_max^mid(C) - x_min^mid(C))

    Args:
        states:      (n_obj, 3)
        state_prior: (n_obj,)
        k:           interpolation parameter in [0, 1]

    Returns:
        scalar threshold
    """
    x_min_mid, x_max_mid = get_midrange_extrema_from_context(
        states, state_prior, q_low=q_low, q_high=q_high
    )
    return x_max_mid - k * (x_max_mid - x_min_mid)


# ---------------------------------------------------------------------------
# Atomic semantics
# ---------------------------------------------------------------------------

def get_size_semval(size, threshold, wf):
    """P(size adjective applies | object size, threshold, blur wf)."""
    eps = 1e-8
    denom = wf * jnp.sqrt(size ** 2 + threshold ** 2 + eps)
    return dist.Normal(0.0, 1.0).cdf((size - threshold) / denom)


def adjMeaning(word, states, current_state_prior, color_semvalue=0.8, wf=0.6, k=0.5):
    """
    Deterministic lexical meaning ⟦w⟧_C(s) for one adjective w across all objects.

    word == 0 → size ("big")
    word == 1 → color ("blue")

    Returns a vector of shape (n_obj,) with semantic values in [0, 1].
    JAX-safe (lax.cond), suitable for vmap/jit.
    """
    def color_branch(_):
        x_color = states[:, 1]  # 0/1 feature
        return x_color * color_semvalue + (1.0 - x_color) * (1.0 - color_semvalue)

    def size_branch(_):
        threshold = get_threshold_k_midrange_jax(states, current_state_prior, k)
        eps = 1e-8
        denom = wf * jnp.sqrt(states[:, 0] ** 2 + threshold ** 2 + eps)
        return dist.Normal(0.0, 1.0).cdf((states[:, 0] - threshold) / denom)

    return lax.cond(word == 1, color_branch, size_branch, operand=None)


# ---------------------------------------------------------------------------
# Literal listeners
# ---------------------------------------------------------------------------

def literal_listener_one_word(states, color_semvalue=0.90, wf=1.0, k=0.5,
                               form_semvalue=None):
    """
    Literal listener for single-word utterances.

    Returns (2, n_obj): row 0 = L(s | 'big'), row 1 = L(s | 'blue').
    `form_semvalue` is accepted but ignored (form not used).
    """
    n_obj = states.shape[0]
    prior0 = jnp.ones(n_obj) / n_obj
    tokens = jnp.array([0, 1], dtype=jnp.int32)  # 0=big, 1=blue

    def update_token(token):
        m = adjMeaning(token, states, prior0, color_semvalue, wf, k)
        unnorm = prior0 * m
        return unnorm / jnp.clip(unnorm.sum(), 1e-20)

    return jax.vmap(update_token)(tokens)  # (2, n_obj)


def literal_listener_recursive(word_length, states, color_semvalue=0.90, wf=1.0,
                                k=0.5, form_semvalue=None, sample_based=None):
    """
    Incremental literal listener using backward functional semantics.

    Utterances:
      row 0: "big blue"  encoded as tokens [0, 1]
      row 1: "blue big"  encoded as tokens [1, 0]

    Returns (2, n_obj).
    `form_semvalue` and `sample_based` are accepted but ignored.
    """
    assert word_length == 2, "Only 2-word utterances are supported."
    n_obj = states.shape[0]
    prior0 = jnp.ones(n_obj) / n_obj

    utterances = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)

    def update_one_token(prior, token):
        m = adjMeaning(token, states, prior, color_semvalue, wf, k)
        unnorm = prior * m
        return unnorm / jnp.clip(unnorm.sum(), 1e-20), None

    def interpret_utterance(tokens):
        tokens_rev = tokens[::-1]  # right-to-left backward composition
        final_belief, _ = lax.scan(update_one_token, prior0, tokens_rev)
        return final_belief  # (n_obj,)

    return jax.vmap(interpret_utterance)(utterances)  # (2, n_obj)


def literal_listener_recursive_frozen(word_length, states, color_semvalue=0.90,
                                       wf=1.0, k=0.5):
    """
    Same as literal_listener_recursive but with STATIC size semantics.

    Size threshold is always computed from the uniform prior, not from
    the running posterior. This removes context-recursive threshold
    adaptation while preserving incremental belief update.

    Returns (2, n_obj).
    """
    assert word_length == 2, "Only 2-word utterances are supported."
    n_obj = states.shape[0]
    prior0 = jnp.ones(n_obj) / n_obj

    utterances = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)

    def update_one_token(prior, token):
        # Size semantics always use uniform prior (frozen threshold)
        m = adjMeaning(token, states, prior0, color_semvalue, wf, k)
        unnorm = prior * m
        return unnorm / jnp.clip(unnorm.sum(), 1e-20), None

    def interpret_utterance(tokens):
        tokens_rev = tokens[::-1]
        final_belief, _ = lax.scan(update_one_token, prior0, tokens_rev)
        return final_belief

    return jax.vmap(interpret_utterance)(utterances)  # (2, n_obj)


# ---------------------------------------------------------------------------
# Speakers
# ---------------------------------------------------------------------------

def speaker_one_word(states, alpha=1, bias=0, color_semvalue=0.98,
                     form_semvalue=None, wf=0.6, k=0.5):
    """One-word speaker. Returns (n_obj, 2)."""
    listener = literal_listener_one_word(states, color_semvalue, wf, k)
    bias_weights = jnp.array([0.0, bias])
    util_speaker = jnp.log(jnp.clip(listener.T, 1e-20, 1.0)) - bias_weights
    return jax.nn.softmax(alpha * util_speaker, axis=-1)


def speaker_recursive(word_length, states, alpha=1.0, bias=0.0,
                      color_semvalue=0.98, form_semvalue=None, wf=0.6, k=0.5):
    """
    Truly incremental RSA speaker (chain-rule over tokens).

    Returns (n_obj, 2):
      col 0: P_inc("big blue"  | s_j)
      col 1: P_inc("blue big"  | s_j)

    `form_semvalue` is accepted but ignored.
    """
    assert word_length == 2

    # Literal listeners
    L1 = literal_listener_one_word(states, color_semvalue, wf, k)   # (2, n_obj)
    L2 = literal_listener_recursive(word_length, states, color_semvalue, wf, k)  # (2, n_obj)

    L1_T = L1.T  # (n_obj, 2): col 0 = L(s|big), col 1 = L(s|blue)
    L2_T = L2.T  # (n_obj, 2): col 0 = L(s|big blue), col 1 = L(s|blue big)

    L1_big  = L1_T[:, 0]
    L1_blue = L1_T[:, 1]
    L2_bigblue = L2_T[:, 0]
    L2_bluebig = L2_T[:, 1]

    eps = 1e-20

    # Step 1: choose first token
    num_big  = jnp.power(jnp.clip(L1_big,  eps, 1.0), alpha)
    num_blue = jnp.power(jnp.clip(L1_blue, eps, 1.0), alpha)
    Z1 = jnp.clip(num_big + num_blue, eps)
    S1_big  = num_big  / Z1
    S1_blue = num_blue / Z1

    # Step 2a: given prefix 'big', choose stop-or-blue
    num_bigblue  = jnp.power(jnp.clip(L2_bigblue, eps, 1.0), alpha)
    num_stop_big = jnp.power(jnp.clip(L1_big,     eps, 1.0), alpha)
    S2_blue_given_big = num_bigblue / jnp.clip(num_bigblue + num_stop_big, eps)

    # Step 2b: given prefix 'blue', choose stop-or-big
    num_bluebig   = jnp.power(jnp.clip(L2_bluebig, eps, 1.0), alpha)
    num_stop_blue = jnp.power(jnp.clip(L1_blue,    eps, 1.0), alpha)
    S2_big_given_blue = num_bluebig / jnp.clip(num_bluebig + num_stop_blue, eps)

    # Path probabilities (chain rule)
    P_chain_bigblue = S1_big  * S2_blue_given_big   # (n_obj,)
    P_chain_bluebig = S1_blue * S2_big_given_blue   # (n_obj,)

    P_chain = jnp.clip(
        jnp.stack([P_chain_bigblue, P_chain_bluebig], axis=-1), eps, 1.0
    )  # (n_obj, 2)

    # Utterance-level cost (bias against "blue big")
    utt_cost = jnp.array([0.0, bias])
    util = jnp.log(P_chain) - utt_cost
    return jax.nn.softmax(util, axis=-1)  # (n_obj, 2)


def global_speaker(states, alpha=1.0, bias=0.0, color_semvalue=0.98,
                   form_semvalue=None, wf=0.6, k=0.5):
    """
    Global (non-incremental) RSA speaker.

    Returns (n_obj, 2):
      col 0: P_global("big blue"  | s_j)
      col 1: P_global("blue big"  | s_j)

    `form_semvalue` is accepted but ignored.
    """
    listener = literal_listener_recursive(2, states, color_semvalue, wf, k)  # (2, n_obj)
    eps = 1e-20
    utt_cost = jnp.array([0.0, bias])
    util_speaker = jnp.log(jnp.clip(listener.T, eps, 1.0)) - utt_cost  # (n_obj, 2)
    return jax.nn.softmax(alpha * util_speaker, axis=-1)  # (n_obj, 2)


def speaker_recursive_frozen(word_length, states, alpha=1.0, bias=0.0,
                              color_semvalue=0.98, form_semvalue=None,
                              wf=0.6, k=0.5):
    """
    Incremental RSA speaker with STATIC (frozen) size semantics.

    Same chain-rule as speaker_recursive, but uses
    literal_listener_recursive_frozen for 2-word listeners.

    Returns (n_obj, 2).
    """
    assert word_length == 2

    L1 = literal_listener_one_word(states, color_semvalue, wf, k)
    L2 = literal_listener_recursive_frozen(word_length, states, color_semvalue, wf, k)

    L1_T = L1.T
    L2_T = L2.T

    L1_big  = L1_T[:, 0]
    L1_blue = L1_T[:, 1]
    L2_bigblue = L2_T[:, 0]
    L2_bluebig = L2_T[:, 1]

    eps = 1e-20

    num_big  = jnp.power(jnp.clip(L1_big,  eps, 1.0), alpha)
    num_blue = jnp.power(jnp.clip(L1_blue, eps, 1.0), alpha)
    Z1 = jnp.clip(num_big + num_blue, eps)
    S1_big  = num_big  / Z1
    S1_blue = num_blue / Z1

    num_bigblue  = jnp.power(jnp.clip(L2_bigblue, eps, 1.0), alpha)
    num_stop_big = jnp.power(jnp.clip(L1_big,     eps, 1.0), alpha)
    S2_blue_given_big = num_bigblue / jnp.clip(num_bigblue + num_stop_big, eps)

    num_bluebig   = jnp.power(jnp.clip(L2_bluebig, eps, 1.0), alpha)
    num_stop_blue = jnp.power(jnp.clip(L1_blue,    eps, 1.0), alpha)
    S2_big_given_blue = num_bluebig / jnp.clip(num_bluebig + num_stop_blue, eps)

    P_chain_bigblue = S1_big  * S2_blue_given_big
    P_chain_bluebig = S1_blue * S2_big_given_blue

    P_chain = jnp.clip(
        jnp.stack([P_chain_bigblue, P_chain_bluebig], axis=-1), eps, 1.0
    )

    utt_cost = jnp.array([0.0, bias])
    util = jnp.log(P_chain) - utt_cost
    return jax.nn.softmax(util, axis=-1)  # (n_obj, 2)


def global_speaker_static(states, alpha=1.0, bias=0.0, color_semvalue=0.98,
                           form_semvalue=None, wf=0.6, k=0.5):
    """
    Global RSA speaker with STATIC (frozen) size semantics.

    Same as global_speaker, but uses literal_listener_recursive_frozen.

    Returns (n_obj, 2).
    """
    listener = literal_listener_recursive_frozen(2, states, color_semvalue, wf, k)
    eps = 1e-20
    utt_cost = jnp.array([0.0, bias])
    util_speaker = jnp.log(jnp.clip(listener.T, eps, 1.0)) - utt_cost
    return jax.nn.softmax(alpha * util_speaker, axis=-1)  # (n_obj, 2)


# ---------------------------------------------------------------------------
# Pragmatic listener  L1(referent | utterance, context)
# This is the communicative success measure (Option B).
# Returns (2, n_obj):
#   row 0: L1(s | "big blue")
#   row 1: L1(s | "blue big")
# ---------------------------------------------------------------------------

def pragmatic_listener(states, alpha=1, bias=0, color_semvalue=0.98,
                       form_semvalue=None, wf=0.6, k=0.5,
                       speaker="global_speaker", word_length=2):
    """
    Pragmatic listener via Bayes' rule over speaker distribution.

    L1(r | u) ∝ S1(u | r) * P(r)   [uniform prior over referents]

    Returns (2, n_obj):
      result[0, 0] = L1(referent | "big blue")  ← communicative success (Option B)
      result[1, 0] = L1(referent | "blue big")
    """
    prior_probs = jnp.ones((2, states.shape[0]))  # uniform (2, n_obj)

    if speaker == "global_speaker":
        softmax_result = global_speaker(states, alpha, bias, color_semvalue,
                                        form_semvalue, wf, k)
    elif speaker == "global_speaker_static":
        softmax_result = global_speaker_static(states, alpha, bias, color_semvalue,
                                                form_semvalue, wf, k)
    elif speaker == "incremental_speaker_static":
        softmax_result = speaker_recursive_frozen(word_length, states, alpha, bias,
                                                   color_semvalue, form_semvalue, wf, k)
    else:  # "incremental_speaker"
        softmax_result = speaker_recursive(word_length, states, alpha, bias,
                                           color_semvalue, form_semvalue, wf, k)

    # Transpose to (2, n_obj), then normalize over objects for each utterance
    return normalize(jnp.transpose(softmax_result) * prior_probs)  # (2, n_obj)


# ---------------------------------------------------------------------------
# Dataset helpers  (encode_states index already fixed: +5/+11/+17)
# ---------------------------------------------------------------------------

def encode_states(line):
    states = []
    for i in range(6):
        color = 1 if line.iloc[11 + i] == "blue" else 0
        form  = 1 if line.iloc[17 + i] == "circle" else 0
        new_obj = (line.iloc[5 + i], color, form)
        states.append(new_obj)
    return jnp.array(states)


def import_dataset(file_path="../01-dataset/01-slider-data-preprocessed.csv"):
    df = pd.read_csv(file_path)
    df = df[df['combination'] == 'dimension_color']
    df.reset_index(inplace=True, drop=True)
    df["states"] = df.apply(lambda row: encode_states(row), axis=1)
    df.prefer_first_1st = jnp.clip(df.prefer_first_1st.to_numpy(), 0, 100)
    df.prefer_first_1st = df.prefer_first_1st / 100
    train = df
    states_train   = jnp.stack([cell for cell in train.states])
    empirical_train = jnp.array(train.prefer_first_1st.to_numpy())
    return states_train, empirical_train, df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_core_rsa():
    file_path = "../01-dataset/01-slider-data-preprocessed.csv"
    df = pd.read_csv(file_path)
    df = df[df['combination'] == 'dimension_color']
    df.reset_index(inplace=True, drop=True)
    df_experiment = df.copy()
    df_experiment["states"] = df_experiment.apply(lambda row: encode_states(row), axis=1)
    df_experiment.prefer_first_1st = jnp.clip(df_experiment.prefer_first_1st.to_numpy(), 0, 100)
    df_experiment.prefer_first_1st = df_experiment.prefer_first_1st / 100
    print(df_experiment.prefer_first_1st.describe())

    index = 15
    states_manuell = jnp.array([[10., 1., 1.],
                                 [10., 1., 1.],
                                 [ 3., 1., 1.],
                                 [ 3., 1., 0.],
                                 [ 3., 1., 0.],
                                 [ 1., 0., 1.]], dtype=jnp.float32)
    states_example = states_manuell
    condition   = df_experiment.iloc[index, df_experiment.columns.get_loc("conditions")]
    distribution = df_experiment.iloc[index, df_experiment.columns.get_loc("sharpness")]
    preference  = df_experiment.iloc[index, df_experiment.columns.get_loc("prefer_first_1st")]
    print(states_example)
    print(condition + " " + distribution)
    print(preference)
    print(f"literal listener one word:      {literal_listener_one_word(states_example)}")
    print(f"literal listener two words:     {literal_listener_recursive(2, states_example)}")
    print(f"speaker one word:               {speaker_one_word(states_example)}")
    print(f"speaker two words global:       {global_speaker(states_example)}")
    print("________________________________________")
    print(f"speaker two words incremental:  {speaker_recursive(2, states_example)}")
    print("________________________________________")
    print(f"pragmatic listener (global):    {pragmatic_listener(states_example)}")
    print("________________________________________")
    print(f"pragmatic listener (incremental): {pragmatic_listener(states_example, speaker='incremental_speaker')}")


def test_threshold():
    states_train, empirical_train, df = import_dataset()
    states_example = states_train[46]
    n_obj = states_example.shape[0]
    prior = jnp.ones(n_obj) / n_obj
    threshold = get_threshold_k_midrange_jax(states_example, prior)
    print(f"midrange threshold: {threshold}")


# ---------------------------------------------------------------------------
# Legacy inference model (kept for reference; uses speaker_recursive API)
# ---------------------------------------------------------------------------

vectorized_speaker = jax.vmap(
    speaker_recursive,
    in_axes=(None, 0, None, None, None, None, None, None)
)

def model_inc_utt_parallel_normal(states=None, data=None):
    alpha        = numpyro.sample("alpha",        dist.HalfNormal(5))
    color_semvalue = numpyro.sample("color_semvalue", dist.Uniform(0.5, 1))
    k            = numpyro.sample("k",            dist.Uniform(0, 1))
    wf           = numpyro.sample("wf",           dist.Uniform(0, 1))
    bias         = numpyro.sample("bias",         dist.HalfNormal(5))
    steepness    = numpyro.sample("steepness",    dist.HalfNormal(0.5))
    sigma        = numpyro.sample("sigma",        dist.Uniform(0, 0.1))

    with numpyro.plate("data", len(states)):
        model_prob     = vectorized_speaker(2, states, alpha, bias, color_semvalue, None, wf, k)
        slider_predict = jax.vmap(link_logit, in_axes=(0, None))(model_prob[:, 0, 0], steepness)
        slider_predict = jnp.clip(slider_predict, 1e-5, 1 - 1e-5)
        if data is not None:
            data = jnp.clip(data, 1e-5, 1 - 1e-5)
        numpyro.sample("obs", dist.TruncatedNormal(
            slider_predict, sigma, low=1e-5, high=1 - 1e-5), obs=data)


if __name__ == "__main__":
    test_core_rsa()
    # test_threshold()
