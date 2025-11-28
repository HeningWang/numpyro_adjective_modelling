import argparse
import pandas as pd
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict, List, Any, Sequence, Callable
from jax import jit, vmap
import jax
from jax import lax
from jax import random
from jax.random import PRNGKey, split
from functools import partial
import matplotlib.pyplot as plt
import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro import param, sample
from numpyro.distributions import constraints, HalfNormal, Normal, Uniform

from numpyro import handlers
from numpyro.infer import MCMC, NUTS, HMC, MixedHMC
from numpyro.infer import Predictive
from sklearn.model_selection import train_test_split
from helper import import_dataset, normalize, build_utterance_prior_jax
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)
print(jax.__version__)
print(jax.devices())

import arviz as az

# ========================
# Global Variables (Setup)
# ========================
utterance_list = import_dataset()["unique_utterances"]  # shape (U,3)

LM_PRIOR_15 = jnp.array([
    0.1669514, 0.16733019, 0.12160929, 0.11005973, 0.09253279,
    0.07532827, 0.02494562, 0.03780574, 0.05690099, 0.02470998,
    0.02651604, 0.01232579, 0.03122547, 0.0363892, 0.01536951
])

# ========================
def utterance_cost_jax(beta: float = 1.0) -> jnp.ndarray:
    """
    Utterance-level cost derived from an external LM prior.
    
    Args:
        beta: temperature parameter controlling strength of LM cost.
              beta = 0 -> uniform (no cost)
              beta = 1 -> original LM prior
              beta > 1 -> sharper LM preferences
    
    Returns:
        costs: jnp.ndarray of shape (15,) with utterance costs
    """

    eps = 1e-12  # numerical stability

    # Temperature-scaled prior: P_beta(u) ∝ P_LM(u)^beta
    scaled_prior = jnp.power(jnp.clip(LM_PRIOR_15, eps), beta)
    scaled_prior = scaled_prior / jnp.sum(scaled_prior)

    # Cost = negative log probability
    costs = -jnp.log(jnp.clip(scaled_prior, eps))

    return costs

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
) -> jnp.ndarray:
    """
    Graded size semantics with mid-range threshold θ_k(C) and noise w_f.

    ⟦size⟧_C(s) = 1 - Φ( (x - θ_k) / (w_f * sqrt(x^2 + θ_k^2)) )
    """
    # Avoid zero division; eps not strictly necessary but helps numerics.
    eps = 1e-8
    denom = w_f * jnp.sqrt(size**2 + theta_k**2 + eps)
    z = (size - theta_k) / denom
    # Standard normal CDF at z
    return 1.0 - dist.Normal(0.0, 1.0).cdf(z)

def _size_meaning_vector(
    states: jnp.ndarray,   # (n_obj, 3)
    prior: jnp.ndarray,    # (n_obj,)
    k: float,
    wf: float,
    q_low: float = 0.2,
    q_high: float = 0.8,
) -> jnp.ndarray:
    """Compute the size meaning vector for all objects in the current context.

    Given a contextual prior C over states ("prior"), we first compute the
    mid-range extrema x_min^mid(C), x_max^mid(C) and the corresponding
    threshold θ_k(C), and then apply the graded size semantics with slack wf
    to each object size.
    """
    # Extract sizes (first dimension of the state tuple)
    sizes = states[:, 0]  # (n_obj,)

    # Context-dependent mid-range threshold θ_k(C)
    theta_k = get_threshold_k_midrange_jax(
        states=states,
        state_prior=prior,
        k=k,
        q_low=q_low,
        q_high=q_high,
    )

    # Graded size semantics relative to θ_k(C)
    m_size = size_semantic_value_midrange(
        size=sizes,
        theta_k=theta_k,
        w_f=wf,
    )  # (n_obj,)

    return m_size


def incremental_semantics_jax(
    states:       jnp.ndarray,   # (n_obj, 3)
    color_sem:    float = 0.95,
    form_sem:     float = 0.8,
    k:            float = 0.5,
    wf:           float = 0.5,
    state_prior:  jnp.ndarray = None,
    utterances:   jnp.ndarray = None,  # (n_utt, T)
) -> jnp.ndarray:
    """
    Compute M_listener[u, s] = P(s | u) for all utterances using backward
    functional (right-to-left) semantics, optimized so that only size
    depends on the evolving context.

    Args:
        states:       (n_obj, 3) array of object states.
                      columns: [size, color, form].
        color_sem:    semantic sharpness for color.
        form_sem:     semantic sharpness for form.
        k:            size-threshold parameter.
        wf:           slack parameter for size semantics.
        state_prior:  (n_obj,) prior over states (optional).
        utterances:   (n_utt, T) int-coded utterances; -1 = padding,
                      0 = size, 1 = color, 2 = form.

    Returns:
        M: (n_utt, n_obj) array, each row normalized: P(s | u_i).
    """
    if utterances is None:
        global utterance_list
        utterances = utterance_list  # (n_utt, T)

    n_utt, T = utterances.shape
    n_obj = states.shape[0]

    # -------------------------
    # Precompute color/form semantics (context-independent)
    # -------------------------
    colors = states[:, 1]
    forms  = states[:, 2]

    color_vec = jnp.where(colors == 1, color_sem, 1.0 - color_sem)  # (n_obj,)
    form_vec  = jnp.where(forms  == 1, form_sem,  1.0 - form_sem)   # (n_obj,)

    # -------------------------
    # Prior P_T(s) shared across utterances
    # -------------------------
    if state_prior is None:
        state_prior = jnp.ones(n_obj) / n_obj  # (n_obj,)
    # Broadcast prior over utterances: (n_utt, n_obj)
    prior0 = jnp.broadcast_to(state_prior, (n_utt, n_obj))

    # -------------------------
    # Scan over tokens from right to left.
    # tokens_rev: (T, n_utt)
    # -------------------------
    tokens_rev = jnp.flip(utterances, axis=1).T

    def update_one_utterance(prior_i, token_i):
        """
        Update for a single utterance i at one token position.
        prior_i : (n_obj,)
        token_i : scalar int (0=size, 1=color, 2=form, -1=padding)
        """
        def skip(_):
            # padding: no-op
            return prior_i

        def apply(_):
            # token_i in {0,1,2}: choose size/color/form branch
            def size_branch(_):
                m = _size_meaning_vector(states, prior_i, k, wf)  # (n_obj,)
                return prior_i * m

            def color_branch(_):
                return prior_i * color_vec

            def form_branch(_):
                return prior_i * form_vec

            # token_i is 0,1,2 here
            return lax.switch(
                token_i,
                (size_branch, color_branch, form_branch),
                operand=None
            )

        # if token_i < 0 → padding → skip
        return lax.cond(token_i < 0, skip, apply, operand=None)

    def step(prior_all, tokens_t):
        """
        prior_all: (n_utt, n_obj) = beliefs for all utterances at current step
        tokens_t:  (n_utt,)      = token at this position for each utterance
        """
        posterior_all = jax.vmap(update_one_utterance)(prior_all, tokens_t)
        return posterior_all, None

    # -------------------------
    # Scan over token positions (right -> left)
    # -------------------------
    final, _ = lax.scan(step, prior0, tokens_rev)
    # final_beliefs has shape (T, n_utt, n_obj);

    # Normalize rows to get P(s | u)
    row_sums = jnp.clip(jnp.sum(final, axis=1, keepdims=True), 1e-20)
    M = final / row_sums
    return M

def global_speaker(
    states: jnp.ndarray,               # shape (n_objs,3)
    alpha: float = 1.0,
    color_semval: float = 0.95,
    wf: float = 1.0,
    beta: float = 1.0,
):
    """
    Output: P(utterance | referent) using global RSA semantics with LM-based cost.

    For each context (set of states), we first compute the incremental literal
    listener M_listener[u, s] = P(s | u) using context-dependent semantics.
    The global speaker then combines informativeness with an utterance-level
    cost derived from an external language-model prior.

    Args:
        states:           array of object states, shape (n_obj, 3).
        alpha:            RSA rationality parameter scaling informativeness.
        color_semval:     semantic sharpness for color.
        k:                size-threshold parameter.
        bias_subjectivity: (currently unused; kept for API compatibility).
        bias_length:      real-valued parameter, interpreted as log-temperature
                          for the LM-based cost (beta = exp(bias_length)).
        utt_prior:        deprecated / ignored (kept for API compatibility).

    Returns:
        M_speaker: jnp.ndarray of shape (n_obj, n_utt) with P(u | s_j).
    """
    eps = 1e-8

    # ----- 1) LM-based utterance cost -----
    # Interpret bias_length as log-temperature: beta = exp(bias_length) > 0
    costs = utterance_cost_jax(beta=beta)  # shape (n_utt,)

    # ----- 2) Incremental literal listener matrix M_listener: (n_utt, n_obj) -----
    # Only size semantics depend on context; color/form are context-independent.
    M_listener = incremental_semantics_jax(
        states=states,
        color_sem=color_semval,
        wf=wf,
    )  # rows = u, cols = s

    # ----- 3) Utilities per state (row) and utterance (col) -----
    log_L = jnp.log(jnp.clip(M_listener.T, eps, 1.0))  # (n_obj, n_utt)

    # Utility: U(s, u) = alpha * log L(s | u) - cost(u)
    util = alpha * log_L - costs[None, :]              # broadcast costs over states

    # ----- 4) Softmax over utterances for each state row -----
    M_speaker = jax.nn.softmax(util, axis=-1)          # (n_obj, n_utt)

    # Referent is always index 0; return only P(u | s_referent)
    referent_index = 0
    probs = M_speaker[referent_index, :]
    return probs

vectorized_global_speaker = jax.vmap(global_speaker, in_axes=(0, # states, along the first axis, i.e. one trial of the experiment
                                                              None, # alpha,
                                                              None, # color_semval
                                                              None, # k
                                                              None, # beta
                                                              )) 

# ========================
# Incremental Speaker
# ========================
# ========================
# Prefix helpers for incremental speaker
# ========================
# Vocabulary indices for adjectives: 0=size (D), 1=color (C), 2=form (F)
VOCAB_SIZE = 3
n_utt, T = utterance_list.shape

# We build:
#  PREFIX_UTTS[t, u, a, :]  = utterance encoding prefix(u,<t) + a at position t, padded with -1
#  CANDIDATE_MASK[t, u, a]  = True iff a is an admissible next token at step t for utterance u
#  ACTIVE_POS[t, u]         = True iff position t is actually used (no padding) in utterance u

prefix_utts_np = np.full((T, n_utt, VOCAB_SIZE, T), -1, dtype=np.int32)
cand_mask_np   = np.zeros((T, n_utt, VOCAB_SIZE), dtype=bool)
active_np      = np.zeros((T, n_utt), dtype=bool)

for t in range(T):
    for u in range(n_utt):
        tokens = np.array(utterance_list[u])
        token_t = tokens[t]
        # If padding at this position, nothing happens here
        if token_t < 0:
            continue

        active_np[t, u] = True

        # Tokens already used in the prefix u_{<t}
        used = set(int(x) for x in tokens[:t] if x >= 0)

        # Candidates: any vocab item not yet used
        for a in range(VOCAB_SIZE):
            if a in used:
                continue
            cand_mask_np[t, u, a] = True

            # Build prefix+candidate sequence (truncate after t)
            seq = np.full(T, -1, dtype=np.int32)
            if t > 0:
                seq[:t] = tokens[:t]
            seq[t] = a
            prefix_utts_np[t, u, a, :] = seq

PREFIX_UTTS    = jnp.asarray(prefix_utts_np)   # (T, n_utt, 3, T)
CANDIDATE_MASK = jnp.asarray(cand_mask_np)     # (T, n_utt, 3)
ACTIVE_POS     = jnp.asarray(active_np)        # (T, n_utt)
# ========================
def incremental_speaker(
    states: jnp.ndarray,
    alpha: float = 1.0,
    color_semval: float = 0.95,
    wf: float = 1.0,
    beta: float = 1.0,
) -> jnp.ndarray:
    """
    Incremental RSA speaker S_inc(u | s*).

    For a given context (states) and intended referent s* (always index 0),
    this returns a distribution over utterances using a token-by-token
    chain-rule construction:

        S_inc(u | s*) ∝ P_beta(u) · ∏_t S_t(w_t | u_{<t}, s*)

    where:
      - P_beta(u) is a temperature-scaled LM prior derived from LM_PRIOR_15,
      - S_t(w_t | u_{<t}, s*) is a local softmax over admissible next tokens,
        driven purely by incremental informativeness.
    """
    eps = 1e-8
    referent_index = 0

    n_utt, T = utterance_list.shape

    # ---- LM-based utterance prior P_beta(u) ----
    lm = jnp.clip(LM_PRIOR_15, eps, 1.0)
    scaled = lm ** beta
    P_beta = scaled / jnp.sum(scaled)  # (n_utt,)

    def step(scores: jnp.ndarray, t: jnp.ndarray):
        """
        One incremental step over position t (0 .. T-1).

        scores : (n_utt,) — cumulative product of local probabilities so far.
        t      : scalar int — current position.
        """
        # Precomputed prefix+candidate utterances for this position
        prefix_utts_t = PREFIX_UTTS[t]     # (n_utt, 3, T)
        cand_mask_t   = CANDIDATE_MASK[t]  # (n_utt, 3)
        active_t      = ACTIVE_POS[t]      # (n_utt,)
        tokens_t      = utterance_list[:, t]  # (n_utt,)

        # Build candidate utterances as a flat batch: (n_utt*3, T)
        cand_utts_flat = prefix_utts_t.reshape(-1, T)

        # Literal listener for all candidate prefixes
        # M_flat: (n_utt*3, n_obj), rows = utterances, cols = states
        M_flat = incremental_semantics_jax(
            states=states,
            color_sem=color_semval,
            wf=wf,
            utterances=cand_utts_flat,
        )
        L_flat = M_flat[:, referent_index]           # (n_utt*3,)
        L_vals = L_flat.reshape(n_utt, VOCAB_SIZE)   # (n_utt,3)

        # Masked softmax over candidates a in {0,1,2}
        logits = jnp.where(
            cand_mask_t,
            alpha * jnp.log(jnp.clip(L_vals, eps, 1.0)),
            -1e9,  # effectively exclude non-candidates
        )
        local_probs = jax.nn.softmax(logits, axis=-1)  # (n_utt,3)

        # Probability of the actually realized token at position t
        # (clamp -1 to 0; inactive rows are handled with active_t)
        token_idx = jnp.clip(tokens_t, 0, VOCAB_SIZE - 1)
        chosen = jnp.take_along_axis(
            local_probs,
            token_idx[:, None],
            axis=1,
        )[:, 0]  # (n_utt,)

        # For padded positions, we do not update the score (multiply by 1)
        chosen = jnp.where(active_t, chosen, 1.0)

        new_scores = scores * chosen
        return new_scores, None

    # Scan over positions t = 0 .. T-1
    init_scores = jnp.ones((n_utt,))
    final_scores, _ = lax.scan(step, init_scores, jnp.arange(T))  # final_scores: (n_utt,)

    # ---- Combine utterance-level LM prior and incremental score ----
    unnorm = P_beta * final_scores
    probs = unnorm / jnp.clip(jnp.sum(unnorm), eps, None)  # (n_utt,)
    return probs

vectorized_incremental_speaker = jax.vmap(incremental_speaker, in_axes=(0, # states, along the first axis, i.e. one trial of the experiment
                                                              None, # alpha,
                                                              None, # color_semval
                                                              None, # wf
                                                            None, # beta
                                                              )) 

def likelihood_function_global_speaker(states = None, empirical = None):
    # Initialize the parameter priors
    alpha = numpyro.sample("alpha", dist.HalfNormal(1.0))
    color_semval = numpyro.sample("color_semvalue", dist.Beta(4, 1))
    size_semval = numpyro.sample("size_semval", dist.Normal(0, 1))
    gamma = 2.0  # fixed scaling hyperparameter (tune if needed)
    wf = numpyro.deterministic("wf", jnp.exp(-gamma * size_semval))
    beta  = numpyro.sample("beta", dist.HalfNormal(0.5))

    # Define the likelihood function
    with numpyro.plate("data", len(states)):
        # Get vectorized global speaker output for all states
        # For single output, it is shape (n_utt, n_obj)
        probs = vectorized_global_speaker(states, alpha, color_semval, wf, beta) #shape (nbatch_size, n_utt, n_obj)
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)

def likelihood_function_incremental_speaker(states = None, empirical = None):
    # Initialize the parameter priors
    alpha = numpyro.sample("alpha", dist.HalfNormal(1.0))
    color_semval = numpyro.sample("color_semvalue", dist.Beta(4, 1))
    size_semval = numpyro.sample("size_semval", dist.Normal(0, 1))
    gamma = 2.0  # fixed scaling hyperparameter (tune if needed)
    wf = numpyro.deterministic("wf", jnp.exp(-gamma * size_semval))
    beta  = numpyro.sample("beta", dist.HalfNormal(0.5))

    # Define the likelihood function
    with numpyro.plate("data", len(states)):
        # Get vectorized incremental speaker output for all states
        # For single output, it is shape (n_utt, n_obj)
        probs = vectorized_incremental_speaker(states, alpha, color_semval, wf, beta) #shape (nbatch_size, n_utt, n_obj)
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)

 # ========================
def run_inference(speaker_type: str = "global",
                    num_warmup: int = 100,
                    num_samples: int = 250,
                    num_chains: int = 4,
                  ):
    # Numpyro setup for remote server on GPU
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(num_chains)

    # Setup output file name
    output_file_name = f"./inference_data/mcmc_results_{speaker_type}_speaker_warmup{num_warmup}_samples{num_samples}_chains{num_chains}"

    # Import dataset
    data = import_dataset()
    states_train = data["states_train"]
    empirical_train_seq_flat = data["empirical_seq_flat"]

    # Some printing for debugging
    print("States train shape:", states_train.shape)
    print("Empirical train flat shape:", empirical_train_seq_flat.shape)
    print("Output file name:" , output_file_name)

    # Define the MCMC kernel and the number of samples
    rng_key = random.PRNGKey(4711)
    rng_key, rng_key_ = random.split(rng_key)

    if speaker_type == "global":
        # Use the global speaker likelihood function
        model = likelihood_function_global_speaker
    elif speaker_type == "incremental":
        # Use the incremental speaker likelihood function
        model = likelihood_function_incremental_speaker
    kernel = NUTS(model, target_accept_prob=0.9, max_tree_depth=10)
    mcmc = MCMC(kernel, num_warmup=num_warmup,num_samples=num_samples, num_chains=num_chains)
    mcmc.run(rng_key_, states_train, empirical_train_seq_flat)

    # print the summary of the posterior distribution
    mcmc.print_summary()

    # Get the MCMC samples and convert to a numpyro ArviZ InferenceData object
    posterior_samples = mcmc.get_samples() 
    posterior_predictive = Predictive(model, posterior_samples)(
    PRNGKey(1), states_train
    )
    prior = Predictive(model, num_samples=500)(
        PRNGKey(2), states_train
    )   

    N = states_train.shape[0]  # 3196

    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords={"item": np.arange(N)},
        dims={"obs": ["item"]},
    )

    # Write the inference data to a netcdf file
    az.to_netcdf(numpyro_data, output_file_name)


def test():
    """
    Main function to run the script.
    """
    # Import dataset
    data = import_dataset()
    states_train = data["states_train"]
    empirical_train_seq = data["empirical_seq"]
    empirical_train_flat = data["empirical_flat"]
    empirical_train_seq_flat = data["empirical_seq_flat"]
    uttSeq_list = jnp.unique(empirical_train_seq, axis=0)  # shape (U, L), U ≤ N

    # Get example state and utterance
    example_index = 3
    example_state = states_train[2]
    example_empirical = empirical_train_seq_flat[example_index]
    example_empirical_seq = empirical_train_seq[example_index]

    # Print example state and utterance
    print("Example state:", example_state)
    # print("Example empirical utterance:", example_empirical)
    # print("Example empirical utterance sequence:", example_empirical_seq)
    # print("Seq to Flat mapping:", uttSeq_list)

    # Compute the incremental semantics for the example state
    example_incremental_semantics = incremental_semantics_jax(example_state, 0.95, 0.95, 0.5, 0.5)
    print("Example incremental semantics:", example_incremental_semantics)
    # Compute the global speaker for the example state
    example_global_speaker = global_speaker(example_state, 0.5, 0.95, 0.5)
    print("Example global speaker:", example_global_speaker[0,])
    # Compute the incremental speaker for the example state
    example_incremental_speaker = incremental_speaker(example_state, 0.5, 0.95, 0.5)
    print("Example incremental speaker:", example_incremental_speaker)
    #print("Example incremental speaker shape:", example_incremental_speaker.shape)

    # example_states_array = states_train[0:2]
    # # Test the vectorized global speaker
    # example_vectorized_global_speaker = vectorized_global_speaker(example_states_array, 1, 0.95, 0.5, 1)
    # utt_probs_conditionedReferent = example_vectorized_global_speaker[:,0,:] # Get the probs of utterances given the first state, referent is always the first state
    # print("Example vectorized global speaker distilled result shape:", utt_probs_conditionedReferent.shape)
    # print("Example vectorized global speaker distilled result shape:", utt_probs_conditionedReferent)

    # # Test the vectorized incremental speaker
    # example_vectorized_incremental_speaker = vectorized_incremental_speaker(example_states_array, 1, 0.95, 0.5, 1)
    # utt_probs_conditionedReferent = example_vectorized_incremental_speaker # Get the probs of utterances given the first state, referent is always the first state
    # print("Example vectorized incremental speaker distilled result shape:", utt_probs_conditionedReferent.shape)
    # print("Example vectorized incremental speaker distilled result shape:", utt_probs_conditionedReferent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speaker inference with NumPyro.")
    parser.add_argument("--speaker_type", type=str, choices=["global", "incremental"], default="global",
                        help="Choose the speaker model type.")
    parser.add_argument("--num_samples", type=int, default=250, help="Number of posterior samples.")
    parser.add_argument("--num_warmup", type=int, default=750, help="Number of warm-up iterations.")
    parser.add_argument("--num_chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--test", action="store_true", help="Run test function and exit.")

    args = parser.parse_args()

    if args.test:
        test()
    else:
        run_inference(
            speaker_type=args.speaker_type,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
        )