"""Production experiment RSA model library.

This module defines speaker models, listener functions, semantics, likelihood
functions for the production experiment. It is a pure library with no CLI or
side effects — use run_inference.py as the entry point.
"""
import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict, List, Any, Sequence, Callable
from jax import jit, vmap
import jax
from jax import lax
from jax import random
from jax.random import PRNGKey, beta, split
from functools import partial
import matplotlib.pyplot as plt
import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro import param, sample
from numpyro.distributions import constraints, HalfNormal, Normal, Uniform

from numpyro import handlers
from numpyro.infer import MCMC, NUTS, HMC, MixedHMC, init_to_value
from numpyro.infer import Predictive
from sklearn.model_selection import train_test_split
from helper import import_dataset, import_dataset_hier, normalize, build_utterance_prior_jax
from principled_features import ORDER_ONLY_LM_RESID_15 as ORDER_ONLY_LM_RESID_15_NP


print(jax.__version__)
print(jax.devices())

import arviz as az

# ========================
# Global Variables (Setup)
# ========================
# Utterance list shape: (15, 3), int-coded; -1 = padding, 0=D, 1=C, 2=F
utterance_list = import_dataset()["unique_utterances"]  # shape (U,3)

UTTERANCE_LABELS = [  
    "D", "DC", "DCF", "DF", "DFC",  
    "C", "CD", "CDF", "CF", "CFD",  
    "F", "FD", "FDC", "FC", "FCD",  
]  

# LM prior over the 15 utterances in the same order as utterance_list  
# Order: D, DC, DCF, DF, DFC, C, CD, CDF, CF, CFD, F, FD, FDC, FC, FCD 
LM_PRIOR_15 = jnp.array([
    0.1669514, 0.16733019, 0.12160929, 0.11005973, 0.09253279,
    0.07532827, 0.02494562, 0.03780574, 0.05690099, 0.02470998,
    0.02651604, 0.01232579, 0.03122547, 0.0363892, 0.01536951
])

# =============================================================================
# 2. UTTERANCE COST  (LM-derived, temperature-scaled)  
# =============================================================================  

def utterance_cost_jax(beta: float = 1.0) -> jnp.ndarray:  
    """  
    Utterance cost = −log P_LM(u)^β (normalised).  

    β = 0  → uniform cost (no LM influence)  
    β = 1  → raw LM prior  
    β > 1  → sharper LM preference (D-family cheaper, F-family costlier)  

    Returns  
    -------  
    costs : (15,)  positive floats; lower = cheaper utterance  
    """  
    eps = 1e-12  
    scaled = jnp.power(jnp.clip(LM_PRIOR_15, eps), beta)  
    scaled = scaled / jnp.sum(scaled)  
    return -jnp.log(jnp.clip(scaled, eps))          # (15,)  

# =============================================================================  
# 3. SIZE SEMANTICS  (context-sensitive, graded)  
# =============================================================================  

def compute_size_semantics(
    states:      jnp.ndarray,   # (n_obj, 3)
    state_prior: jnp.ndarray,   # (n_obj,)  sums to 1
    k:           float,         # threshold interpolation ∈ (0, 1)
    wf:          float,         # sigmoid width / slack > 0
    q_low:       float = 0.2,
    q_high:      float = 0.8,
) -> jnp.ndarray:  
    """  
    Graded size semantics ⟦D⟧_C(s) via prior-weighted CDF thresholding.  

    Steps  
    -----  
    1. Sort objects by size; compute prior-weighted CDF.  
    2. Identify mid-range band [x_min_mid, x_max_mid] at quantiles  
       (q_low, q_high).  
    3. Threshold  θ_k = x_max_mid − k·(x_max_mid − x_min_mid)  
         k = 0  →  θ_k = x_max_mid  (only largest objects are "big")  
         k = 1  →  θ_k = x_min_mid  (most objects qualify as "big")  
    4. Graded membership  Φ((x − θ_k) / (wf·√(x² + θ_k²))).  

    Returns  
    -------  
    m_size : (n_obj,)  ∈ (0, 1)  
    """  
    sizes        = states[:, 0]  
    idx          = jnp.argsort(sizes)  
    sizes_sorted = sizes[idx]  
    cdf          = jnp.cumsum(state_prior[idx])  

    x_min_mid = sizes_sorted[jnp.argmax(cdf >= q_low)]  
    x_max_mid = sizes_sorted[jnp.argmax(cdf >= q_high)]  
    theta_k   = x_max_mid - k * (x_max_mid - x_min_mid)  

    eps   = 1e-8  
    denom = wf * jnp.sqrt(sizes ** 2 + theta_k ** 2 + eps)  
    z     = (sizes - theta_k) / denom  
    return 0.5 * (1.0 + lax.erf(z / jnp.sqrt(2.0)))    # (n_obj,) 


# ── Pre compute size semantic values ──────────────────────────────────────────────────────  
# Precompute sizes as a JAX constant — avoids re-slicing inside vmap  
states = import_dataset()["states_train"]  # (N, 3)
SIZES = jnp.array(np.array(states)[:, :, 0])   # (N, n_obj) — fixed at trace time  

def compute_size_semantics_fast(
    sizes:     jnp.ndarray,        # (n_obj,)
    posterior: jnp.ndarray,        # (n_obj,)  current per-utt posterior
    k:         float,              # threshold interpolation ∈ (0, 1)
    wf:        float,              # sigmoid width > 0
    q_low:     float = 0.2,
    q_high:    float = 0.8,
) -> jnp.ndarray:                  # (n_obj,) ∈ (0, 1)

    eps = 1e-8

    # ── Step 1: sort by size, compute posterior-weighted CDF ──────────────────
    idx          = jnp.argsort(sizes)            # (n_obj,)
    sizes_sorted = sizes[idx]                    # (n_obj,)
    post_sorted  = posterior[idx]                # (n_obj,)

    # Normalise posterior in case it doesn't sum to exactly 1.0
    post_sorted  = post_sorted / (jnp.sum(post_sorted) + eps)
    cdf          = jnp.cumsum(post_sorted)       # (n_obj,) ∈ (0, 1]

    # ── Step 2: find x_min_mid, x_max_mid via first crossing (searchsorted) ──
    idx_low  = jnp.minimum(jnp.searchsorted(cdf, q_low, side="left"), sizes_sorted.shape[0] - 1)
    idx_high = jnp.minimum(jnp.searchsorted(cdf, q_high, side="left"), sizes_sorted.shape[0] - 1)
    x_min_mid = sizes_sorted[idx_low]           # size at q_low quantile
    x_max_mid = sizes_sorted[idx_high]          # size at q_high quantile

    # ── Step 3: context-dependent threshold ───────────────────────────────────
    #   k = 0  →  θ_k = x_max_mid  (only the largest objects are "big")
    #   k = 0.5→  θ_k = midpoint   (medium threshold)
    #   k = 1  →  θ_k = x_min_mid  (even small objects qualify as "big")
    theta_k = x_max_mid - k * (x_max_mid - x_min_mid)       # scalar

    # ── Step 4: graded membership (original scale-adaptive denominator) ───────
    #   denom = wf · √(x² + θ_k²)   ← NOT wf·σ_context
    #   This scales with the absolute magnitudes of sizes and threshold
    denom = wf * jnp.sqrt(sizes ** 2 + theta_k ** 2 + eps)  # (n_obj,)
    z     = (sizes - theta_k) / denom                # (n_obj,)

    return 0.5 * (1.0 + lax.erf(z / jnp.sqrt(2.0)))         # (n_obj,) ∈ (0,1)


def compute_size_semantics_fast_presorted(
    sizes:        jnp.ndarray,      # (n_obj,)
    sort_idx:     jnp.ndarray,      # (n_obj,)
    sizes_sorted: jnp.ndarray,      # (n_obj,)
    posterior:    jnp.ndarray,      # (n_obj,)
    k:            float,
    wf:           float,
    q_low:        float = 0.2,
    q_high:       float = 0.8,
) -> jnp.ndarray:
    eps = 1e-8

    post_sorted = posterior[sort_idx]
    post_sorted = post_sorted / (jnp.sum(post_sorted) + eps)
    cdf = jnp.cumsum(post_sorted)

    idx_low = jnp.minimum(jnp.searchsorted(cdf, q_low, side="left"), sizes_sorted.shape[0] - 1)
    idx_high = jnp.minimum(jnp.searchsorted(cdf, q_high, side="left"), sizes_sorted.shape[0] - 1)

    x_min_mid = sizes_sorted[idx_low]
    x_max_mid = sizes_sorted[idx_high]
    theta_k = x_max_mid - k * (x_max_mid - x_min_mid)

    denom = wf * jnp.sqrt(sizes ** 2 + theta_k ** 2 + eps)
    z = (sizes - theta_k) / denom
    return 0.5 * (1.0 + lax.erf(z / jnp.sqrt(2.0)))


def compute_size_semantics_comparison_class(
    states:     jnp.ndarray,  # (n_obj, 3)
    class_mask: jnp.ndarray,  # (n_obj,), hard comparison-class indicator
    k:          float,
    wf:         float,
    q_low:      float = 0.2,
    q_high:     float = 0.8,
) -> jnp.ndarray:
    """Size semantics with thresholds computed over a hard comparison class.

    The mask is supplied by determinate colour/form adjectives to the right of
    the size adjective.  If that class is empty, the display-wide class is used.
    """
    n_obj = states.shape[0]
    mask = class_mask.astype(jnp.float32)
    mask_total = jnp.sum(mask)
    fallback_prior = jnp.ones(n_obj, dtype=jnp.float32) / n_obj
    class_prior = mask / jnp.clip(mask_total, 1e-8)
    state_prior = jnp.where(mask_total > 0.0, class_prior, fallback_prior)
    return compute_size_semantics(states, state_prior, k, wf, q_low, q_high)


def _literal_listener_comparison_class_one(
    states:      jnp.ndarray,
    utterance:   jnp.ndarray,
    color_sem:   float,
    form_sem:    float,
    k:           float,
    wf:          float,
    state_prior: jnp.ndarray,
) -> jnp.ndarray:
    """Literal listener for one utterance under comparison-class size semantics."""
    eps = 1e-8
    colors = states[:, 1]
    forms = states[:, 2]
    color_vec = jnp.where(colors == 1, color_sem, 1.0 - color_sem)
    form_vec = jnp.where(forms == 1, form_sem, 1.0 - form_sem)
    color_class = (colors == 1).astype(jnp.float32)
    form_class = (forms == 1).astype(jnp.float32)

    def step(carry, token_i):
        posterior_i, class_mask_i = carry

        def skip(_):
            return posterior_i, class_mask_i

        def apply(_):
            size_vec = compute_size_semantics_comparison_class(
                states,
                class_mask_i,
                k,
                wf,
            )

            def apply_size(__):
                return posterior_i * size_vec, class_mask_i

            def apply_color(__):
                return posterior_i * color_vec, class_mask_i * color_class

            def apply_form(__):
                return posterior_i * form_vec, class_mask_i * form_class

            return lax.switch(
                token_i,
                [apply_size, apply_color, apply_form],
                operand=None,
            )

        return lax.cond(token_i < 0, skip, apply, operand=None), None

    tokens_rev = jnp.flip(utterance)
    init_class = jnp.ones(states.shape[0], dtype=jnp.float32)
    final, _ = lax.scan(step, (state_prior, init_class), tokens_rev)
    posterior, _ = final
    return posterior / jnp.clip(jnp.sum(posterior), eps)


def literal_listener_comparison_class_batch(
    states:      jnp.ndarray,
    utterances:  jnp.ndarray,
    color_sem:   float = 0.95,
    form_sem:    float = 0.80,
    k:           float = 0.5,
    wf:          float = 0.5,
    state_prior: jnp.ndarray = None,
) -> jnp.ndarray:
    """Literal listener matrix for a batch of utterances."""
    if state_prior is None:
        state_prior = jnp.ones(states.shape[0], dtype=jnp.float32) / states.shape[0]

    return jax.vmap(
        _literal_listener_comparison_class_one,
        in_axes=(None, 0, None, None, None, None, None),
    )(states, utterances, color_sem, form_sem, k, wf, state_prior)


def incremental_semantics_jax_comparison_class(
    states:      jnp.ndarray,
    color_sem:   float = 0.95,
    form_sem:    float = 0.80,
    k:           float = 0.5,
    wf:          float = 0.5,
    state_prior: jnp.ndarray = None,
    utterances:  jnp.ndarray = None,
) -> jnp.ndarray:
    """Literal listener with size thresholds from right-context comparison classes."""
    if utterances is None:
        utterances = utterance_list
    return literal_listener_comparison_class_batch(
        states=states,
        utterances=utterances,
        color_sem=color_sem,
        form_sem=form_sem,
        k=k,
        wf=wf,
        state_prior=state_prior,
    )



# =============================================================================  
# 4. INCREMENTAL LITERAL LISTENER  (right-to-left scan)  
# =============================================================================  

def incremental_semantics_jax(
    states:      jnp.ndarray,          # (n_obj, 3)
    color_sem:   float = 0.95,         # ∈ (0.5, 1)
    form_sem:    float = 0.80,         # ∈ (0.5, 1)
    k:           float = 0.5,          # ∈ (0, 1)
    wf:          float = 0.5,          # > 0
    state_prior: jnp.ndarray = None,   # (n_obj,)
    utterances:  jnp.ndarray = None,   # (n_utt, T)
) -> jnp.ndarray:  
    """  
    Literal listener matrix M[u, s] = P(s | u).  

    The scan processes tokens right-to-left so that each successive word  
    narrows the posterior in a psycholinguistically plausible order.  
    Size semantics are context-recursive: recomputed from the running  
    posterior at each scan step so that the threshold adapts to the  
    current belief state.  Colour and form are pre-computed once.  

    Returns  
    -------  
    M : (n_utt, n_obj)  each row sums to 1  
    """  
    if utterances is None:  
        utterances = utterance_list                     # (15, 3)  

    n_utt, T = utterances.shape  
    n_obj    = states.shape[0]  

    if state_prior is None:  
        state_prior = jnp.ones(n_obj) / n_obj  

    # ------------------------------------------------------------------  
    # Pre-compute context-INDEPENDENT semantics (colour / form) once  
    # ------------------------------------------------------------------  
    colors    = states[:, 1]  
    forms     = states[:, 2]  
    color_vec = jnp.where(colors == 1, color_sem, 1.0 - color_sem)  # (n_obj,)  
    form_vec  = jnp.where(forms  == 1, form_sem,  1.0 - form_sem)   # (n_obj,)  
    # Size semantics are context-recursive: recomputed from the running  
    # posterior at each scan step (see update_one below).  

    # ------------------------------------------------------------------  
    # Scan: process tokens right → left  
    # ------------------------------------------------------------------  
    prior0     = jnp.broadcast_to(state_prior, (n_utt, n_obj))      # (n_utt, n_obj)  
    tokens_rev = jnp.flip(utterances, axis=1).T                      # (T, n_utt)  

    def update_one(prior_i, token_i):  
        """Update a single utterance's running posterior by one token."""  
        # Normalise running posterior for size-semantics context  
        prior_norm = prior_i / jnp.clip(prior_i.sum(), 1e-20)  
        size_vec = compute_size_semantics(states, prior_norm, k, wf)  
        def skip(_):   return prior_i  
        def apply(_):  
            return lax.switch(  
                token_i,  
                [lambda _: prior_i * size_vec,  
                 lambda _: prior_i * color_vec,  
                 lambda _: prior_i * form_vec],  
                operand=None,  
            )  
        return lax.cond(token_i < 0, skip, apply, operand=None)  

    def step(prior_all, tokens_t):  
        """Apply one token position across all utterances in parallel."""  
        return jax.vmap(update_one)(prior_all, tokens_t), None  

    final, _ = lax.scan(step, prior0, tokens_rev)   # (n_utt, n_obj)  

    row_sums = jnp.clip(final.sum(axis=1, keepdims=True), 1e-20)  
    return final / row_sums                          # (n_utt, n_obj)


def incremental_semantics_jax_frozen(
    states:      jnp.ndarray,          # (n_obj, 3)
    color_sem:   float = 0.95,         # ∈ (0.5, 1)
    form_sem:    float = 0.80,         # ∈ (0.5, 1)
    k:           float = 0.5,          # ∈ (0, 1)
    wf:          float = 0.5,          # > 0
    state_prior: jnp.ndarray = None,   # (n_obj,)
    utterances:  jnp.ndarray = None,   # (n_utt, T)
) -> jnp.ndarray:
    """
    Literal listener matrix M[u, s] = P(s | u) with STATIC size semantics.

    Same as incremental_semantics_jax, but size semantics are computed once
    from the initial uniform prior and held fixed throughout composition.
    This removes context-recursive threshold adaptation while preserving
    incremental belief update for the posterior itself.

    Returns
    -------
    M : (n_utt, n_obj)  each row sums to 1
    """
    if utterances is None:
        utterances = utterance_list

    n_utt, T = utterances.shape
    n_obj    = states.shape[0]

    if state_prior is None:
        state_prior = jnp.ones(n_obj) / n_obj

    # Pre-compute ALL semantics once (size uses the initial prior, not running)
    colors    = states[:, 1]
    forms     = states[:, 2]
    color_vec = jnp.where(colors == 1, color_sem, 1.0 - color_sem)
    form_vec  = jnp.where(forms  == 1, form_sem,  1.0 - form_sem)
    size_vec  = compute_size_semantics(states, state_prior, k, wf)

    prior0     = jnp.broadcast_to(state_prior, (n_utt, n_obj))
    tokens_rev = jnp.flip(utterances, axis=1).T

    def update_one(prior_i, token_i):
        def skip(_):   return prior_i
        def apply(_):
            return lax.switch(
                token_i,
                [lambda _: prior_i * size_vec,
                 lambda _: prior_i * color_vec,
                 lambda _: prior_i * form_vec],
                operand=None,
            )
        return lax.cond(token_i < 0, skip, apply, operand=None)

    def step(prior_all, tokens_t):
        return jax.vmap(update_one)(prior_all, tokens_t), None

    final, _ = lax.scan(step, prior0, tokens_rev)

    row_sums = jnp.clip(final.sum(axis=1, keepdims=True), 1e-20)
    return final / row_sums


# =============================================================================
# 5. GLOBAL RSA SPEAKER
# =============================================================================

def global_speaker(
    states:       jnp.ndarray,   # (n_obj, 3)  one trial
    alpha:        float = 3.0,   # rationality
    color_sem:    float = 0.95,  # ∈ (0.5, 1)
    form_sem:     float = 0.80,  # ∈ (0.5, 1)
    k:            float = 0.5,   # ∈ (0, 1)
    wf:           float = 0.5,   # > 0
    beta:         float = 1.0,   # LM cost temperature
    gamma:        float = 0.0,   # length bias
    epsilon:      float = 0.01,  # lapse rate
) -> jnp.ndarray:
    """
    P(u | s_referent) under global RSA with LM-based utterance cost.

    Utility:  U(u, s) = α · log L(s | u) − cost(u) + γ · N_WORDS
    Speaker:  P(u | s) ∝ exp U(u, s), then mixed with uniform lapse.

    The referent is always states[0].

    Returns
    -------
    probs : (n_utt,)  probability over utterances for the referent
    """
    eps = 1e-8

    costs      = utterance_cost_jax(beta=beta)           # (n_utt,)
    M_listener = incremental_semantics_jax(              # (n_utt, n_obj)
        states      = states,
        color_sem   = color_sem,
        form_sem    = form_sem,
        k           = k,
        wf          = wf,
    )

    log_L = jnp.log(jnp.clip(M_listener.T, eps))        # (n_obj, n_utt)
    util  = alpha * log_L - costs[None, :] + gamma * N_WORDS[None, :]  # (n_obj, n_utt)
    M_speaker = jax.nn.softmax(util, axis=-1)            # (n_obj, n_utt)
    M_speaker = (1.0 - epsilon) * M_speaker + epsilon / M_listener.shape[0]  # lapse

    return M_speaker[0, :]                               # referent = index 0  

# Vectorise over trials (axis 0 of states batch)
vectorized_global_speaker = jax.vmap(
    global_speaker,
    in_axes=(0,    # states  — one trial per row
             None, # alpha
             None, # color_sem
             None, # form_sem
             None, # k
             None, # wf
             None, # beta
             None, # gamma
             None, # epsilon
             ),
)


def global_speaker_static(
    states:       jnp.ndarray,
    alpha:        float = 3.0,
    color_sem:    float = 0.95,
    form_sem:     float = 0.80,
    k:            float = 0.5,
    wf:           float = 0.5,
    beta:         float = 1.0,
    gamma:        float = 0.0,
    epsilon:      float = 0.01,
) -> jnp.ndarray:
    """
    P(u | s_referent) under global RSA with STATIC size semantics.

    Same as global_speaker, but uses incremental_semantics_jax_frozen so that
    the size threshold is computed once from the uniform prior and not
    recomputed from the running posterior during composition.
    """
    eps = 1e-8

    costs      = utterance_cost_jax(beta=beta)
    M_listener = incremental_semantics_jax_frozen(
        states    = states,
        color_sem = color_sem,
        form_sem  = form_sem,
        k         = k,
        wf        = wf,
    )

    log_L = jnp.log(jnp.clip(M_listener.T, eps))
    util  = alpha * log_L - costs[None, :] + gamma * N_WORDS[None, :]
    M_speaker = jax.nn.softmax(util, axis=-1)
    M_speaker = (1.0 - epsilon) * M_speaker + epsilon / M_listener.shape[0]

    return M_speaker[0, :]


vectorized_global_speaker_static = jax.vmap(
    global_speaker_static,
    in_axes=(0,    # states
             None, # alpha
             None, # color_sem
             None, # form_sem
             None, # k
             None, # wf
             None, # beta
             None, # gamma
             None, # epsilon
             ),
)


# ── Pre JIT ──────────────────────────────────────────────────────
@jax.jit
def jitted_global_speaker(states, alpha, color_semval, form_semval, k, wf, beta, gamma, epsilon):
    return vectorized_global_speaker(
        states, alpha, color_semval, form_semval, k, wf, beta, gamma, epsilon
    )

@jax.jit
def jitted_global_speaker_static(states, alpha, color_semval, form_semval, k, wf, beta, gamma, epsilon):
    return vectorized_global_speaker_static(
        states, alpha, color_semval, form_semval, k, wf, beta, gamma, epsilon
    )

def likelihood_function_global_speaker(states=None, empirical=None):
    # ── Semantic parameters (fixed from global_static posterior) ──────────────
    color_sem = 0.971
    form_sem  = 0.50

    k  = 0.5
    wf = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)

    alpha   = numpyro.sample("alpha", dist.HalfNormal(5.0))
    gamma   = numpyro.sample("gamma", dist.Normal(0.0, 1.0))
    epsilon = numpyro.sample("epsilon", dist.Beta(1.0, 50.0))

    with numpyro.plate("data", len(states)):
        probs = jitted_global_speaker(
            states, alpha, color_sem, form_sem, k, wf, beta, gamma, epsilon
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
# ========================
# Incremental Speaker
# ========================
# =============================================================================
# PRECOMPUTED STATIC STRUCTURES (run once, CPU, before any JAX tracing)
# =============================================================================
VOCAB_SIZE = 3
n_utt, T   = utterance_list.shape

# ── Original prefix helpers (unchanged) ──────────────────────────────────────
prefix_utts_np = np.full((T, n_utt, VOCAB_SIZE, T), -1, dtype=np.int32)
cand_mask_np   = np.zeros((T, n_utt, VOCAB_SIZE),    dtype=bool)
active_np      = np.zeros((T, n_utt),                dtype=bool)

for t in range(T):
    for u in range(n_utt):
        tokens  = np.array(utterance_list[u])
        token_t = tokens[t]
        if token_t < 0:
            continue
        active_np[t, u] = True
        used = set(int(x) for x in tokens[:t] if x >= 0)
        for a in range(VOCAB_SIZE):
            if a in used:
                continue
            cand_mask_np[t, u, a] = True
            seq = np.full(T, -1, dtype=np.int32)
            if t > 0:
                seq[:t] = tokens[:t]
            seq[t] = a
            prefix_utts_np[t, u, a, :] = seq

PREFIX_UTTS    = jnp.asarray(prefix_utts_np)    # (T, n_utt, 3, T)
CANDIDATE_MASK = jnp.asarray(cand_mask_np)      # (T, n_utt, 3)
ACTIVE_POS     = jnp.asarray(active_np)         # (T, n_utt)

# ── NEW: token presence masks → replaces inner lax.scan ───────────────────────
# TOKEN_PRESENT[t, u, a, v] = True iff vocab item v appears in PREFIX_UTTS[t,u,a,:]
# Lets us replace scan-over-tokens with a vectorised dot product
token_present_np = np.zeros((T, n_utt, VOCAB_SIZE, VOCAB_SIZE), dtype=np.float32)
for t in range(T):
    for u in range(n_utt):
        for a in range(VOCAB_SIZE):
            seq = prefix_utts_np[t, u, a, :]
            valid = seq[seq >= 0]
            for v in range(VOCAB_SIZE):
                token_present_np[t, u, a, v] = float(v in valid)

TOKEN_PRESENT = jnp.asarray(token_present_np)   # (T, n_utt, 3, 3)
# TOKEN_PRESENT[t, u, a, :] = which vocab items are in prefix+candidate sequence
# replaces the inner scan(apply_tok) with a single einsum

# ── Actual tokens at each position for posterior update ───────────────────────
# ACTUAL_TOK[t, u] = tokens_t[u], clamped
actual_tok_np = np.clip(np.array(utterance_list).T, 0, VOCAB_SIZE - 1).astype(np.int32)
ACTUAL_TOK    = jnp.asarray(actual_tok_np)       # (T, n_utt)

# ── One-hot for actual tokens → used in posterior update ─────────────────────
# ACTUAL_TOK_ONEHOT[t, u, v] = 1 if tokens_t[u] == v else 0
actual_onehot_np = np.zeros((T, n_utt, VOCAB_SIZE), dtype=np.float32)
for t in range(T):
    for u in range(n_utt):
        tok = int(utterance_list[u, t])
        if tok >= 0:
            actual_onehot_np[t, u, tok] = 1.0
ACTUAL_TOK_ONEHOT = jnp.asarray(actual_onehot_np)   # (T, n_utt, 3)

# ── Number of words per utterance (for length bias) ──────────────────────────
N_WORDS = jnp.sum(jnp.array(utterance_list) >= 0, axis=1).astype(jnp.float32)  # (n_utt,)

# LM residual after removing mean log probability within each length class.
# This lets the LM prior encode within-length naturalness/order while separate
# length parameters encode over-/under-specification pressure.
_log_lm_np = np.log(np.clip(np.asarray(LM_PRIOR_15), 1e-12, None)).astype(np.float32)
_n_words_np = np.asarray(N_WORDS)
_lm_resid_np = np.zeros_like(_log_lm_np, dtype=np.float32)
for _length in np.unique(_n_words_np):
    _mask = _n_words_np == _length
    _lm_resid_np[_mask] = _log_lm_np[_mask] - np.mean(_log_lm_np[_mask])
LOG_LM_RESID_15 = jnp.asarray(_lm_resid_np)
# Raw LM log-prob (length-cumulative) for contextual model variant: lets a single
# LM coefficient absorb both within-length naturalness AND length pressure.
LOG_LM_RAW_15 = jnp.asarray(_log_lm_np)
LOG_LM_ORDER_ONLY_15 = jnp.asarray(ORDER_ONLY_LM_RESID_15_NP, dtype=jnp.float32)
BASE_VISUAL_SALIENCE = jnp.asarray([0.0, 1.0, 0.25], dtype=jnp.float32)
BLUR_WF_MULTIPLIER = 2.0

# ── First word of each utterance (for first-word intercepts) ─────────────────
# D=0, C=1, F=2 → one-hot masks for first-word bias
FIRST_WORD = jnp.array(utterance_list)[:, 0]  # (n_utt,) first token of each utterance

# ── F-present mask (the previous variant form-as-redundant-modifier boost) ────────────────
# 1 if the utterance contains the F (form, token index 2) word at any position.
F_PRESENT_15 = jnp.asarray(
    (np.asarray(utterance_list) == 2).any(axis=1).astype(np.float32)
)  # (n_utt,)

# ── Joint per-utterance dimension presence (global 2x2 speaker) ──────────────
# FULL_PRESENT_15[u, d] = 1 iff utterance u asserts dimension d anywhere
# (d: 0=size/D, 1=colour/C, 2=form/F). The global (non-incremental) speaker's
# literal listener conditions on ALL of an utterance's asserted dims jointly.
FULL_PRESENT_15 = jnp.asarray(np.stack([
    (np.asarray(utterance_list) == d).any(axis=1).astype(np.float32)
    for d in range(VOCAB_SIZE)
], axis=1))  # (n_utt, 3)

# COMPLETION_MASK[t, u, a, v] = True iff terminal utterance v is reachable
# after extending utterance u's prefix at position t with adjective a.
# This supports a planned-prefix incremental variant while preserving the
# original 15 terminal utterance inventory.
completion_mask_np = np.zeros((T, n_utt, VOCAB_SIZE, n_utt), dtype=bool)
utterance_list_np = np.asarray(utterance_list)

for t in range(T):
    for u in range(n_utt):
        for a in range(VOCAB_SIZE):
            if not cand_mask_np[t, u, a]:
                continue
            prefix = prefix_utts_np[t, u, a]
            prefix_valid = prefix[prefix >= 0]
            prefix_len = len(prefix_valid)
            for v in range(n_utt):
                terminal = utterance_list_np[v]
                terminal_valid = terminal[terminal >= 0]
                if len(terminal_valid) < prefix_len:
                    continue
                completion_mask_np[t, u, a, v] = bool(
                    np.array_equal(terminal_valid[:prefix_len], prefix_valid)
                )

COMPLETION_MASK = jnp.asarray(completion_mask_np)

# ── levers ───────────────────────────────────────────────────────────
# (2) Non-canonical-order mask: 1 if BOTH colour (1) and form (2) appear AND
#     form precedes colour (F-before-C). These are exactly {DFC, FDC, FC, FCD}
#     — the violations of the canonical colour-before-form adjective order
#     (Cinque 1994; Scott 2002; Sproat & Shih 1991). All ≈0 in human data.
def _f_before_c(row: np.ndarray) -> float:
    pos = {}
    for p, tok in enumerate(row):
        if tok >= 0 and tok not in pos:
            pos[tok] = p
    return 1.0 if (1 in pos and 2 in pos and pos[2] < pos[1]) else 0.0


F_BEFORE_C_15 = jnp.asarray(
    np.array([_f_before_c(r) for r in np.asarray(utterance_list)], dtype=np.float32)
)  # (n_utt,)

# (1) 3-word mask for the erdc-gated over-specification penalty. The per-word
#     length bonus γ·(N−1) cannot fix the erdc DF deficit (attenuating it
#     dumps mass into the 1-word D, not the 2-word DF — verified by sweep);
#     a flat 3-word penalty redistributes to DF proportionally instead.
IS_3WORD_15 = jnp.asarray((np.asarray(N_WORDS) >= 3.0).astype(np.float32))  # (n_utt,)

# ── C-initial mask for global-speaker v5 lambda_C boost ──────────────────────
# 1 if the utterance starts with C (mention index 1), else 0.
C_INITIAL_MASK = (FIRST_WORD == 1).astype(jnp.float32)  # (n_utt,)

# ── Canonical-ordering mask (C1): 1 if utterance violates size < colour < form
# surface order; 0 if canonical. D=0, C=1, F=2 → earlier index must come first.
# Canonical: D, C, F, DC, DF, CF, DCF (indices 0,1,3,5,8,10 plus DCF=2).
# Non-canonical: CD(6), CDF(7), CFD(9), DFC(4), FD(11), FDC(12), FC(13), FCD(14).
_canon_list = np.array(utterance_list, dtype=np.int64)  # (n_utt, 3), pad=-1
_noncanon = np.zeros(n_utt, dtype=np.float32)
for _u in range(n_utt):
    toks = [t for t in _canon_list[_u] if t >= 0]
    if any(toks[i] > toks[i + 1] for i in range(len(toks) - 1)):
        _noncanon[_u] = 1.0
NONCANON_MASK = jnp.asarray(_noncanon)                  # (n_utt,)
# Binary masks: which utterances start with C / F (D is reference)
STARTS_C = (FIRST_WORD == 1).astype(jnp.float32)  # (n_utt,)
STARTS_F = (FIRST_WORD == 2).astype(jnp.float32)  # (n_utt,)

# =============================================================================
# INCREMENTAL SPEAKER
# =============================================================================

def incremental_speaker(
    states:       jnp.ndarray,
    alpha_D:      float = 3.0,
    alpha_C:      float = 3.0,
    alpha_F:      float = 3.0,
    color_semval: float = 0.95,
    form_semval:  float = 0.80,
    k:            float = 0.50,
    wf:           float = 1.00,
    beta:         float = 1.00,
    gamma:        float = 0.0,
    epsilon:      float = 0.01,
) -> jnp.ndarray:

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]
    alpha_vec      = jnp.array([alpha_D, alpha_C, alpha_F])

    # ── NEW: extract sizes once here ──────────────────────────────────────────
    # states[:, 0] would be re-executed inside every vmap call without this
    sizes = states[:, 0]                                              # (n_obj,)
    size_sort_idx = jnp.argsort(sizes)                                # (n_obj,)
    sizes_sorted = sizes[size_sort_idx]                               # (n_obj,)
    # ─────────────────────────────────────────────────────────────────────────

    # ── LM utterance prior ────────────────────────────────────────────────────
    lm_scaled  = jnp.power(jnp.clip(LM_PRIOR_15, eps), beta)
    log_P_beta = jnp.log(lm_scaled / jnp.sum(lm_scaled))            # (n_utt,)

    # ── Context-independent log-semantics (color, form) ───────────────────────
    colors = states[:, 1]
    forms  = states[:, 2]

    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )                                                                  # (n_obj,)
    log_form_sem  = jnp.log(
        jnp.where(forms  == 1, form_semval,  1.0 - form_semval)  + eps
    )                                                                  # (n_obj,)

    # ── Scan carry ────────────────────────────────────────────────────────────
    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))          # (n_utt, n_obj)

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]    # (n_utt, 3)
        active_t    = ACTIVE_POS[t]        # (n_utt,)

        # ── Step 1: size log-semantics per utterance (reuse pre-sorted sizes) ─
        def size_log_sem_for_utt(post):
            sv = compute_size_semantics_fast_presorted(
                sizes,
                size_sort_idx,
                sizes_sorted,
                post,
                k,
                wf,
            )
            return jnp.log(jnp.clip(sv, eps))

        size_log_sems = jax.vmap(size_log_sem_for_utt)(per_utt_posts)   # (n_utt, n_obj)

        # ── Step 2: full log-sem table  (n_utt, 3, n_obj) ────────────────────
        log_sem_static = jnp.stack(
            [log_color_sem, log_form_sem], axis=0                    # (2, n_obj)
        )
        log_sem_table = jnp.concatenate([
            size_log_sems[:, None, :],                               # (n_utt, 1, n_obj)
            jnp.broadcast_to(
                log_sem_static[None, :, :], (n_utt, 2, n_obj)
            ),                                                        # (n_utt, 2, n_obj)
        ], axis=1)                                                    # (n_utt, 3, n_obj)

        # ── Step 3: candidate scores via TOKEN_PRESENT einsum ─────────────────
        token_pres_t = TOKEN_PRESENT[t]                              # (n_utt, 3, 3)

        log_prod_sem = jnp.einsum(
            "uav, uvo -> uao",
            token_pres_t,      # (n_utt, 3, 3)
            log_sem_table,     # (n_utt, 3, n_obj)
        )                                                             # (n_utt, 3, n_obj)

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))     # (n_utt, n_obj)
        log_prior = log_per_utt_posts

        log_updated = log_prior[:, None, :] + log_prod_sem           # (n_utt, 3, n_obj)

        log_Z    = jax.scipy.special.logsumexp(log_updated, axis=-1) # (n_utt, 3)
        log_norm = log_updated - log_Z[:, :, None]                   # (n_utt, 3, n_obj)
        log_L_ref = log_norm[:, :, referent_index]                   # (n_utt, 3)

        # ── Step 4: masked softmax ─────────────────────────────────────────────
        logits      = jnp.where(
            cand_mask_t,
            alpha_vec[None, :] * log_L_ref,
            -1e9,
        )
        local_probs = jax.nn.softmax(logits, axis=-1)                # (n_utt, 3)

        # ── Step 5: probability of actual token ───────────────────────────────
        chosen = jnp.sum(
            local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1
        )                                                             # (n_utt,)
        chosen     = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        # ── Step 6: posterior update ───────────────────────────────────────────
        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo",
            ACTUAL_TOK_ONEHOT[t],    # (n_utt, 3)
            log_sem_table,           # (n_utt, 3, n_obj)
        )                                                             # (n_utt, n_obj)

        log_updated_post = log_per_utt_posts + jnp.where(
            active_t[:, None],
            selected_log_sem,
            0.0,
        )
        log_Z_post        = jax.scipy.special.logsumexp(
            log_updated_post, axis=-1, keepdims=True
        )
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)  # (n_utt, n_obj)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(
        step,
        (init_scores, init_posts),
        jnp.arange(T),
    )

    log_unnorm = log_P_beta + log_final_scores + gamma * N_WORDS      # (n_utt,)
    model_probs = jax.nn.softmax(log_unnorm)                           # (n_utt,)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


def incremental_speaker_frozen(
    states:       jnp.ndarray,
    alpha_D:      float = 3.0,
    alpha_C:      float = 3.0,
    alpha_F:      float = 3.0,
    color_semval: float = 0.95,
    form_semval:  float = 0.80,
    k:            float = 0.50,
    wf:           float = 1.00,
    beta:         float = 1.00,
    gamma:        float = 0.0,
    epsilon:      float = 0.01,
) -> jnp.ndarray:

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]
    alpha_vec      = jnp.array([alpha_D, alpha_C, alpha_F])

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    lm_scaled  = jnp.power(jnp.clip(LM_PRIOR_15, eps), beta)
    log_P_beta = jnp.log(lm_scaled / jnp.sum(lm_scaled))

    colors = states[:, 1]
    forms  = states[:, 2]

    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem  = jnp.log(
        jnp.where(forms  == 1, form_semval,  1.0 - form_semval)  + eps
    )

    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))

    def size_log_sem_for_utt(post):
        sv = compute_size_semantics_fast_presorted(
            sizes,
            size_sort_idx,
            sizes_sorted,
            post,
            k,
            wf,
        )
        return jnp.log(jnp.clip(sv, eps))

    size_log_sems_frozen = jax.vmap(size_log_sem_for_utt)(init_posts)

    log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
    log_sem_table_frozen = jnp.concatenate([
        size_log_sems_frozen[:, None, :],
        jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
    ], axis=1)

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]
        active_t    = ACTIVE_POS[t]

        token_pres_t = TOKEN_PRESENT[t]
        log_prod_sem = jnp.einsum("uav, uvo -> uao", token_pres_t, log_sem_table_frozen)

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
        log_updated = log_per_utt_posts[:, None, :] + log_prod_sem

        log_Z = jax.scipy.special.logsumexp(log_updated, axis=-1)
        log_norm = log_updated - log_Z[:, :, None]
        log_L_ref = log_norm[:, :, referent_index]

        logits = jnp.where(cand_mask_t, alpha_vec[None, :] * log_L_ref, -1e9)
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo",
            ACTUAL_TOK_ONEHOT[t],
            log_sem_table_frozen,
        )

        log_updated_post = log_per_utt_posts + jnp.where(active_t[:, None], selected_log_sem, 0.0)
        log_Z_post = jax.scipy.special.logsumexp(log_updated_post, axis=-1, keepdims=True)
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(step, (init_scores, init_posts), jnp.arange(T))

    log_unnorm = log_P_beta + log_final_scores + gamma * N_WORDS
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


def incremental_speaker_lookahead(
    states:       jnp.ndarray,
    alpha:        float = 3.0,
    color_semval: float = 0.95,
    form_semval:  float = 0.80,
    k:            float = 0.50,
    wf:           float = 1.00,
    beta:         float = 1.00,
) -> jnp.ndarray:
    """Incremental speaker with d=1 lookahead at step 0, greedy at steps 1+.

    At step 0, each candidate first token is scored by the BEST 2-token
    continuation (max over valid second tokens), not by immediate
    informativeness.  Steps 1+ are identical to the greedy incremental speaker.
    """

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted  = sizes[size_sort_idx]

    lm_scaled  = jnp.power(jnp.clip(LM_PRIOR_15, eps), beta)
    log_P_beta = jnp.log(lm_scaled / jnp.sum(lm_scaled))

    colors = states[:, 1]
    forms  = states[:, 2]
    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem = jnp.log(
        jnp.where(forms == 1, form_semval, 1.0 - form_semval) + eps
    )

    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))

    # ── Lookahead precomputation ────────────────────────────────────
    # Size semantics at step 0 (from uniform prior)
    size_sem_0 = compute_size_semantics_fast_presorted(
        sizes, size_sort_idx, sizes_sorted, uniform, k, wf
    )
    log_size_sem_0 = jnp.log(jnp.clip(size_sem_0, eps))

    # Per-dimension log-semantics: (3, n_obj)  [D=size, C=color, F=form]
    log_sem_all = jnp.stack([log_size_sem_0, log_color_sem, log_form_sem])

    # Posterior after each possible first token: post_a ∝ uniform · sem(a)
    log_post_1 = jnp.log(uniform + eps)[None, :] + log_sem_all   # (3, n_obj)
    log_post_1 = log_post_1 - jax.scipy.special.logsumexp(
        log_post_1, axis=-1, keepdims=True
    )
    post_1 = jnp.exp(log_post_1)                                  # (3, n_obj)

    # Recursive size semantics recomputed from each updated posterior
    def _size_after(post):
        sv = compute_size_semantics_fast_presorted(
            sizes, size_sort_idx, sizes_sorted, post, k, wf
        )
        return jnp.log(jnp.clip(sv, eps))
    log_size_after = jax.vmap(_size_after)(post_1)                 # (3, n_obj)

    # Step-1 sem table conditioned on first token:
    # log_sem_1[a, dim, obj]  (3, 3, n_obj)
    log_sem_1 = jnp.stack([
        log_size_after,
        jnp.broadcast_to(log_color_sem[None, :], (3, n_obj)),
        jnp.broadcast_to(log_form_sem[None, :],  (3, n_obj)),
    ], axis=1)

    # L(s* | first=a, second=a') for all 3×3 pairs
    log_joint = log_post_1[:, None, :] + log_sem_1                 # (3, 3, n_obj)
    log_Z_la  = jax.scipy.special.logsumexp(log_joint, axis=-1)    # (3, 3)
    log_L_2   = log_joint[:, :, referent_index] - log_Z_la         # (3, 3)

    # Mask self-repetition (can't use same dimension twice)
    valid = 1.0 - jnp.eye(3)
    log_L_2_masked = jnp.where(valid > 0, log_L_2, -1e9)

    # Lookahead value per first token = max over valid second tokens
    lookahead_val = jnp.max(log_L_2_masked, axis=-1)              # (3,)

    # ── Scan (step 0 → lookahead, steps 1-2 → greedy) ──────────────
    def step(carry, t):
        log_scores, per_utt_posts = carry
        cand_mask_t = CANDIDATE_MASK[t]
        active_t    = ACTIVE_POS[t]

        # Size semantics per utterance (recursive)
        def size_log_sem_for_utt(post):
            sv = compute_size_semantics_fast_presorted(
                sizes, size_sort_idx, sizes_sorted, post, k, wf,
            )
            return jnp.log(jnp.clip(sv, eps))
        size_log_sems = jax.vmap(size_log_sem_for_utt)(per_utt_posts)

        log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
        log_sem_table = jnp.concatenate([
            size_log_sems[:, None, :],
            jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
        ], axis=1)

        # Greedy candidate scores
        token_pres_t = TOKEN_PRESENT[t]
        log_prod_sem = jnp.einsum(
            "uav, uvo -> uao", token_pres_t, log_sem_table
        )
        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
        log_updated = log_per_utt_posts[:, None, :] + log_prod_sem
        log_Z_step  = jax.scipy.special.logsumexp(log_updated, axis=-1)
        log_L_ref   = log_updated[:, :, referent_index] - log_Z_step

        greedy_logits = jnp.where(cand_mask_t, alpha * log_L_ref, -1e9)

        # Lookahead logits (broadcast: same value for all utterances at step 0)
        la_logits = jnp.where(
            cand_mask_t,
            alpha * jnp.broadcast_to(lookahead_val[None, :], (n_utt, 3)),
            -1e9,
        )

        logits      = jnp.where(t == 0, la_logits, greedy_logits)
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen     = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen     = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        # Posterior update
        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo", ACTUAL_TOK_ONEHOT[t], log_sem_table,
        )
        log_updated_post = log_per_utt_posts + jnp.where(
            active_t[:, None], selected_log_sem, 0.0,
        )
        log_Z_post        = jax.scipy.special.logsumexp(
            log_updated_post, axis=-1, keepdims=True
        )
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(
        step, (init_scores, init_posts), jnp.arange(T),
    )

    log_unnorm = log_P_beta + log_final_scores
    return jax.nn.softmax(log_unnorm)


# =============================================================================
# EXTENDED INCREMENTAL SPEAKER (per-dim alpha, length bias, lapse rate)
# =============================================================================

def incremental_speaker_extended(
    states:       jnp.ndarray,
    alpha:        float = 3.0,
    color_semval: float = 0.971,
    form_semval:  float = 0.50,
    k:            float = 0.50,
    wf:           float = 1.00,
    beta:         float = 1.00,
    gamma:        float = 0.00,
    epsilon:      float = 0.01,
    mu_C:         float = 0.00,
    mu_F:         float = 0.00,
) -> jnp.ndarray:
    """Incremental speaker with shared rationality, step-level mention biases,
    length bias, and lapse rate.

    Parameters
    ----------
    alpha : shared rationality (scales informativeness for all dimensions)
    gamma : length bias (positive → prefer longer utterances)
    epsilon : lapse rate (mixture weight on uniform)
    mu_C, mu_F : step-level mention biases (D = 0 reference).
        Positive values increase the tendency to mention that dimension
        regardless of informativeness (captures redundant modification).
    """

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    mu_vec = jnp.array([0.0, mu_C, mu_F])                     # (3,)

    lm_scaled  = jnp.power(jnp.clip(LM_PRIOR_15, eps), beta)
    log_P_beta = jnp.log(lm_scaled / jnp.sum(lm_scaled))

    colors = states[:, 1]
    forms  = states[:, 2]

    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem  = jnp.log(
        jnp.where(forms  == 1, form_semval,  1.0 - form_semval)  + eps
    )

    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]
        active_t    = ACTIVE_POS[t]

        # Size log-semantics per utterance (recursive)
        def size_log_sem_for_utt(post):
            sv = compute_size_semantics_fast_presorted(
                sizes, size_sort_idx, sizes_sorted, post, k, wf,
            )
            return jnp.log(jnp.clip(sv, eps))

        size_log_sems = jax.vmap(size_log_sem_for_utt)(per_utt_posts)

        log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
        log_sem_table = jnp.concatenate([
            size_log_sems[:, None, :],
            jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
        ], axis=1)

        # Candidate scores via TOKEN_PRESENT einsum
        token_pres_t = TOKEN_PRESENT[t]
        log_prod_sem = jnp.einsum(
            "uav, uvo -> uao", token_pres_t, log_sem_table
        )

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
        log_updated = log_per_utt_posts[:, None, :] + log_prod_sem

        log_Z    = jax.scipy.special.logsumexp(log_updated, axis=-1)
        log_norm = log_updated - log_Z[:, :, None]
        log_L_ref = log_norm[:, :, referent_index]

        # Shared alpha * informativeness + per-dimension mention bias
        logits = jnp.where(
            cand_mask_t,
            alpha * log_L_ref + mu_vec[None, :],
            -1e9,
        )
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen     = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        # Posterior update
        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo", ACTUAL_TOK_ONEHOT[t], log_sem_table,
        )
        log_updated_post = log_per_utt_posts + jnp.where(
            active_t[:, None], selected_log_sem, 0.0,
        )
        log_Z_post        = jax.scipy.special.logsumexp(
            log_updated_post, axis=-1, keepdims=True
        )
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(
        step, (init_scores, init_posts), jnp.arange(T),
    )

    # Length bias (mu already applied at each step inside the scan)
    log_unnorm = log_P_beta + log_final_scores + gamma * N_WORDS

    # Model probs before lapse
    model_probs = jax.nn.softmax(log_unnorm)

    # Lapse: mix with uniform
    probs = (1.0 - epsilon) * model_probs + epsilon / n_utt

    return probs


def incremental_speaker_v5(
    states:                jnp.ndarray,
    is_colour_sufficient:  float,
    is_sharp:              float,
    alpha_D:               float = 3.0,
    alpha_C:               float = 3.0,
    alpha_F:               float = 3.0,
    lambda_C:              float = 0.0,
    color_semval:          float = 0.95,
    form_semval:           float = 0.80,
    k:                     float = 0.50,
    wf:                    float = 1.00,
    beta:                  float = 1.00,
    gamma_1:               float = 0.0,
    gamma_2:               float = 0.0,
    delta_gamma_1:         float = 0.0,
    delta_gamma_2:         float = 0.0,
    eta_1:                 float = 0.0,
    eta_2:                 float = 0.0,
    mu_noncanon:           float = 0.0,
    epsilon:               float = 0.01,
) -> jnp.ndarray:
    """Incremental speaker v5: condition-gated lambda_C boost (R1: first-step only) +
    saturating two-step length bias with optional condition-dependent offsets
    (F1: delta_gamma_* applied additively on colour-sufficient trials to allow the
    length bonus to shrink there, pushing mass toward bare C over C-prefix compounds)."""

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]
    alpha_vec      = jnp.array([alpha_D, alpha_C, alpha_F])

    # ── NEW: extract sizes once here ──────────────────────────────────────────
    # states[:, 0] would be re-executed inside every vmap call without this
    sizes = states[:, 0]                                              # (n_obj,)
    size_sort_idx = jnp.argsort(sizes)                                # (n_obj,)
    sizes_sorted = sizes[size_sort_idx]                               # (n_obj,)
    # ─────────────────────────────────────────────────────────────────────────

    # ── LM utterance prior ────────────────────────────────────────────────────
    lm_scaled  = jnp.power(jnp.clip(LM_PRIOR_15, eps), beta)
    log_P_beta = jnp.log(lm_scaled / jnp.sum(lm_scaled))            # (n_utt,)

    # ── Context-independent log-semantics (color, form) ───────────────────────
    colors = states[:, 1]
    forms  = states[:, 2]

    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )                                                                  # (n_obj,)
    log_form_sem  = jnp.log(
        jnp.where(forms  == 1, form_semval,  1.0 - form_semval)  + eps
    )                                                                  # (n_obj,)

    # ── Scan carry ────────────────────────────────────────────────────────────
    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))          # (n_utt, n_obj)

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]    # (n_utt, 3)
        active_t    = ACTIVE_POS[t]        # (n_utt,)

        # ── Step 1: size log-semantics per utterance (reuse pre-sorted sizes) ─
        def size_log_sem_for_utt(post):
            sv = compute_size_semantics_fast_presorted(
                sizes,
                size_sort_idx,
                sizes_sorted,
                post,
                k,
                wf,
            )
            return jnp.log(jnp.clip(sv, eps))

        size_log_sems = jax.vmap(size_log_sem_for_utt)(per_utt_posts)   # (n_utt, n_obj)

        # ── Step 2: full log-sem table  (n_utt, 3, n_obj) ────────────────────
        log_sem_static = jnp.stack(
            [log_color_sem, log_form_sem], axis=0                    # (2, n_obj)
        )
        log_sem_table = jnp.concatenate([
            size_log_sems[:, None, :],                               # (n_utt, 1, n_obj)
            jnp.broadcast_to(
                log_sem_static[None, :, :], (n_utt, 2, n_obj)
            ),                                                        # (n_utt, 2, n_obj)
        ], axis=1)                                                    # (n_utt, 3, n_obj)

        # ── Step 3: candidate scores via TOKEN_PRESENT einsum ─────────────────
        token_pres_t = TOKEN_PRESENT[t]                              # (n_utt, 3, 3)

        log_prod_sem = jnp.einsum(
            "uav, uvo -> uao",
            token_pres_t,      # (n_utt, 3, 3)
            log_sem_table,     # (n_utt, 3, n_obj)
        )                                                             # (n_utt, 3, n_obj)

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))     # (n_utt, n_obj)
        log_prior = log_per_utt_posts

        log_updated = log_prior[:, None, :] + log_prod_sem           # (n_utt, 3, n_obj)

        log_Z    = jax.scipy.special.logsumexp(log_updated, axis=-1) # (n_utt, 3)
        log_norm = log_updated - log_Z[:, :, None]                   # (n_utt, 3, n_obj)
        log_L_ref = log_norm[:, :, referent_index]                   # (n_utt, 3)

        # ── Step 4: masked softmax (with condition-gated lambda_C boost) ──────
        # R1: boost active only at the first mention step (t == 0) so that bare-C
        # benefits but C-prefix compounds (CF, CDF) are not multiplied through.
        first_step_gate = (t == 0).astype(jnp.float32)
        boost_vec = jnp.array(
            [0.0, lambda_C * is_colour_sufficient * first_step_gate, 0.0]
        )                                                              # (3,)
        logits_with_boost = jnp.where(
            cand_mask_t,
            alpha_vec[None, :] * log_L_ref + boost_vec[None, :],
            -1e9,
        )
        local_probs = jax.nn.softmax(logits_with_boost, axis=-1)     # (n_utt, 3)

        # ── Step 5: probability of actual token ───────────────────────────────
        chosen = jnp.sum(
            local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1
        )                                                             # (n_utt,)
        chosen     = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        # ── Step 6: posterior update ───────────────────────────────────────────
        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo",
            ACTUAL_TOK_ONEHOT[t],    # (n_utt, 3)
            log_sem_table,           # (n_utt, 3, n_obj)
        )                                                             # (n_utt, n_obj)

        log_updated_post = log_per_utt_posts + jnp.where(
            active_t[:, None],
            selected_log_sem,
            0.0,
        )
        log_Z_post        = jax.scipy.special.logsumexp(
            log_updated_post, axis=-1, keepdims=True
        )
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)  # (n_utt, n_obj)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(
        step,
        (init_scores, init_posts),
        jnp.arange(T),
    )

    # ── Saturating length bias with F1 (colour-sufficient) + F2 (sharpness) offsets ──
    k_extra = jnp.maximum(N_WORDS - 1, 0)
    gamma_1_eff = gamma_1 + delta_gamma_1 * is_colour_sufficient + eta_1 * is_sharp
    gamma_2_eff = gamma_2 + delta_gamma_2 * is_colour_sufficient + eta_2 * is_sharp
    length_bonus = (
        gamma_1_eff * (k_extra >= 1).astype(jnp.float32)
        + gamma_2_eff * (k_extra >= 2).astype(jnp.float32)
    )
    noncanon_bonus = mu_noncanon * NONCANON_MASK                      # (n_utt,)
    log_unnorm = log_P_beta + log_final_scores + length_bonus + noncanon_bonus
    model_probs = jax.nn.softmax(log_unnorm)                          # (n_utt,)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


def incremental_speaker_frozen_v5(
    states:                jnp.ndarray,
    is_colour_sufficient:  float,
    is_sharp:              float,
    alpha_D:               float = 3.0,
    alpha_C:               float = 3.0,
    alpha_F:               float = 3.0,
    lambda_C:              float = 0.0,
    color_semval:          float = 0.95,
    form_semval:           float = 0.80,
    k:                     float = 0.50,
    wf:                    float = 1.00,
    beta:                  float = 1.00,
    gamma_1:               float = 0.0,
    gamma_2:               float = 0.0,
    delta_gamma_1:         float = 0.0,
    delta_gamma_2:         float = 0.0,
    eta_1:                 float = 0.0,
    eta_2:                 float = 0.0,
    mu_noncanon:           float = 0.0,
    epsilon:               float = 0.01,
) -> jnp.ndarray:
    """v5 with FROZEN (context-fixed) size semantics."""
    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]
    alpha_vec      = jnp.array([alpha_D, alpha_C, alpha_F])

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    lm_scaled  = jnp.power(jnp.clip(LM_PRIOR_15, eps), beta)
    log_P_beta = jnp.log(lm_scaled / jnp.sum(lm_scaled))

    colors = states[:, 1]
    forms  = states[:, 2]

    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem  = jnp.log(
        jnp.where(forms  == 1, form_semval,  1.0 - form_semval)  + eps
    )

    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))

    def size_log_sem_for_utt(post):
        sv = compute_size_semantics_fast_presorted(
            sizes, size_sort_idx, sizes_sorted, post, k, wf,
        )
        return jnp.log(jnp.clip(sv, eps))

    size_log_sems_frozen = jax.vmap(size_log_sem_for_utt)(init_posts)
    log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
    log_sem_table_frozen = jnp.concatenate([
        size_log_sems_frozen[:, None, :],
        jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
    ], axis=1)

    def step(carry, t):
        log_scores, per_utt_posts = carry
        cand_mask_t = CANDIDATE_MASK[t]
        active_t    = ACTIVE_POS[t]

        token_pres_t = TOKEN_PRESENT[t]
        log_prod_sem = jnp.einsum("uav, uvo -> uao", token_pres_t, log_sem_table_frozen)

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
        log_updated = log_per_utt_posts[:, None, :] + log_prod_sem
        log_Z = jax.scipy.special.logsumexp(log_updated, axis=-1)
        log_norm = log_updated - log_Z[:, :, None]
        log_L_ref = log_norm[:, :, referent_index]

        # R1: lambda_C active only at step 0
        first_step_gate = (t == 0).astype(jnp.float32)
        boost_vec = jnp.array(
            [0.0, lambda_C * is_colour_sufficient * first_step_gate, 0.0]
        )
        logits = jnp.where(
            cand_mask_t,
            alpha_vec[None, :] * log_L_ref + boost_vec[None, :],
            -1e9,
        )
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo", ACTUAL_TOK_ONEHOT[t], log_sem_table_frozen,
        )
        log_updated_post = log_per_utt_posts + jnp.where(active_t[:, None], selected_log_sem, 0.0)
        log_Z_post = jax.scipy.special.logsumexp(log_updated_post, axis=-1, keepdims=True)
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)
        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(step, (init_scores, init_posts), jnp.arange(T))

    k_extra = jnp.maximum(N_WORDS - 1, 0)
    gamma_1_eff = gamma_1 + delta_gamma_1 * is_colour_sufficient + eta_1 * is_sharp
    gamma_2_eff = gamma_2 + delta_gamma_2 * is_colour_sufficient + eta_2 * is_sharp
    length_bonus = (
        gamma_1_eff * (k_extra >= 1).astype(jnp.float32)
        + gamma_2_eff * (k_extra >= 2).astype(jnp.float32)
    )
    noncanon_bonus = mu_noncanon * NONCANON_MASK
    log_unnorm = log_P_beta + log_final_scores + length_bonus + noncanon_bonus
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


def global_speaker_v5(
    states:                jnp.ndarray,
    is_colour_sufficient:  float,
    is_sharp:              float,
    alpha:                 float = 3.0,
    lambda_C:              float = 0.0,
    color_sem:             float = 0.95,
    form_sem:              float = 0.80,
    k:                     float = 0.5,
    wf:                    float = 1.0,
    beta:                  float = 1.0,
    gamma_1:               float = 0.0,
    gamma_2:               float = 0.0,
    delta_gamma_1:         float = 0.0,
    delta_gamma_2:         float = 0.0,
    eta_1:                 float = 0.0,
    eta_2:                 float = 0.0,
    mu_noncanon:           float = 0.0,
    epsilon:               float = 0.01,
) -> jnp.ndarray:
    """Global speaker with v5 mechanisms: lambda_C lifts C-initial utterances on
    colour-sufficient trials; saturating two-step length bias with condition
    offsets (F1); canonical-ordering penalty (C1)."""
    eps = 1e-8
    costs = utterance_cost_jax(beta=beta)
    M_listener = incremental_semantics_jax(
        states=states, color_sem=color_sem, form_sem=form_sem, k=k, wf=wf,
    )
    log_L = jnp.log(jnp.clip(M_listener.T, eps))              # (n_obj, n_utt)

    k_extra = jnp.maximum(N_WORDS - 1, 0)
    gamma_1_eff = gamma_1 + delta_gamma_1 * is_colour_sufficient + eta_1 * is_sharp
    gamma_2_eff = gamma_2 + delta_gamma_2 * is_colour_sufficient + eta_2 * is_sharp
    length_bonus = (
        gamma_1_eff * (k_extra >= 1).astype(jnp.float32)
        + gamma_2_eff * (k_extra >= 2).astype(jnp.float32)
    )                                                          # (n_utt,)
    c_boost = lambda_C * is_colour_sufficient * C_INITIAL_MASK  # (n_utt,)
    noncanon_bonus = mu_noncanon * NONCANON_MASK               # (n_utt,)

    util = (
        alpha * log_L
        - costs[None, :]
        + length_bonus[None, :]
        + c_boost[None, :]
        + noncanon_bonus[None, :]
    )
    M_speaker = jax.nn.softmax(util, axis=-1)
    M_speaker = (1.0 - epsilon) * M_speaker + epsilon / M_listener.shape[0]
    return M_speaker[0, :]


def global_speaker_static_v5(
    states:                jnp.ndarray,
    is_colour_sufficient:  float,
    is_sharp:              float,
    alpha:                 float = 3.0,
    lambda_C:              float = 0.0,
    color_sem:             float = 0.95,
    form_sem:              float = 0.80,
    k:                     float = 0.5,
    wf:                    float = 1.0,
    beta:                  float = 1.0,
    gamma_1:               float = 0.0,
    gamma_2:               float = 0.0,
    delta_gamma_1:         float = 0.0,
    delta_gamma_2:         float = 0.0,
    eta_1:                 float = 0.0,
    eta_2:                 float = 0.0,
    mu_noncanon:           float = 0.0,
    epsilon:               float = 0.01,
) -> jnp.ndarray:
    """Global static (frozen size semantics) with v5 mechanisms."""
    eps = 1e-8
    costs = utterance_cost_jax(beta=beta)
    M_listener = incremental_semantics_jax_frozen(
        states=states, color_sem=color_sem, form_sem=form_sem, k=k, wf=wf,
    )
    log_L = jnp.log(jnp.clip(M_listener.T, eps))

    k_extra = jnp.maximum(N_WORDS - 1, 0)
    gamma_1_eff = gamma_1 + delta_gamma_1 * is_colour_sufficient + eta_1 * is_sharp
    gamma_2_eff = gamma_2 + delta_gamma_2 * is_colour_sufficient + eta_2 * is_sharp
    length_bonus = (
        gamma_1_eff * (k_extra >= 1).astype(jnp.float32)
        + gamma_2_eff * (k_extra >= 2).astype(jnp.float32)
    )
    c_boost = lambda_C * is_colour_sufficient * C_INITIAL_MASK
    noncanon_bonus = mu_noncanon * NONCANON_MASK

    util = (
        alpha * log_L
        - costs[None, :]
        + length_bonus[None, :]
        + c_boost[None, :]
        + noncanon_bonus[None, :]
    )
    M_speaker = jax.nn.softmax(util, axis=-1)
    M_speaker = (1.0 - epsilon) * M_speaker + epsilon / M_listener.shape[0]
    return M_speaker[0, :]


def incremental_speaker_contextual(
    states:                jnp.ndarray,
    sufficient_dim:        int,
    has_one_word_solution: float,
    is_sharp:              float,
    alpha_D:               float = 3.0,
    alpha_C:               float = 3.0,
    alpha_F:               float = 3.0,
    lambda_suff:           float = 0.0,
    color_semval:          float = 0.95,
    form_semval:           float = 0.80,
    k:                     float = 0.50,
    wf:                    float = 1.00,
    beta_lm:               float = 1.00,
    epsilon:               float = 0.01,
) -> jnp.ndarray:
    """Incremental speaker with a context-sensitive production layer.

    the base contextual variant: the LM term uses raw GPT-2 log-probability (LOG_LM_RAW_15) so the
    cumulative per-token negative log-prob naturally penalizes longer
    utterances, absorbing what was previously six separate gamma_* coefficients
    (gamma_1/2 length, gamma_oneword_1/2 one-word availability, gamma_sharp_1/2
    perceptual sharpness). Only first-word sufficiency boost (lambda_suff)
    remains as an explicit context term — it is theoretically the
    "lead-with-the-disambiguator" prior and is not in the LM signal.
    """

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]
    alpha_vec      = jnp.array([alpha_D, alpha_C, alpha_F])

    # has_one_word_solution and is_sharp are kept in the signature for stable
    # vmap-in-axes but are no longer consumed: the raw LM prior absorbs both
    # one-word and sharpness-conditioned length pressure.
    del has_one_word_solution
    del is_sharp

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    log_lm_raw = beta_lm * LOG_LM_RAW_15

    colors = states[:, 1]
    forms  = states[:, 2]
    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem  = jnp.log(
        jnp.where(forms  == 1, form_semval,  1.0 - form_semval)  + eps
    )

    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]
        active_t    = ACTIVE_POS[t]

        def size_log_sem_for_utt(post):
            sv = compute_size_semantics_fast_presorted(
                sizes,
                size_sort_idx,
                sizes_sorted,
                post,
                k,
                wf,
            )
            return jnp.log(jnp.clip(sv, eps))

        size_log_sems = jax.vmap(size_log_sem_for_utt)(per_utt_posts)

        log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
        log_sem_table = jnp.concatenate([
            size_log_sems[:, None, :],
            jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
        ], axis=1)

        token_pres_t = TOKEN_PRESENT[t]
        log_prod_sem = jnp.einsum(
            "uav, uvo -> uao",
            token_pres_t,
            log_sem_table,
        )

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
        log_updated = log_per_utt_posts[:, None, :] + log_prod_sem
        log_Z = jax.scipy.special.logsumexp(log_updated, axis=-1)
        log_norm = log_updated - log_Z[:, :, None]
        log_L_ref = log_norm[:, :, referent_index]

        first_step_gate = (t == 0).astype(jnp.float32)
        suff_boost_vec = lambda_suff * first_step_gate * jnp.array([
            sufficient_dim == 0,
            sufficient_dim == 1,
            sufficient_dim == 2,
        ], dtype=jnp.float32)
        logits = jnp.where(
            cand_mask_t,
            alpha_vec[None, :] * log_L_ref + suff_boost_vec[None, :],
            -1e9,
        )
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo",
            ACTUAL_TOK_ONEHOT[t],
            log_sem_table,
        )
        log_updated_post = log_per_utt_posts + jnp.where(
            active_t[:, None],
            selected_log_sem,
            0.0,
        )
        log_Z_post = jax.scipy.special.logsumexp(
            log_updated_post, axis=-1, keepdims=True
        )
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(
        step,
        (init_scores, init_posts),
        jnp.arange(T),
    )

    log_unnorm = log_lm_raw + log_final_scores
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


def incremental_speaker_contextual_lambdaunc(
    states:                jnp.ndarray,
    sufficient_dim:        int,
    has_one_word_solution: float,
    is_sharp:              float,
    alpha_D:               float = 3.0,
    alpha_C:               float = 3.0,
    alpha_F:               float = 3.0,
    lambda_suff:           float = 0.0,
    color_semval:          float = 0.95,
    form_semval:           float = 0.80,
    k:                     float = 0.50,
    wf:                    float = 1.00,
    beta_lm:               float = 1.00,
    lambda_uncertainty:    float = 0.0,
    epsilon:               float = 0.01,
) -> jnp.ndarray:
    """Contextual speaker with a principled listener-uncertainty cost term.

    Identical to ``incremental_speaker_contextual`` (the base contextual variant) except the
    speaker bears an implicit cost proportional to the residual listener
    uncertainty their utterance leaves over the candidate referents:

        log_unnorm = beta_lm * LOG_LM_RAW_15
                   + log_final_scores
                   - lambda_uncertainty * (1 - P_listener_final(target | u))

    where ``P_listener_final[u]`` is the listener's posterior on the referent
    after consuming utterance ``u`` (the second carry slot of ``lax.scan`` that
    the base contextual variant discards).  As ``lambda_uncertainty -> 0`` this reduces exactly to
    the base contextual variant.  As it grows, longer utterances that drive the listener posterior
    toward 1 are preferred — capturing the over-specification preference that
    the pre-loop baseline's six ``gamma_*`` step-coefficients were curve-
    fitting, with one theoretically grounded coefficient instead of six.
    """

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]
    alpha_vec      = jnp.array([alpha_D, alpha_C, alpha_F])

    del has_one_word_solution
    del is_sharp

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    log_lm_raw = beta_lm * LOG_LM_RAW_15

    colors = states[:, 1]
    forms  = states[:, 2]
    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem  = jnp.log(
        jnp.where(forms  == 1, form_semval,  1.0 - form_semval)  + eps
    )

    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]
        active_t    = ACTIVE_POS[t]

        def size_log_sem_for_utt(post):
            sv = compute_size_semantics_fast_presorted(
                sizes,
                size_sort_idx,
                sizes_sorted,
                post,
                k,
                wf,
            )
            return jnp.log(jnp.clip(sv, eps))

        size_log_sems = jax.vmap(size_log_sem_for_utt)(per_utt_posts)

        log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
        log_sem_table = jnp.concatenate([
            size_log_sems[:, None, :],
            jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
        ], axis=1)

        token_pres_t = TOKEN_PRESENT[t]
        log_prod_sem = jnp.einsum(
            "uav, uvo -> uao",
            token_pres_t,
            log_sem_table,
        )

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
        log_updated = log_per_utt_posts[:, None, :] + log_prod_sem
        log_Z = jax.scipy.special.logsumexp(log_updated, axis=-1)
        log_norm = log_updated - log_Z[:, :, None]
        log_L_ref = log_norm[:, :, referent_index]

        first_step_gate = (t == 0).astype(jnp.float32)
        suff_boost_vec = lambda_suff * first_step_gate * jnp.array([
            sufficient_dim == 0,
            sufficient_dim == 1,
            sufficient_dim == 2,
        ], dtype=jnp.float32)
        logits = jnp.where(
            cand_mask_t,
            alpha_vec[None, :] * log_L_ref + suff_boost_vec[None, :],
            -1e9,
        )
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo",
            ACTUAL_TOK_ONEHOT[t],
            log_sem_table,
        )
        log_updated_post = log_per_utt_posts + jnp.where(
            active_t[:, None],
            selected_log_sem,
            0.0,
        )
        log_Z_post = jax.scipy.special.logsumexp(
            log_updated_post, axis=-1, keepdims=True
        )
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, per_utt_posts_final), _ = lax.scan(
        step,
        (init_scores, init_posts),
        jnp.arange(T),
    )

    p_listener_final = per_utt_posts_final[:, referent_index]   # (n_utt,)
    uncertainty_cost = lambda_uncertainty * (1.0 - p_listener_final)
    log_unnorm = log_lm_raw + log_final_scores - uncertainty_cost
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


def incremental_speaker_simplified(
    states:                jnp.ndarray,
    sufficient_dim:        int,
    has_one_word_solution: float,
    is_sharp:              float,
    alpha:                 float = 3.0,
    beta_order:            float = 1.0,
    lambda_frontload:      float = 0.0,
    gamma_uncertainty_len: float = 0.0,
    color_semval:          float = 0.59,
    form_semval:           float = 0.50,
    k:                     float = 0.50,
    wf:                    float = 0.6856,
    epsilon:               float = 0.01,
    order_scores:          jnp.ndarray = LOG_LM_RESID_15,
) -> jnp.ndarray:
    """Simplified production speaker with three broad pressures.

    The model keeps one RSA scale, one baseline order prior, one first-word
    salience/frontloading pressure, and one uncertainty-gated length pressure.
    It intentionally avoids residual-specific terms from the extended
    production model.
    """

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]

    del has_one_word_solution

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    log_order_prior = beta_order * order_scores

    colors = states[:, 1]
    forms  = states[:, 2]
    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem  = jnp.log(
        jnp.where(forms  == 1, form_semval,  1.0 - form_semval)  + eps
    )

    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]
        active_t    = ACTIVE_POS[t]

        def size_log_sem_for_utt(post):
            sv = compute_size_semantics_fast_presorted(
                sizes,
                size_sort_idx,
                sizes_sorted,
                post,
                k,
                wf,
            )
            return jnp.log(jnp.clip(sv, eps))

        size_log_sems = jax.vmap(size_log_sem_for_utt)(per_utt_posts)

        log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
        log_sem_table = jnp.concatenate([
            size_log_sems[:, None, :],
            jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
        ], axis=1)

        token_pres_t = TOKEN_PRESENT[t]
        log_prod_sem = jnp.einsum(
            "uav, uvo -> uao",
            token_pres_t,
            log_sem_table,
        )

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
        log_updated = log_per_utt_posts[:, None, :] + log_prod_sem
        log_Z = jax.scipy.special.logsumexp(log_updated, axis=-1)
        log_norm = log_updated - log_Z[:, :, None]
        log_L_ref = log_norm[:, :, referent_index]

        first_step_gate = (t == 0).astype(jnp.float32)
        frontload_vec = lambda_frontload * first_step_gate * jnp.array([
            sufficient_dim == 0,
            sufficient_dim == 1,
            sufficient_dim == 2,
        ], dtype=jnp.float32)
        logits = jnp.where(
            cand_mask_t,
            alpha * log_L_ref + frontload_vec[None, :],
            -1e9,
        )
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo",
            ACTUAL_TOK_ONEHOT[t],
            log_sem_table,
        )
        log_updated_post = log_per_utt_posts + jnp.where(
            active_t[:, None],
            selected_log_sem,
            0.0,
        )
        log_Z_post = jax.scipy.special.logsumexp(
            log_updated_post, axis=-1, keepdims=True
        )
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(
        step,
        (init_scores, init_posts),
        jnp.arange(T),
    )

    blur_gate = 1.0 - is_sharp
    length_bonus = gamma_uncertainty_len * blur_gate * jnp.maximum(N_WORDS - 1.0, 0.0)
    log_unnorm = log_order_prior + log_final_scores + length_bonus
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


def _visual_salience_scores(
    states: jnp.ndarray,
    is_sharp: float,
    base_visual_salience: jnp.ndarray = BASE_VISUAL_SALIENCE,
) -> jnp.ndarray:
    """Soft non-pragmatic visual salience for D/C/F, centered by trial."""
    sizes = states[:, 0]
    colors = states[:, 1]
    forms = states[:, 2]

    size_scale = jnp.std(sizes) + 1e-8
    size_margin = (sizes[0] - jnp.max(sizes[1:])) / size_scale
    size_contrast = jax.nn.sigmoid(size_margin) * (0.5 + 0.5 * is_sharp)

    color_contrast = 1.0 - jnp.mean((colors[1:] == colors[0]).astype(jnp.float32))
    form_contrast = 1.0 - jnp.mean((forms[1:] == forms[0]).astype(jnp.float32))

    raw = base_visual_salience + jnp.array(
        [size_contrast, color_contrast, form_contrast],
        dtype=jnp.float32,
    )
    return raw - jnp.mean(raw)


def _size_entropy_for_wf(states: jnp.ndarray, k: float, wf: float) -> jnp.ndarray:
    """Entropy of the size-word listener posterior under a given size width."""
    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]
    uniform = jnp.ones(states.shape[0]) / states.shape[0]
    membership = compute_size_semantics_fast_presorted(
        sizes,
        size_sort_idx,
        sizes_sorted,
        uniform,
        k,
        wf,
    )
    post = membership / jnp.clip(jnp.sum(membership), 1e-8)
    post = jnp.clip(post, 1e-8, 1.0)
    return -jnp.sum(post * jnp.log(post))


def _size_uncertainty_excess(states: jnp.ndarray, is_sharp: float,
                             k: float, wf: float) -> jnp.ndarray:
    """Extra size uncertainty caused by degraded perceptual reliability."""
    sharp_entropy = _size_entropy_for_wf(states, k, wf)
    wf_eff = wf * (1.0 + (1.0 - is_sharp) * (BLUR_WF_MULTIPLIER - 1.0))
    effective_entropy = _size_entropy_for_wf(states, k, wf_eff)
    return jnp.maximum(effective_entropy - sharp_entropy, 0.0)


def _salience_continuation_load(salience_vec: jnp.ndarray) -> jnp.ndarray:
    """Relative salience of adjectives that are followed by another adjective."""
    salient = jnp.maximum(salience_vec, 0.0)
    continued_tokens = ACTUAL_TOK_ONEHOT[:-1] * ACTIVE_POS[1:, :, None]
    return jnp.einsum("tuv,v->u", continued_tokens, salient)


def incremental_speaker_principled(
    states:                jnp.ndarray,
    sufficient_dim:        int,
    has_one_word_solution: float,
    is_sharp:              float,
    alpha:                 float = 3.0,
    beta_order:            float = 1.0,
    lambda_salience:       float = 0.0,
    rho_salience_stop:     float = 0.0,
    gamma_uncertainty_len: float = 0.0,
    color_semval:          float = 0.59,
    form_semval:           float = 0.50,
    k:                     float = 0.50,
    wf:                    float = 0.6856,
    epsilon:               float = 0.01,
    order_scores:          jnp.ndarray = LOG_LM_ORDER_ONLY_15,
    base_visual_salience:  jnp.ndarray = BASE_VISUAL_SALIENCE,
    recursive:             bool = True,
    size_context_mode:     str = "posterior",
) -> jnp.ndarray:
    """Simplified speaker with order-only LM prior and derived visual features."""

    if size_context_mode not in ("posterior", "comparison_class"):
        raise ValueError(
            f"Unknown size_context_mode {size_context_mode!r}; "
            "expected 'posterior' or 'comparison_class'."
        )

    eps = 1e-8
    referent_index = 0
    n_obj = states.shape[0]

    del sufficient_dim, has_one_word_solution

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    log_order_prior = beta_order * order_scores
    salience_vec = _visual_salience_scores(states, is_sharp, base_visual_salience)

    colors = states[:, 1]
    forms = states[:, 2]
    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem = jnp.log(
        jnp.where(forms == 1, form_semval, 1.0 - form_semval) + eps
    )

    uniform = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts = jnp.broadcast_to(uniform, (n_utt, n_obj))

    def size_log_sem_for_utt(post):
        sv = compute_size_semantics_fast_presorted(
            sizes,
            size_sort_idx,
            sizes_sorted,
            post,
            k,
            wf,
        )
        return jnp.log(jnp.clip(sv, eps))

    size_log_sems_static = jax.vmap(size_log_sem_for_utt)(init_posts)

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]
        active_t = ACTIVE_POS[t]

        if size_context_mode == "comparison_class" and recursive:
            candidate_seqs = jnp.reshape(PREFIX_UTTS[t], (n_utt * VOCAB_SIZE, T))
            candidate_posts = jnp.reshape(
                literal_listener_comparison_class_batch(
                    states,
                    candidate_seqs,
                    color_semval,
                    form_semval,
                    k,
                    wf,
                ),
                (n_utt, VOCAB_SIZE, n_obj),
            )
            log_L_ref = jnp.log(jnp.clip(candidate_posts[:, :, referent_index], eps))
        else:
            size_log_sems_recursive = jax.vmap(size_log_sem_for_utt)(per_utt_posts)
            size_log_sems = size_log_sems_recursive if recursive else size_log_sems_static

            log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
            log_sem_table = jnp.concatenate([
                size_log_sems[:, None, :],
                jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
            ], axis=1)

            token_pres_t = TOKEN_PRESENT[t]
            log_prod_sem = jnp.einsum(
                "uav, uvo -> uao",
                token_pres_t,
                log_sem_table,
            )

            log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
            log_updated = log_per_utt_posts[:, None, :] + log_prod_sem
            log_Z = jax.scipy.special.logsumexp(log_updated, axis=-1)
            log_norm = log_updated - log_Z[:, :, None]
            log_L_ref = log_norm[:, :, referent_index]

        salience_boost = lambda_salience * salience_vec
        logits = jnp.where(
            cand_mask_t,
            alpha * log_L_ref + salience_boost[None, :],
            -1e9,
        )
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        if size_context_mode == "comparison_class" and recursive:
            actual_idx = ACTUAL_TOK[t][:, None, None]
            selected_posts = jnp.take_along_axis(
                candidate_posts,
                jnp.broadcast_to(actual_idx, (n_utt, 1, n_obj)),
                axis=1,
            )[:, 0, :]
            new_per_utt_posts = jnp.where(
                active_t[:, None],
                selected_posts,
                per_utt_posts,
            )
        else:
            selected_log_sem = jnp.einsum(
                "uv, uvo -> uo",
                ACTUAL_TOK_ONEHOT[t],
                log_sem_table,
            )
            log_updated_post = log_per_utt_posts + jnp.where(
                active_t[:, None],
                selected_log_sem,
                0.0,
            )
            log_Z_post = jax.scipy.special.logsumexp(
                log_updated_post, axis=-1, keepdims=True
            )
            new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(
        step,
        (init_scores, init_posts),
        jnp.arange(T),
    )

    size_uncertainty = _size_uncertainty_excess(states, is_sharp, k, wf)
    length_bonus = gamma_uncertainty_len * size_uncertainty * jnp.maximum(N_WORDS - 1.0, 0.0)
    salience_stop_cost = rho_salience_stop * _salience_continuation_load(salience_vec)
    log_unnorm = log_order_prior + log_final_scores + length_bonus - salience_stop_cost
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


def _principled_terminal_log_ref(
    states: jnp.ndarray,
    color_semval: float,
    form_semval: float,
    k: float,
    wf: float,
    recursive: bool,
    size_context_mode: str,
) -> jnp.ndarray:
    if size_context_mode == "comparison_class" and recursive:
        listener_fn = incremental_semantics_jax_comparison_class
    else:
        listener_fn = incremental_semantics_jax if recursive else incremental_semantics_jax_frozen
    listener = listener_fn(
        states=states,
        color_sem=color_semval,
        form_sem=form_semval,
        k=k,
        wf=wf,
    )
    return jnp.log(jnp.clip(listener[:, 0], 1e-8))


def incremental_speaker_principled_planned_prefix(
    states:                jnp.ndarray,
    sufficient_dim:        int,
    has_one_word_solution: float,
    is_sharp:              float,
    alpha:                 float = 3.0,
    beta_order:            float = 1.0,
    lambda_salience:       float = 0.0,
    rho_salience_stop:     float = 0.0,
    planning_scale:        float = 0.0,
    gamma_uncertainty_len: float = 0.0,
    color_semval:          float = 0.59,
    form_semval:           float = 0.50,
    k:                     float = 0.50,
    wf:                    float = 0.6856,
    epsilon:               float = 0.01,
    order_scores:          jnp.ndarray = LOG_LM_ORDER_ONLY_15,
    base_visual_salience:  jnp.ndarray = BASE_VISUAL_SALIENCE,
    recursive:             bool = True,
    size_context_mode:     str = "posterior",
) -> jnp.ndarray:
    """Principled incremental speaker with lookahead over reachable utterances.

    Local adjective choices retain the current greedy informativeness term, but
    candidate prefixes also receive a soft option value over complete
    utterances reachable from that prefix.  Setting ``planning_scale`` to zero
    recovers the original principled incremental speaker.
    """

    if size_context_mode not in ("posterior", "comparison_class"):
        raise ValueError(
            f"Unknown size_context_mode {size_context_mode!r}; "
            "expected 'posterior' or 'comparison_class'."
        )

    eps = 1e-8
    referent_index = 0
    n_obj = states.shape[0]

    del sufficient_dim, has_one_word_solution

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    log_order_prior = beta_order * order_scores
    salience_vec = _visual_salience_scores(states, is_sharp, base_visual_salience)

    colors = states[:, 1]
    forms = states[:, 2]
    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem = jnp.log(
        jnp.where(forms == 1, form_semval, 1.0 - form_semval) + eps
    )

    uniform = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts = jnp.broadcast_to(uniform, (n_utt, n_obj))

    def size_log_sem_for_utt(post):
        sv = compute_size_semantics_fast_presorted(
            sizes,
            size_sort_idx,
            sizes_sorted,
            post,
            k,
            wf,
        )
        return jnp.log(jnp.clip(sv, eps))

    size_log_sems_static = jax.vmap(size_log_sem_for_utt)(init_posts)
    terminal_log_ref = _principled_terminal_log_ref(
        states,
        color_semval,
        form_semval,
        k,
        wf,
        recursive,
        size_context_mode,
    )
    size_uncertainty = _size_uncertainty_excess(states, is_sharp, k, wf)
    length_bonus = gamma_uncertainty_len * size_uncertainty * jnp.maximum(N_WORDS - 1.0, 0.0)
    salience_stop_cost = rho_salience_stop * _salience_continuation_load(salience_vec)
    terminal_utility = (
        alpha * terminal_log_ref
        + log_order_prior
        + length_bonus
        - salience_stop_cost
    )

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]
        active_t = ACTIVE_POS[t]

        if size_context_mode == "comparison_class" and recursive:
            candidate_seqs = jnp.reshape(PREFIX_UTTS[t], (n_utt * VOCAB_SIZE, T))
            candidate_posts = jnp.reshape(
                literal_listener_comparison_class_batch(
                    states,
                    candidate_seqs,
                    color_semval,
                    form_semval,
                    k,
                    wf,
                ),
                (n_utt, VOCAB_SIZE, n_obj),
            )
            log_L_ref = jnp.log(jnp.clip(candidate_posts[:, :, referent_index], eps))
        else:
            size_log_sems_recursive = jax.vmap(size_log_sem_for_utt)(per_utt_posts)
            size_log_sems = size_log_sems_recursive if recursive else size_log_sems_static

            log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
            log_sem_table = jnp.concatenate([
                size_log_sems[:, None, :],
                jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
            ], axis=1)

            token_pres_t = TOKEN_PRESENT[t]
            log_prod_sem = jnp.einsum(
                "uav, uvo -> uao",
                token_pres_t,
                log_sem_table,
            )

            log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
            log_updated = log_per_utt_posts[:, None, :] + log_prod_sem
            log_Z = jax.scipy.special.logsumexp(log_updated, axis=-1)
            log_norm = log_updated - log_Z[:, :, None]
            log_L_ref = log_norm[:, :, referent_index]

        completion_values = jax.scipy.special.logsumexp(
            jnp.where(
                COMPLETION_MASK[t],
                terminal_utility[None, None, :],
                -1e9,
            ),
            axis=-1,
        )
        salience_boost = lambda_salience * salience_vec
        logits = jnp.where(
            cand_mask_t,
            alpha * log_L_ref
            + salience_boost[None, :]
            + planning_scale * completion_values,
            -1e9,
        )
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        if size_context_mode == "comparison_class" and recursive:
            actual_idx = ACTUAL_TOK[t][:, None, None]
            selected_posts = jnp.take_along_axis(
                candidate_posts,
                jnp.broadcast_to(actual_idx, (n_utt, 1, n_obj)),
                axis=1,
            )[:, 0, :]
            new_per_utt_posts = jnp.where(
                active_t[:, None],
                selected_posts,
                per_utt_posts,
            )
        else:
            selected_log_sem = jnp.einsum(
                "uv, uvo -> uo",
                ACTUAL_TOK_ONEHOT[t],
                log_sem_table,
            )
            log_updated_post = log_per_utt_posts + jnp.where(
                active_t[:, None],
                selected_log_sem,
                0.0,
            )
            log_Z_post = jax.scipy.special.logsumexp(
                log_updated_post, axis=-1, keepdims=True
            )
            new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(
        step,
        (init_scores, init_posts),
        jnp.arange(T),
    )

    log_unnorm = log_order_prior + log_final_scores + length_bonus - salience_stop_cost
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


def _apply_principled_response_policy(
    probs: jnp.ndarray,
    sufficient_dim: int,
    has_one_word_solution: float,
    is_sharp: float,
    is_colour_sufficient: float,
    lambda_sufficient_single: float,
    lambda_reliability_form: float,
    lambda_sufficient_form_pair: float = 0.0,
    lambda_three_word_penalty: float = 0.0,
    lambda_sharp_form_suppression: float = 0.0,
    lambda_size_sharp_single_bonus: float = 0.0,
    lambda_size_sharp_form_pair_penalty: float = 0.0,
) -> jnp.ndarray:
    dim_count = jnp.sum(FULL_PRESENT_15, axis=1)
    single_dim = (dim_count == 1.0).astype(jnp.float32)
    two_dim = (dim_count == 2.0).astype(jnp.float32)
    dim_id = jnp.argmax(FULL_PRESENT_15, axis=1)
    sufficient_match = (
        (sufficient_dim >= 0)
        & (dim_id == sufficient_dim)
        & (N_WORDS == 1.0)
    ).astype(jnp.float32)
    sufficient_single_bonus = (
        lambda_sufficient_single
        * has_one_word_solution
        * single_dim
        * sufficient_match
    )
    reliability_form_bonus = (
        lambda_reliability_form
        * (1.0 - is_colour_sufficient)
        * F_PRESENT_15
    )
    safe_sufficient_dim = jnp.maximum(sufficient_dim, 0)
    sufficient_present = FULL_PRESENT_15[:, safe_sufficient_dim]
    sufficient_form_pair_bonus = (
        lambda_sufficient_form_pair
        * has_one_word_solution
        * (sufficient_dim >= 0)
        * (sufficient_dim != 2)
        * two_dim
        * sufficient_present
        * F_PRESENT_15
    )
    three_word_penalty = lambda_three_word_penalty * (N_WORDS == 3.0)
    sharp_form_penalty = (
        lambda_sharp_form_suppression
        * is_sharp
        * has_one_word_solution
        * (sufficient_dim >= 0)
        * (sufficient_dim != 2)
        * F_PRESENT_15
    )
    size_sharp_gate = is_sharp * has_one_word_solution * (sufficient_dim == 0)
    exact_size_single = (N_WORDS == 1.0) * FULL_PRESENT_15[:, 0]
    exact_size_form_pair = (
        (N_WORDS == 2.0)
        * FULL_PRESENT_15[:, 0]
        * (1.0 - FULL_PRESENT_15[:, 1])
        * FULL_PRESENT_15[:, 2]
    )
    size_sharp_single_bonus = (
        lambda_size_sharp_single_bonus * size_sharp_gate * exact_size_single
    )
    size_sharp_form_pair_penalty = (
        lambda_size_sharp_form_pair_penalty * size_sharp_gate * exact_size_form_pair
    )
    logits = (
        jnp.log(jnp.clip(probs, 1e-12))
        + sufficient_single_bonus
        + reliability_form_bonus
        + sufficient_form_pair_bonus
        + size_sharp_single_bonus
        - three_word_penalty
        - sharp_form_penalty
        - size_sharp_form_pair_penalty
    )
    return jax.nn.softmax(logits)


def incremental_speaker_principled_response_policy(
    states:                    jnp.ndarray,
    sufficient_dim:            int,
    has_one_word_solution:     float,
    is_sharp:                  float,
    is_colour_sufficient:      float,
    alpha:                     float = 3.0,
    beta_order:                float = 1.0,
    lambda_salience:           float = 0.0,
    rho_salience_stop:         float = 0.0,
    lambda_sufficient_single:  float = 0.0,
    lambda_reliability_form:   float = 0.0,
    lambda_sufficient_form_pair: float = 0.0,
    lambda_three_word_penalty:   float = 0.0,
    lambda_sharp_form_suppression: float = 0.0,
    lambda_size_sharp_single_bonus: float = 0.0,
    lambda_size_sharp_form_pair_penalty: float = 0.0,
    gamma_uncertainty_len:     float = 0.0,
    color_semval:              float = 0.59,
    form_semval:               float = 0.50,
    k:                         float = 0.50,
    wf:                        float = 0.6856,
    epsilon:                   float = 0.01,
    order_scores:              jnp.ndarray = LOG_LM_ORDER_ONLY_15,
    base_visual_salience:      jnp.ndarray = BASE_VISUAL_SALIENCE,
    recursive:                 bool = True,
    size_context_mode:         str = "posterior",
) -> jnp.ndarray:
    """Principled incremental speaker plus two response-policy pressures."""
    base_probs = incremental_speaker_principled(
        states,
        sufficient_dim,
        has_one_word_solution,
        is_sharp,
        alpha,
        beta_order,
        lambda_salience,
        rho_salience_stop,
        gamma_uncertainty_len,
        color_semval,
        form_semval,
        k,
        wf,
        epsilon,
        order_scores,
        base_visual_salience,
        recursive=recursive,
        size_context_mode=size_context_mode,
    )
    return _apply_principled_response_policy(
        base_probs,
        sufficient_dim,
        has_one_word_solution,
        is_sharp,
        is_colour_sufficient,
        lambda_sufficient_single,
        lambda_reliability_form,
        lambda_sufficient_form_pair,
        lambda_three_word_penalty,
        lambda_sharp_form_suppression,
        lambda_size_sharp_single_bonus,
        lambda_size_sharp_form_pair_penalty,
    )


def global_speaker_principled(
    states:                jnp.ndarray,
    sufficient_dim:        int,
    has_one_word_solution: float,
    is_sharp:              float,
    alpha:                 float = 3.0,
    beta_order:            float = 1.0,
    lambda_salience:       float = 0.0,
    rho_salience_stop:     float = 0.0,
    gamma_uncertainty_len: float = 0.0,
    color_semval:          float = 0.59,
    form_semval:           float = 0.50,
    k:                     float = 0.50,
    wf:                    float = 0.6856,
    epsilon:               float = 0.01,
    order_scores:          jnp.ndarray = LOG_LM_ORDER_ONLY_15,
    base_visual_salience:  jnp.ndarray = BASE_VISUAL_SALIENCE,
    recursive:             bool = True,
    size_context_mode:     str = "posterior",
) -> jnp.ndarray:
    """Global utterance-choice counterpart of the principled speaker."""

    if size_context_mode not in ("posterior", "comparison_class"):
        raise ValueError(
            f"Unknown size_context_mode {size_context_mode!r}; "
            "expected 'posterior' or 'comparison_class'."
        )

    del sufficient_dim, has_one_word_solution

    if size_context_mode == "comparison_class" and recursive:
        listener_fn = incremental_semantics_jax_comparison_class
    else:
        listener_fn = incremental_semantics_jax if recursive else incremental_semantics_jax_frozen
    listener = listener_fn(
        states=states,
        color_sem=color_semval,
        form_sem=form_semval,
        k=k,
        wf=wf,
    )
    log_L_ref = jnp.log(jnp.clip(listener[:, 0], 1e-8))

    salience_vec = _visual_salience_scores(states, is_sharp, base_visual_salience)
    utterance_salience = jnp.einsum("ud,d->u", FULL_PRESENT_15, salience_vec)
    salience_boost = lambda_salience * utterance_salience

    size_uncertainty = _size_uncertainty_excess(states, is_sharp, k, wf)
    length_bonus = gamma_uncertainty_len * size_uncertainty * jnp.maximum(N_WORDS - 1.0, 0.0)
    salience_stop_cost = rho_salience_stop * _salience_continuation_load(salience_vec)

    log_unnorm = (
        beta_order * order_scores
        + alpha * log_L_ref
        + salience_boost
        + length_bonus
        - salience_stop_cost
    )
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


def global_speaker_principled_response_policy(
    states:                    jnp.ndarray,
    sufficient_dim:            int,
    has_one_word_solution:     float,
    is_sharp:                  float,
    is_colour_sufficient:      float,
    alpha:                     float = 3.0,
    beta_order:                float = 1.0,
    lambda_salience:           float = 0.0,
    rho_salience_stop:         float = 0.0,
    lambda_sufficient_single:  float = 0.0,
    lambda_reliability_form:   float = 0.0,
    lambda_sufficient_form_pair: float = 0.0,
    lambda_three_word_penalty:   float = 0.0,
    lambda_sharp_form_suppression: float = 0.0,
    lambda_size_sharp_single_bonus: float = 0.0,
    lambda_size_sharp_form_pair_penalty: float = 0.0,
    gamma_uncertainty_len:     float = 0.0,
    color_semval:              float = 0.59,
    form_semval:               float = 0.50,
    k:                         float = 0.50,
    wf:                        float = 0.6856,
    epsilon:                   float = 0.01,
    order_scores:              jnp.ndarray = LOG_LM_ORDER_ONLY_15,
    base_visual_salience:      jnp.ndarray = BASE_VISUAL_SALIENCE,
    recursive:                 bool = True,
    size_context_mode:         str = "posterior",
) -> jnp.ndarray:
    """Principled global speaker plus the shared utterance-level policy."""
    base_probs = global_speaker_principled(
        states,
        sufficient_dim,
        has_one_word_solution,
        is_sharp,
        alpha,
        beta_order,
        lambda_salience,
        rho_salience_stop,
        gamma_uncertainty_len,
        color_semval,
        form_semval,
        k,
        wf,
        epsilon,
        order_scores,
        base_visual_salience,
        recursive=recursive,
        size_context_mode=size_context_mode,
    )
    return _apply_principled_response_policy(
        base_probs,
        sufficient_dim,
        has_one_word_solution,
        is_sharp,
        is_colour_sufficient,
        lambda_sufficient_single,
        lambda_reliability_form,
        lambda_sufficient_form_pair,
        lambda_three_word_penalty,
        lambda_sharp_form_suppression,
        lambda_size_sharp_single_bonus,
        lambda_size_sharp_form_pair_penalty,
    )


# ── Vectorise over trials ──────────────────────────────────────────────────────
vectorized_incremental_speaker = jax.vmap(
    incremental_speaker,
    in_axes=(0,    # states      — one trial per row
             None, # alpha_D
             None, # alpha_C
             None, # alpha_F
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta
             None, # gamma
             None, # epsilon
             ),
)

vectorized_incremental_speaker_frozen = jax.vmap(
    incremental_speaker_frozen,
    in_axes=(0,    # states
             None, # alpha_D
             None, # alpha_C
             None, # alpha_F
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta
             None, # gamma
             None, # epsilon
             ),
)

vectorized_incremental_speaker_lookahead = jax.vmap(
    incremental_speaker_lookahead,
    in_axes=(0, None, None, None, None, None, None),
)

# ── Pre JIT ──────────────────────────────────────────────────────
@jax.jit
def jitted_speaker(states, alpha_D, alpha_C, alpha_F, color_semval, form_semval, k, wf, beta, gamma, epsilon):
    return vectorized_incremental_speaker(
        states, alpha_D, alpha_C, alpha_F, color_semval, form_semval, k, wf, beta, gamma, epsilon
    )

@jax.jit
def jitted_speaker_frozen(states, alpha_D, alpha_C, alpha_F, color_semval, form_semval, k, wf, beta, gamma, epsilon):
    return vectorized_incremental_speaker_frozen(
        states, alpha_D, alpha_C, alpha_F, color_semval, form_semval, k, wf, beta, gamma, epsilon
    )

@jax.jit
def jitted_speaker_lookahead(states, alpha, color_semval, form_semval, k, wf, beta):
    return vectorized_incremental_speaker_lookahead(
        states, alpha, color_semval, form_semval, k, wf, beta
    )

# ── Extended speaker vectorize + JIT ─────────────────────────────────────────
vectorized_incremental_speaker_extended = jax.vmap(
    incremental_speaker_extended,
    in_axes=(0,    # states
             None, # alpha
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta
             None, # gamma
             None, # epsilon
             None, # mu_C
             None, # mu_F
             ),
)

@jax.jit
def jitted_speaker_extended(states, alpha, color_semval, form_semval,
                             k, wf, beta, gamma, epsilon, mu_C, mu_F):
    return vectorized_incremental_speaker_extended(
        states, alpha, color_semval, form_semval, k, wf, beta,
        gamma, epsilon, mu_C, mu_F,
    )

# Hierarchical: per-trial alpha
vectorized_incremental_speaker_extended_hier = jax.vmap(
    incremental_speaker_extended,
    in_axes=(0,    # states
             0,    # alpha  ← per-trial
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta
             None, # gamma
             None, # epsilon
             None, # mu_C
             None, # mu_F
             ),
)

@jax.jit
def jitted_speaker_extended_hier(states, alpha_per_trial, color_semval,
                                  form_semval, k, wf, beta, gamma, epsilon,
                                  mu_C, mu_F):
    return vectorized_incremental_speaker_extended_hier(
        states, alpha_per_trial, color_semval, form_semval, k, wf, beta,
        gamma, epsilon, mu_C, mu_F,
    )

# Hierarchical: per-trial alpha_D, alpha_C, alpha_F and per-trial is_colour_sufficient
vectorized_incremental_speaker_v5_hier = jax.vmap(
    incremental_speaker_v5,
    in_axes=(0,    # states
             0,    # is_colour_sufficient ← per-trial
             0,    # is_sharp ← per-trial
             0,    # alpha_D ← per-trial
             0,    # alpha_C ← per-trial
             0,    # alpha_F ← per-trial
             None, # lambda_C
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta
             None, # gamma_1
             None, # gamma_2
             None, # delta_gamma_1
             None, # delta_gamma_2
             None, # eta_1
             None, # eta_2
             None, # mu_noncanon
             None, # epsilon
             ),
)

@jax.jit
def jitted_speaker_v5_hier(
    states, is_colour_sufficient, is_sharp,
    alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
    lambda_C, color_semval, form_semval, k, wf, beta,
    gamma_1, gamma_2, delta_gamma_1, delta_gamma_2, eta_1, eta_2, mu_noncanon, epsilon,
):
    return vectorized_incremental_speaker_v5_hier(
        states, is_colour_sufficient, is_sharp,
        alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
        lambda_C, color_semval, form_semval, k, wf, beta,
        gamma_1, gamma_2, delta_gamma_1, delta_gamma_2, eta_1, eta_2, mu_noncanon, epsilon,
    )


# Contextual compromise model: incremental recursive, per-trial alpha + generic
# context predictors.
vectorized_incremental_speaker_contextual_hier = jax.vmap(
    incremental_speaker_contextual,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # alpha_D
             0,    # alpha_C
             0,    # alpha_F
             None, # lambda_suff
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta_lm
             None, # epsilon
             ),
)

@jax.jit
def jitted_speaker_contextual_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
    lambda_suff, color_semval, form_semval, k, wf, beta_lm,
    epsilon,
):
    return vectorized_incremental_speaker_contextual_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
        lambda_suff, color_semval, form_semval, k, wf, beta_lm,
        epsilon,
    )


# Contextual + listener-uncertainty cost: same as the base contextual variant vmap but with
# one extra broadcast slot for the scalar lambda_uncertainty parameter.
vectorized_incremental_speaker_contextual_lambdaunc_hier = jax.vmap(
    incremental_speaker_contextual_lambdaunc,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # alpha_D
             0,    # alpha_C
             0,    # alpha_F
             None, # lambda_suff
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta_lm
             None, # lambda_uncertainty
             None, # epsilon
             ),
)

@jax.jit
def jitted_speaker_contextual_lambdaunc_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
    lambda_suff, color_semval, form_semval, k, wf, beta_lm,
    lambda_uncertainty, epsilon,
):
    return vectorized_incremental_speaker_contextual_lambdaunc_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
        lambda_suff, color_semval, form_semval, k, wf, beta_lm,
        lambda_uncertainty, epsilon,
    )


vectorized_incremental_speaker_simplified_hier = jax.vmap(
    incremental_speaker_simplified,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # alpha
             None, # beta_order
             None, # lambda_frontload
             None, # gamma_uncertainty_len
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # epsilon
             None, # order_scores
             ),
)

@jax.jit
def jitted_speaker_simplified_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_per_trial, beta_order, lambda_frontload, gamma_uncertainty_len,
    color_semval, form_semval, k, wf, epsilon, order_scores,
):
    return vectorized_incremental_speaker_simplified_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        alpha_per_trial, beta_order, lambda_frontload, gamma_uncertainty_len,
        color_semval, form_semval, k, wf, epsilon, order_scores,
    )


vectorized_incremental_speaker_principled_hier = jax.vmap(
    incremental_speaker_principled,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # alpha
             None, # beta_order
             None, # lambda_salience
             None, # rho_salience_stop
             None, # gamma_uncertainty_len
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # epsilon
             None, # order_scores
             None, # base_visual_salience
             None, # recursive
             None, # size_context_mode
             ),
)

vectorized_incremental_speaker_principled_planned_hier = jax.vmap(
    incremental_speaker_principled_planned_prefix,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # alpha
             None, # beta_order
             None, # lambda_salience
             None, # rho_salience_stop
             None, # planning_scale
             None, # gamma_uncertainty_len
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # epsilon
             None, # order_scores
             None, # base_visual_salience
             None, # recursive
             None, # size_context_mode
             ),
)

vectorized_incremental_speaker_principled_response_policy_hier = jax.vmap(
    incremental_speaker_principled_response_policy,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # is_colour_sufficient
             0,    # alpha
             None, # beta_order
             None, # lambda_salience
             None, # rho_salience_stop
             None, # lambda_sufficient_single
             None, # lambda_reliability_form
             None, # lambda_sufficient_form_pair
             None, # lambda_three_word_penalty
             None, # lambda_sharp_form_suppression
             None, # lambda_size_sharp_single_bonus
             None, # lambda_size_sharp_form_pair_penalty
             None, # gamma_uncertainty_len
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # epsilon
             None, # order_scores
             None, # base_visual_salience
             None, # recursive
             None, # size_context_mode
             ),
)

@partial(jax.jit, static_argnames=("recursive", "size_context_mode"))
def jitted_speaker_principled_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_per_trial, beta_order, lambda_salience, rho_salience_stop,
    gamma_uncertainty_len,
    color_semval, form_semval, k, wf, epsilon, order_scores,
    base_visual_salience, recursive=True, size_context_mode="posterior",
):
    return vectorized_incremental_speaker_principled_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        alpha_per_trial, beta_order, lambda_salience, rho_salience_stop,
        gamma_uncertainty_len, color_semval, form_semval, k, wf, epsilon,
        order_scores, base_visual_salience, recursive, size_context_mode,
    )


@partial(jax.jit, static_argnames=("recursive", "size_context_mode"))
def jitted_speaker_principled_planned_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_per_trial, beta_order, lambda_salience, rho_salience_stop,
    planning_scale, gamma_uncertainty_len,
    color_semval, form_semval, k, wf, epsilon, order_scores,
    base_visual_salience, recursive=True, size_context_mode="posterior",
):
    return vectorized_incremental_speaker_principled_planned_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        alpha_per_trial, beta_order, lambda_salience, rho_salience_stop,
        planning_scale, gamma_uncertainty_len, color_semval, form_semval, k,
        wf, epsilon, order_scores, base_visual_salience, recursive,
        size_context_mode,
    )


@partial(jax.jit, static_argnames=("recursive", "size_context_mode"))
def jitted_speaker_principled_response_policy_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp, is_colour_sufficient,
    alpha_per_trial, beta_order, lambda_salience, rho_salience_stop,
    lambda_sufficient_single, lambda_reliability_form,
    lambda_sufficient_form_pair, lambda_three_word_penalty,
    lambda_sharp_form_suppression, lambda_size_sharp_single_bonus,
    lambda_size_sharp_form_pair_penalty,
    gamma_uncertainty_len,
    color_semval, form_semval, k, wf, epsilon, order_scores,
    base_visual_salience, recursive=True, size_context_mode="posterior",
):
    return vectorized_incremental_speaker_principled_response_policy_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        is_colour_sufficient, alpha_per_trial, beta_order, lambda_salience,
        rho_salience_stop, lambda_sufficient_single, lambda_reliability_form,
        lambda_sufficient_form_pair, lambda_three_word_penalty,
        lambda_sharp_form_suppression, lambda_size_sharp_single_bonus,
        lambda_size_sharp_form_pair_penalty,
        gamma_uncertainty_len, color_semval, form_semval, k, wf, epsilon,
        order_scores, base_visual_salience, recursive, size_context_mode,
    )


vectorized_global_speaker_principled_hier = jax.vmap(
    global_speaker_principled,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # alpha
             None, # beta_order
             None, # lambda_salience
             None, # rho_salience_stop
             None, # gamma_uncertainty_len
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # epsilon
             None, # order_scores
             None, # base_visual_salience
             None, # recursive
             None, # size_context_mode
             ),
)


vectorized_global_speaker_principled_response_policy_hier = jax.vmap(
    global_speaker_principled_response_policy,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # is_colour_sufficient
             0,    # alpha
             None, # beta_order
             None, # lambda_salience
             None, # rho_salience_stop
             None, # lambda_sufficient_single
             None, # lambda_reliability_form
             None, # lambda_sufficient_form_pair
             None, # lambda_three_word_penalty
             None, # lambda_sharp_form_suppression
             None, # lambda_size_sharp_single_bonus
             None, # lambda_size_sharp_form_pair_penalty
             None, # gamma_uncertainty_len
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # epsilon
             None, # order_scores
             None, # base_visual_salience
             None, # recursive
             None, # size_context_mode
             ),
)


@partial(jax.jit, static_argnames=("recursive", "size_context_mode"))
def jitted_global_speaker_principled_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_per_trial, beta_order, lambda_salience, rho_salience_stop,
    gamma_uncertainty_len,
    color_semval, form_semval, k, wf, epsilon, order_scores,
    base_visual_salience, recursive=True, size_context_mode="posterior",
):
    return vectorized_global_speaker_principled_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        alpha_per_trial, beta_order, lambda_salience, rho_salience_stop,
        gamma_uncertainty_len, color_semval, form_semval, k, wf, epsilon,
        order_scores, base_visual_salience, recursive, size_context_mode,
    )


@partial(jax.jit, static_argnames=("recursive", "size_context_mode"))
def jitted_global_speaker_principled_response_policy_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp, is_colour_sufficient,
    alpha_per_trial, beta_order, lambda_salience, rho_salience_stop,
    lambda_sufficient_single, lambda_reliability_form,
    lambda_sufficient_form_pair, lambda_three_word_penalty,
    lambda_sharp_form_suppression, lambda_size_sharp_single_bonus,
    lambda_size_sharp_form_pair_penalty,
    gamma_uncertainty_len,
    color_semval, form_semval, k, wf, epsilon, order_scores,
    base_visual_salience, recursive=True, size_context_mode="posterior",
):
    return vectorized_global_speaker_principled_response_policy_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        is_colour_sufficient, alpha_per_trial, beta_order, lambda_salience,
        rho_salience_stop, lambda_sufficient_single, lambda_reliability_form,
        lambda_sufficient_form_pair, lambda_three_word_penalty,
        lambda_sharp_form_suppression, lambda_size_sharp_single_bonus,
        lambda_size_sharp_form_pair_penalty,
        gamma_uncertainty_len, color_semval, form_semval, k, wf, epsilon,
        order_scores, base_visual_salience, recursive, size_context_mode,
    )


# ── v5: incremental frozen (context-fixed) — per-trial alpha + flag ─────────
vectorized_incremental_speaker_frozen_v5_hier = jax.vmap(
    incremental_speaker_frozen_v5,
    in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None, None,
             None, None, None, None, None, None, None, None),
)

@jax.jit
def jitted_speaker_frozen_v5_hier(
    states, is_colour_sufficient, is_sharp,
    alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
    lambda_C, color_semval, form_semval, k, wf, beta,
    gamma_1, gamma_2, delta_gamma_1, delta_gamma_2, eta_1, eta_2, mu_noncanon, epsilon,
):
    return vectorized_incremental_speaker_frozen_v5_hier(
        states, is_colour_sufficient, is_sharp,
        alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
        lambda_C, color_semval, form_semval, k, wf, beta,
        gamma_1, gamma_2, delta_gamma_1, delta_gamma_2, eta_1, eta_2, mu_noncanon, epsilon,
    )


# ── v5: global (context-updating) — per-trial alpha (single) + flag ─────────
vectorized_global_speaker_v5_hier = jax.vmap(
    global_speaker_v5,
    in_axes=(0, 0, 0, 0, None, None, None, None, None, None,
             None, None, None, None, None, None, None, None),
)

@jax.jit
def jitted_global_speaker_v5_hier(
    states, is_colour_sufficient, is_sharp, alpha_per_trial,
    lambda_C, color_semval, form_semval, k, wf, beta,
    gamma_1, gamma_2, delta_gamma_1, delta_gamma_2, eta_1, eta_2, mu_noncanon, epsilon,
):
    return vectorized_global_speaker_v5_hier(
        states, is_colour_sufficient, is_sharp, alpha_per_trial,
        lambda_C, color_semval, form_semval, k, wf, beta,
        gamma_1, gamma_2, delta_gamma_1, delta_gamma_2, eta_1, eta_2, mu_noncanon, epsilon,
    )


# ── v5: global static (context-fixed) ──────────────────────────────────────
vectorized_global_speaker_static_v5_hier = jax.vmap(
    global_speaker_static_v5,
    in_axes=(0, 0, 0, 0, None, None, None, None, None, None,
             None, None, None, None, None, None, None, None),
)

@jax.jit
def jitted_global_speaker_static_v5_hier(
    states, is_colour_sufficient, is_sharp, alpha_per_trial,
    lambda_C, color_semval, form_semval, k, wf, beta,
    gamma_1, gamma_2, delta_gamma_1, delta_gamma_2, eta_1, eta_2, mu_noncanon, epsilon,
):
    return vectorized_global_speaker_static_v5_hier(
        states, is_colour_sufficient, is_sharp, alpha_per_trial,
        lambda_C, color_semval, form_semval, k, wf, beta,
        gamma_1, gamma_2, delta_gamma_1, delta_gamma_2, eta_1, eta_2, mu_noncanon, epsilon,
    )


# Warm up JIT with dummy values
_dummy_states = jnp.ones((len(states), 6, 3))
try:
    _ = jitted_speaker(_dummy_states, 3.0, 3.0, 3.0, 0.95, 0.80, 0.5, 1.0, 1.0, 0.0, 0.01)
    _.block_until_ready()
    _ = jitted_speaker_frozen(_dummy_states, 3.0, 3.0, 3.0, 0.95, 0.80, 0.5, 1.0, 1.0, 0.0, 0.01)
    _.block_until_ready()
except Exception as e:
    print(f"JIT warmup skipped: {e}")



# =============================================================================
# HIERARCHICAL VMAPS  (map over states AND alpha — one alpha per trial)
# =============================================================================

vectorized_global_speaker_hier = jax.vmap(
    global_speaker,
    in_axes=(0,    # states
             0,    # alpha  ← per-trial (participant-specific)
             None, # color_sem
             None, # form_sem
             None, # k
             None, # wf
             None, # beta
             None, # gamma
             None, # epsilon
             ),
)

@jax.jit
def jitted_global_speaker_hier(states, alpha_per_trial, color_semval, form_semval, k, wf, beta, gamma, epsilon):
    return vectorized_global_speaker_hier(
        states, alpha_per_trial, color_semval, form_semval, k, wf, beta, gamma, epsilon
    )

vectorized_incremental_speaker_hier = jax.vmap(
    incremental_speaker,
    in_axes=(0,    # states
             0,    # alpha_D  ← per-trial
             0,    # alpha_C  ← per-trial
             0,    # alpha_F  ← per-trial
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta
             None, # gamma
             None, # epsilon
             ),
)

@jax.jit
def jitted_speaker_hier(states, alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                         color_semval, form_semval, k, wf, beta, gamma, epsilon):
    return vectorized_incremental_speaker_hier(
        states, alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
        color_semval, form_semval, k, wf, beta, gamma, epsilon
    )

# --- Static-semantics hierarchical vmaps ---

vectorized_global_speaker_static_hier = jax.vmap(
    global_speaker_static,
    in_axes=(0,    # states
             0,    # alpha  ← per-trial
             None, # color_sem
             None, # form_sem
             None, # k
             None, # wf
             None, # beta
             None, # gamma
             None, # epsilon
             ),
)

@jax.jit
def jitted_global_speaker_static_hier(states, alpha_per_trial, color_semval, form_semval, k, wf, beta, gamma, epsilon):
    return vectorized_global_speaker_static_hier(
        states, alpha_per_trial, color_semval, form_semval, k, wf, beta, gamma, epsilon
    )

vectorized_incremental_speaker_frozen_hier = jax.vmap(
    incremental_speaker_frozen,
    in_axes=(0,    # states
             0,    # alpha_D  ← per-trial
             0,    # alpha_C  ← per-trial
             0,    # alpha_F  ← per-trial
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta
             None, # gamma
             None, # epsilon
             ),
)

@jax.jit
def jitted_speaker_frozen_hier(states, alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                                color_semval, form_semval, k, wf, beta, gamma, epsilon):
    return vectorized_incremental_speaker_frozen_hier(
        states, alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
        color_semval, form_semval, k, wf, beta, gamma, epsilon
    )

vectorized_incremental_speaker_lookahead_hier = jax.vmap(
    incremental_speaker_lookahead,
    in_axes=(0, 0, None, None, None, None, None),
)

@jax.jit
def jitted_speaker_lookahead_hier(states, alpha_per_trial, color_semval, form_semval, k, wf, beta):
    return vectorized_incremental_speaker_lookahead_hier(
        states, alpha_per_trial, color_semval, form_semval, k, wf, beta
    )


def likelihood_function_global_speaker_hier(
    states=None, empirical=None,
    participant_idx=None, n_participants=None,
):
    """Global speaker with per-participant random effects on alpha."""
    color_sem = 0.971
    form_sem  = 0.50
    k         = 0.5
    wf        = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)

    alpha   = numpyro.sample("alpha", dist.HalfNormal(5.0))
    # gamma and epsilon fixed at incremental posterior means (not identifiable in global)
    gamma   = 2.32
    epsilon = 0.23
    tau     = numpyro.sample("tau",   dist.HalfNormal(0.2))

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    alpha_per_trial = jnp.maximum(alpha + delta[participant_idx], 0.0)

    with numpyro.plate("data", len(states)):
        probs = jitted_global_speaker_hier(
            states, alpha_per_trial, color_sem, form_sem, k, wf, beta, gamma, epsilon
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


# ── Reported model (paper): single alpha, log_beta, tau, delta ───────────

def _make_reported_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0):
    """Factory for the reported model with fixed semantics."""
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None):
        alpha    = numpyro.sample("alpha", dist.HalfNormal(5.0))
        log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
        beta     = jnp.exp(log_beta)
        tau      = numpyro.sample("tau", dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_per_trial = jnp.maximum(alpha + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_hier(
                states, alpha_per_trial, alpha_per_trial, alpha_per_trial,
                color_semval, form_semval, k, wf, beta, 0.0, 0.0,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_reported_hier = _make_reported_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
)

likelihood_function_reported_lowcol_hier = _make_reported_model(
    color_semval=0.85, form_semval=0.50, k=0.5, wf=1.0,
)


# ── Extended v1: per-dim alpha, gamma, epsilon ───────────────────────────

def _make_extended_v1_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0):
    """Factory for extended v1 with fixed semantics."""
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None):
        log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
        beta     = jnp.exp(log_beta)

        alpha_D  = numpyro.sample("alpha_D", dist.HalfNormal(5.0))
        alpha_C  = numpyro.sample("alpha_C", dist.HalfNormal(5.0))
        alpha_F  = numpyro.sample("alpha_F", dist.HalfNormal(5.0))
        gamma    = numpyro.sample("gamma", dist.Normal(0.0, 1.0))
        epsilon  = numpyro.sample("epsilon", dist.Beta(1.0, 50.0))
        tau      = numpyro.sample("tau",   dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_hier(
                states, alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                color_semval, form_semval, k, wf, beta, gamma, epsilon
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_incremental_speaker_hier = _make_extended_v1_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
)

likelihood_function_incremental_speaker_lowcol_hier = _make_extended_v1_model(
    color_semval=0.85, form_semval=0.50, k=0.5, wf=1.0,
)


# ── Contextual compromise model: generic condition-sensitive production layer ─

def _make_contextual_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0):
    """C2-contextual (the base contextual variant): per-dim alpha + RAW LM + lambda_suff. No gammas."""
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None):
        log_beta_lm = numpyro.sample("log_beta_lm", dist.Normal(0.0, 0.5))
        beta_lm     = jnp.exp(log_beta_lm)

        alpha_D         = numpyro.sample("alpha_D",         dist.HalfNormal(5.0))
        alpha_C         = numpyro.sample("alpha_C",         dist.HalfNormal(5.0))
        alpha_F         = numpyro.sample("alpha_F",         dist.HalfNormal(5.0))
        lambda_suff     = numpyro.sample("lambda_suff",     dist.Normal(0.0, 1.0))
        epsilon         = numpyro.sample("epsilon",         dist.Beta(1.0, 50.0))
        tau             = numpyro.sample("tau",             dist.HalfNormal(0.2))

        # Centered parameterization for delta. Non-centered was tried (see
        # commit f9e5afe) but caused chain-stuck multimodality on delta[30]
        # (r-hat=1.30); the funnel here is weak because each subject has
        # enough trials to identify their offset.
        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_contextual_hier(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_suff, color_semval, form_semval, k, wf, beta_lm,
                epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_contextual_hier = _make_contextual_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
)


def _make_contextual_lambdaunc_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
):
    """Contextual + listener-uncertainty cost (one principled parameter on top
    of the base contextual variant).

    Adds ``lambda_uncertainty ~ HalfNormal(2.0)``: the speaker bears a cost
    proportional to the residual listener uncertainty after producing each
    utterance, replacing the ad-hoc 6-gamma length-bonus structure with one
    theoretically grounded coefficient (audience-design RSA).
    """
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None):
        log_beta_lm = numpyro.sample("log_beta_lm", dist.Normal(0.0, 0.5))
        beta_lm     = jnp.exp(log_beta_lm)

        alpha_D            = numpyro.sample("alpha_D",            dist.HalfNormal(5.0))
        alpha_C            = numpyro.sample("alpha_C",            dist.HalfNormal(5.0))
        alpha_F            = numpyro.sample("alpha_F",            dist.HalfNormal(5.0))
        lambda_suff        = numpyro.sample("lambda_suff",        dist.Normal(0.0, 1.0))
        lambda_uncertainty = numpyro.sample("lambda_uncertainty", dist.HalfNormal(2.0))
        epsilon            = numpyro.sample("epsilon",            dist.Beta(1.0, 50.0))
        tau                = numpyro.sample("tau",                dist.HalfNormal(0.2))

        # Same centered delta parameterization as the base contextual variant — funnel is weak
        # because each subject has enough trials to identify their offset.
        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_contextual_lambdaunc_hier(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_suff, color_semval, form_semval, k, wf, beta_lm,
                lambda_uncertainty, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_contextual_lambdaunc_hier = _make_contextual_lambdaunc_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
)


def _make_simplified_model(
    color_semval=0.59,
    form_semval=0.50,
    k=0.5,
    wf=0.6856,
    order_mode="lm_resid",
    drop: tuple = (),
):
    """Three-pressure production model for anti-overfitting refits."""
    _valid_modes = {"lm_resid", "lm_raw", "hand_order", "none"}
    if order_mode not in _valid_modes:
        raise ValueError(f"Unsupported simplified order_mode {order_mode!r}; "
                         f"expected {sorted(_valid_modes)}")
    _valid_drops = {"frontload", "uncertainty_len"}
    drop = frozenset(drop)
    _bad = drop - _valid_drops
    if _bad:
        raise ValueError(f"Unsupported simplified drop(s): {sorted(_bad)}; "
                         f"supported: {sorted(_valid_drops)}")

    if order_mode == "lm_resid":
        order_scores = LOG_LM_RESID_15
    elif order_mode == "lm_raw":
        order_scores = LOG_LM_RAW_15
    elif order_mode == "hand_order":
        order_scores = -NONCANON_MASK
    else:
        order_scores = jnp.zeros_like(LOG_LM_RESID_15)

    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None):
        alpha = numpyro.sample("alpha", dist.HalfNormal(5.0))
        if order_mode == "none":
            beta_order = 0.0
        else:
            log_beta_order = numpyro.sample("log_beta_order", dist.Normal(0.0, 0.5))
            beta_order = jnp.exp(log_beta_order)
        lambda_frontload = (
            0.0 if "frontload" in drop
            else numpyro.sample("lambda_frontload", dist.Normal(0.0, 1.0))
        )
        gamma_uncertainty_len = (
            0.0 if "uncertainty_len" in drop
            else numpyro.sample("gamma_uncertainty_len", dist.HalfNormal(2.0))
        )
        epsilon = numpyro.sample("epsilon", dist.Beta(1.0, 50.0))
        tau = numpyro.sample("tau", dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_per_trial = jnp.maximum(alpha + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_simplified_hier(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_per_trial, beta_order, lambda_frontload,
                gamma_uncertainty_len, color_semval, form_semval, k, wf,
                epsilon, order_scores,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_simplified_lm_resid_hier = _make_simplified_model(
    order_mode="lm_resid",
)
likelihood_function_simplified_lm_raw_hier = _make_simplified_model(
    order_mode="lm_raw",
)
likelihood_function_simplified_hand_order_hier = _make_simplified_model(
    order_mode="hand_order",
)
likelihood_function_simplified_no_frontload_hier = _make_simplified_model(
    order_mode="lm_resid", drop=("frontload",),
)
likelihood_function_simplified_no_uncertainty_len_hier = _make_simplified_model(
    order_mode="lm_resid", drop=("uncertainty_len",),
)
likelihood_function_simplified_no_order_hier = _make_simplified_model(
    order_mode="none",
)


PRINCIPLED_PRIOR_PROFILES = {
    "default": {
        "alpha_scale": 5.0,
        "log_beta_order_sd": 0.5,
        "lambda_salience_scale": 2.0,
        "rho_salience_stop_scale": 2.0,
        "planning_scale": 1.0,
        "lambda_sufficient_single_scale": 2.0,
        "lambda_reliability_form_scale": 2.0,
        "lambda_sufficient_form_pair_scale": 1.5,
        "lambda_three_word_penalty_scale": 1.5,
        "lambda_sharp_form_suppression_scale": 1.5,
        "lambda_size_sharp_single_bonus_scale": 1.5,
        "lambda_size_sharp_form_pair_penalty_scale": 1.5,
        "gamma_uncertainty_len_scale": 2.0,
        "tau_scale": 0.2,
    },
    "regularized": {
        "alpha_scale": 3.0,
        "log_beta_order_sd": 0.35,
        "lambda_salience_scale": 1.0,
        "rho_salience_stop_scale": 0.75,
        "planning_scale": 0.75,
        "lambda_sufficient_single_scale": 1.5,
        "lambda_reliability_form_scale": 1.5,
        "lambda_sufficient_form_pair_scale": 1.0,
        "lambda_three_word_penalty_scale": 1.0,
        "lambda_sharp_form_suppression_scale": 1.0,
        "lambda_size_sharp_single_bonus_scale": 1.0,
        "lambda_size_sharp_form_pair_penalty_scale": 1.0,
        "gamma_uncertainty_len_scale": 1.0,
        "tau_scale": 0.15,
    },
    "strong_regularized": {
        "alpha_scale": 2.0,
        "log_beta_order_sd": 0.25,
        "lambda_salience_scale": 0.75,
        "rho_salience_stop_scale": 0.5,
        "planning_scale": 0.5,
        "lambda_sufficient_single_scale": 1.0,
        "lambda_reliability_form_scale": 1.0,
        "lambda_sufficient_form_pair_scale": 0.75,
        "lambda_three_word_penalty_scale": 0.75,
        "lambda_sharp_form_suppression_scale": 0.75,
        "lambda_size_sharp_single_bonus_scale": 0.75,
        "lambda_size_sharp_form_pair_penalty_scale": 0.75,
        "gamma_uncertainty_len_scale": 0.75,
        "tau_scale": 0.10,
    },
}


def _make_principled_model(
    drop: tuple = (),
    salience_stop: bool = False,
    planned_prefix: bool = False,
    response_policy: bool = False,
    bounded_form: bool = False,
    sharp_form_suppression: bool = False,
    size_sharp_policy: bool = False,
    prior_profile: str = "default",
    cell: str = "inc_rec",
    fixed_epsilon: float | None = None,
    size_context_mode: str = "posterior",
):
    """Order-only LM + soft salience + derived size-uncertainty model."""
    _valid_drops = {"order", "salience", "uncertainty_len"}
    drop = frozenset(drop)
    _bad = drop - _valid_drops
    if _bad:
        raise ValueError(f"Unsupported principled drop(s): {sorted(_bad)}; "
                         f"supported: {sorted(_valid_drops)}")
    if salience_stop and "salience" in drop:
        raise ValueError("salience_stop requires the salience mechanism.")
    if prior_profile not in PRINCIPLED_PRIOR_PROFILES:
        raise ValueError(
            f"Unsupported principled prior profile '{prior_profile}'; "
            f"supported: {sorted(PRINCIPLED_PRIOR_PROFILES)}"
        )
    _valid_cells = {"inc_rec", "inc_static", "glob_rec", "glob_static"}
    if cell not in _valid_cells:
        raise ValueError(f"Unsupported principled 2x2 cell {cell!r}; "
                         f"supported: {sorted(_valid_cells)}")
    if planned_prefix and cell in ("glob_rec", "glob_static"):
        raise ValueError("planned_prefix is defined for incremental 2x2 cells.")
    if planned_prefix and response_policy:
        raise ValueError("planned_prefix and response_policy are separate variants.")
    if bounded_form and not response_policy:
        raise ValueError("bounded_form requires response_policy.")
    if sharp_form_suppression and not response_policy:
        raise ValueError("sharp_form_suppression requires response_policy.")
    if size_sharp_policy and not response_policy:
        raise ValueError("size_sharp_policy requires response_policy.")
    _valid_size_context_modes = {"posterior", "comparison_class"}
    if size_context_mode not in _valid_size_context_modes:
        raise ValueError(
            f"Unsupported principled size_context_mode {size_context_mode!r}; "
            f"supported: {sorted(_valid_size_context_modes)}"
        )

    priors = PRINCIPLED_PRIOR_PROFILES[prior_profile]
    order_scores = jnp.zeros_like(LOG_LM_ORDER_ONLY_15) if "order" in drop else LOG_LM_ORDER_ONLY_15
    if planned_prefix:
        speaker_fn = jitted_speaker_principled_planned_hier
    elif response_policy:
        speaker_fn = (
            jitted_global_speaker_principled_response_policy_hier
            if cell in ("glob_rec", "glob_static")
            else jitted_speaker_principled_response_policy_hier
        )
    else:
        speaker_fn = (
            jitted_global_speaker_principled_hier
            if cell in ("glob_rec", "glob_static")
            else jitted_speaker_principled_hier
        )
    recursive = cell in ("inc_rec", "glob_rec")

    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None,
              is_colour_sufficient=None):
        alpha = numpyro.sample("alpha", dist.HalfNormal(priors["alpha_scale"]))
        if "order" in drop:
            beta_order = 0.0
        else:
            log_beta_order = numpyro.sample(
                "log_beta_order",
                dist.Normal(0.0, priors["log_beta_order_sd"]),
            )
            beta_order = jnp.exp(log_beta_order)
        lambda_salience = (
            0.0 if "salience" in drop
            else numpyro.sample(
                "lambda_salience",
                dist.HalfNormal(priors["lambda_salience_scale"]),
            )
        )
        rho_salience_stop = (
            numpyro.sample(
                "rho_salience_stop",
                dist.HalfNormal(priors["rho_salience_stop_scale"]),
            )
            if salience_stop else 0.0
        )
        planning_scale = (
            numpyro.sample(
                "planning_scale",
                dist.HalfNormal(priors["planning_scale"]),
            )
            if planned_prefix else 0.0
        )
        lambda_sufficient_single = (
            numpyro.sample(
                "lambda_sufficient_single",
                dist.HalfNormal(priors["lambda_sufficient_single_scale"]),
            )
            if response_policy else 0.0
        )
        lambda_reliability_form = (
            numpyro.sample(
                "lambda_reliability_form",
                dist.HalfNormal(priors["lambda_reliability_form_scale"]),
            )
            if response_policy else 0.0
        )
        lambda_sufficient_form_pair = (
            numpyro.sample(
                "lambda_sufficient_form_pair",
                dist.HalfNormal(priors["lambda_sufficient_form_pair_scale"]),
            )
            if bounded_form else 0.0
        )
        lambda_three_word_penalty = (
            numpyro.sample(
                "lambda_three_word_penalty",
                dist.HalfNormal(priors["lambda_three_word_penalty_scale"]),
            )
            if bounded_form else 0.0
        )
        lambda_sharp_form_suppression = (
            numpyro.sample(
                "lambda_sharp_form_suppression",
                dist.HalfNormal(priors["lambda_sharp_form_suppression_scale"]),
            )
            if sharp_form_suppression else 0.0
        )
        lambda_size_sharp_single_bonus = (
            numpyro.sample(
                "lambda_size_sharp_single_bonus",
                dist.HalfNormal(priors["lambda_size_sharp_single_bonus_scale"]),
            )
            if size_sharp_policy else 0.0
        )
        lambda_size_sharp_form_pair_penalty = (
            numpyro.sample(
                "lambda_size_sharp_form_pair_penalty",
                dist.HalfNormal(priors["lambda_size_sharp_form_pair_penalty_scale"]),
            )
            if size_sharp_policy else 0.0
        )
        gamma_uncertainty_len = (
            0.0 if "uncertainty_len" in drop
            else numpyro.sample(
                "gamma_uncertainty_len",
                dist.HalfNormal(priors["gamma_uncertainty_len_scale"]),
            )
        )
        epsilon = (
            numpyro.deterministic("epsilon", jnp.asarray(fixed_epsilon))
            if fixed_epsilon is not None
            else numpyro.sample("epsilon", dist.Beta(1.0, 50.0))
        )
        tau = numpyro.sample("tau", dist.HalfNormal(priors["tau_scale"]))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_per_trial = jnp.maximum(alpha + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            if planned_prefix:
                probs = speaker_fn(
                    states, sufficient_dim, has_one_word_solution, is_sharp,
                    alpha_per_trial, beta_order, lambda_salience,
                    rho_salience_stop, planning_scale, gamma_uncertainty_len,
                    0.59, 0.50, 0.50, 0.6856, epsilon, order_scores,
                    BASE_VISUAL_SALIENCE, recursive=recursive,
                    size_context_mode=size_context_mode,
                )
            elif response_policy:
                probs = speaker_fn(
                    states, sufficient_dim, has_one_word_solution, is_sharp,
                    is_colour_sufficient, alpha_per_trial, beta_order,
                    lambda_salience, rho_salience_stop, lambda_sufficient_single,
                    lambda_reliability_form, lambda_sufficient_form_pair,
                    lambda_three_word_penalty, lambda_sharp_form_suppression,
                    lambda_size_sharp_single_bonus,
                    lambda_size_sharp_form_pair_penalty,
                    gamma_uncertainty_len, 0.59, 0.50, 0.50, 0.6856,
                    epsilon, order_scores,
                    BASE_VISUAL_SALIENCE, recursive=recursive,
                    size_context_mode=size_context_mode,
                )
            else:
                probs = speaker_fn(
                    states, sufficient_dim, has_one_word_solution, is_sharp,
                    alpha_per_trial, beta_order, lambda_salience, rho_salience_stop,
                    gamma_uncertainty_len, 0.59, 0.50, 0.50, 0.6856,
                    epsilon, order_scores, BASE_VISUAL_SALIENCE,
                    recursive=recursive,
                    size_context_mode=size_context_mode,
                )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_principled_hier = _make_principled_model()
likelihood_function_principled_no_order_hier = _make_principled_model(drop=("order",))
likelihood_function_principled_no_salience_hier = _make_principled_model(drop=("salience",))
likelihood_function_principled_no_uncertainty_len_hier = _make_principled_model(
    drop=("uncertainty_len",),
)
likelihood_function_principled_salience_stop_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
)
likelihood_function_principled_salience_stop_regularized_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="regularized",
)
likelihood_function_principled_salience_stop_regularized_2x2_inc_rec_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="regularized",
    cell="inc_rec",
)
likelihood_function_principled_salience_stop_regularized_2x2_inc_static_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="regularized",
    cell="inc_static",
)
likelihood_function_principled_salience_stop_regularized_2x2_glob_rec_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="regularized",
    cell="glob_rec",
)
likelihood_function_principled_salience_stop_regularized_2x2_glob_static_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="regularized",
    cell="glob_static",
)
likelihood_function_principled_salience_stop_regularized_2x2_glob_rec_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="regularized",
    cell="glob_rec",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_2x2_glob_static_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="regularized",
    cell="glob_static",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_tmcc_2x2_inc_rec_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="regularized",
    cell="inc_rec",
    size_context_mode="comparison_class",
)
likelihood_function_principled_salience_stop_regularized_tmcc_2x2_inc_static_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="regularized",
    cell="inc_static",
    size_context_mode="comparison_class",
)
likelihood_function_principled_salience_stop_regularized_plannedprefix_2x2_inc_rec_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    planned_prefix=True,
    prior_profile="regularized",
    cell="inc_rec",
)
likelihood_function_principled_salience_stop_regularized_plannedprefix_2x2_inc_static_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    planned_prefix=True,
    prior_profile="regularized",
    cell="inc_static",
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_inc_rec_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    prior_profile="regularized",
    cell="inc_rec",
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_inc_static_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    prior_profile="regularized",
    cell="inc_static",
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_glob_rec_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    prior_profile="regularized",
    cell="glob_rec",
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_glob_static_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    prior_profile="regularized",
    cell="glob_static",
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_inc_rec_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    prior_profile="regularized",
    cell="inc_rec",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_inc_static_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    prior_profile="regularized",
    cell="inc_static",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_glob_rec_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    prior_profile="regularized",
    cell="glob_rec",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_glob_static_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    prior_profile="regularized",
    cell="glob_static",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    prior_profile="regularized",
    cell="inc_rec",
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    prior_profile="regularized",
    cell="inc_static",
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    prior_profile="regularized",
    cell="glob_rec",
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    prior_profile="regularized",
    cell="glob_static",
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    prior_profile="regularized",
    cell="inc_rec",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    prior_profile="regularized",
    cell="inc_static",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_rec_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    sharp_form_suppression=True,
    prior_profile="regularized",
    cell="inc_rec",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_static_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    sharp_form_suppression=True,
    prior_profile="regularized",
    cell="inc_static",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_rec_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    size_sharp_policy=True,
    prior_profile="regularized",
    cell="inc_rec",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_static_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    size_sharp_policy=True,
    prior_profile="regularized",
    cell="inc_static",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_rec_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    size_sharp_policy=True,
    prior_profile="regularized",
    cell="glob_rec",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_static_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    size_sharp_policy=True,
    prior_profile="regularized",
    cell="glob_static",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    prior_profile="regularized",
    cell="glob_rec",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    response_policy=True,
    bounded_form=True,
    prior_profile="regularized",
    cell="glob_static",
    fixed_epsilon=0.003,
)
likelihood_function_principled_salience_stop_regularized_tmcc_2x2_glob_rec_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="regularized",
    cell="glob_rec",
    size_context_mode="comparison_class",
)
likelihood_function_principled_salience_stop_regularized_tmcc_2x2_glob_static_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="regularized",
    cell="glob_static",
    size_context_mode="comparison_class",
)
likelihood_function_principled_salience_stop_regularized_tmcc_2x2_glob_rec_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="regularized",
    cell="glob_rec",
    fixed_epsilon=0.003,
    size_context_mode="comparison_class",
)
likelihood_function_principled_salience_stop_regularized_tmcc_2x2_glob_static_fixedeps_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="regularized",
    cell="glob_static",
    fixed_epsilon=0.003,
    size_context_mode="comparison_class",
)
likelihood_function_principled_salience_stop_strong_regularized_hier = _make_principled_model(
    drop=("uncertainty_len",),
    salience_stop=True,
    prior_profile="strong_regularized",
)


def _make_contextual_freewf_model(color_semval=0.971, form_semval=0.50, k=0.5):
    """the base contextual variant speaker but with ``wf`` (size-semantics noise scale) learned.

    the previous variant in the parameter-tuning sweep. The pre-loop default ``wf = 1.0`` saturates
    the size semantics into its flat regime — sharp and blurred trials get
    nearly identical listener confidences (RSA target ≈ 0.23 vs 0.21 for
    erdc) despite the empirical target-distractor gap being 6.66 vs 4.15.

    Putting ``log_wf ~ Normal(-1.0, 0.5)`` (prior bulk wf ≈ 0.22 – 0.61)
    lets the data find a wf in the erf's discriminative range, which
    propagates the existing sharp/blurred encoding in the size values
    through to a meaningful listener-confidence asymmetry. No new output
    coefficient; the entire mechanism lives in one re-tuned representation
    parameter. +1 free param vs the base contextual variant.
    """
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None):
        log_beta_lm = numpyro.sample("log_beta_lm", dist.Normal(0.0, 0.5))
        beta_lm     = jnp.exp(log_beta_lm)

        log_wf      = numpyro.sample("log_wf",      dist.Normal(-1.0, 0.5))
        wf          = jnp.exp(log_wf)

        alpha_D     = numpyro.sample("alpha_D",     dist.HalfNormal(5.0))
        alpha_C     = numpyro.sample("alpha_C",     dist.HalfNormal(5.0))
        alpha_F     = numpyro.sample("alpha_F",     dist.HalfNormal(5.0))
        lambda_suff = numpyro.sample("lambda_suff", dist.Normal(0.0, 1.0))
        epsilon     = numpyro.sample("epsilon",     dist.Beta(1.0, 50.0))
        tau         = numpyro.sample("tau",         dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_contextual_hier(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_suff, color_semval, form_semval, k, wf, beta_lm,
                epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_contextual_freewf_hier = _make_contextual_freewf_model(
    color_semval=0.971, form_semval=0.50, k=0.5,
)


# size semantics with a fixed reference scale R that
# anchors the comparison and breaks the function's scale-invariance.
SIZE_ANCHOR_R = 5.0

# parsimonious fixed wf, set to the posterior median of the previous variant
# (R²-winning variant on dc subset). Lets the anchored+2-gamma model carry
# the same wf-tuned representation without paying a free parameter for it.
# Source: arviz median of log_wf in a prior fit was -0.3775 → wf = 0.6856.
WF_FIXED_ITER11_MEDIAN = 0.6856

# the previous variant (contextual_pcalpha_formmod) posterior median of log_beta_lm, used
# only by the a prior variant β_lm-fixed ABLATION (contextual_pcalpha_canon_betafixed)
# to test how much of lambda_noncanon is genuinely additional vs. signal the
# free β_lm would otherwise reallocate. Source: arviz median of log_beta_lm
# in the a prior variant NC = 1.907718 → beta_lm = 6.737698.
LOG_BETA_LM_FIXED_ITER17 = 1.907718


def incremental_speaker_contextual_anchored(
    states:                jnp.ndarray,
    sufficient_dim:        int,
    has_one_word_solution: float,
    is_sharp:              float,
    alpha_D:               float = 3.0,
    alpha_C:               float = 3.0,
    alpha_F:               float = 3.0,
    lambda_suff:           float = 0.0,
    color_semval:          float = 0.95,
    form_semval:           float = 0.80,
    k:                     float = 0.50,
    wf:                    float = 1.00,
    beta_lm:               float = 1.00,
    epsilon:               float = 0.01,
) -> jnp.ndarray:
    """the base contextual variant speaker with size semantics anchored to an absolute scale R.

    The only structural change vs ``incremental_speaker_contextual`` is the
    size-semantics denominator:

        denom = wf * sqrt(sizes² + theta_k² + R²)        (anchored)
        denom = wf * sqrt(sizes² + theta_k²)             (the base contextual variant)

    With ``R = SIZE_ANCHOR_R = 5.0``, the denominator can never collapse
    below ``wf * R``, so the listener cannot saturate the size discriminant
    when the local cluster is tight (which is exactly what was happening in
    the base contextual variant with theta_k pinned to the distractor cluster). Sharp trials
    with their larger target-distractor gaps now produce reliably higher
    z than blurred trials. No new free parameters — R is a fixed grounded
    scale (the typical "small" size in the dataset).
    """

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]
    alpha_vec      = jnp.array([alpha_D, alpha_C, alpha_F])

    del has_one_word_solution
    del is_sharp

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    log_lm_raw = beta_lm * LOG_LM_RAW_15

    colors = states[:, 1]
    forms  = states[:, 2]
    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem  = jnp.log(
        jnp.where(forms  == 1, form_semval,  1.0 - form_semval)  + eps
    )

    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))

    def _anchored_size_sem(sizes_arr, post):
        post_sorted = post[size_sort_idx]
        post_sorted = post_sorted / (jnp.sum(post_sorted) + eps)
        cdf = jnp.cumsum(post_sorted)
        idx_low  = jnp.minimum(jnp.searchsorted(cdf, 0.2, side="left"),
                                sizes_sorted.shape[0] - 1)
        idx_high = jnp.minimum(jnp.searchsorted(cdf, 0.8, side="left"),
                                sizes_sorted.shape[0] - 1)
        x_min_mid = sizes_sorted[idx_low]
        x_max_mid = sizes_sorted[idx_high]
        theta_k   = x_max_mid - k * (x_max_mid - x_min_mid)
        denom     = wf * jnp.sqrt(
            sizes_arr ** 2 + theta_k ** 2 + SIZE_ANCHOR_R ** 2 + eps
        )
        z         = (sizes_arr - theta_k) / denom
        return 0.5 * (1.0 + lax.erf(z / jnp.sqrt(2.0)))

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]
        active_t    = ACTIVE_POS[t]

        def size_log_sem_for_utt(post):
            sv = _anchored_size_sem(sizes, post)
            return jnp.log(jnp.clip(sv, eps))

        size_log_sems = jax.vmap(size_log_sem_for_utt)(per_utt_posts)

        log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
        log_sem_table = jnp.concatenate([
            size_log_sems[:, None, :],
            jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
        ], axis=1)

        token_pres_t = TOKEN_PRESENT[t]
        log_prod_sem = jnp.einsum(
            "uav, uvo -> uao",
            token_pres_t,
            log_sem_table,
        )

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
        log_updated = log_per_utt_posts[:, None, :] + log_prod_sem
        log_Z = jax.scipy.special.logsumexp(log_updated, axis=-1)
        log_norm = log_updated - log_Z[:, :, None]
        log_L_ref = log_norm[:, :, referent_index]

        first_step_gate = (t == 0).astype(jnp.float32)
        suff_boost_vec = lambda_suff * first_step_gate * jnp.array([
            sufficient_dim == 0,
            sufficient_dim == 1,
            sufficient_dim == 2,
        ], dtype=jnp.float32)
        logits = jnp.where(
            cand_mask_t,
            alpha_vec[None, :] * log_L_ref + suff_boost_vec[None, :],
            -1e9,
        )
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo",
            ACTUAL_TOK_ONEHOT[t],
            log_sem_table,
        )
        log_updated_post = log_per_utt_posts + jnp.where(
            active_t[:, None],
            selected_log_sem,
            0.0,
        )
        log_Z_post = jax.scipy.special.logsumexp(
            log_updated_post, axis=-1, keepdims=True
        )
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(
        step,
        (init_scores, init_posts),
        jnp.arange(T),
    )

    log_unnorm = log_lm_raw + log_final_scores
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


vectorized_incremental_speaker_contextual_anchored_hier = jax.vmap(
    incremental_speaker_contextual_anchored,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # alpha_D
             0,    # alpha_C
             0,    # alpha_F
             None, # lambda_suff
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta_lm
             None, # epsilon
             ),
)


@jax.jit
def jitted_speaker_contextual_anchored_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
    lambda_suff, color_semval, form_semval, k, wf, beta_lm,
    epsilon,
):
    return vectorized_incremental_speaker_contextual_anchored_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
        lambda_suff, color_semval, form_semval, k, wf, beta_lm,
        epsilon,
    )


def incremental_speaker_contextual_anchored_gamma(
    states:                jnp.ndarray,
    sufficient_dim:        int,
    has_one_word_solution: float,
    is_sharp:              float,
    alpha_D:               float = 3.0,
    alpha_C:               float = 3.0,
    alpha_F:               float = 3.0,
    lambda_suff:           float = 0.0,
    color_semval:          float = 0.95,
    form_semval:           float = 0.80,
    k:                     float = 0.50,
    wf:                    float = 1.00,
    beta_lm:               float = 1.00,
    gamma_base:            float = 0.0,
    gamma_oneword:         float = 0.0,
    epsilon:               float = 0.01,
) -> jnp.ndarray:
    """Anchored size semantics + compact 2-gamma length bonus.

    Reuses the anchored size kernel from
    ``incremental_speaker_contextual_anchored`` (denominator includes the
    fixed reference R via ``SIZE_ANCHOR_R``) and adds a *single linear*
    length-bonus term gated on ``has_one_word_solution``:

        gamma_eff   = gamma_base + gamma_oneword * has_one_word_solution
        length_bonus[u] = gamma_eff * max(N_WORDS[u] - 1, 0)
        log_unnorm = beta_lm * LOG_LM_RAW_15 + log_final_scores + length_bonus

    With ``has_one_word_solution = 1`` (erdc / zrdc), the effective bonus
    collapses to ``gamma_base + gamma_oneword`` (baseline data has those
    cancel to ≈ 0). With ``has_one_word_solution = 0`` (brdc), the bonus
    is just ``gamma_base`` (baseline data has this strongly positive,
    pushing the model toward DCF-type long phrases). The compact 2-gamma
    form captures the same shape the 6-gamma baseline used, dropping the
    redundant length-3 tier and the is_sharp modulation.
    """

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]
    alpha_vec      = jnp.array([alpha_D, alpha_C, alpha_F])

    del is_sharp

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    log_lm_raw = beta_lm * LOG_LM_RAW_15

    colors = states[:, 1]
    forms  = states[:, 2]
    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem  = jnp.log(
        jnp.where(forms  == 1, form_semval,  1.0 - form_semval)  + eps
    )

    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))

    def _anchored_size_sem(sizes_arr, post):
        post_sorted = post[size_sort_idx]
        post_sorted = post_sorted / (jnp.sum(post_sorted) + eps)
        cdf = jnp.cumsum(post_sorted)
        idx_low  = jnp.minimum(jnp.searchsorted(cdf, 0.2, side="left"),
                                sizes_sorted.shape[0] - 1)
        idx_high = jnp.minimum(jnp.searchsorted(cdf, 0.8, side="left"),
                                sizes_sorted.shape[0] - 1)
        x_min_mid = sizes_sorted[idx_low]
        x_max_mid = sizes_sorted[idx_high]
        theta_k   = x_max_mid - k * (x_max_mid - x_min_mid)
        denom     = wf * jnp.sqrt(
            sizes_arr ** 2 + theta_k ** 2 + SIZE_ANCHOR_R ** 2 + eps
        )
        z         = (sizes_arr - theta_k) / denom
        return 0.5 * (1.0 + lax.erf(z / jnp.sqrt(2.0)))

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]
        active_t    = ACTIVE_POS[t]

        def size_log_sem_for_utt(post):
            sv = _anchored_size_sem(sizes, post)
            return jnp.log(jnp.clip(sv, eps))

        size_log_sems = jax.vmap(size_log_sem_for_utt)(per_utt_posts)

        log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
        log_sem_table = jnp.concatenate([
            size_log_sems[:, None, :],
            jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
        ], axis=1)

        token_pres_t = TOKEN_PRESENT[t]
        log_prod_sem = jnp.einsum(
            "uav, uvo -> uao",
            token_pres_t,
            log_sem_table,
        )

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
        log_updated = log_per_utt_posts[:, None, :] + log_prod_sem
        log_Z = jax.scipy.special.logsumexp(log_updated, axis=-1)
        log_norm = log_updated - log_Z[:, :, None]
        log_L_ref = log_norm[:, :, referent_index]

        first_step_gate = (t == 0).astype(jnp.float32)
        suff_boost_vec = lambda_suff * first_step_gate * jnp.array([
            sufficient_dim == 0,
            sufficient_dim == 1,
            sufficient_dim == 2,
        ], dtype=jnp.float32)
        logits = jnp.where(
            cand_mask_t,
            alpha_vec[None, :] * log_L_ref + suff_boost_vec[None, :],
            -1e9,
        )
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo",
            ACTUAL_TOK_ONEHOT[t],
            log_sem_table,
        )
        log_updated_post = log_per_utt_posts + jnp.where(
            active_t[:, None],
            selected_log_sem,
            0.0,
        )
        log_Z_post = jax.scipy.special.logsumexp(
            log_updated_post, axis=-1, keepdims=True
        )
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(
        step,
        (init_scores, init_posts),
        jnp.arange(T),
    )

    # Compact 2-gamma length bonus, gated on has_one_word_solution.
    gamma_eff   = gamma_base + gamma_oneword * has_one_word_solution
    length_bonus = gamma_eff * jnp.maximum(N_WORDS - 1.0, 0.0)

    log_unnorm = log_lm_raw + log_final_scores + length_bonus
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


vectorized_incremental_speaker_contextual_anchored_gamma_hier = jax.vmap(
    incremental_speaker_contextual_anchored_gamma,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # alpha_D
             0,    # alpha_C
             0,    # alpha_F
             None, # lambda_suff
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta_lm
             None, # gamma_base
             None, # gamma_oneword
             None, # epsilon
             ),
)


@jax.jit
def jitted_speaker_contextual_anchored_gamma_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
    lambda_suff, color_semval, form_semval, k, wf, beta_lm,
    gamma_base, gamma_oneword, epsilon,
):
    return vectorized_incremental_speaker_contextual_anchored_gamma_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
        lambda_suff, color_semval, form_semval, k, wf, beta_lm,
        gamma_base, gamma_oneword, epsilon,
    )


def _make_contextual_anchored_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0):
    """the base contextual variant priors, anchored size semantics (R = 5.0). 0 new params."""
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None):
        log_beta_lm = numpyro.sample("log_beta_lm", dist.Normal(0.0, 0.5))
        beta_lm     = jnp.exp(log_beta_lm)

        alpha_D     = numpyro.sample("alpha_D",     dist.HalfNormal(5.0))
        alpha_C     = numpyro.sample("alpha_C",     dist.HalfNormal(5.0))
        alpha_F     = numpyro.sample("alpha_F",     dist.HalfNormal(5.0))
        lambda_suff = numpyro.sample("lambda_suff", dist.Normal(0.0, 1.0))
        epsilon     = numpyro.sample("epsilon",     dist.Beta(1.0, 50.0))
        tau         = numpyro.sample("tau",         dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_contextual_anchored_hier(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_suff, color_semval, form_semval, k, wf, beta_lm,
                epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_contextual_anchored_hier = _make_contextual_anchored_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
)


def _make_contextual_freewf_anchored_model(color_semval=0.971, form_semval=0.50, k=0.5):
    """anchored size semantics + learned ``wf``. +1 free param."""
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None):
        log_beta_lm = numpyro.sample("log_beta_lm", dist.Normal(0.0, 0.5))
        beta_lm     = jnp.exp(log_beta_lm)

        log_wf      = numpyro.sample("log_wf",      dist.Normal(-1.0, 0.5))
        wf          = jnp.exp(log_wf)

        alpha_D     = numpyro.sample("alpha_D",     dist.HalfNormal(5.0))
        alpha_C     = numpyro.sample("alpha_C",     dist.HalfNormal(5.0))
        alpha_F     = numpyro.sample("alpha_F",     dist.HalfNormal(5.0))
        lambda_suff = numpyro.sample("lambda_suff", dist.Normal(0.0, 1.0))
        epsilon     = numpyro.sample("epsilon",     dist.Beta(1.0, 50.0))
        tau         = numpyro.sample("tau",         dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_contextual_anchored_hier(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_suff, color_semval, form_semval, k, wf, beta_lm,
                epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_contextual_freewf_anchored_hier = _make_contextual_freewf_anchored_model(
    color_semval=0.971, form_semval=0.50, k=0.5,
)


def _make_contextual_anchored_gamma_model(color_semval=0.971, form_semval=0.50, k=0.5):
    """anchored size semantics + free wf + compact 2-gamma length bonus.

    Three new free coefficients vs the base contextual variant: ``log_wf`` (size-semantics
    sensitivity), ``gamma_base`` (per-extra-token length bonus baseline),
    and ``gamma_oneword`` (modulation when the trial has a one-word
    solution). Total 10 named parameters, vs the base contextual variant's 7 and the 6-gamma
    baseline's 13.

    Same speaker as ``incremental_speaker_contextual_anchored`` plus the
    linear length-bonus aggregation. The combination targets both the
    structural fit (R²) — via the gamma length-bonus that pushes brdc
    trials toward DCF without affecting erdc/zrdc — and the lapse rate
    (ε) — via the anchored representation that breaks size-semantics
    scale-invariance.
    """
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None):
        log_beta_lm   = numpyro.sample("log_beta_lm",   dist.Normal(0.0, 0.5))
        beta_lm       = jnp.exp(log_beta_lm)

        log_wf        = numpyro.sample("log_wf",        dist.Normal(-1.0, 0.5))
        wf            = jnp.exp(log_wf)

        alpha_D       = numpyro.sample("alpha_D",       dist.HalfNormal(5.0))
        alpha_C       = numpyro.sample("alpha_C",       dist.HalfNormal(5.0))
        alpha_F       = numpyro.sample("alpha_F",       dist.HalfNormal(5.0))
        lambda_suff   = numpyro.sample("lambda_suff",   dist.Normal(0.0, 1.0))
        gamma_base    = numpyro.sample("gamma_base",    dist.Normal(0.0, 2.0))
        gamma_oneword = numpyro.sample("gamma_oneword", dist.Normal(0.0, 2.0))
        epsilon       = numpyro.sample("epsilon",       dist.Beta(1.0, 50.0))
        tau           = numpyro.sample("tau",           dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_contextual_anchored_gamma_hier(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_suff, color_semval, form_semval, k, wf, beta_lm,
                gamma_base, gamma_oneword, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_contextual_anchored_gamma_hier = _make_contextual_anchored_gamma_model(
    color_semval=0.971, form_semval=0.50, k=0.5,
)


def _make_contextual_anchored_gamma_fixedwf_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
):
    """same as the previous variant but with ``wf`` hardcoded at the previous variant
    posterior median (0.6856) instead of sampled. Drops 1 free parameter
    (no more ``log_wf`` sampling), bringing the model to 9 named coefficients
    while keeping the anchored size semantics and the compact 2-gamma
    length-bonus structure that recovered the baseline R².
    """
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None):
        log_beta_lm   = numpyro.sample("log_beta_lm",   dist.Normal(0.0, 0.5))
        beta_lm       = jnp.exp(log_beta_lm)

        alpha_D       = numpyro.sample("alpha_D",       dist.HalfNormal(5.0))
        alpha_C       = numpyro.sample("alpha_C",       dist.HalfNormal(5.0))
        alpha_F       = numpyro.sample("alpha_F",       dist.HalfNormal(5.0))
        lambda_suff   = numpyro.sample("lambda_suff",   dist.Normal(0.0, 1.0))
        gamma_base    = numpyro.sample("gamma_base",    dist.Normal(0.0, 2.0))
        gamma_oneword = numpyro.sample("gamma_oneword", dist.Normal(0.0, 2.0))
        epsilon       = numpyro.sample("epsilon",       dist.Beta(1.0, 50.0))
        tau           = numpyro.sample("tau",           dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_contextual_anchored_gamma_hier(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_suff, color_semval, form_semval, k, wf, beta_lm,
                gamma_base, gamma_oneword, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_contextual_anchored_gamma_fixedwf_hier = (
    _make_contextual_anchored_gamma_fixedwf_model(
        color_semval=0.971, form_semval=0.50, k=0.5,
        wf=WF_FIXED_ITER11_MEDIAN,
    )
)


def _make_contextual_anchored_gamma_fixedwf_pcalpha_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
):
    """+ hierarchical alpha by ``(participant × condition)``.

    Replaces the per-participant ``delta`` random effect with a per-
    (participant × condition) one, non-centered for stability with the
    larger latent space (113 × n_conditions cells; in the dc subset
    n_conditions = 3 → 339 latents).

        tau ~ HalfNormal(0.2)
        delta_raw[p, c] ~ Normal(0, 1)
        delta[p, c] = tau * delta_raw[p, c]
        per_trial_offset = delta[participant_idx, condition_idx]
        alpha_X_per_trial = max(alpha_X + per_trial_offset, 0)   for X in {D, C, F}

    Theory: the residual erdc D over-prediction (P(F | first=D) = 28% model
    vs 42% human) localized in the per-step diagnostic was not closed by
    free form_semval  — the model's alpha pattern is the
    bottleneck. Allowing alpha to drift differently per (participant,
    condition) lets the posterior find that subjects in erdc trials need
    different overall alpha than in zrdc / brdc, without committing to a
    hand-specified per-suff_dim coefficient.

    Named-coefficient count: 9 (same as the previous variant). The (P × C) cells add
    +226 latents beyond the previous variant — within numpyro's plate machinery, not
    new named-scalar parameters.
    """
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None,
              condition_idx=None, n_conditions=None):
        log_beta_lm   = numpyro.sample("log_beta_lm",   dist.Normal(0.0, 0.5))
        beta_lm       = jnp.exp(log_beta_lm)

        alpha_D       = numpyro.sample("alpha_D",       dist.HalfNormal(5.0))
        alpha_C       = numpyro.sample("alpha_C",       dist.HalfNormal(5.0))
        alpha_F       = numpyro.sample("alpha_F",       dist.HalfNormal(5.0))
        lambda_suff   = numpyro.sample("lambda_suff",   dist.Normal(0.0, 1.0))
        gamma_base    = numpyro.sample("gamma_base",    dist.Normal(0.0, 2.0))
        gamma_oneword = numpyro.sample("gamma_oneword", dist.Normal(0.0, 2.0))
        epsilon       = numpyro.sample("epsilon",       dist.Beta(1.0, 50.0))
        tau           = numpyro.sample("tau",           dist.HalfNormal(0.2))

        # Non-centered per-(participant × condition) random effect. plate
        # dim=-1 is the innermost (rightmost) axis = conditions; dim=-2 is
        # the next = participants. delta_raw has shape (P, C).
        with numpyro.plate("conditions_p", n_conditions, dim=-1):
            with numpyro.plate("participants", n_participants, dim=-2):
                delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, 1.0))
        delta = numpyro.deterministic("delta", delta_raw * tau)

        per_trial_offset = delta[participant_idx, condition_idx]
        alpha_D_per_trial = jnp.maximum(alpha_D + per_trial_offset, 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + per_trial_offset, 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + per_trial_offset, 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_contextual_anchored_gamma_hier(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_suff, color_semval, form_semval, k, wf, beta_lm,
                gamma_base, gamma_oneword, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_contextual_anchored_gamma_fixedwf_pcalpha_hier = (
    _make_contextual_anchored_gamma_fixedwf_pcalpha_model(
        color_semval=0.971, form_semval=0.50, k=0.5,
        wf=WF_FIXED_ITER11_MEDIAN,
    )
)


def incremental_speaker_contextual_anchored_gamma_sharpbonus(
    states:                jnp.ndarray,
    sufficient_dim:        int,
    has_one_word_solution: float,
    is_sharp:              float,
    alpha_D:               float = 3.0,
    alpha_C:               float = 3.0,
    alpha_F:               float = 3.0,
    lambda_suff:           float = 0.0,
    color_semval:          float = 0.95,
    form_semval:           float = 0.80,
    k:                     float = 0.50,
    wf:                    float = 1.00,
    beta_lm:               float = 1.00,
    gamma_base:            float = 0.0,
    gamma_oneword:         float = 0.0,
    gamma_sharp:           float = 0.0,
    epsilon:               float = 0.01,
) -> jnp.ndarray:
    """anchored speaker + sharpness-gated length-bonus boost.

    Identical to ``incremental_speaker_contextual_anchored_gamma`` ,
    except the length-bonus aggregation adds one extra positive coefficient
    that fires on blurred trials:

        gamma_eff = gamma_base
                  + gamma_oneword * has_one_word_solution
                  + gamma_sharp   * (1 - is_sharp)
        length_bonus[u] = gamma_eff * max(N_WORDS[u] - 1, 0)

    Sharp trials: gamma_eff = gamma_base + gamma_oneword * has_one_word_solution
                  (unchanged from the previous variant).
    Blurred trials: gamma_eff gets an extra positive term gamma_sharp, pushing
                    the speaker toward longer utterances on blurred trials
                    across all conditions.

    Targets the dominant residual on the merged main :
    erdc-blurred over-stopping (P(STOP | first=D) = 34% model vs 7% human).
    Speaker-side mechanism — captures over-specification under perceptual
    ambiguity by adding length-bonus rather than perturbing listener-side
    semantics (which the previous variant's blur_R_inflation failed to identify).

    Risk: applies to ALL blurred trials. zrdc-blurred already shows the model
    slightly under-predicting bare C (residual -0.111); a positive
    gamma_sharp would push the model toward longer utterances there too. The
    data-vs-prior balance resolves the trade-off.
    """

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]
    alpha_vec      = jnp.array([alpha_D, alpha_C, alpha_F])

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    log_lm_raw = beta_lm * LOG_LM_RAW_15

    colors = states[:, 1]
    forms  = states[:, 2]
    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem  = jnp.log(
        jnp.where(forms  == 1, form_semval,  1.0 - form_semval)  + eps
    )

    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))

    def _anchored_size_sem(sizes_arr, post):
        post_sorted = post[size_sort_idx]
        post_sorted = post_sorted / (jnp.sum(post_sorted) + eps)
        cdf = jnp.cumsum(post_sorted)
        idx_low  = jnp.minimum(jnp.searchsorted(cdf, 0.2, side="left"),
                                sizes_sorted.shape[0] - 1)
        idx_high = jnp.minimum(jnp.searchsorted(cdf, 0.8, side="left"),
                                sizes_sorted.shape[0] - 1)
        x_min_mid = sizes_sorted[idx_low]
        x_max_mid = sizes_sorted[idx_high]
        theta_k   = x_max_mid - k * (x_max_mid - x_min_mid)
        denom     = wf * jnp.sqrt(
            sizes_arr ** 2 + theta_k ** 2 + SIZE_ANCHOR_R ** 2 + eps
        )
        z         = (sizes_arr - theta_k) / denom
        return 0.5 * (1.0 + lax.erf(z / jnp.sqrt(2.0)))

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]
        active_t    = ACTIVE_POS[t]

        def size_log_sem_for_utt(post):
            sv = _anchored_size_sem(sizes, post)
            return jnp.log(jnp.clip(sv, eps))

        size_log_sems = jax.vmap(size_log_sem_for_utt)(per_utt_posts)

        log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
        log_sem_table = jnp.concatenate([
            size_log_sems[:, None, :],
            jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
        ], axis=1)

        token_pres_t = TOKEN_PRESENT[t]
        log_prod_sem = jnp.einsum(
            "uav, uvo -> uao",
            token_pres_t,
            log_sem_table,
        )

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
        log_updated = log_per_utt_posts[:, None, :] + log_prod_sem
        log_Z = jax.scipy.special.logsumexp(log_updated, axis=-1)
        log_norm = log_updated - log_Z[:, :, None]
        log_L_ref = log_norm[:, :, referent_index]

        first_step_gate = (t == 0).astype(jnp.float32)
        suff_boost_vec = lambda_suff * first_step_gate * jnp.array([
            sufficient_dim == 0,
            sufficient_dim == 1,
            sufficient_dim == 2,
        ], dtype=jnp.float32)
        logits = jnp.where(
            cand_mask_t,
            alpha_vec[None, :] * log_L_ref + suff_boost_vec[None, :],
            -1e9,
        )
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo",
            ACTUAL_TOK_ONEHOT[t],
            log_sem_table,
        )
        log_updated_post = log_per_utt_posts + jnp.where(
            active_t[:, None],
            selected_log_sem,
            0.0,
        )
        log_Z_post = jax.scipy.special.logsumexp(
            log_updated_post, axis=-1, keepdims=True
        )
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(
        step,
        (init_scores, init_posts),
        jnp.arange(T),
    )

    # 3-gamma length bonus: base, has_one_word_solution modulation, AND
    # sharpness modulation (positive when blurred).
    blur_gate = 1.0 - is_sharp
    gamma_eff = (
        gamma_base
        + gamma_oneword * has_one_word_solution
        + gamma_sharp * blur_gate
    )
    length_bonus = gamma_eff * jnp.maximum(N_WORDS - 1.0, 0.0)

    log_unnorm = log_lm_raw + log_final_scores + length_bonus
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


vectorized_incremental_speaker_contextual_anchored_gamma_sharpbonus_hier = jax.vmap(
    incremental_speaker_contextual_anchored_gamma_sharpbonus,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # alpha_D
             0,    # alpha_C
             0,    # alpha_F
             None, # lambda_suff
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta_lm
             None, # gamma_base
             None, # gamma_oneword
             None, # gamma_sharp
             None, # epsilon
             ),
)


@jax.jit
def jitted_speaker_contextual_anchored_gamma_sharpbonus_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
    lambda_suff, color_semval, form_semval, k, wf, beta_lm,
    gamma_base, gamma_oneword, gamma_sharp, epsilon,
):
    return vectorized_incremental_speaker_contextual_anchored_gamma_sharpbonus_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
        lambda_suff, color_semval, form_semval, k, wf, beta_lm,
        gamma_base, gamma_oneword, gamma_sharp, epsilon,
    )


def incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod(
    states:                jnp.ndarray,
    sufficient_dim:        int,
    has_one_word_solution: float,
    is_sharp:              float,
    alpha_D:               float = 3.0,
    alpha_C:               float = 3.0,
    alpha_F:               float = 3.0,
    lambda_suff:           float = 0.0,
    lambda_form_mod:       float = 0.0,
    color_semval:          float = 0.95,
    form_semval:           float = 0.80,
    k:                     float = 0.50,
    wf:                    float = 1.00,
    beta_lm:               float = 1.00,
    gamma_base:            float = 0.0,
    gamma_oneword:         float = 0.0,
    gamma_sharp:           float = 0.0,
    epsilon:               float = 0.01,
) -> jnp.ndarray:
    """(pcalpha + gammasharp) + erdc-gated F-present boost.

    Identical to ``incremental_speaker_contextual_anchored_gamma_sharpbonus``
    except one extra utterance-level coefficient is added to ``log_unnorm``
    (alongside the LM and length-bonus terms) that boosts every utterance
    CONTAINING the F (form) word, but only on erdc trials
    (``sufficient_dim == 0``):

        log_unnorm[u] = beta_lm * LM[u] + rsa[u] + length_bonus[u]
                      + lambda_form_mod * 1[sufficient_dim == 0] * F_PRESENT[u]

    Mechanism / motivation (per-step + sweep decomposition of the previous variant):

    - In erdc-blurred humans overwhelmingly produce F-containing descriptions
      (DF 36%, DCF 30%) over no-F ones (D 6%, DC 4%): when the sufficient
      dimension is hard-to-perceive size, speakers add the salient orthogonal
      form feature. the previous variant inverts this (DC 26%, DCF 20%, DF 6%).
    - A step-2 F-token logit boost was rejected: it inflates the
      non-canonical DFC (D-F-C, human 3%) because it rewards F at position 2,
      fighting the LM's canonical C-before-F order. An utterance-level
      F-present term instead lifts DCF and DFC together and lets the LM keep
      the canonical DCF > DFC ordering.
    - The boost MUST be erdc-gated: an ungated F-present term destroys
      zrdc bare-C (human 63%, drops to 6% in a sweep) because zrdc speakers
      do NOT add form when colour alone (easy to perceive) suffices. Gating
      on ``sufficient_dim == 0`` leaves zrdc and brdc untouched (verified:
      identical predictions for any lambda_form_mod).

    At the sweep optimum (~1.5-2) this fixes the erdc D/DC over-prediction
    and DCF under-prediction (the bulk of the previous variant erdc residual mass) with
    zero side-effects on zrdc/brdc. The residual erdc DF deficit is
    LM-prior-limited (GPT-2 suppresses bare "D F") and is left for a separate
    LM-downweight lever.

    +1 named coefficient over the previous variant -> 11 named + 339 latents.
    """

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]
    alpha_vec      = jnp.array([alpha_D, alpha_C, alpha_F])

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    log_lm_raw = beta_lm * LOG_LM_RAW_15

    colors = states[:, 1]
    forms  = states[:, 2]
    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem  = jnp.log(
        jnp.where(forms  == 1, form_semval,  1.0 - form_semval)  + eps
    )

    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))

    def _anchored_size_sem(sizes_arr, post):
        post_sorted = post[size_sort_idx]
        post_sorted = post_sorted / (jnp.sum(post_sorted) + eps)
        cdf = jnp.cumsum(post_sorted)
        idx_low  = jnp.minimum(jnp.searchsorted(cdf, 0.2, side="left"),
                                sizes_sorted.shape[0] - 1)
        idx_high = jnp.minimum(jnp.searchsorted(cdf, 0.8, side="left"),
                                sizes_sorted.shape[0] - 1)
        x_min_mid = sizes_sorted[idx_low]
        x_max_mid = sizes_sorted[idx_high]
        theta_k   = x_max_mid - k * (x_max_mid - x_min_mid)
        denom     = wf * jnp.sqrt(
            sizes_arr ** 2 + theta_k ** 2 + SIZE_ANCHOR_R ** 2 + eps
        )
        z         = (sizes_arr - theta_k) / denom
        return 0.5 * (1.0 + lax.erf(z / jnp.sqrt(2.0)))

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]
        active_t    = ACTIVE_POS[t]

        def size_log_sem_for_utt(post):
            sv = _anchored_size_sem(sizes, post)
            return jnp.log(jnp.clip(sv, eps))

        size_log_sems = jax.vmap(size_log_sem_for_utt)(per_utt_posts)

        log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
        log_sem_table = jnp.concatenate([
            size_log_sems[:, None, :],
            jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
        ], axis=1)

        token_pres_t = TOKEN_PRESENT[t]
        log_prod_sem = jnp.einsum(
            "uav, uvo -> uao",
            token_pres_t,
            log_sem_table,
        )

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
        log_updated = log_per_utt_posts[:, None, :] + log_prod_sem
        log_Z = jax.scipy.special.logsumexp(log_updated, axis=-1)
        log_norm = log_updated - log_Z[:, :, None]
        log_L_ref = log_norm[:, :, referent_index]

        first_step_gate = (t == 0).astype(jnp.float32)
        suff_boost_vec = lambda_suff * first_step_gate * jnp.array([
            sufficient_dim == 0,
            sufficient_dim == 1,
            sufficient_dim == 2,
        ], dtype=jnp.float32)
        logits = jnp.where(
            cand_mask_t,
            alpha_vec[None, :] * log_L_ref + suff_boost_vec[None, :],
            -1e9,
        )
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo",
            ACTUAL_TOK_ONEHOT[t],
            log_sem_table,
        )
        log_updated_post = log_per_utt_posts + jnp.where(
            active_t[:, None],
            selected_log_sem,
            0.0,
        )
        log_Z_post = jax.scipy.special.logsumexp(
            log_updated_post, axis=-1, keepdims=True
        )
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)

        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(
        step,
        (init_scores, init_posts),
        jnp.arange(T),
    )

    blur_gate = 1.0 - is_sharp
    gamma_eff = (
        gamma_base
        + gamma_oneword * has_one_word_solution
        + gamma_sharp * blur_gate
    )
    length_bonus = gamma_eff * jnp.maximum(N_WORDS - 1.0, 0.0)

    # Erdc-gated (sufficient_dim == 0) utterance-level F-present boost.
    erdc_gate = (sufficient_dim == 0).astype(jnp.float32)
    form_present_bonus = lambda_form_mod * erdc_gate * F_PRESENT_15

    log_unnorm = log_lm_raw + log_final_scores + length_bonus + form_present_bonus
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


vectorized_incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod_hier = jax.vmap(
    incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # alpha_D
             0,    # alpha_C
             0,    # alpha_F
             None, # lambda_suff
             None, # lambda_form_mod
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta_lm
             None, # gamma_base
             None, # gamma_oneword
             None, # gamma_sharp
             None, # epsilon
             ),
)


@jax.jit
def jitted_speaker_contextual_anchored_gamma_sharpbonus_formmod_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
    lambda_suff, lambda_form_mod, color_semval, form_semval, k, wf, beta_lm,
    gamma_base, gamma_oneword, gamma_sharp, epsilon,
):
    return vectorized_incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
        lambda_suff, lambda_form_mod, color_semval, form_semval, k, wf, beta_lm,
        gamma_base, gamma_oneword, gamma_sharp, epsilon,
    )


def _make_contextual_pcalpha_formmod_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
):
    """(pcalpha) + gammasharp + erdc-gated F-present boost.

    Adds ``lambda_form_mod ~ Normal(0, 2)`` over the previous variant and routes through
    ``jitted_speaker_contextual_anchored_gamma_sharpbonus_formmod_hier``.

    The prior SD (2.0) matches ``gamma_base``/``gamma_oneword``. The the previous variant
    sweep puts the erdc-optimal value
    around 1.5-2 nats; Normal(0, 2) covers that without being strongly
    informative and lets the data pull it back toward 0 if the gate is wrong.

    +1 named coefficient over the previous variant -> 11 named + 339 latents.
    """
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None,
              condition_idx=None, n_conditions=None):
        log_beta_lm   = numpyro.sample("log_beta_lm",   dist.Normal(0.0, 0.5))
        beta_lm       = jnp.exp(log_beta_lm)

        alpha_D       = numpyro.sample("alpha_D",       dist.HalfNormal(5.0))
        alpha_C       = numpyro.sample("alpha_C",       dist.HalfNormal(5.0))
        alpha_F       = numpyro.sample("alpha_F",       dist.HalfNormal(5.0))
        lambda_suff   = numpyro.sample("lambda_suff",   dist.Normal(0.0, 1.0))
        lambda_form_mod = numpyro.sample("lambda_form_mod", dist.Normal(0.0, 2.0))
        gamma_base    = numpyro.sample("gamma_base",    dist.Normal(0.0, 2.0))
        gamma_oneword = numpyro.sample("gamma_oneword", dist.Normal(0.0, 2.0))
        gamma_sharp   = numpyro.sample("gamma_sharp",   dist.HalfNormal(2.0))
        epsilon       = numpyro.sample("epsilon",       dist.Beta(1.0, 50.0))
        tau           = numpyro.sample("tau",           dist.HalfNormal(0.2))

        # Non-centered (P × C) random effect on the shared per-trial alpha offset.
        with numpyro.plate("conditions_p", n_conditions, dim=-1):
            with numpyro.plate("participants", n_participants, dim=-2):
                delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, 1.0))
        delta = numpyro.deterministic("delta", delta_raw * tau)

        per_trial_offset = delta[participant_idx, condition_idx]
        alpha_D_per_trial = jnp.maximum(alpha_D + per_trial_offset, 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + per_trial_offset, 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + per_trial_offset, 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_contextual_anchored_gamma_sharpbonus_formmod_hier(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_suff, lambda_form_mod, color_semval, form_semval, k, wf,
                beta_lm, gamma_base, gamma_oneword, gamma_sharp, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_contextual_pcalpha_formmod_hier = _make_contextual_pcalpha_formmod_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
)


def incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon(
    states:                jnp.ndarray,
    sufficient_dim:        int,
    has_one_word_solution: float,
    is_sharp:              float,
    alpha_D:               float = 3.0,
    alpha_C:               float = 3.0,
    alpha_F:               float = 3.0,
    lambda_suff:           float = 0.0,
    lambda_form_mod:       float = 0.0,
    gamma_len3_erdc:       float = 0.0,
    lambda_noncanon:       float = 0.0,
    color_semval:          float = 0.95,
    form_semval:           float = 0.80,
    k:                     float = 0.50,
    wf:                    float = 1.00,
    beta_lm:               float = 1.00,
    gamma_base:            float = 0.0,
    gamma_oneword:         float = 0.0,
    gamma_sharp:           float = 0.0,
    epsilon:               float = 0.01,
    recursive:             bool  = True,
) -> jnp.ndarray:
    """(formmod) + two penalties from the a prior variant diagnostic.

    Adds, at the utterance level alongside the LM / length / form-present
    terms:

        - gamma_len3_erdc * 1[sufficient_dim == 0] * IS_3WORD[u]   (lever 1)
        - lambda_noncanon *                          F_BEFORE_C[u] (lever 2)

    Both coefficients ~ HalfNormal(2.0) and enter as **penalties** (subtracted).

    the base contextual variant — erdc-gated 3-word penalty. The a prior variant residual is a DF↔DCF/DFC
    mass split: the per-word length bonus γ·(N−1) in erdc/blurred (≈+0.89/word)
    over-rewards the 3rd token, so the model spends DF's mass on DCF/DFC. A
    per-word γ attenuation was rejected by sweep (it dumps mass into the
    1-word D, not the 2-word DF). A flat 3-word penalty redistributes to DF
    proportionally: in the sweep it lifts erdc/blurred DF 0.14→0.22 at
    coefficient ≈1, with DFC collapsing toward the human ~0.03.

    extension parameters 2 — global non-canonical-order penalty. F_BEFORE_C = {DFC, FDC, FC,
    FCD}: utterances violating canonical colour-before-form adjective order
    (Cinque 1994; Scott 2002; Sproat & Shih 1991), all ≈0 in human data. The
    a prior variant DFC over-production is an RSA cheap-F-continuation artefact (small
    alpha_F), NOT an LM problem (the LM residual already prefers DCF > DFC).
    The penalty is global: canonical order is a universal constraint and the
    sweep confirmed zero side-effects on zrdc (untouched) and brdc (DFC nudged
    0.043→0.016, harmless).

    Both LM-based fixes for these residuals were tested and rejected
    : uniform LM scaling collapses the
    distribution, and LM length/ordering decomposition makes DF monotonically
    worse — the anti-DF signal is inseparable from the LM's
    mass-concentration role. The residuals are γ-length + RSA-ordering, not
    LM-encoded.

    +2 named coefficients over the previous variant -> 13 named + 339 latents.
    """

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]
    alpha_vec      = jnp.array([alpha_D, alpha_C, alpha_F])

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    log_lm_raw = beta_lm * LOG_LM_RAW_15

    colors = states[:, 1]
    forms  = states[:, 2]
    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem  = jnp.log(
        jnp.where(forms  == 1, form_semval,  1.0 - form_semval)  + eps
    )

    uniform     = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(n_utt)
    init_posts  = jnp.broadcast_to(uniform, (n_utt, n_obj))

    def _anchored_size_sem(sizes_arr, post):
        post_sorted = post[size_sort_idx]
        post_sorted = post_sorted / (jnp.sum(post_sorted) + eps)
        cdf = jnp.cumsum(post_sorted)
        idx_low  = jnp.minimum(jnp.searchsorted(cdf, 0.2, side="left"),
                                sizes_sorted.shape[0] - 1)
        idx_high = jnp.minimum(jnp.searchsorted(cdf, 0.8, side="left"),
                                sizes_sorted.shape[0] - 1)
        x_min_mid = sizes_sorted[idx_low]
        x_max_mid = sizes_sorted[idx_high]
        theta_k   = x_max_mid - k * (x_max_mid - x_min_mid)
        denom     = wf * jnp.sqrt(
            sizes_arr ** 2 + theta_k ** 2 + SIZE_ANCHOR_R ** 2 + eps
        )
        z         = (sizes_arr - theta_k) / denom
        return 0.5 * (1.0 + lax.erf(z / jnp.sqrt(2.0)))

    def step(carry, t):
        log_scores, per_utt_posts = carry

        cand_mask_t = CANDIDATE_MASK[t]
        active_t    = ACTIVE_POS[t]

        def size_log_sem_for_utt(post):
            sv = _anchored_size_sem(sizes, post)
            return jnp.log(jnp.clip(sv, eps))

        size_log_sems = jax.vmap(size_log_sem_for_utt)(per_utt_posts)

        log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
        log_sem_table = jnp.concatenate([
            size_log_sems[:, None, :],
            jnp.broadcast_to(log_sem_static[None, :, :], (n_utt, 2, n_obj)),
        ], axis=1)

        token_pres_t = TOKEN_PRESENT[t]
        log_prod_sem = jnp.einsum(
            "uav, uvo -> uao",
            token_pres_t,
            log_sem_table,
        )

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
        log_updated = log_per_utt_posts[:, None, :] + log_prod_sem
        log_Z = jax.scipy.special.logsumexp(log_updated, axis=-1)
        log_norm = log_updated - log_Z[:, :, None]
        log_L_ref = log_norm[:, :, referent_index]

        first_step_gate = (t == 0).astype(jnp.float32)
        suff_boost_vec = lambda_suff * first_step_gate * jnp.array([
            sufficient_dim == 0,
            sufficient_dim == 1,
            sufficient_dim == 2,
        ], dtype=jnp.float32)
        logits = jnp.where(
            cand_mask_t,
            alpha_vec[None, :] * log_L_ref + suff_boost_vec[None, :],
            -1e9,
        )
        local_probs = jax.nn.softmax(logits, axis=-1)

        chosen = jnp.sum(local_probs * ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo",
            ACTUAL_TOK_ONEHOT[t],
            log_sem_table,
        )
        log_updated_post = log_per_utt_posts + jnp.where(
            active_t[:, None],
            selected_log_sem,
            0.0,
        )
        log_Z_post = jax.scipy.special.logsumexp(
            log_updated_post, axis=-1, keepdims=True
        )
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)

        # 2x2 SEMANTICS factor: recursive carries the Bayesian-updated
        # posterior across tokens (default = best model); static (recursive
        # =False) freezes it at the uniform prior every step (contextual
        # analogue of incremental_speaker_frozen). Python bool closed over
        # -> compile-time branch, no traced control flow.
        carried_posts = new_per_utt_posts if recursive else per_utt_posts
        return (log_scores + log_chosen, carried_posts), None

    (log_final_scores, _), _ = lax.scan(
        step,
        (init_scores, init_posts),
        jnp.arange(T),
    )

    blur_gate = 1.0 - is_sharp
    gamma_eff = (
        gamma_base
        + gamma_oneword * has_one_word_solution
        + gamma_sharp * blur_gate
    )
    length_bonus = gamma_eff * jnp.maximum(N_WORDS - 1.0, 0.0)

    # Erdc-gated (sufficient_dim == 0) utterance-level F-present boost .
    erdc_gate = (sufficient_dim == 0).astype(jnp.float32)
    form_present_bonus = lambda_form_mod * erdc_gate * F_PRESENT_15

    # the previous variant penalties (subtracted).
    len3_penalty     = gamma_len3_erdc * erdc_gate * IS_3WORD_15
    noncanon_penalty = lambda_noncanon * F_BEFORE_C_15

    log_unnorm = (
        log_lm_raw
        + log_final_scores
        + length_bonus
        + form_present_bonus
        - len3_penalty
        - noncanon_penalty
    )
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / n_utt


vectorized_incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier = jax.vmap(
    incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # alpha_D
             0,    # alpha_C
             0,    # alpha_F
             None, # lambda_suff
             None, # lambda_form_mod
             None, # gamma_len3_erdc
             None, # lambda_noncanon
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta_lm
             None, # gamma_base
             None, # gamma_oneword
             None, # gamma_sharp
             None, # epsilon
             None, # recursive (2x2 semantics factor; static Python bool)
             ),
)


@partial(jax.jit, static_argnames=("recursive",))
def jitted_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
    lambda_suff, lambda_form_mod, gamma_len3_erdc, lambda_noncanon,
    color_semval, form_semval, k, wf, beta_lm,
    gamma_base, gamma_oneword, gamma_sharp, epsilon, recursive=True,
):
    return vectorized_incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
        lambda_suff, lambda_form_mod, gamma_len3_erdc, lambda_noncanon,
        color_semval, form_semval, k, wf, beta_lm,
        gamma_base, gamma_oneword, gamma_sharp, epsilon, recursive,
    )


def global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_D=3.0, alpha_C=3.0, alpha_F=3.0, lambda_suff=0.0,
    lambda_form_mod=0.0, gamma_len3_erdc=0.0, lambda_noncanon=0.0,
    color_semval=0.95, form_semval=0.80, k=0.50, wf=1.00, beta_lm=1.00,
    gamma_base=0.0, gamma_oneword=0.0, gamma_sharp=0.0, epsilon=0.01,
    recursive: bool = True,
):
    """GLOBAL counterpart of the contextual-canon speaker (2x2 speaker factor).

    The incremental speaker accrues utility token-by-token while a literal
    listener's posterior is Bayesian-updated across tokens. The GLOBAL speaker
    instead computes ONE joint literal-listener posterior over the *full*
    utterance (conditioning on all of an utterance's asserted dimensions at
    once, FULL_PRESENT_15), then:

      - recursive=True  : one RSA pragmatic layer over the 15 candidate
                          utterances (S1 then L1) — the SEMANTICS factor's
                          "recursive" for the global speaker.
      - recursive=False : score the literal listener directly (literal-only).

    Every utterance-level term (LM prior, first-word suff boost, length,
    form-present, len3 / non-canonical penalties, epsilon mixing) is IDENTICAL
    to the incremental speaker, so the ONLY structural difference is the
    utility-accrual (joint vs sequential). Same fixed-constant / 10-named
    parameter inventory ⇒ a parameter-matched 2x2 cell.
    """
    eps = 1e-8
    referent_index = 0
    n_obj = states.shape[0]
    alpha_vec = jnp.array([alpha_D, alpha_C, alpha_F])
    alpha_bar = jnp.mean(alpha_vec)  # one effective rationality (joint speaker)

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]
    colors = states[:, 1]
    forms = states[:, 2]
    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps)
    log_form_sem = jnp.log(
        jnp.where(forms == 1, form_semval, 1.0 - form_semval) + eps)
    log_lm_raw = beta_lm * LOG_LM_RAW_15

    uniform = jnp.ones(n_obj) / n_obj

    # Anchored size semantics under the uniform prior (no incremental carry).
    post_sorted = uniform[size_sort_idx]
    post_sorted = post_sorted / (jnp.sum(post_sorted) + eps)
    cdf = jnp.cumsum(post_sorted)
    idx_low = jnp.minimum(jnp.searchsorted(cdf, 0.2, side="left"),
                          sizes_sorted.shape[0] - 1)
    idx_high = jnp.minimum(jnp.searchsorted(cdf, 0.8, side="left"),
                           sizes_sorted.shape[0] - 1)
    x_min_mid = sizes_sorted[idx_low]
    x_max_mid = sizes_sorted[idx_high]
    theta_k = x_max_mid - k * (x_max_mid - x_min_mid)
    denom = wf * jnp.sqrt(sizes ** 2 + theta_k ** 2 + SIZE_ANCHOR_R ** 2 + eps)
    z = (sizes - theta_k) / denom
    size_sem = 0.5 * (1.0 + lax.erf(z / jnp.sqrt(2.0)))
    log_size_sem = jnp.log(jnp.clip(size_sem, eps))

    # (3 dims, n_obj) log-semantics table; joint listener conditions on all
    # asserted dims of each utterance at once.
    log_sem_table = jnp.stack([log_size_sem, log_color_sem, log_form_sem],
                              axis=0)                       # (3, n_obj)
    log_joint = jnp.einsum("ud, do -> uo",
                           FULL_PRESENT_15, log_sem_table)  # (n_utt, n_obj)
    log_post = jnp.log(uniform)[None, :] + log_joint
    log_post = log_post - jax.scipy.special.logsumexp(
        log_post, axis=-1, keepdims=True)
    log_L0_ref = log_post[:, referent_index]                # (n_utt,)

    if recursive:
        log_S1 = alpha_bar * log_post
        log_S1 = log_S1 - jax.scipy.special.logsumexp(
            log_S1, axis=0, keepdims=True)                  # normalise / utt
        log_L1 = log_S1 + jnp.log(uniform)[None, :]
        log_L1 = log_L1 - jax.scipy.special.logsumexp(
            log_L1, axis=-1, keepdims=True)
        log_score = alpha_bar * log_L1[:, referent_index]
    else:
        log_score = alpha_bar * log_L0_ref

    # First-word boost: utterances whose first asserted dim == sufficient_dim
    # (FIRST_WORD is the per-utterance first-token/dimension index).
    suff_boost = lambda_suff * (FIRST_WORD == sufficient_dim).astype(jnp.float32)

    blur_gate = 1.0 - is_sharp
    gamma_eff = (gamma_base
                 + gamma_oneword * has_one_word_solution
                 + gamma_sharp * blur_gate)
    length_bonus = gamma_eff * jnp.maximum(N_WORDS - 1.0, 0.0)
    erdc_gate = (sufficient_dim == 0).astype(jnp.float32)
    form_present_bonus = lambda_form_mod * erdc_gate * F_PRESENT_15
    len3_penalty = gamma_len3_erdc * erdc_gate * IS_3WORD_15
    noncanon_penalty = lambda_noncanon * F_BEFORE_C_15

    log_unnorm = (log_lm_raw + log_score + suff_boost + length_bonus
                  + form_present_bonus - len3_penalty - noncanon_penalty)
    model_probs = jax.nn.softmax(log_unnorm)
    n_utt_local = log_unnorm.shape[0]
    return (1.0 - epsilon) * model_probs + epsilon / n_utt_local


vectorized_global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier = jax.vmap(
    global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon,
    in_axes=(0,    # states
             0,    # sufficient_dim
             0,    # has_one_word_solution
             0,    # is_sharp
             0,    # alpha_D
             0,    # alpha_C
             0,    # alpha_F
             None, # lambda_suff
             None, # lambda_form_mod
             None, # gamma_len3_erdc
             None, # lambda_noncanon
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta_lm
             None, # gamma_base
             None, # gamma_oneword
             None, # gamma_sharp
             None, # epsilon
             None, # recursive (2x2 semantics factor; static Python bool)
             ),
)


@partial(jax.jit, static_argnames=("recursive",))
def jitted_global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
    lambda_suff, lambda_form_mod, gamma_len3_erdc, lambda_noncanon,
    color_semval, form_semval, k, wf, beta_lm,
    gamma_base, gamma_oneword, gamma_sharp, epsilon, recursive=True,
):
    return vectorized_global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier(
        states, sufficient_dim, has_one_word_solution, is_sharp,
        alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
        lambda_suff, lambda_form_mod, gamma_len3_erdc, lambda_noncanon,
        color_semval, form_semval, k, wf, beta_lm,
        gamma_base, gamma_oneword, gamma_sharp, epsilon, recursive,
    )


def _make_contextual_pcalpha_canon_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
):
    """(formmod) + erdc 3-word penalty + canonical-order penalty.

    Adds ``gamma_len3_erdc ~ HalfNormal(2.0)`` and
    ``lambda_noncanon ~ HalfNormal(2.0)`` over the previous variant. HalfNormal (positive,
    entering as subtracted penalties) because both directions are
    theory/diagnostic-determined — over-specification penalty under sufficient
    size, and the canonical colour-before-form constraint — exactly as
    ``gamma_sharp`` uses HalfNormal for a sign-determined length term.

    +2 named coefficients over the previous variant -> 13 named + 339 latents.
    """
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None,
              condition_idx=None, n_conditions=None):
        log_beta_lm   = numpyro.sample("log_beta_lm",   dist.Normal(0.0, 0.5))
        beta_lm       = jnp.exp(log_beta_lm)

        alpha_D       = numpyro.sample("alpha_D",       dist.HalfNormal(5.0))
        alpha_C       = numpyro.sample("alpha_C",       dist.HalfNormal(5.0))
        alpha_F       = numpyro.sample("alpha_F",       dist.HalfNormal(5.0))
        lambda_suff   = numpyro.sample("lambda_suff",   dist.Normal(0.0, 1.0))
        lambda_form_mod = numpyro.sample("lambda_form_mod", dist.Normal(0.0, 2.0))
        gamma_len3_erdc = numpyro.sample("gamma_len3_erdc", dist.HalfNormal(2.0))
        lambda_noncanon = numpyro.sample("lambda_noncanon", dist.HalfNormal(2.0))
        gamma_base    = numpyro.sample("gamma_base",    dist.Normal(0.0, 2.0))
        gamma_oneword = numpyro.sample("gamma_oneword", dist.Normal(0.0, 2.0))
        gamma_sharp   = numpyro.sample("gamma_sharp",   dist.HalfNormal(2.0))
        epsilon       = numpyro.sample("epsilon",       dist.Beta(1.0, 50.0))
        tau           = numpyro.sample("tau",           dist.HalfNormal(0.2))

        # Non-centered (P × C) random effect on the shared per-trial alpha offset.
        with numpyro.plate("conditions_p", n_conditions, dim=-1):
            with numpyro.plate("participants", n_participants, dim=-2):
                delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, 1.0))
        delta = numpyro.deterministic("delta", delta_raw * tau)

        per_trial_offset = delta[participant_idx, condition_idx]
        alpha_D_per_trial = jnp.maximum(alpha_D + per_trial_offset, 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + per_trial_offset, 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + per_trial_offset, 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_suff, lambda_form_mod, gamma_len3_erdc, lambda_noncanon,
                color_semval, form_semval, k, wf,
                beta_lm, gamma_base, gamma_oneword, gamma_sharp, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_contextual_pcalpha_canon_hier = _make_contextual_pcalpha_canon_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
)


def _make_contextual_pcalpha_canon_betafixed_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
):
    """ABLATION: identical to ``_make_contextual_pcalpha_canon_model``
    but ``beta_lm`` is FIXED at the a prior variant posterior median
    (exp(LOG_BETA_LM_FIXED_ITER17) ≈ 6.738) instead of being sampled.

    Purpose: the canonical-order penalty conceptually overlaps the LM prior
    (GPT-2 already prefers DCF > DFC by ~1.75 nat at the fitted β_lm). In the
    free-β_lm a prior variant fit, corr(log_beta_lm, lambda_noncanon) = +0.43 and β_lm
    dropped 6.75→6.42 when the penalty entered — so lambda_noncanon may be
    partly reallocated LM signal. Pinning β_lm removes that freedom: if
    lambda_noncanon still lands ≈2.5–2.6 and R² holds ≈0.919, the
    canonical-order signal is genuinely ADDITIONAL, not relabelled LM mass.

    Not a ladder iteration — a robustness check. Same 339 latents; 12 named
    (drops log_beta_lm vs a prior variant's 13).
    """
    beta_lm_fixed = float(np.exp(LOG_BETA_LM_FIXED_ITER17))

    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None,
              condition_idx=None, n_conditions=None):
        beta_lm       = jnp.asarray(beta_lm_fixed)  # FIXED (not sampled)

        alpha_D       = numpyro.sample("alpha_D",       dist.HalfNormal(5.0))
        alpha_C       = numpyro.sample("alpha_C",       dist.HalfNormal(5.0))
        alpha_F       = numpyro.sample("alpha_F",       dist.HalfNormal(5.0))
        lambda_suff   = numpyro.sample("lambda_suff",   dist.Normal(0.0, 1.0))
        lambda_form_mod = numpyro.sample("lambda_form_mod", dist.Normal(0.0, 2.0))
        gamma_len3_erdc = numpyro.sample("gamma_len3_erdc", dist.HalfNormal(2.0))
        lambda_noncanon = numpyro.sample("lambda_noncanon", dist.HalfNormal(2.0))
        gamma_base    = numpyro.sample("gamma_base",    dist.Normal(0.0, 2.0))
        gamma_oneword = numpyro.sample("gamma_oneword", dist.Normal(0.0, 2.0))
        gamma_sharp   = numpyro.sample("gamma_sharp",   dist.HalfNormal(2.0))
        epsilon       = numpyro.sample("epsilon",       dist.Beta(1.0, 50.0))
        tau           = numpyro.sample("tau",           dist.HalfNormal(0.2))

        with numpyro.plate("conditions_p", n_conditions, dim=-1):
            with numpyro.plate("participants", n_participants, dim=-2):
                delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, 1.0))
        delta = numpyro.deterministic("delta", delta_raw * tau)

        per_trial_offset = delta[participant_idx, condition_idx]
        alpha_D_per_trial = jnp.maximum(alpha_D + per_trial_offset, 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + per_trial_offset, 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + per_trial_offset, 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_suff, lambda_form_mod, gamma_len3_erdc, lambda_noncanon,
                color_semval, form_semval, k, wf,
                beta_lm, gamma_base, gamma_oneword, gamma_sharp, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_contextual_pcalpha_canon_betafixed_hier = (
    _make_contextual_pcalpha_canon_betafixed_model(
        color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
    )
)


def _make_contextual_pcalpha_canon_parsimony_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
    drop: tuple = (), free: tuple = (), cell: str = "inc_rec",
):
    """Parsimony model: a prior variant (``contextual_pcalpha_canon``) minus its two
    free parameter drops, both PROVEN zero-cost in the May-2026 session:

    1. ``log_beta_lm`` FIXED at the a prior variant posterior median
       (exp(LOG_BETA_LM_FIXED_ITER17) ≈ 6.738) — the β_lm-fixed ablation
       (``contextual_pcalpha_canon_betafixed``) gave R² identical to a prior variant
       and ``lambda_noncanon`` unchanged, so the LM temperature is free to pin.
    2. ``gamma_len3_erdc`` DROPPED entirely — its a prior variant joint posterior
       collapsed to ≈0.04 [0.00,0.10] (data-dead); the speaker fn still takes
       the argument so we simply pass a constant 0.0 (term ≡ 0).

    Net: a prior variant's 13 named → **11 named coefficients + 339 latents**
    (``delta_raw``, 113 participants × 3 conditions). Fixed constants
    unchanged: ``color_semval=0.971``, ``form_semval=0.50``, ``k=0.5``,
    ``wf=WF_FIXED_ITER11_MEDIAN``, plus ``beta_lm`` now constant.

    Expected fit: ≈ a prior variant (R²(all) ≈ R²(emp≥.02) ≈ 0.919) at the smallest
    coefficient set with no remaining data-dead/redundant term — the starting
    point of the parsimony-vs-fit frontier.

    ``drop``: a tuple of named coefficients to PIN at 0.0 (not sampled) for the
    leave-one-out parsimony frontier. Supported:
    ``alpha_C``, ``alpha_F``, ``lambda_suff``, ``gamma_sharp``. Default ``()``
    is the 11-named base parsimony model (behaviour identical to before this
    arg was added — the running base NC is unaffected). Each drop removes one
    sampled coefficient (11→10) by zeroing its contribution.

    ``free``: a tuple of currently-FIXED constants to instead SAMPLE
    (fixed-constant diagnostic). Supported: ``color_semval`` (csv),
    ``form_semval`` (fsv), ``k``, ``wf``. Default ``()`` keeps all four fixed
    (behaviour identical to before this arg was added). Priors: csv, fsv ~
    Uniform(0.5, 0.999) (a semantic value below 0.5 would invert the
    predicate, so the lower bound is the vacuous point); k ~ Uniform(0, 1)
    (anchor fraction); wf via ``log_wf ~ Normal(-1.0, 0.5)`` (the established
    a prior variant freewf convention, prior bulk wf≈0.22–0.61). Each freed constant
    adds one sampled named parameter.
    """
    beta_lm_fixed = float(np.exp(LOG_BETA_LM_FIXED_ITER17))
    drop = frozenset(drop)
    _valid_drops = {"alpha_C", "alpha_F", "lambda_suff", "gamma_sharp"}
    _bad = drop - _valid_drops
    if _bad:
        raise ValueError(f"Unsupported parsimony drop(s): {sorted(_bad)}; "
                          f"supported: {sorted(_valid_drops)}")
    free = frozenset(free)
    _valid_free = {"color_semval", "form_semval", "k", "wf"}
    _bad_free = free - _valid_free
    if _bad_free:
        raise ValueError(f"Unsupported free constant(s): {sorted(_bad_free)}; "
                          f"supported: {sorted(_valid_free)}")

    # 2x2 (speaker × semantics) on the best model. cell selects
    # the speaker-utility-accrual path (incremental vs global) and the
    # listener-recursion flag (recursive vs static). Default "inc_rec" ==
    # the merged best model (behaviour unchanged; existing NCs unaffected).
    _valid_cells = {"inc_rec", "inc_static", "glob_rec", "glob_static"}
    if cell not in _valid_cells:
        raise ValueError(f"Unsupported 2x2 cell {cell!r}; "
                         f"expected {sorted(_valid_cells)}")
    _use_global = cell in ("glob_rec", "glob_static")
    _recursive  = cell in ("inc_rec", "glob_rec")

    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None,
              condition_idx=None, n_conditions=None):
        beta_lm       = jnp.asarray(beta_lm_fixed)  # FIXED (not sampled)

        alpha_D       = numpyro.sample("alpha_D",       dist.HalfNormal(5.0))
        alpha_C = (0.0 if "alpha_C" in drop
                   else numpyro.sample("alpha_C", dist.HalfNormal(5.0)))
        alpha_F = (0.0 if "alpha_F" in drop
                   else numpyro.sample("alpha_F", dist.HalfNormal(5.0)))
        lambda_suff = (0.0 if "lambda_suff" in drop
                       else numpyro.sample("lambda_suff", dist.Normal(0.0, 1.0)))
        lambda_form_mod = numpyro.sample("lambda_form_mod", dist.Normal(0.0, 2.0))
        lambda_noncanon = numpyro.sample("lambda_noncanon", dist.HalfNormal(2.0))
        gamma_base    = numpyro.sample("gamma_base",    dist.Normal(0.0, 2.0))
        gamma_oneword = numpyro.sample("gamma_oneword", dist.Normal(0.0, 2.0))
        gamma_sharp = (0.0 if "gamma_sharp" in drop
                       else numpyro.sample("gamma_sharp", dist.HalfNormal(2.0)))
        epsilon       = numpyro.sample("epsilon",       dist.Beta(1.0, 50.0))
        tau           = numpyro.sample("tau",           dist.HalfNormal(0.2))

        # Fixed semantic/size constants — SAMPLE the ones named in `free`,
        # otherwise use the closure constant (default: all four fixed).
        csv_r = (numpyro.sample("color_semval", dist.Uniform(0.5, 0.999))
                 if "color_semval" in free else color_semval)
        fsv_r = (numpyro.sample("form_semval", dist.Uniform(0.5, 0.999))
                 if "form_semval" in free else form_semval)
        k_r = (numpyro.sample("k", dist.Uniform(0.0, 1.0))
               if "k" in free else k)
        if "wf" in free:
            log_wf = numpyro.sample("log_wf", dist.Normal(-1.0, 0.5))
            wf_r = jnp.exp(log_wf)
        else:
            wf_r = wf

        # gamma_len3_erdc DROPPED (data-dead in a prior variant); term ≡ 0.
        gamma_len3_erdc = 0.0

        with numpyro.plate("conditions_p", n_conditions, dim=-1):
            with numpyro.plate("participants", n_participants, dim=-2):
                delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, 1.0))
        delta = numpyro.deterministic("delta", delta_raw * tau)

        per_trial_offset = delta[participant_idx, condition_idx]
        alpha_D_per_trial = jnp.maximum(alpha_D + per_trial_offset, 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + per_trial_offset, 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + per_trial_offset, 0.0)

        speaker_fn = (
            jitted_global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier
            if _use_global
            else jitted_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier
        )
        with numpyro.plate("data", len(states)):
            probs = speaker_fn(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_suff, lambda_form_mod, gamma_len3_erdc, lambda_noncanon,
                csv_r, fsv_r, k_r, wf_r,
                beta_lm, gamma_base, gamma_oneword, gamma_sharp, epsilon,
                recursive=_recursive,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_contextual_pcalpha_canon_parsimony_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
    )
)

# --- Leave-one-out parsimony frontier: each drops ONE
# coefficient from the 11-named base parsimony model (11→10 named). ---
likelihood_function_contextual_pcalpha_canon_parsimony_no_gammasharp_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("gamma_sharp",),
    )
)
likelihood_function_contextual_pcalpha_canon_parsimony_no_lambdasuff_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("lambda_suff",),
    )
)
likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",),
    )
)

# --- Fixed-constant diagnostic: on top of the RECOMMENDED model
# (drop=alpha_F), free each currently-fixed constant — separately and jointly
# — to test free-vs-fix and locate the data-optimal value. Each freed
# constant adds one sampled named parameter (10 → 11; freeall4 → 14). ---
likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freecsv_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",), free=("color_semval",),
    )
)
likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freefsv_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",), free=("form_semval",),
    )
)
likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freek_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",), free=("k",),
    )
)
likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freewf_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",), free=("wf",),
    )
)
likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freeall4_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",),
        free=("color_semval", "form_semval", "k", "wf"),
    )
)

# Re-fixed colour-semval at the free-csv posterior mean (≈0.59). The free-csv
# refit gained +0.022 R² but mixed poorly (ESS≈191); re-fixing recovers the
# gain at 10 named params with clean sampling — the manuscript-model
# candidate if it confirms. Still all-fixed: 10 named.
likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_csv059_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.59, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",),
    )
)

# --- 2x2 (speaker × semantics) on the best model. All four
# share the identical 10-named inventory + csv=0.59 + fixed beta_lm; they
# differ ONLY in cell= (utility accrual × listener recursion). inc_rec ==
# the merged best model (`…_no_alphaF_csv059`). ---
likelihood_function_contextual_pcalpha_canon_parsimony_2x2_inc_rec_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.59, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",), cell="inc_rec",
    )
)
likelihood_function_contextual_pcalpha_canon_parsimony_2x2_inc_static_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.59, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",), cell="inc_static",
    )
)
likelihood_function_contextual_pcalpha_canon_parsimony_2x2_glob_rec_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.59, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",), cell="glob_rec",
    )
)
likelihood_function_contextual_pcalpha_canon_parsimony_2x2_glob_static_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.59, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",), cell="glob_static",
    )
)


def _make_contextual_pcalpha_gammasharp_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
):
    """(pcalpha) + sharpness-gated length-bonus boost.

    Identical to ``_make_contextual_anchored_gamma_fixedwf_pcalpha_model`` except
    it samples ``gamma_sharp ~ HalfNormal(2.0)`` and uses
    ``jitted_speaker_contextual_anchored_gamma_sharpbonus_hier`` which adds
    ``gamma_sharp * (1 - is_sharp)`` to the length-bonus aggregation.

    Targets the per-step diagnostic finding on the previous variant main: speakers in
    erdc-blurred trials almost never stop after D (7% vs model's 34%), and
    speakers in erdc-sharp trials stop ~35% (which the model already gets
    right). The single new positive coefficient captures the sharpness-
    conditional over-specification at the speaker side rather than the
    listener side (which the previous variant's blur_R_inflation failed to identify).

    +1 named coefficient over the previous variant → 10 named + 339 latents.
    """
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None,
              condition_idx=None, n_conditions=None):
        log_beta_lm   = numpyro.sample("log_beta_lm",   dist.Normal(0.0, 0.5))
        beta_lm       = jnp.exp(log_beta_lm)

        alpha_D       = numpyro.sample("alpha_D",       dist.HalfNormal(5.0))
        alpha_C       = numpyro.sample("alpha_C",       dist.HalfNormal(5.0))
        alpha_F       = numpyro.sample("alpha_F",       dist.HalfNormal(5.0))
        lambda_suff   = numpyro.sample("lambda_suff",   dist.Normal(0.0, 1.0))
        gamma_base    = numpyro.sample("gamma_base",    dist.Normal(0.0, 2.0))
        gamma_oneword = numpyro.sample("gamma_oneword", dist.Normal(0.0, 2.0))
        gamma_sharp   = numpyro.sample("gamma_sharp",   dist.HalfNormal(2.0))
        epsilon       = numpyro.sample("epsilon",       dist.Beta(1.0, 50.0))
        tau           = numpyro.sample("tau",           dist.HalfNormal(0.2))

        # Non-centered (P × C) random effect on the shared per-trial alpha offset.
        with numpyro.plate("conditions_p", n_conditions, dim=-1):
            with numpyro.plate("participants", n_participants, dim=-2):
                delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, 1.0))
        delta = numpyro.deterministic("delta", delta_raw * tau)

        per_trial_offset = delta[participant_idx, condition_idx]
        alpha_D_per_trial = jnp.maximum(alpha_D + per_trial_offset, 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + per_trial_offset, 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + per_trial_offset, 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_contextual_anchored_gamma_sharpbonus_hier(
                states, sufficient_dim, has_one_word_solution, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_suff, color_semval, form_semval, k, wf, beta_lm,
                gamma_base, gamma_oneword, gamma_sharp, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_contextual_pcalpha_gammasharp_hier = _make_contextual_pcalpha_gammasharp_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
)


# =============================================================================
# V5 LIKELIHOOD FACTORIES  (v5: full, v5a: lambda_C only, v5b: gamma only)
# =============================================================================

def _make_v5_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0):
    """Factory for v5: ext-v1 + condition-gated lambda_C + saturating gamma_1, gamma_2."""
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              is_colour_sufficient=None, is_sharp=None):
        log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
        beta     = jnp.exp(log_beta)

        alpha_D       = numpyro.sample("alpha_D",       dist.HalfNormal(5.0))
        alpha_C       = numpyro.sample("alpha_C",       dist.HalfNormal(5.0))
        alpha_F       = numpyro.sample("alpha_F",       dist.HalfNormal(5.0))
        lambda_C      = numpyro.sample("lambda_C",      dist.Normal(0.0, 1.0))
        gamma_1       = numpyro.sample("gamma_1",       dist.Normal(0.0, 1.0))
        gamma_2       = numpyro.sample("gamma_2",       dist.Normal(0.0, 1.0))
        delta_gamma_1 = numpyro.sample("delta_gamma_1", dist.Normal(0.0, 1.0))
        delta_gamma_2 = numpyro.sample("delta_gamma_2", dist.Normal(0.0, 1.0))
        eta_1         = numpyro.sample("eta_1",         dist.Normal(0.0, 1.0))
        eta_2         = numpyro.sample("eta_2",         dist.Normal(0.0, 1.0))
        mu_noncanon   = numpyro.sample("mu_noncanon",   dist.Normal(0.0, 1.0))
        epsilon       = numpyro.sample("epsilon",       dist.Beta(1.0, 50.0))
        tau           = numpyro.sample("tau",           dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_v5_hier(
                states, is_colour_sufficient, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_C, color_semval, form_semval, k, wf, beta,
                gamma_1, gamma_2, delta_gamma_1, delta_gamma_2, eta_1, eta_2, mu_noncanon, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_v5_hier = _make_v5_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
)


def _make_v5_no_lm_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0):
    """v5 (F3): same as v5 (F2) but with LM prior removed (beta = 0).
    Tests whether μ_noncanon + γ terms fully absorb what the LM prior was doing."""
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              is_colour_sufficient=None, is_sharp=None):
        beta = 0.0  # no LM prior: LM_PRIOR^0 = 1 for all utts → log_P_beta constant

        alpha_D       = numpyro.sample("alpha_D",       dist.HalfNormal(5.0))
        alpha_C       = numpyro.sample("alpha_C",       dist.HalfNormal(5.0))
        alpha_F       = numpyro.sample("alpha_F",       dist.HalfNormal(5.0))
        lambda_C      = numpyro.sample("lambda_C",      dist.Normal(0.0, 1.0))
        gamma_1       = numpyro.sample("gamma_1",       dist.Normal(0.0, 1.0))
        gamma_2       = numpyro.sample("gamma_2",       dist.Normal(0.0, 1.0))
        delta_gamma_1 = numpyro.sample("delta_gamma_1", dist.Normal(0.0, 1.0))
        delta_gamma_2 = numpyro.sample("delta_gamma_2", dist.Normal(0.0, 1.0))
        eta_1         = numpyro.sample("eta_1",         dist.Normal(0.0, 1.0))
        eta_2         = numpyro.sample("eta_2",         dist.Normal(0.0, 1.0))
        mu_noncanon   = numpyro.sample("mu_noncanon",   dist.Normal(0.0, 1.0))
        epsilon       = numpyro.sample("epsilon",       dist.Beta(1.0, 50.0))
        tau           = numpyro.sample("tau",           dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_v5_hier(
                states, is_colour_sufficient, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_C, color_semval, form_semval, k, wf, beta,
                gamma_1, gamma_2, delta_gamma_1, delta_gamma_2, eta_1, eta_2, mu_noncanon, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_v5_no_lm_hier = _make_v5_no_lm_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
)


def _make_v5a_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0):
    """v5a: lambda_C added on top of ext-v1; linear gamma retained (gamma_2 := gamma_1)."""
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              is_colour_sufficient=None, is_sharp=None):
        log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
        beta     = jnp.exp(log_beta)

        alpha_D  = numpyro.sample("alpha_D",  dist.HalfNormal(5.0))
        alpha_C  = numpyro.sample("alpha_C",  dist.HalfNormal(5.0))
        alpha_F  = numpyro.sample("alpha_F",  dist.HalfNormal(5.0))
        lambda_C = numpyro.sample("lambda_C", dist.Normal(0.0, 1.0))
        gamma    = numpyro.sample("gamma",    dist.Normal(0.0, 1.0))
        epsilon  = numpyro.sample("epsilon",  dist.Beta(1.0, 50.0))
        tau      = numpyro.sample("tau",      dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_v5_hier(
                states, is_colour_sufficient, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_C, color_semval, form_semval, k, wf, beta,
                gamma, gamma, 0.0, 0.0, 0.0, 0.0, 0.0, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_v5a_hier = _make_v5a_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
)


def _make_v5b_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0):
    """v5b: saturating gamma_1, gamma_2 added on top of ext-v1; no lambda_C."""
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              is_colour_sufficient=None, is_sharp=None):
        log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
        beta     = jnp.exp(log_beta)

        alpha_D  = numpyro.sample("alpha_D", dist.HalfNormal(5.0))
        alpha_C  = numpyro.sample("alpha_C", dist.HalfNormal(5.0))
        alpha_F  = numpyro.sample("alpha_F", dist.HalfNormal(5.0))
        gamma_1  = numpyro.sample("gamma_1", dist.Normal(0.0, 1.0))
        gamma_2  = numpyro.sample("gamma_2", dist.Normal(0.0, 1.0))
        epsilon  = numpyro.sample("epsilon", dist.Beta(1.0, 50.0))
        tau      = numpyro.sample("tau",     dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_v5_hier(
                states, is_colour_sufficient, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                0.0, color_semval, form_semval, k, wf, beta,
                gamma_1, gamma_2, 0.0, 0.0, 0.0, 0.0, 0.0, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_v5b_hier = _make_v5b_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
)


# =============================================================================
# V5 2×2 FACTORIES: incremental-static, global-recursive, global-static
# For global variants: by convention, gamma/delta/epsilon are fixed
# at the v5 (incremental-recursive) posterior means since they are not jointly
# identifiable with the single alpha under a global softmax. λ_C and α (+ τ, δ)
# are sampled.
# =============================================================================

V5_FIXED = dict(
    gamma_1=2.32, gamma_2=1.58,
    delta_gamma_1=-2.16, delta_gamma_2=-0.58,
    eta_1=-0.61, eta_2=-0.34,
    mu_noncanon=-5.08,
    epsilon=0.10,
)


def _make_v5_inc_static_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0):
    """v5 + frozen (context-fixed) incremental speaker."""
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              is_colour_sufficient=None, is_sharp=None):
        log_beta      = numpyro.sample("log_beta",      dist.Normal(0.0, 0.5))
        beta          = jnp.exp(log_beta)
        alpha_D       = numpyro.sample("alpha_D",       dist.HalfNormal(5.0))
        alpha_C       = numpyro.sample("alpha_C",       dist.HalfNormal(5.0))
        alpha_F       = numpyro.sample("alpha_F",       dist.HalfNormal(5.0))
        lambda_C      = numpyro.sample("lambda_C",      dist.Normal(0.0, 1.0))
        gamma_1       = numpyro.sample("gamma_1",       dist.Normal(0.0, 1.0))
        gamma_2       = numpyro.sample("gamma_2",       dist.Normal(0.0, 1.0))
        delta_gamma_1 = numpyro.sample("delta_gamma_1", dist.Normal(0.0, 1.0))
        delta_gamma_2 = numpyro.sample("delta_gamma_2", dist.Normal(0.0, 1.0))
        eta_1         = numpyro.sample("eta_1",         dist.Normal(0.0, 1.0))
        eta_2         = numpyro.sample("eta_2",         dist.Normal(0.0, 1.0))
        mu_noncanon   = numpyro.sample("mu_noncanon",   dist.Normal(0.0, 1.0))
        epsilon       = numpyro.sample("epsilon",       dist.Beta(1.0, 50.0))
        tau           = numpyro.sample("tau",           dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_frozen_v5_hier(
                states, is_colour_sufficient, is_sharp,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_C, color_semval, form_semval, k, wf, beta,
                gamma_1, gamma_2, delta_gamma_1, delta_gamma_2, eta_1, eta_2, mu_noncanon, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_v5_inc_static_hier = _make_v5_inc_static_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
)


def _make_v5_global_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
                          static: bool = False):
    """v5 + global speaker. gamma/delta/epsilon fixed at v5 posterior means
    (not jointly identifiable with single global alpha)."""
    jit_fn = jitted_global_speaker_static_v5_hier if static else jitted_global_speaker_v5_hier

    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              is_colour_sufficient=None, is_sharp=None):
        log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
        beta     = jnp.exp(log_beta)
        alpha    = numpyro.sample("alpha",    dist.HalfNormal(5.0))
        lambda_C = numpyro.sample("lambda_C", dist.Normal(0.0, 1.0))
        tau      = numpyro.sample("tau",      dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_per_trial = jnp.maximum(alpha + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jit_fn(
                states, is_colour_sufficient, is_sharp, alpha_per_trial,
                lambda_C, color_semval, form_semval, k, wf, beta,
                V5_FIXED["gamma_1"], V5_FIXED["gamma_2"],
                V5_FIXED["delta_gamma_1"], V5_FIXED["delta_gamma_2"],
                V5_FIXED["eta_1"], V5_FIXED["eta_2"],
                V5_FIXED["mu_noncanon"], V5_FIXED["epsilon"],
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_v5_global_hier = _make_v5_global_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0, static=False,
)

likelihood_function_v5_global_static_hier = _make_v5_global_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0, static=True,
)


def _make_v5_global_full_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
                                static: bool = False):
    """v5 + global speaker, ALL params sampled (not fixed at V5_FIXED).
    Tests whether the global-side semantic-regime effect survives when
    γ/δγ/η/μ are allowed to move freely."""
    jit_fn = jitted_global_speaker_static_v5_hier if static else jitted_global_speaker_v5_hier

    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              is_colour_sufficient=None, is_sharp=None):
        log_beta      = numpyro.sample("log_beta",      dist.Normal(0.0, 0.5))
        beta          = jnp.exp(log_beta)
        alpha         = numpyro.sample("alpha",         dist.HalfNormal(5.0))
        lambda_C      = numpyro.sample("lambda_C",      dist.Normal(0.0, 1.0))
        gamma_1       = numpyro.sample("gamma_1",       dist.Normal(0.0, 1.0))
        gamma_2       = numpyro.sample("gamma_2",       dist.Normal(0.0, 1.0))
        delta_gamma_1 = numpyro.sample("delta_gamma_1", dist.Normal(0.0, 1.0))
        delta_gamma_2 = numpyro.sample("delta_gamma_2", dist.Normal(0.0, 1.0))
        eta_1         = numpyro.sample("eta_1",         dist.Normal(0.0, 1.0))
        eta_2         = numpyro.sample("eta_2",         dist.Normal(0.0, 1.0))
        mu_noncanon   = numpyro.sample("mu_noncanon",   dist.Normal(0.0, 1.0))
        epsilon       = numpyro.sample("epsilon",       dist.Beta(1.0, 50.0))
        tau           = numpyro.sample("tau",           dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))
        alpha_per_trial = jnp.maximum(alpha + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jit_fn(
                states, is_colour_sufficient, is_sharp, alpha_per_trial,
                lambda_C, color_semval, form_semval, k, wf, beta,
                gamma_1, gamma_2, delta_gamma_1, delta_gamma_2,
                eta_1, eta_2, mu_noncanon, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_v5_global_full_hier = _make_v5_global_full_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0, static=False,
)

likelihood_function_v5_global_static_full_hier = _make_v5_global_full_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0, static=True,
)


def likelihood_function_incremental_speaker(states=None, empirical=None):
    color_semval = 0.971
    form_semval  = 0.50
    k  = 0.5
    wf = 1.0

    alpha_D  = numpyro.sample("alpha_D", dist.HalfNormal(5.0))
    alpha_C  = numpyro.sample("alpha_C", dist.HalfNormal(5.0))
    alpha_F  = numpyro.sample("alpha_F", dist.HalfNormal(5.0))
    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)
    gamma    = numpyro.sample("gamma", dist.Normal(0.0, 1.0))
    epsilon  = numpyro.sample("epsilon", dist.Beta(1.0, 50.0))

    with numpyro.plate("data", len(states)):
        probs = jitted_speaker(
            states, alpha_D, alpha_C, alpha_F, color_semval, form_semval, k, wf, beta, gamma, epsilon
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


# =============================================================================
# STATIC-SEMANTICS LIKELIHOOD FUNCTIONS
# =============================================================================

def likelihood_function_global_speaker_static(states=None, empirical=None):
    """Global speaker with STATIC size semantics (no context recursion)."""
    color_sem = 0.971
    form_sem  = 0.50
    k  = 0.5
    wf = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)
    alpha    = numpyro.sample("alpha", dist.HalfNormal(5.0))
    gamma    = numpyro.sample("gamma", dist.Normal(0.0, 1.0))
    epsilon  = numpyro.sample("epsilon", dist.Beta(1.0, 50.0))

    with numpyro.plate("data", len(states)):
        probs = jitted_global_speaker_static(states, alpha, color_sem, form_sem, k, wf, beta, gamma, epsilon)
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


def likelihood_function_global_speaker_static_hier(
    states=None, empirical=None,
    participant_idx=None, n_participants=None,
):
    """Global speaker hierarchical + STATIC size semantics."""
    color_sem = 0.971
    form_sem  = 0.50
    k  = 0.5
    wf = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)
    alpha    = numpyro.sample("alpha", dist.HalfNormal(5.0))
    # gamma and epsilon fixed at incremental posterior means (not identifiable in global)
    gamma    = 2.32
    epsilon  = 0.23
    tau      = numpyro.sample("tau",   dist.HalfNormal(0.2))

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    alpha_per_trial = jnp.maximum(alpha + delta[participant_idx], 0.0)

    with numpyro.plate("data", len(states)):
        probs = jitted_global_speaker_static_hier(states, alpha_per_trial, color_sem, form_sem, k, wf, beta, gamma, epsilon)
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


def likelihood_function_incremental_speaker_frozen(states=None, empirical=None):
    """Incremental speaker with FROZEN (static) size semantics."""
    color_semval = 0.971
    form_semval  = 0.50
    k  = 0.5
    wf = 1.0

    alpha_D  = numpyro.sample("alpha_D", dist.HalfNormal(5.0))
    alpha_C  = numpyro.sample("alpha_C", dist.HalfNormal(5.0))
    alpha_F  = numpyro.sample("alpha_F", dist.HalfNormal(5.0))
    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)
    gamma    = numpyro.sample("gamma", dist.Normal(0.0, 1.0))
    epsilon  = numpyro.sample("epsilon", dist.Beta(1.0, 50.0))

    with numpyro.plate("data", len(states)):
        probs = jitted_speaker_frozen(states, alpha_D, alpha_C, alpha_F, color_semval, form_semval, k, wf, beta, gamma, epsilon)
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


def likelihood_function_incremental_speaker_frozen_hier(
    states=None, empirical=None,
    participant_idx=None, n_participants=None,
):
    """Incremental speaker hierarchical + FROZEN (static) size semantics."""
    color_semval = 0.971
    form_semval  = 0.50
    k  = 0.5
    wf = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)
    alpha_D  = numpyro.sample("alpha_D", dist.HalfNormal(5.0))
    alpha_C  = numpyro.sample("alpha_C", dist.HalfNormal(5.0))
    alpha_F  = numpyro.sample("alpha_F", dist.HalfNormal(5.0))
    gamma    = numpyro.sample("gamma", dist.Normal(0.0, 1.0))
    epsilon  = numpyro.sample("epsilon", dist.Beta(1.0, 50.0))
    tau      = numpyro.sample("tau",   dist.HalfNormal(0.2))

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
    alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
    alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

    with numpyro.plate("data", len(states)):
        probs = jitted_speaker_frozen_hier(
            states, alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
            color_semval, form_semval, k, wf, beta, gamma, epsilon
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


# =============================================================================
# LM-PRIOR ABLATION LIKELIHOOD FUNCTIONS
# =============================================================================

def likelihood_function_incremental_lm_only_hier(
    states=None, empirical=None,
    participant_idx=None, n_participants=None,
):
    """LM-only ablation: alpha=0 (no RSA informativeness), infer only log_beta."""
    color_semval = 0.971
    form_semval  = 0.50
    k, wf = 0.5, 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)
    gamma    = numpyro.sample("gamma", dist.Normal(0.0, 1.0))
    epsilon  = numpyro.sample("epsilon", dist.Beta(1.0, 50.0))

    alpha_zero = jnp.zeros(len(states))

    with numpyro.plate("data", len(states)):
        probs = jitted_speaker_hier(
            states, alpha_zero, alpha_zero, alpha_zero,
            color_semval, form_semval, k, wf, beta, gamma, epsilon
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


def likelihood_function_incremental_rsa_only_hier(
    states=None, empirical=None,
    participant_idx=None, n_participants=None,
):
    """RSA-only ablation: beta=0 (uniform LM prior), infer alpha + hier."""
    color_semval = 0.971
    form_semval  = 0.50
    k, wf = 0.5, 1.0
    beta = 0.0

    alpha_D  = numpyro.sample("alpha_D", dist.HalfNormal(5.0))
    alpha_C  = numpyro.sample("alpha_C", dist.HalfNormal(5.0))
    alpha_F  = numpyro.sample("alpha_F", dist.HalfNormal(5.0))
    gamma    = numpyro.sample("gamma", dist.Normal(0.0, 1.0))
    epsilon  = numpyro.sample("epsilon", dist.Beta(1.0, 50.0))
    tau      = numpyro.sample("tau",   dist.HalfNormal(0.2))

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
    alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
    alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

    with numpyro.plate("data", len(states)):
        probs = jitted_speaker_hier(
            states, alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
            color_semval, form_semval, k, wf, beta, gamma, epsilon
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


# =============================================================================
# LOOKAHEAD SPEAKER LIKELIHOOD FUNCTION
# =============================================================================

def likelihood_function_incremental_speaker_lookahead_hier(
    states=None, empirical=None,
    participant_idx=None, n_participants=None,
):
    """Incremental speaker with d=1 lookahead, per-participant random effects."""
    color_semval = 0.971
    form_semval  = 0.841
    k, wf = 0.5, 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)

    alpha = numpyro.sample("alpha", dist.HalfNormal(5.0))
    tau   = numpyro.sample("tau",   dist.HalfNormal(0.2))

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    alpha_per_trial = jnp.maximum(alpha + delta[participant_idx], 0.0)

    with numpyro.plate("data", len(states)):
        probs = jitted_speaker_lookahead_hier(
            states, alpha_per_trial, color_semval, form_semval, k, wf, beta
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


# =============================================================================
# EXTENDED SPEAKER LIKELIHOOD FUNCTION
# =============================================================================

def likelihood_function_incremental_speaker_extended_hier(
    states=None, empirical=None,
    participant_idx=None, n_participants=None,
):
    """Extended incremental speaker: per-dim alpha, step-level mention biases,
    length bias, lapse rate."""
    color_semval = 0.971
    form_semval  = 0.50
    k            = 0.5
    wf           = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)

    # Shared rationality
    alpha = numpyro.sample("alpha", dist.HalfNormal(5.0))

    # Step-level mention biases (D = 0 reference)
    mu_C = numpyro.sample("mu_C", dist.Normal(0.0, 2.0))
    mu_F = numpyro.sample("mu_F", dist.Normal(0.0, 2.0))

    # Length bias (positive = prefer longer utterances)
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 1.0))

    # Lapse rate (tighter prior)
    epsilon = numpyro.sample("epsilon", dist.Beta(2.0, 98.0))

    # Hierarchical: participant offset on alpha
    tau = numpyro.sample("tau", dist.HalfNormal(0.2))

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    alpha_per_trial = jnp.maximum(alpha + delta[participant_idx], 0.0)

    with numpyro.plate("data", len(states)):
        probs = jitted_speaker_extended_hier(
            states, alpha_per_trial, color_semval, form_semval,
            k, wf, beta, gamma, epsilon, mu_C, mu_F,
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


# ── Mixture model: two incremental speaker components ────────────────────

def incremental_speaker_mixture(
    states:       jnp.ndarray,
    alpha1_D:     float = 5.0,
    alpha1_C:     float = 3.0,
    alpha1_F:     float = 3.0,
    alpha2_D:     float = 1.0,
    alpha2_C:     float = 1.0,
    alpha2_F:     float = 1.0,
    color_semval: float = 0.971,
    form_semval:  float = 0.50,
    k:            float = 0.50,
    wf:           float = 1.00,
    beta:         float = 1.00,
    gamma1:       float = 0.0,
    gamma2:       float = 0.0,
    epsilon:      float = 0.01,
    pi:           float = 0.50,
) -> jnp.ndarray:
    """Mixture of two incremental speakers with shared semantics/beta.

    Component 1 ("deliberative"): per-dim alphas + gamma1, no lapse.
    Component 2 ("habitual"):     per-dim alphas + gamma2, no lapse.
    The mixture is: pi * comp1 + (1-pi) * comp2, then lapse with epsilon.
    """
    probs1 = incremental_speaker(
        states, alpha1_D, alpha1_C, alpha1_F,
        color_semval, form_semval, k, wf, beta, gamma1, 0.0,
    )
    probs2 = incremental_speaker(
        states, alpha2_D, alpha2_C, alpha2_F,
        color_semval, form_semval, k, wf, beta, gamma2, 0.0,
    )
    mixed = pi * probs1 + (1.0 - pi) * probs2
    return (1.0 - epsilon) * mixed + epsilon / n_utt


# Vmap: per-trial axes for component 1 alphas (hierarchical offsets),
# everything else shared across trials.
vectorized_incremental_speaker_mixture_hier = jax.vmap(
    incremental_speaker_mixture,
    in_axes=(0,    # states
             0,    # alpha1_D  ← per-trial
             0,    # alpha1_C  ← per-trial
             0,    # alpha1_F  ← per-trial
             None, # alpha2_D
             None, # alpha2_C
             None, # alpha2_F
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta
             None, # gamma1
             None, # gamma2
             None, # epsilon
             None, # pi
             ),
)


@jax.jit
def jitted_speaker_mixture_hier(
    states, alpha1_D_pt, alpha1_C_pt, alpha1_F_pt,
    alpha2_D, alpha2_C, alpha2_F,
    color_semval, form_semval, k, wf, beta,
    gamma1, gamma2, epsilon, pi,
):
    return vectorized_incremental_speaker_mixture_hier(
        states, alpha1_D_pt, alpha1_C_pt, alpha1_F_pt,
        alpha2_D, alpha2_C, alpha2_F,
        color_semval, form_semval, k, wf, beta,
        gamma1, gamma2, epsilon, pi,
    )


def likelihood_function_incremental_speaker_mixture_hier(
    states=None, empirical=None,
    participant_idx=None, n_participants=None,
):
    """Mixture of two incremental speakers (per-dim alpha each).

    Component 1 ("deliberative"): higher alphas, hierarchical offsets.
    Component 2 ("habitual"): lower alphas, no hierarchical offsets.
    Shared: beta, semantics, k, wf.  Separate: gamma per component.
    """
    color_semval = 0.971
    form_semval  = 0.50
    k            = 0.5
    wf           = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)

    # Component 1 ("deliberative") — per-dim alpha with hier offsets
    alpha1_D = numpyro.sample("alpha1_D", dist.HalfNormal(5.0))
    alpha1_C = numpyro.sample("alpha1_C", dist.HalfNormal(5.0))
    alpha1_F = numpyro.sample("alpha1_F", dist.HalfNormal(5.0))
    gamma1   = numpyro.sample("gamma1",   dist.Normal(0.0, 1.0))

    # Component 2 ("habitual") — per-dim alpha, no hier offsets
    # Ordering constraint: alpha1_D > alpha2_D to break label switching.
    # Sample gap_D > 0, then alpha2_D = alpha1_D - gap_D (clamped ≥ 0).
    gap_D    = numpyro.sample("gap_D",    dist.HalfNormal(3.0))
    alpha2_D = numpyro.deterministic("alpha2_D", jnp.maximum(alpha1_D - gap_D, 0.0))
    alpha2_C = numpyro.sample("alpha2_C", dist.HalfNormal(3.0))
    alpha2_F = numpyro.sample("alpha2_F", dist.HalfNormal(3.0))
    gamma2   = numpyro.sample("gamma2",   dist.Normal(0.0, 1.0))

    # Mixing weight and lapse
    pi      = numpyro.sample("pi",      dist.Beta(2.0, 2.0))
    epsilon = numpyro.sample("epsilon", dist.Beta(2.0, 98.0))

    # Hierarchical offsets on component 1 only
    tau = numpyro.sample("tau", dist.HalfNormal(0.2))

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    alpha1_D_pt = jnp.maximum(alpha1_D + delta[participant_idx], 0.0)
    alpha1_C_pt = jnp.maximum(alpha1_C + delta[participant_idx], 0.0)
    alpha1_F_pt = jnp.maximum(alpha1_F + delta[participant_idx], 0.0)

    with numpyro.plate("data", len(states)):
        probs = jitted_speaker_mixture_hier(
            states, alpha1_D_pt, alpha1_C_pt, alpha1_F_pt,
            alpha2_D, alpha2_C, alpha2_F,
            color_semval, form_semval, k, wf, beta,
            gamma1, gamma2, epsilon, pi,
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


def likelihood_function_incremental_speaker_mixture_simple_hier(
    states=None, empirical=None,
    participant_idx=None, n_participants=None,
):
    """Simplified mixture: single alpha per component (not per-dim).

    Component 1 ("deliberative"): high alpha, hier offsets, gamma1.
    Component 2 ("habitual"):     low alpha, no hier offsets, gamma2.
    Ordering: alpha1 > alpha2 via gap parameterisation.
    """
    color_semval = 0.971
    form_semval  = 0.50
    k            = 0.5
    wf           = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)

    # Component 1 — single alpha (applied to all dims)
    alpha1  = numpyro.sample("alpha1", dist.HalfNormal(5.0))
    gamma1  = numpyro.sample("gamma1", dist.Normal(0.0, 1.0))

    # Component 2 — alpha1 - gap (ordered)
    gap     = numpyro.sample("gap", dist.HalfNormal(3.0))
    alpha2  = numpyro.deterministic("alpha2", jnp.maximum(alpha1 - gap, 0.0))
    gamma2  = numpyro.sample("gamma2", dist.Normal(0.0, 1.0))

    # Mixing weight and lapse
    pi      = numpyro.sample("pi",      dist.Beta(2.0, 2.0))
    epsilon = numpyro.sample("epsilon", dist.Beta(2.0, 98.0))

    # Hierarchical offsets on component 1 only
    tau = numpyro.sample("tau", dist.HalfNormal(0.2))

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    # Broadcast single alpha to all 3 dims for component 1
    alpha1_pt = jnp.maximum(alpha1 + delta[participant_idx], 0.0)

    with numpyro.plate("data", len(states)):
        probs = jitted_speaker_mixture_hier(
            states, alpha1_pt, alpha1_pt, alpha1_pt,
            alpha2, alpha2, alpha2,
            color_semval, form_semval, k, wf, beta,
            gamma1, gamma2, epsilon, pi,
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


# ========================
def canonicalize_speaker_type(speaker_type: str) -> str:
    """Map legacy CLI aliases to the canonical public model name."""
    if speaker_type == "incremental_frozen":
        return "incremental_static"
    return speaker_type


def test():
    """Quick sanity check of model functions."""
    data = import_dataset()
    states_train = data["states_train"]
    example_state = states_train[2]

    print("Example state:", example_state)
    example_incremental_semantics = incremental_semantics_jax(example_state, 0.95, 0.95, 0.5, 0.5)
    print("Example incremental semantics:", example_incremental_semantics)
    example_global_speaker = global_speaker(example_state, 0.5, 0.95, 0.5)
    print("Example global speaker:", example_global_speaker[0,])
    example_incremental_speaker = incremental_speaker(example_state, 0.5, 0.95, 0.5)
    print("Example incremental speaker:", example_incremental_speaker)
