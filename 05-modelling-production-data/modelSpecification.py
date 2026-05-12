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

# ── First word of each utterance (for first-word intercepts) ─────────────────
# D=0, C=1, F=2 → one-hot masks for first-word bias
FIRST_WORD = jnp.array(utterance_list)[:, 0]  # (n_utt,) first token of each utterance

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

    Lever 1: the LM term uses raw GPT-2 log-probability (LOG_LM_RAW_15) so the
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

    Identical to ``incremental_speaker_contextual`` (Lever 1) except the
    speaker bears an implicit cost proportional to the residual listener
    uncertainty their utterance leaves over the candidate referents:

        log_unnorm = beta_lm * LOG_LM_RAW_15
                   + log_final_scores
                   - lambda_uncertainty * (1 - P_listener_final(target | u))

    where ``P_listener_final[u]`` is the listener's posterior on the referent
    after consuming utterance ``u`` (the second carry slot of ``lax.scan`` that
    Lever 1 discards).  As ``lambda_uncertainty -> 0`` this reduces exactly to
    Lever 1.  As it grows, longer utterances that drive the listener posterior
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


# Contextual + listener-uncertainty cost: same as the Lever-1 vmap but with
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
    """C2-contextual (Lever 1): per-dim alpha + RAW LM + lambda_suff. No gammas."""
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
    of Lever 1).

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

        # Same centered delta parameterization as Lever 1 — funnel is weak
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


def _make_contextual_freewf_model(color_semval=0.971, form_semval=0.50, k=0.5):
    """Lever-1 speaker but with ``wf`` (size-semantics noise scale) learned.

    Iter 8 in the R²-first loop. The pre-loop default ``wf = 1.0`` saturates
    the size semantics into its flat regime — sharp and blurred trials get
    nearly identical listener confidences (RSA target ≈ 0.23 vs 0.21 for
    erdc) despite the empirical target-distractor gap being 6.66 vs 4.15.

    Putting ``log_wf ~ Normal(-1.0, 0.5)`` (prior bulk wf ≈ 0.22 – 0.61)
    lets the data find a wf in the erf's discriminative range, which
    propagates the existing sharp/blurred encoding in the size values
    through to a meaningful listener-confidence asymmetry. No new output
    coefficient; the entire mechanism lives in one re-tuned representation
    parameter. +1 free param vs Lever 1.
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


# Iter 9 + Iter 10: size semantics with a fixed reference scale R that
# anchors the comparison and breaks the function's scale-invariance.
SIZE_ANCHOR_R = 5.0

# Iter 12: parsimonious fixed wf, set to the posterior median of Iter 11
# (R²-winning variant on dc subset). Lets the anchored+2-gamma model carry
# the same wf-tuned representation without paying a free parameter for it.
# Source: arviz median of log_wf in iter11 NC was -0.3775 → wf = 0.6856.
WF_FIXED_ITER11_MEDIAN = 0.6856


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
    """Lever-1 speaker with size semantics anchored to an absolute scale R.

    The only structural change vs ``incremental_speaker_contextual`` is the
    size-semantics denominator:

        denom = wf * sqrt(sizes² + theta_k² + R²)        (anchored)
        denom = wf * sqrt(sizes² + theta_k²)             (Lever 1)

    With ``R = SIZE_ANCHOR_R = 5.0``, the denominator can never collapse
    below ``wf * R``, so the listener cannot saturate the size discriminant
    when the local cluster is tight (which is exactly what was happening in
    Lever 1 with theta_k pinned to the distractor cluster). Sharp trials
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
    """Iter 9: Lever-1 priors, anchored size semantics (R = 5.0). 0 new params."""
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
    """Iter 10: anchored size semantics + learned ``wf``. +1 free param."""
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
    """Iter 11: anchored size semantics + free wf + compact 2-gamma length bonus.

    Three new free coefficients vs Lever 1: ``log_wf`` (size-semantics
    sensitivity), ``gamma_base`` (per-extra-token length bonus baseline),
    and ``gamma_oneword`` (modulation when the trial has a one-word
    solution). Total 10 named parameters, vs Lever 1's 7 and the 6-gamma
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
    """Iter 12: same as Iter 11 but with ``wf`` hardcoded at the Iter 11
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
    """Iter 14: Iter 12 + hierarchical alpha by ``(participant × condition)``.

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
    free form_semval (Iter 13) — the model's alpha pattern is the
    bottleneck. Allowing alpha to drift differently per (participant,
    condition) lets the posterior find that subjects in erdc trials need
    different overall alpha than in zrdc / brdc, without committing to a
    hand-specified per-suff_dim coefficient.

    Named-coefficient count: 9 (same as Iter 12). The (P × C) cells add
    +226 latents beyond Iter 12 — within numpyro's plate machinery, not
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
# For global variants: per memo §5.1 convention, gamma/delta/epsilon are fixed
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
