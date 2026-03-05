import os
os.environ["JAX_PLATFORMS"] = "cpu"        # ← force CPU before JAX init  
os.environ["JAX_TRACEBACK_FILTERING"] = "off"  # optional: full tracebacks  
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"  


import argparse
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

# Fixed sharpness-conditioned size gain (Option B, non-inferred start values)
GAMMA_BLURRED = 0.9
GAMMA_SHARP = 2

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
    gamma:       float = 1.0,   # size gain / discriminability
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
    z     = gamma * (sizes - theta_k) / denom  
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
    gamma:     float = 1.0,        # size gain / discriminability
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
    z     = gamma * (sizes - theta_k) / denom                # (n_obj,)

    return 0.5 * (1.0 + lax.erf(z / jnp.sqrt(2.0)))         # (n_obj,) ∈ (0,1)


def compute_size_semantics_fast_presorted(
    sizes:        jnp.ndarray,      # (n_obj,)
    sort_idx:     jnp.ndarray,      # (n_obj,)
    sizes_sorted: jnp.ndarray,      # (n_obj,)
    posterior:    jnp.ndarray,      # (n_obj,)
    k:            float,
    wf:           float,
    gamma:        float = 1.0,
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
    z = gamma * (sizes - theta_k) / denom
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
    gamma:       float = 1.0,          # size gain / discriminability
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
        size_vec = compute_size_semantics(states, prior_norm, k, wf, gamma=gamma)  
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
    gamma:        float = 1.0,   # size gain / discriminability
) -> jnp.ndarray:  
    """  
    P(u | s_referent) under global RSA with LM-based utterance cost.  

    Utility:  U(u, s) = α · log L(s | u) − cost(u)  
    Speaker:  P(u | s) ∝ exp U(u, s)  

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
        gamma       = gamma,
    )  

    log_L = jnp.log(jnp.clip(M_listener.T, eps))        # (n_obj, n_utt)  
    util  = alpha * log_L - costs[None, :]               # (n_obj, n_utt)  
    M_speaker = jax.nn.softmax(util, axis=-1)            # (n_obj, n_utt)  

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
             0,    # gamma (per-trial sharpness-conditioned gain)
             ),  
)  

# ── Pre JIT ──────────────────────────────────────────────────────  
@jax.jit  
def jitted_global_speaker(states, alpha, color_semval, form_semval, k, wf, beta, gamma):  
    return vectorized_global_speaker(  
        states, alpha, color_semval, form_semval, k, wf, beta, gamma  
    )  
def likelihood_function_global_speaker(states = None, empirical = None, sharpness_idx = None):
    # # ── Semantic parameters ──────────────────────────────────────────────────  
    # phi_color = numpyro.sample("phi_color", dist.HalfNormal(2.0))  
    # color_sem = jax.nn.sigmoid(phi_color)          # ∈ (0.5, 1)  

    # phi_form  = numpyro.sample("phi_form",  dist.HalfNormal(1.0))  
    # form_sem  = jax.nn.sigmoid(phi_form)           # ∈ (0.5, 1)  

    # fixed color and form semantics
    color_sem = 0.8
    form_sem  = 0.7

    # Infer k
    # phi_k     = numpyro.sample("phi_k",     dist.Normal(0.0, 1.0))  
    # k         = jax.nn.sigmoid(phi_k)             # ∈ (0, 1), prior mean 0.5  

    # Fixed k
    k = 0.5

    # Infer wf
    # log_wf = numpyro.sample("log_wf", dist.TruncatedNormal(0.0, 0.5, low=-1.0, high=1.0))
    # wf     = jnp.exp(log_wf)                   # prior mean ≈ 1.0

    # Fixed wf
    wf        = 1.0

    # Infe beta
    log_beta  = numpyro.sample("log_beta",  dist.Normal(0.0, 0.5))  
    beta      = jnp.exp(log_beta)                  # prior mean ≈ 1.0

    if sharpness_idx is None:
        sharpness_idx = jnp.zeros((len(states),), dtype=jnp.float32)
    gamma = jnp.where(sharpness_idx > 0.5, GAMMA_SHARP, GAMMA_BLURRED)

    # Infer alpha
    alpha = numpyro.sample("alpha", dist.HalfNormal(5.0))  # prior mean ≈ 4.0, but with long tail

    #  ── Fixed parameters ─────────────────────────────────
    # beta      = 1.0
    # alpha     = 3.0     

    # Define the likelihood function
    with numpyro.plate("data", len(states)):
        # Get vectorized global speaker output for all states
        # For single output, it is shape (n_utt, n_obj)
        probs = jitted_global_speaker(  
            states,  
            alpha,  
            color_sem,  
            form_sem,  
            k,  
            wf,  
            beta,  
            gamma,
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

# =============================================================================  
# INCREMENTAL SPEAKER  
# =============================================================================  

def incremental_speaker(
    states:       jnp.ndarray,
    alpha:        float = 3.0,
    color_semval: float = 0.95,
    form_semval:  float = 0.80,
    k:            float = 0.50,
    wf:           float = 1.00,
    beta:         float = 1.00,
    gamma:        float = 1.00,
) -> jnp.ndarray:

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]

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
                gamma,
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
            alpha * log_L_ref,
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

    log_unnorm = log_P_beta + log_final_scores                        # (n_utt,)
    return jax.nn.softmax(log_unnorm)                                 # (n_utt,)


def incremental_speaker_frozen(
    states:       jnp.ndarray,
    alpha:        float = 3.0,
    color_semval: float = 0.95,
    form_semval:  float = 0.80,
    k:            float = 0.50,
    wf:           float = 1.00,
    beta:         float = 1.00,
    gamma:        float = 1.00,
) -> jnp.ndarray:

    eps            = 1e-8
    referent_index = 0
    n_obj          = states.shape[0]

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
            gamma,
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

        logits = jnp.where(cand_mask_t, alpha * log_L_ref, -1e9)
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

    log_unnorm = log_P_beta + log_final_scores
    return jax.nn.softmax(log_unnorm)

# ── Vectorise over trials ──────────────────────────────────────────────────────  
vectorized_incremental_speaker = jax.vmap(  
    incremental_speaker,  
    in_axes=(0,    # states      — one trial per row  
             None, # alpha  
             None, # color_semval  
             None, # form_semval  
             None, # k  
             None, # wf  
             None, # beta  
             0,    # gamma (per-trial sharpness-conditioned gain)
             ),  
)  

vectorized_incremental_speaker_frozen = jax.vmap(
    incremental_speaker_frozen,
    in_axes=(0,
             None,
             None,
             None,
             None,
             None,
             None,
             0,
             ),
)

# ── Pre JIT ──────────────────────────────────────────────────────  
@jax.jit  
def jitted_speaker(states, alpha, color_semval, form_semval, k, wf, beta, gamma):  
    return vectorized_incremental_speaker(  
        states, alpha, color_semval, form_semval, k, wf, beta, gamma  
    )  


@jax.jit
def jitted_speaker_frozen(states, alpha, color_semval, form_semval, k, wf, beta, gamma):
    return vectorized_incremental_speaker_frozen(
        states, alpha, color_semval, form_semval, k, wf, beta, gamma
    )

# Warm it up with dummy values matching your actual shapes/dtypes  
_dummy_states = jnp.ones((len(states), 6, 3))  
_dummy_gamma = jnp.ones((len(states),), dtype=jnp.float32)
_ = jitted_speaker(_dummy_states, 3.0, 0.95, 0.80, 0.5, 1.0, 1.0, _dummy_gamma)  
_.block_until_ready()  
_ = jitted_speaker_frozen(_dummy_states, 3.0, 0.95, 0.80, 0.5, 1.0, 1.0, _dummy_gamma)
_.block_until_ready()



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
             0,    # gamma (per-trial sharpness-conditioned gain)
             ),
)

@jax.jit
def jitted_global_speaker_hier(states, alpha_per_trial, color_semval, form_semval, k, wf, beta, gamma):
    return vectorized_global_speaker_hier(
        states, alpha_per_trial, color_semval, form_semval, k, wf, beta, gamma
    )

vectorized_incremental_speaker_hier = jax.vmap(
    incremental_speaker,
    in_axes=(0,    # states
             0,    # alpha  ← per-trial
             None, # color_semval
             None, # form_semval
             None, # k
             None, # wf
             None, # beta
             0,    # gamma
             ),
)

@jax.jit
def jitted_speaker_hier(states, alpha_per_trial, color_semval, form_semval, k, wf, beta, gamma):
    return vectorized_incremental_speaker_hier(
        states, alpha_per_trial, color_semval, form_semval, k, wf, beta, gamma
    )


def likelihood_function_global_speaker_hier(
    states=None, empirical=None, sharpness_idx=None,
    participant_idx=None, n_participants=None,
):
    """Global speaker with per-participant random effects on alpha.

    Population priors:
        alpha    ~ HalfNormal(5.0)
        log_beta ~ Normal(0.0, 0.5)
    Participant-level random intercepts:
        tau     ~ HalfNormal(0.2)   # SD of rationality shifts
        delta_i ~ Normal(0, tau)    # per-participant alpha offset
    Observation model:
        alpha_j = max(alpha + delta[participant_idx[j]], 0)
        obs_j  ~ Categorical(global_speaker(state_j, alpha_j, ...))
    """
    color_sem = 0.8
    form_sem  = 0.7
    k         = 0.5
    wf        = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)

    alpha = numpyro.sample("alpha", dist.HalfNormal(5.0))
    tau   = numpyro.sample("tau",   dist.HalfNormal(0.2))

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    if sharpness_idx is None:
        sharpness_idx = jnp.zeros((len(states),), dtype=jnp.float32)
    gamma = jnp.where(sharpness_idx > 0.5, GAMMA_SHARP, GAMMA_BLURRED)

    alpha_per_trial = jnp.maximum(alpha + delta[participant_idx], 0.0)

    with numpyro.plate("data", len(states)):
        probs = jitted_global_speaker_hier(
            states, alpha_per_trial, color_sem, form_sem, k, wf, beta, gamma
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


def likelihood_function_incremental_speaker_hier(
    states=None, empirical=None, sharpness_idx=None,
    participant_idx=None, n_participants=None,
):
    """Incremental speaker with per-participant random effects on alpha.

    Same structure as likelihood_function_global_speaker_hier but uses
    the incremental RSA speaker.
    """
    color_semval = 0.8
    form_semval  = 0.7
    k            = 0.5
    wf           = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)

    alpha = numpyro.sample("alpha", dist.HalfNormal(5.0))
    tau   = numpyro.sample("tau",   dist.HalfNormal(0.2))

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    if sharpness_idx is None:
        sharpness_idx = jnp.zeros((len(states),), dtype=jnp.float32)
    gamma = jnp.where(sharpness_idx > 0.5, GAMMA_SHARP, GAMMA_BLURRED)

    alpha_per_trial = jnp.maximum(alpha + delta[participant_idx], 0.0)

    with numpyro.plate("data", len(states)):
        probs = jitted_speaker_hier(
            states, alpha_per_trial, color_semval, form_semval, k, wf, beta, gamma
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


def likelihood_function_incremental_speaker(states = None, empirical = None, sharpness_idx = None, incremental_fast_mode: bool = False):
    # ── Semantic parameters ───────────────────────────────────────────────────  
    # # Infer color and form semantics; size semantics are computed from these via the
    # phi_color  = numpyro.sample("phi_color", dist.HalfNormal(2.0))  
    # color_semval = jax.nn.sigmoid(phi_color)           # ∈ (0.5, 1)  

    # phi_form   = numpyro.sample("phi_form",  dist.HalfNormal(1.0))  
    # form_semval  = jax.nn.sigmoid(phi_form)            # ∈ (0.5, 1)  

    # Fixed color and form semantics
    color_semval = 0.8
    form_semval  = 0.7
    # Infer k
    # k    = numpyro.sample("k",     dist.Beta(2.0, 2.0))
    # Fixed k
    k = 0.5

    # Infer wf 
    # log_wf = numpyro.sample("log_wf", dist.TruncatedNormal(0.0, 0.5,  low=-1.0,  high=1.0))  
    # wf        = jnp.exp(log_wf)                   # prior mean ≈ 1.0  
    # Fixed wf
    wf = 1.0

    # Infer alpha
    alpha = numpyro.sample("alpha", dist.HalfNormal(5.0))  # prior mean ≈ 4.0, but with long tail
    
    # Fixed alpha
    #alpha  = 3.0    # fixed 

    # Infer beta
    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)                 # prior mean ≈ 1.0 

    if sharpness_idx is None:
        sharpness_idx = jnp.zeros((len(states),), dtype=jnp.float32)
    gamma = jnp.where(sharpness_idx > 0.5, GAMMA_SHARP, GAMMA_BLURRED)

    # Fixed beta
    #beta   = 1.0    # fixed: posterior was ~ exp(-0.03) ≈ 1.0  

    # Define the likelihood function
    with numpyro.plate("data", len(states)):
        # Get vectorized incremental speaker output for all states
        # For single output, it is shape (n_utt, n_obj)
        speaker_fn = jitted_speaker_frozen if incremental_fast_mode else jitted_speaker
        probs = speaker_fn(
            states,
            alpha,
            color_semval,
            form_semval,
            k,
            wf,
            beta,
            gamma,
        )                                              # (N, n_utt)
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


# ========================
# Gamma-inferred variants
# ========================

def likelihood_function_global_speaker_gamma(states=None, empirical=None, sharpness_idx=None):
    """Global speaker with inferred size-discriminability (gamma) per sharpness condition."""
    color_sem = 0.8
    form_sem  = 0.7
    k         = 0.5
    wf        = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)
    alpha    = numpyro.sample("alpha", dist.HalfNormal(5.0))

    log_gamma_blurred = numpyro.sample("log_gamma_blurred", dist.Normal(jnp.log(GAMMA_BLURRED), 0.5))
    log_gamma_sharp   = numpyro.sample("log_gamma_sharp",   dist.Normal(jnp.log(GAMMA_SHARP),   0.5))

    if sharpness_idx is None:
        sharpness_idx = jnp.zeros((len(states),), dtype=jnp.float32)
    gamma = jnp.where(sharpness_idx > 0.5, jnp.exp(log_gamma_sharp), jnp.exp(log_gamma_blurred))

    with numpyro.plate("data", len(states)):
        probs = jitted_global_speaker(states, alpha, color_sem, form_sem, k, wf, beta, gamma)
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


def likelihood_function_incremental_speaker_gamma(states=None, empirical=None, sharpness_idx=None,
                                                   incremental_fast_mode: bool = False):
    """Incremental speaker with inferred size-discriminability (gamma) per sharpness condition."""
    color_semval = 0.8
    form_semval  = 0.7
    k    = 0.5
    wf   = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)
    alpha    = numpyro.sample("alpha", dist.HalfNormal(5.0))

    log_gamma_blurred = numpyro.sample("log_gamma_blurred", dist.Normal(jnp.log(GAMMA_BLURRED), 0.5))
    log_gamma_sharp   = numpyro.sample("log_gamma_sharp",   dist.Normal(jnp.log(GAMMA_SHARP),   0.5))

    if sharpness_idx is None:
        sharpness_idx = jnp.zeros((len(states),), dtype=jnp.float32)
    gamma = jnp.where(sharpness_idx > 0.5, jnp.exp(log_gamma_sharp), jnp.exp(log_gamma_blurred))

    with numpyro.plate("data", len(states)):
        speaker_fn = jitted_speaker_frozen if incremental_fast_mode else jitted_speaker
        probs = speaker_fn(states, alpha, color_semval, form_semval, k, wf, beta, gamma)
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


def likelihood_function_global_speaker_hier_gamma(
    states=None, empirical=None, sharpness_idx=None,
    participant_idx=None, n_participants=None,
):
    """Global speaker hierarchical + inferred gamma."""
    color_sem = 0.8
    form_sem  = 0.7
    k         = 0.5
    wf        = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)
    alpha    = numpyro.sample("alpha", dist.HalfNormal(5.0))
    tau      = numpyro.sample("tau",   dist.HalfNormal(0.2))

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    log_gamma_blurred = numpyro.sample("log_gamma_blurred", dist.Normal(jnp.log(GAMMA_BLURRED), 0.5))
    log_gamma_sharp   = numpyro.sample("log_gamma_sharp",   dist.Normal(jnp.log(GAMMA_SHARP),   0.5))

    if sharpness_idx is None:
        sharpness_idx = jnp.zeros((len(states),), dtype=jnp.float32)
    gamma = jnp.where(sharpness_idx > 0.5, jnp.exp(log_gamma_sharp), jnp.exp(log_gamma_blurred))

    alpha_per_trial = jnp.maximum(alpha + delta[participant_idx], 0.0)

    with numpyro.plate("data", len(states)):
        probs = jitted_global_speaker_hier(
            states, alpha_per_trial, color_sem, form_sem, k, wf, beta, gamma
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


def likelihood_function_incremental_speaker_hier_gamma(
    states=None, empirical=None, sharpness_idx=None,
    participant_idx=None, n_participants=None,
):
    """Incremental speaker hierarchical + inferred gamma."""
    color_semval = 0.8
    form_semval  = 0.7
    k            = 0.5
    wf           = 1.0

    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
    beta     = jnp.exp(log_beta)
    alpha    = numpyro.sample("alpha", dist.HalfNormal(5.0))
    tau      = numpyro.sample("tau",   dist.HalfNormal(0.2))

    with numpyro.plate("participants", n_participants):
        delta = numpyro.sample("delta", dist.Normal(0.0, tau))

    log_gamma_blurred = numpyro.sample("log_gamma_blurred", dist.Normal(jnp.log(GAMMA_BLURRED), 0.5))
    log_gamma_sharp   = numpyro.sample("log_gamma_sharp",   dist.Normal(jnp.log(GAMMA_SHARP),   0.5))

    if sharpness_idx is None:
        sharpness_idx = jnp.zeros((len(states),), dtype=jnp.float32)
    gamma = jnp.where(sharpness_idx > 0.5, jnp.exp(log_gamma_sharp), jnp.exp(log_gamma_blurred))

    alpha_per_trial = jnp.maximum(alpha + delta[participant_idx], 0.0)

    with numpyro.plate("data", len(states)):
        probs = jitted_speaker_hier(
            states, alpha_per_trial, color_semval, form_semval, k, wf, beta, gamma
        )
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)


# ========================
def run_inference(speaker_type: str = "global",
                    num_warmup: int = 100,
                    num_samples: int = 250,
                    num_chains: int = 4,
                    incremental_fast_mode: bool = False,
                    infer_gamma: bool = False,
                    device: str = "cpu",
                  ):

    # Setup output file name
    gamma_suffix = "_gamma" if infer_gamma else ""
    output_file_name = f"./inference_data/mcmc_results_{speaker_type}_speaker{gamma_suffix}_warmup{num_warmup}_samples{num_samples}_chains{num_chains}.nc"

    # Delete if exists  
    if os.path.exists(output_file_name):  
        os.remove(output_file_name)  
        print(f"Deleted existing file: {output_file_name}")  

     
    # Import dataset
    data = import_dataset()
    states_train = data["states_train"]
    empirical_train_seq_flat = data["empirical_seq_flat"]
    sharpness_idx = data["sharpness_idx"]

    # Some printing for debugging
    print("States train shape:", states_train.shape)
    print("Empirical train flat shape:", empirical_train_seq_flat.shape)
    print("Sharpness index shape:", sharpness_idx.shape)
    print("Output file name:" , output_file_name)
    
    # run_from_map
    MAP_GLOBAL = {"phi_color": 1.25, "phi_form": 1.0}  
    MAP_INC    = {"phi_color": 1.20, "phi_form": 0.7} 
    # Define the MCMC kernel and the number of samples
    rng_key = random.PRNGKey(4711)
    rng_key, rng_key_ = random.split(rng_key)

    if speaker_type == "global":
        model = likelihood_function_global_speaker_gamma if infer_gamma else likelihood_function_global_speaker
        target_accept_prob = 0.9
        max_tree_depth = 3
    elif speaker_type == "incremental":
        model = likelihood_function_incremental_speaker_gamma if infer_gamma else likelihood_function_incremental_speaker
        target_accept_prob = 0.85
        max_tree_depth = 2
    else:
        raise ValueError(f"Unknown speaker_type: {speaker_type}")
    
    kernel = NUTS(model, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=num_warmup,num_samples=num_samples, num_chains=num_chains,
                    chain_method="vectorized",           # ← vectorized chains for speed
                    )
    if speaker_type == "incremental":
        mcmc.run(rng_key_, states_train, empirical_train_seq_flat, sharpness_idx, incremental_fast_mode)
    else:
        mcmc.run(rng_key_, states_train, empirical_train_seq_flat, sharpness_idx)

    # print the summary of the posterior distribution
    mcmc.print_summary()

    # Get the MCMC samples and convert to a numpyro ArviZ InferenceData object
    posterior_samples = mcmc.get_samples() 
    if speaker_type == "incremental":
        posterior_predictive = Predictive(model, posterior_samples)(
            PRNGKey(1), states_train, None, sharpness_idx, incremental_fast_mode
        )
        prior = Predictive(model, num_samples=500)(
            PRNGKey(2), states_train, None, sharpness_idx, incremental_fast_mode
        )
    else:
        posterior_predictive = Predictive(model, posterior_samples)(
            PRNGKey(1), states_train, None, sharpness_idx
        )
        prior = Predictive(model, num_samples=500)(
            PRNGKey(2), states_train, None, sharpness_idx
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

def run_inference_hier(
    speaker_type: str = "global",
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    infer_gamma: bool = False,
    device: str = "cpu",
):
    """Run MCMC for the hierarchical (random participant alpha) speaker model.

    Saves results to
        ./inference_data/mcmc_results_{speaker_type}_speaker_hier[_gamma]_warmup{W}_samples{S}_chains{C}.nc
    """
    gamma_suffix = "_gamma" if infer_gamma else ""
    output_file_name = (
        f"./inference_data/mcmc_results_{speaker_type}_speaker_hier{gamma_suffix}"
        f"_warmup{num_warmup}_samples{num_samples}_chains{num_chains}.nc"
    )
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
        print(f"Deleted existing file: {output_file_name}")

    data = import_dataset_hier()
    states_train          = data["states_train"]
    empirical_seq_flat    = data["empirical_seq_flat"]
    sharpness_idx         = data["sharpness_idx"]
    participant_idx       = data["participant_idx"]
    n_participants        = data["n_participants"]

    print(f"Hierarchical model: {n_participants} participants, {len(states_train)} observations")
    print(f"Output file: {output_file_name}")

    rng_key = random.PRNGKey(4711)
    rng_key, rng_key_ = random.split(rng_key)

    if speaker_type == "global":
        model = likelihood_function_global_speaker_hier_gamma if infer_gamma else likelihood_function_global_speaker_hier
        target_accept_prob = 0.9
        max_tree_depth     = 3
    elif speaker_type == "incremental":
        model = likelihood_function_incremental_speaker_hier_gamma if infer_gamma else likelihood_function_incremental_speaker_hier
        target_accept_prob = 0.85
        max_tree_depth     = 2
    else:
        raise ValueError(f"Unknown speaker_type: {speaker_type}")

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
        states_train, empirical_seq_flat, sharpness_idx,
        participant_idx, n_participants,
    )
    mcmc.print_summary(exclude_deterministic=False)

    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model, posterior_samples)(
        PRNGKey(1), states_train, None, sharpness_idx, participant_idx, n_participants
    )
    prior = Predictive(model, num_samples=500)(
        PRNGKey(2), states_train, None, sharpness_idx, participant_idx, n_participants
    )

    N = states_train.shape[0]
    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords={
            "item":         np.arange(N),
            "participants": np.arange(n_participants),
        },
        dims={"obs": ["item"], "delta": ["participants"]},
    )
    az.to_netcdf(numpyro_data, output_file_name)
    assert os.path.exists(output_file_name), f"Save failed: {output_file_name} not found"
    print(f"Saved: {output_file_name}")


def profile_likelihood_function(speaker_type: str = "global"):
        # ── Run this BEFORE the full inference ───────────────────────────────────────
    # Grid search over (phi_color, phi_k) to visualise the log-posterior surface

    phi_color_grid = np.linspace(0.5, 3.5, 40)
    phi_k_grid     = np.linspace(-3.0, 5.0, 40)
    phi_form_fixed = 0.85   # fix phi_form at plausible value
    states_jnp = import_dataset()["states_train"].astype(jnp.float32)
    empirical_all = import_dataset()["empirical_seq_flat"].astype(jnp.int32)
    log_post = np.zeros((40, 40))

    for i, pc in enumerate(phi_color_grid):
        for j, pk in enumerate(phi_k_grid):

            color_sv = float(jax.nn.sigmoid(pc))
            form_sv  = float(jax.nn.sigmoid(phi_form_fixed))
            k_val    = float(jax.nn.sigmoid(pk))

            if speaker_type == "global":
                speaker = vectorized_global_speaker
            elif speaker_type == "incremental":
                speaker = jitted_speaker
            probs    = np.array(speaker(
                states_jnp, 3.0, color_sv, form_sv, k_val, 1.0, 1.0
            ))

            # Log likelihood
            ll = np.sum(np.log(probs[np.arange(len(empirical_all)),
                                    empirical_all] + 1e-8))

            # Log prior (Normal(2.0, 0.75) for phi_color, Normal(0, 1) for phi_k)
            lp = (
                -0.5 * ((pc - 2.0) / 0.75)**2
                - 0.5 * ((pk - 0.0) / 1.0)**2
            )
            log_post[i, j] = ll + lp

    # ── Plot ──────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    cf = ax.contourf(
        phi_k_grid, phi_color_grid, log_post,
        levels=40, cmap="viridis"
    )
    plt.colorbar(cf, ax=ax, label="Log posterior")
    ax.set_xlabel("phi_k  (→ k = sigmoid(phi_k))", fontsize=11)
    ax.set_ylabel("phi_color  (→ color_semval = sigmoid(phi_color))", fontsize=11)
    ax.set_title(
        "Log-posterior surface: phi_color × phi_k\n"
        "(phi_form fixed at 0.85)\n"
        "Multiple peaks = true multimodality,  single ridge = sampler issue",
        fontsize=10
    )

    # Annotate k values on x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    k_ticks = [-2, -1, 0, 1, 2, 3, 4]
    ax2.set_xticks(k_ticks)
    ax2.set_xticklabels([f"{jax.nn.sigmoid(jnp.array(float(v))):.2f}" for v in k_ticks])
    ax2.set_xlabel("k = sigmoid(phi_k)", fontsize=10)

    plt.tight_layout()
    plt.savefig("./figures/log_posterior_surface_global.png", dpi=150, bbox_inches="tight")
    plt.show()
def profile_likelihood_function_color_form(speaker_type: str = "global"):
        # ── Confirm bimodality BEFORE fixing sampler ──────────────────────────────────
    phi_color_grid = np.linspace(0.5, 3.5, 50)
    phi_form_grid  = np.linspace(-2.0, 3.0, 50)
    log_post_2d    = np.zeros((50, 50))
    states_jnp = import_dataset()["states_train"].astype(jnp.float32)
    empirical_all = import_dataset()["empirical_seq_flat"].astype(jnp.int32)
    if speaker_type == "global":
        speaker = vectorized_global_speaker
    elif speaker_type == "incremental":
        speaker = jitted_speaker

    for i, pc in enumerate(phi_color_grid):
        for j, pf in enumerate(phi_form_grid):
            color_sv = float(jax.nn.sigmoid(pc))
            form_sv  = float(jax.nn.sigmoid(pf))

            probs = np.array(speaker(
                states_jnp, 3.0, color_sv, form_sv, 1.0, 1.0, 1.0
            ))
            ll = np.sum(np.log(
                probs[np.arange(len(empirical_all)), empirical_all] + 1e-8
            ))
            lp = (
                -0.5 * ((pc - 2.0) / 0.75)**2
            + -0.5 * ((pf - 1.0) / 0.75)**2
            )
            log_post_2d[i, j] = ll + lp

    # ── Plot ──────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    cf = ax.contourf(phi_form_grid, phi_color_grid, log_post_2d,
                    levels=40, cmap="viridis")
    plt.colorbar(cf, ax=ax, label="Log posterior")
    ax.set_xlabel("phi_form → form_semval = sigmoid(phi_form)")
    ax.set_ylabel("phi_color → color_semval = sigmoid(phi_color)")
    ax.set_title(
        "Profile posterior: phi_color × phi_form\n"
        "Single peak = sampler issue  |  Two peaks = true bimodality"
    )
    # Mark all local maxima
    from scipy.ndimage import maximum_filter
    local_max = (log_post_2d == maximum_filter(log_post_2d, size=5))
    rows, cols = np.where(local_max & (log_post_2d > log_post_2d.max() - 50))
    for r, c in zip(rows, cols):
        ax.scatter(phi_form_grid[c], phi_color_grid[r],
                color="red", s=100, marker="*", zorder=5)
    plt.tight_layout()
    plt.savefig("./figures/profile_post_color_form_gb.png", dpi=150)
    plt.show()

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
    parser.add_argument("--speaker_type", type=str, choices=["global", "incremental"], default="incremental",
                        help="Choose the speaker model type.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of posterior samples.")
    parser.add_argument("--num_warmup", type=int, default=500, help="Number of warm-up iterations.")
    parser.add_argument("--num_chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--incremental_fast_mode", action="store_true",
                        help="Use faster approximate incremental speaker with frozen size-semantics updates.")
    parser.add_argument("--test", action="store_true", help="Run test function and exit.")
    parser.add_argument(
        "--hierarchical", action="store_true",
        help="Run hierarchical model with random per-participant alpha intercepts.",
    )
    parser.add_argument(
        "--infer_gamma", action="store_true",
        help="Infer gamma (size discriminability) per sharpness condition instead of using fixed values.",
    )

    args = parser.parse_args()

    # Place to test functions
    #profile_likelihood_function(args.speaker_type)

    if args.test:
        test()
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
            incremental_fast_mode=args.incremental_fast_mode,
        )