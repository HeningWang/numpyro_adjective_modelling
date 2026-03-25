#!/usr/bin/env python
"""
02-simulation-fast.py — Optimised single-process RSA simulation.

Optimisations over 01-simulation-random-states.py:
  1. One JAX process: JAX start-up + JIT compilation happens once, not 1440×.
  2. Batched JAX calls: for each (speaker, nobj) pair the entire
     (sd_spread × color_semvalue × k × wf × sample) grid is executed in a
     single vmapped call instead of one Python subprocess per parameter combo.

Total outer Python iterations: 2 semantics × 2 speakers × 8 nobj = 32.
Each iteration fires one JIT-compiled vmap over (n_sd=3, P=125, S=1000) contexts.
Total output rows: 2 × 2 × 8 × 5 × 5 × 5 × 3 × 1000 = 12,000,000.
"""

import itertools
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.stats

import core_rsa  # local import — also initialises JAX once


# ── helpers copied from 01-simulation-random-states.py ───────────────────────
def modify_referent(context):
    """
    Place the biggest object at index 0 and set its color=1, form=1.
    Input/output: (n_obj, 3).
    """
    max_index = jnp.argmax(context[:, 0])
    modified = context.at[0, 0].set(context[max_index, 0])
    modified = modified.at[max_index, 0].set(context[0, 0])
    modified = modified.at[0, 1].set(1)
    modified = modified.at[0, 2].set(1)
    return modified


# ── fixed constants ──────────────────────────────────────────────────────────
ALPHA        = 1.0
BIAS         = 0.0
WORLD_LENGTH = 2
SAMPLE_SIZE  = 1000
SIZE_DIST    = "normal"
OBJ_UPPER    = 30
OBJ_LOWER    = 1
OBJ_P        = 0.5

# ── parameter sweep ──────────────────────────────────────────────────────────
NOBJ_LIST           = list(range(2, 31, 4))                     # [2, 6, …, 30]  8 values
SPEAKER_LIST        = ["incremental_speaker", "global_speaker"]  # 2 values
SEMANTICS_LIST      = ["recursive", "static"]                    # 2 values
COLOR_SEMVALUE_LIST = [round(0.90 + 0.02 * i, 10) for i in range(5)]  # [0.90 … 0.98]  5 values
K_LIST              = [round(0.10 + 0.20 * i, 10) for i in range(5)]  # [0.1 … 0.9]    5 values
WF_LIST             = [0.1, 0.2, 0.3, 0.5, 0.8]                # perceptual blur  5 values
SD_SPREAD_LIST      = [2.0, 7.75, 15.0]                        # 3 values

# Map (semantics, speaker) → pragmatic_listener speaker string
SPEAKER_KEY_MAP = {
    ("recursive", "incremental_speaker"): "incremental_speaker",
    ("recursive", "global_speaker"):      "global_speaker",
    ("static",    "incremental_speaker"): "incremental_speaker_static",
    ("static",    "global_speaker"):      "global_speaker_static",
}

# Total rows = 2 * 2 * 8 * 5 * 5 * 5 * 3 * 1000 = 12,000,000
OUTPUT_FILE = "../04-simulation-w-randomstates/simulation_full_run.csv"


# ── context generation (NumPy / scipy — outside JAX trace) ───────────────────
def generate_modified_contexts(nobj: int, sd_spread: float, n: int) -> jnp.ndarray:
    """
    Sample n random contexts of shape (n, nobj, 3) and apply modify_referent.
    Columns: [size, color, form].
    """
    mean = (OBJ_UPPER + OBJ_LOWER) / 2.0
    a    = (OBJ_LOWER - mean) / sd_spread
    b    = (OBJ_UPPER - mean) / sd_spread
    sizes  = scipy.stats.truncnorm.rvs(
        a=a, b=b, loc=mean, scale=sd_spread, size=(n, nobj)
    )
    colors = np.random.binomial(1, OBJ_P, size=(n, nobj)).astype(np.float32)
    forms  = np.random.binomial(1, OBJ_P, size=(n, nobj)).astype(np.float32)
    raw = jnp.array(
        np.stack([sizes.astype(np.float32), colors, forms], axis=-1)
    )  # (n, nobj, 3)
    return jax.vmap(modify_referent)(raw)  # (n, nobj, 3)


# ── build per-speaker batched inference function ─────────────────────────────
def make_batch_fn(speaker: str):
    """
    Returns a JIT-compiled function:
        fn(all_states, csvs, ks, wfs) → (n_sd, P, S, 2)

    all_states : (n_sd, S, nobj, 3)
    csvs       : (P,)   color_semvalue for each param combo
    ks         : (P,)   k for each param combo
    wfs        : (P,)   wf for each param combo
    output     : (n_sd, P, S, 2)  last dim = [probs_big_blue, probs_blue_big]

    Speaker type is captured in the closure; no Python branching inside JAX.
    """
    # Innermost: single context × single scalar params → (2,)
    def one_context(state, csv, k, wf):
        pl = core_rsa.pragmatic_listener(
            state, ALPHA, BIAS, csv, None, wf, k,
            speaker, WORLD_LENGTH
        )
        # pl: (2, nobj); communicative success = column 0 (referent)
        return jnp.array([pl[0, 0], pl[1, 0]])  # (2,)

    # Level 1: vmap over S samples — state varies, params fixed
    vmap_S = jax.vmap(one_context, in_axes=(0, None, None, None))

    # Level 2: vmap over P param combos — states fixed, params vary
    vmap_P = jax.vmap(vmap_S, in_axes=(None, 0, 0, 0))

    # Level 3: vmap over n_sd sd_spread contexts — params fixed
    vmap_SD = jax.vmap(vmap_P, in_axes=(0, None, None, None))

    return jax.jit(vmap_SD)


# ── output helpers ────────────────────────────────────────────────────────────
def build_df(results: jnp.ndarray, speaker: str, semantics: str,
             nobj: int) -> pd.DataFrame:
    """
    results : (n_sd, P, S, 2)
    Returns a DataFrame with n_sd * P * S rows.
    """
    n_sd = len(SD_SPREAD_LIST)
    S    = SAMPLE_SIZE

    # Reconstruct the same meshgrid used to build param arrays
    csv_grid, k_grid, wf_grid = np.meshgrid(COLOR_SEMVALUE_LIST, K_LIST, WF_LIST)
    csvs_flat = csv_grid.ravel()   # (P,)
    ks_flat   = k_grid.ravel()     # (P,)
    wfs_flat  = wf_grid.ravel()    # (P,)
    P = len(csvs_flat)

    results_np = np.asarray(results)  # pull from device once

    cols: dict = {k: [] for k in [
        "probs_big_blue", "probs_blue_big",
        "alpha", "bias", "nobj", "color_semvalue",
        "wf", "k", "speaker", "semantics", "size_distribution",
        "sd_spread", "sample_size", "world_length",
    ]}

    for i_sd, sd_spread in enumerate(SD_SPREAD_LIST):
        for i_p in range(P):
            probs = results_np[i_sd, i_p]  # (S, 2)
            cols["probs_big_blue"].extend(probs[:, 0].tolist())
            cols["probs_blue_big"].extend(probs[:, 1].tolist())
            cols["alpha"]            .extend([ALPHA]                * S)
            cols["bias"]             .extend([BIAS]                 * S)
            cols["nobj"]             .extend([nobj]                 * S)
            cols["color_semvalue"]   .extend([float(csvs_flat[i_p])] * S)
            cols["wf"]               .extend([float(wfs_flat[i_p])] * S)
            cols["k"]                .extend([float(ks_flat[i_p])]  * S)
            cols["speaker"]          .extend([speaker]              * S)
            cols["semantics"]        .extend([semantics]            * S)
            cols["size_distribution"].extend([SIZE_DIST]            * S)
            cols["sd_spread"]        .extend([sd_spread]            * S)
            cols["sample_size"]      .extend([S]                    * S)
            cols["world_length"]     .extend([WORLD_LENGTH]         * S)

    return pd.DataFrame(cols)


def write_df(df: pd.DataFrame, first: bool) -> None:
    if first:
        df.to_csv(OUTPUT_FILE, index=False)
    else:
        df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # Param-combo arrays — built once, reused every iteration
    csv_grid, k_grid, wf_grid = np.meshgrid(COLOR_SEMVALUE_LIST, K_LIST, WF_LIST)
    csvs_jnp = jnp.array(csv_grid.ravel(), dtype=jnp.float32)
    ks_jnp   = jnp.array(k_grid.ravel(),   dtype=jnp.float32)
    wfs_jnp  = jnp.array(wf_grid.ravel(),  dtype=jnp.float32)

    n_total   = len(SEMANTICS_LIST) * len(SPEAKER_LIST) * len(NOBJ_LIST)
    counter   = 0
    first     = True
    t0        = time.time()

    # Remove old output file so header is written fresh
    if os.path.isfile(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    for semantics in SEMANTICS_LIST:
        for speaker in SPEAKER_LIST:
            speaker_key = SPEAKER_KEY_MAP[(semantics, speaker)]
            print(f"\n=== Semantics: {semantics} | Speaker: {speaker} "
                  f"(key={speaker_key}) — building JIT ===")
            batch_fn = make_batch_fn(speaker_key)

            for nobj in NOBJ_LIST:
                counter += 1
                t_iter = time.time()

                # 1. Generate contexts for every sd_spread (NumPy, outside JAX)
                all_states = jnp.stack([
                    generate_modified_contexts(nobj, sd_spread, SAMPLE_SIZE)
                    for sd_spread in SD_SPREAD_LIST
                ])  # (n_sd, S, nobj, 3)

                # 2. One batched JAX call over (n_sd=3, P=125, S=1000)
                results = batch_fn(all_states, csvs_jnp, ks_jnp, wfs_jnp)
                results.block_until_ready()  # (n_sd, P, S, 2)

                # 3. Build DataFrame and write
                df_block = build_df(results, speaker, semantics, nobj)
                write_df(df_block, first)
                first = False

                elapsed       = time.time() - t_iter
                total_elapsed = time.time() - t0
                print(
                    f"  [{counter:2d}/{n_total}]  sem={semantics:<10s}  "
                    f"speaker={speaker:<22s}  nobj={nobj:2d}"
                    f"  rows={len(df_block):>7,}  iter={elapsed:.1f}s  "
                    f"total={total_elapsed:.0f}s"
                )

    print(f"\nDone. Output: {OUTPUT_FILE}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
