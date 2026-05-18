"""Decompose iter16 P(utt) into LM, RSA, length-bonus contributions per utterance,
for representative dc-subset trials.

Goal: localize WHY model under-predicts DF and DCF (and over-predicts D, DFC)
in erdc/blurred. Prints the contribution of each forward-pass term so we can
see which dominates.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import arviz as az
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import helper  # noqa: E402

helper.CONDITIONS_OF_INTEREST = ("erdc", "zrdc", "brdc")

import modelSpecification as ms  # noqa: E402

NC = HERE / "inference_data" / (
    "mcmc_results_contextual_pcalpha_gammasharp_speaker_hier_dc_warmup4000_samples2000_chains4_vast_iter16.nc"
)

UTT_LABELS = ["D", "DC", "DCF", "DF", "DFC", "C", "CD", "CDF", "CF",
              "CFD", "F", "FD", "FDC", "FC", "FCD"]


def posterior_medians(idata):
    post = idata.posterior
    return {
        n: float(np.median(post[n].values))
        for n in [
            "alpha_D", "alpha_C", "alpha_F",
            "lambda_suff", "gamma_base", "gamma_oneword", "gamma_sharp",
            "epsilon", "log_beta_lm",
        ]
    }


def decompose_single_trial(state, sufficient_dim, has_one_word_solution,
                           is_sharp, params, color_semval=0.971,
                           form_semval=0.50, k=0.5, wf=None):
    """Run the speaker once, returning a dict mapping utterance->components.

    Components: rsa (sum log_chosen along steps), lm (beta*log_lm),
    length (gamma_eff * (N-1)), unnorm (sum of the three), prob (after softmax+epsilon mix).
    """
    if wf is None:
        wf = float(ms.WF_FIXED_ITER11_MEDIAN)

    alpha_D = jnp.float32(params["alpha_D"])
    alpha_C = jnp.float32(params["alpha_C"])
    alpha_F = jnp.float32(params["alpha_F"])
    lambda_suff = jnp.float32(params["lambda_suff"])
    gamma_base = jnp.float32(params["gamma_base"])
    gamma_oneword = jnp.float32(params["gamma_oneword"])
    gamma_sharp = jnp.float32(params["gamma_sharp"])
    epsilon = jnp.float32(params["epsilon"])
    beta_lm = jnp.float32(np.exp(params["log_beta_lm"]))

    # Replicate the forward pass but expose components.
    from jax import lax
    import jax.scipy.special as jsp

    eps = 1e-8
    n_obj = state.shape[0]
    alpha_vec = jnp.array([alpha_D, alpha_C, alpha_F])

    sizes = state[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]

    log_lm_raw = beta_lm * ms.LOG_LM_RAW_15

    colors = state[:, 1]
    forms = state[:, 2]
    log_color_sem = jnp.log(
        jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps
    )
    log_form_sem = jnp.log(
        jnp.where(forms == 1, form_semval, 1.0 - form_semval) + eps
    )

    uniform = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(ms.n_utt)
    init_posts = jnp.broadcast_to(uniform, (ms.n_utt, n_obj))

    SIZE_ANCHOR_R = ms.SIZE_ANCHOR_R

    def _anchored_size_sem(sizes_arr, post):
        post_sorted = post[size_sort_idx]
        post_sorted = post_sorted / (jnp.sum(post_sorted) + eps)
        cdf = jnp.cumsum(post_sorted)
        idx_low = jnp.minimum(jnp.searchsorted(cdf, 0.2, side="left"),
                              sizes_sorted.shape[0] - 1)
        idx_high = jnp.minimum(jnp.searchsorted(cdf, 0.8, side="left"),
                               sizes_sorted.shape[0] - 1)
        x_min_mid = sizes_sorted[idx_low]
        x_max_mid = sizes_sorted[idx_high]
        theta_k = x_max_mid - k * (x_max_mid - x_min_mid)
        denom = wf * jnp.sqrt(sizes_arr ** 2 + theta_k ** 2 + SIZE_ANCHOR_R ** 2 + eps)
        z = (sizes_arr - theta_k) / denom
        return 0.5 * (1.0 + lax.erf(z / jnp.sqrt(2.0)))

    referent_index = 0

    def step(carry, t):
        log_scores, per_utt_posts = carry
        cand_mask_t = ms.CANDIDATE_MASK[t]
        active_t = ms.ACTIVE_POS[t]

        def size_log_sem_for_utt(post):
            sv = _anchored_size_sem(sizes, post)
            return jnp.log(jnp.clip(sv, eps))

        from jax import vmap
        size_log_sems = vmap(size_log_sem_for_utt)(per_utt_posts)

        log_sem_static = jnp.stack([log_color_sem, log_form_sem], axis=0)
        log_sem_table = jnp.concatenate([
            size_log_sems[:, None, :],
            jnp.broadcast_to(log_sem_static[None, :, :], (ms.n_utt, 2, n_obj)),
        ], axis=1)

        token_pres_t = ms.TOKEN_PRESENT[t]
        log_prod_sem = jnp.einsum("uav, uvo -> uao", token_pres_t, log_sem_table)

        log_per_utt_posts = jnp.log(jnp.clip(per_utt_posts, eps))
        log_updated = log_per_utt_posts[:, None, :] + log_prod_sem
        log_Z = jsp.logsumexp(log_updated, axis=-1)
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
        local_probs = jnp.exp(logits - jsp.logsumexp(logits, axis=-1, keepdims=True))

        chosen = jnp.sum(local_probs * ms.ACTUAL_TOK_ONEHOT[t], axis=-1)
        chosen = jnp.where(active_t, chosen, 1.0)
        log_chosen = jnp.where(active_t, jnp.log(jnp.clip(chosen, eps)), 0.0)

        selected_log_sem = jnp.einsum(
            "uv, uvo -> uo", ms.ACTUAL_TOK_ONEHOT[t], log_sem_table
        )
        log_updated_post = log_per_utt_posts + jnp.where(
            active_t[:, None], selected_log_sem, 0.0,
        )
        log_Z_post = jsp.logsumexp(log_updated_post, axis=-1, keepdims=True)
        new_per_utt_posts = jnp.exp(log_updated_post - log_Z_post)
        return (log_scores + log_chosen, new_per_utt_posts), None

    (log_final_scores, _), _ = lax.scan(
        step, (init_scores, init_posts), jnp.arange(ms.T)
    )

    blur_gate = 1.0 - is_sharp
    gamma_eff = (
        gamma_base
        + gamma_oneword * has_one_word_solution
        + gamma_sharp * blur_gate
    )
    length_bonus = gamma_eff * jnp.maximum(ms.N_WORDS - 1.0, 0.0)

    log_unnorm = log_lm_raw + log_final_scores + length_bonus
    model_probs = jnp.exp(log_unnorm - jsp.logsumexp(log_unnorm))
    final_probs = (1.0 - epsilon) * model_probs + epsilon / ms.n_utt

    return {
        "lm": np.asarray(log_lm_raw),
        "rsa": np.asarray(log_final_scores),
        "length": np.asarray(length_bonus),
        "unnorm": np.asarray(log_unnorm),
        "probs": np.asarray(final_probs),
        "gamma_eff": float(gamma_eff),
    }


def pick_one_trial(df, conditions, sharpness):
    matches = df[(df["conditions"] == conditions) & (df["sharpness"] == sharpness)]
    return matches.iloc[0]


def main() -> None:
    idata = az.from_netcdf(str(NC))
    params = posterior_medians(idata)
    print("Iter16 posterior medians:")
    for k_, v_ in params.items():
        print(f"  {k_:>15}: {v_:+.4f}")
    print(f"  beta_lm (exp):   {np.exp(params['log_beta_lm']):.3f}")
    print()

    data = helper.import_dataset()
    df = data["df"].reset_index(drop=True)
    states = data["states_train"]
    cond_pairs = df["conditions"].astype(str).str[-2:]
    relevant_property = df["relevant_property"].astype(str)
    sufficient_dim_np = np.full(len(df), -1, dtype=np.int32)
    first_mask = (relevant_property == "first").to_numpy()
    second_mask = (relevant_property == "second").to_numpy()
    sufficient_dim_np[first_mask] = cond_pairs[first_mask].str[0].map(
        helper.CONDITION_DIM_TO_INDEX
    ).to_numpy(dtype=np.int32)
    sufficient_dim_np[second_mask] = cond_pairs[second_mask].str[1].map(
        helper.CONDITION_DIM_TO_INDEX
    ).to_numpy(dtype=np.int32)
    has_one_word_solution_np = (sufficient_dim_np >= 0).astype(np.float32)
    is_sharp_np = (df["sharpness"].astype(str) == "sharp").to_numpy(dtype=np.float32)

    trials_to_check = [
        ("erdc", "blurred"),
        ("erdc", "sharp"),
        ("zrdc", "blurred"),
    ]

    for cond, sharpness in trials_to_check:
        mask = (df["conditions"] == cond) & (df["sharpness"] == sharpness)
        idx = int(np.flatnonzero(mask.to_numpy())[0])
        state = states[idx]
        sd = int(sufficient_dim_np[idx])
        how = float(has_one_word_solution_np[idx])
        isharp = float(is_sharp_np[idx])

        d = decompose_single_trial(state, sd, how, isharp, params)
        print(f"=== {cond}/{sharpness}  (trial {idx})  sufficient_dim={sd}  is_sharp={int(isharp)}  has_one_word={int(how)} ===")
        print(f"    gamma_eff = {d['gamma_eff']:+.3f}")
        print(f"    {'utt':<5}{'lm':>10}{'rsa':>10}{'length':>10}{'unnorm':>10}{'prob':>10}")
        order = np.argsort(-d["probs"])
        for u in order:
            print(f"    {UTT_LABELS[u]:<5}{d['lm'][u]:>+10.3f}{d['rsa'][u]:>+10.3f}{d['length'][u]:>+10.3f}{d['unnorm'][u]:>+10.3f}{d['probs'][u]:>10.4f}")
        print()


if __name__ == "__main__":
    main()
