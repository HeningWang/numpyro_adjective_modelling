"""Iter 18 diagnostic: can an erdc-gated LM-downweight lift the LM-limited
erdc/DF residual without breaking iter 17's wins?

Runs the iter 17 (formmod) speaker at posterior medians, then sweeps a
multiplicative LM scale ``s`` applied to the LM term ON erdc trials only
(zrdc/brdc untouched — option C is suffdim-gated). For each s, reports the
erdc D-initial cells vs human, and confirms erdc/sharp + zrdc/brdc behaviour.

Decision output: the s that best matches human erdc/blurred (esp. DF), and
whether lifting DF costs the iter-17 gains (D/DC/DCF) or inflates DFC.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import arviz as az
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jax import lax, vmap

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import helper  # noqa: E402

helper.CONDITIONS_OF_INTEREST = ("erdc", "zrdc", "brdc")
import modelSpecification as ms  # noqa: E402

NC = HERE / "inference_data" / (
    "mcmc_results_contextual_pcalpha_formmod_speaker_hier_dc_"
    "warmup4000_samples2000_chains4.nc"
)
UTT = ["D", "DC", "DCF", "DF", "DFC", "C", "CD", "CDF", "CF", "CFD",
       "F", "FD", "FDC", "FC", "FCD"]


def medians(idata):
    p = idata.posterior
    g = lambda n: float(np.median(p[n].values))
    return dict(
        alpha_D=g("alpha_D"), alpha_C=g("alpha_C"), alpha_F=g("alpha_F"),
        lambda_suff=g("lambda_suff"), lambda_form_mod=g("lambda_form_mod"),
        gamma_base=g("gamma_base"), gamma_oneword=g("gamma_oneword"),
        gamma_sharp=g("gamma_sharp"), epsilon=g("epsilon"),
        beta_lm=float(np.exp(g("log_beta_lm"))),
    )


def speaker_with_lm_scale(state, sd, how, isharp, P, lm_scale,
                          color_semval=0.971, form_semval=0.50, k=0.5):
    """Iter 17 forward pass with an erdc-gated multiplicative LM scale.

    lm_scale applies to the LM term only when sufficient_dim == 0 (erdc).
    """
    wf = float(ms.WF_FIXED_ITER11_MEDIAN)
    eps = 1e-8
    n_obj = state.shape[0]
    alpha_vec = jnp.array([P["alpha_D"], P["alpha_C"], P["alpha_F"]])
    sizes = state[:, 0]
    ssi = jnp.argsort(sizes)
    ss = sizes[ssi]

    erdc = 1.0 if int(sd) == 0 else 0.0
    eff_lm_scale = (1.0 - erdc) * 1.0 + erdc * lm_scale
    log_lm_raw = eff_lm_scale * P["beta_lm"] * ms.LOG_LM_RAW_15

    colors, forms = state[:, 1], state[:, 2]
    log_color_sem = jnp.log(jnp.where(colors == 1, color_semval, 1 - color_semval) + eps)
    log_form_sem = jnp.log(jnp.where(forms == 1, form_semval, 1 - form_semval) + eps)
    uniform = jnp.ones(n_obj) / n_obj
    init_scores = jnp.zeros(ms.n_utt)
    init_posts = jnp.broadcast_to(uniform, (ms.n_utt, n_obj))

    def asize(arr, post):
        ps = post[ssi]; ps = ps / (jnp.sum(ps) + eps)
        cdf = jnp.cumsum(ps)
        il = jnp.minimum(jnp.searchsorted(cdf, 0.2, side="left"), ss.shape[0] - 1)
        ih = jnp.minimum(jnp.searchsorted(cdf, 0.8, side="left"), ss.shape[0] - 1)
        xmin, xmax = ss[il], ss[ih]
        th = xmax - k * (xmax - xmin)
        den = wf * jnp.sqrt(arr ** 2 + th ** 2 + ms.SIZE_ANCHOR_R ** 2 + eps)
        return 0.5 * (1.0 + lax.erf((arr - th) / den / jnp.sqrt(2.0)))

    def step(carry, t):
        ls, pup = carry
        cm, at = ms.CANDIDATE_MASK[t], ms.ACTIVE_POS[t]
        sls = vmap(lambda po: jnp.log(jnp.clip(asize(sizes, po), eps)))(pup)
        st = jnp.stack([log_color_sem, log_form_sem], 0)
        tbl = jnp.concatenate([sls[:, None, :],
                               jnp.broadcast_to(st[None], (ms.n_utt, 2, n_obj))], 1)
        lps = jnp.einsum("uav,uvo->uao", ms.TOKEN_PRESENT[t], tbl)
        lpu = jnp.log(jnp.clip(pup, eps))
        lu = lpu[:, None, :] + lps
        lz = jsp.logsumexp(lu, -1)
        lref = (lu - lz[:, :, None])[:, :, 0]
        fsg = (t == 0).astype(jnp.float32)
        sbv = P["lambda_suff"] * fsg * jnp.array(
            [sd == 0, sd == 1, sd == 2], jnp.float32)
        logits = jnp.where(cm, alpha_vec[None] * lref + sbv[None], -1e9)
        lp = jax.nn.softmax(logits, -1)
        ch = jnp.sum(lp * ms.ACTUAL_TOK_ONEHOT[t], -1)
        ch = jnp.where(at, ch, 1.0)
        lch = jnp.where(at, jnp.log(jnp.clip(ch, eps)), 0.0)
        sels = jnp.einsum("uv,uvo->uo", ms.ACTUAL_TOK_ONEHOT[t], tbl)
        lup = lpu + jnp.where(at[:, None], sels, 0.0)
        lzp = jsp.logsumexp(lup, -1, keepdims=True)
        return (ls + lch, jnp.exp(lup - lzp)), None

    (lfs, _), _ = lax.scan(step, (init_scores, init_posts), jnp.arange(ms.T))
    blur = 1.0 - isharp
    geff = P["gamma_base"] + P["gamma_oneword"] * how + P["gamma_sharp"] * blur
    length_bonus = geff * jnp.maximum(ms.N_WORDS - 1.0, 0.0)
    fpb = P["lambda_form_mod"] * erdc * ms.F_PRESENT_15
    lunorm = log_lm_raw + lfs + length_bonus + fpb
    mp = jax.nn.softmax(lunorm)
    return np.asarray((1 - P["epsilon"]) * mp + P["epsilon"] / ms.n_utt)


def main():
    P = medians(az.from_netcdf(str(NC)))
    print("iter17 medians:", {k: round(v, 3) for k, v in P.items()})

    d = helper.import_dataset()
    df = d["df"].reset_index(drop=True)
    states = d["states_train"]
    cp = df["conditions"].astype(str).str[-2:]
    rp = df["relevant_property"].astype(str)
    sd = np.full(len(df), -1, np.int32)
    fmk = (rp == "first").to_numpy()
    smk = (rp == "second").to_numpy()
    sd[fmk] = cp[fmk].str[0].map(helper.CONDITION_DIM_TO_INDEX).to_numpy(np.int32)
    sd[smk] = cp[smk].str[1].map(helper.CONDITION_DIM_TO_INDEX).to_numpy(np.int32)
    how = (sd >= 0).astype(np.float32)
    isharp = (df["sharpness"].astype(str) == "sharp").to_numpy(np.float32)

    HUMAN = {
        ("erdc", "blurred"): dict(D=.056, DC=.043, DCF=.300, DF=.365, DFC=.032),
        ("erdc", "sharp"):   dict(D=.269, DC=.046, DCF=.205, DF=.231, DFC=.022),
        ("zrdc", "blurred"): dict(C=.628, CF=.263, CDF=.005),
        ("brdc", "blurred"): dict(DCF=.309, CF=.270, DC=.069, DF=.085, F=.117),
    }
    scales = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    for (cond, sh), hum in HUMAN.items():
        m = ((df["conditions"] == cond) & (df["sharpness"] == sh)).to_numpy()
        i = int(np.flatnonzero(m)[0])
        print(f"\n=== {cond}/{sh}  (LM scale gated to erdc only) ===")
        print(f"{'utt':<5}{'human':>8}" + "".join(f"{f's={s}':>9}" for s in scales))
        R = {s: speaker_with_lm_scale(states[i], int(sd[i]), float(how[i]),
                                      float(isharp[i]), P, s) for s in scales}
        keys = ["D", "DC", "DCF", "DF", "DFC"] if cond != "zrdc" else ["C", "CF", "CDF"]
        for kk in keys:
            u = UTT.index(kk)
            h = hum.get(kk, float("nan"))
            print(f"{kk:<5}{h:>8.3f}" + "".join(f"{R[s][u]:>9.4f}" for s in scales))


if __name__ == "__main__":
    main()
