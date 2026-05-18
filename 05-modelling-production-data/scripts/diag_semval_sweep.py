"""Fixed-constant sensitivity pre-screen for the recommended reporting model
(`contextual_pcalpha_canon_parsimony_no_alphaF`, 10 named).

Replays the model's posterior through the REAL speaker factory at a grid of
the four fixed constants — `color_semval` (csv), `form_semval` (fsv), `k`
(size-threshold anchor), `wf` (size-sigmoid sharpness) — and recomputes the
ladder PPC R²(all)/R²(emp≥.02). Cheap (~1-2 min, CPU, no MCMC) PRE-SCREEN to
locate the landscape and pick grid points for the authoritative joint refits.

⚠️ Static-sweep caveat (project lesson, memo §7.6): the other 10 named params
are held at values fit *with* the current constants, so this OVER-promises —
treat it as a landscape sketch, not the verdict. The free/re-fixed joint
refits (run_inference, CSP) are authoritative.

Method (exact within the caveat): no speaker reimplementation. We
  1. reload the recommended NC's posterior sample sites,
  2. rebuild the dc-subset model inputs (import_dataset_hier + the same
     subset filter run_inference uses) and VALIDATE them against the NC's
     observed_data (obs codes / N / n_participants must match exactly —
     hard assert, so any data-prep drift fails loudly), then
  3. for each grid point, instantiate the parsimony factory at those
     constants and draw posterior-predictive obs via numpyro Predictive,
     scoring with the shared ppc_lever1_review machinery.

Produces (gitignored): results/contextual_dc/semval_sweep.csv
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
import numpy as np
import pandas as pd
from jax import random
from numpyro.infer import Predictive
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

import helper  # noqa: E402

helper.CONDITIONS_OF_INTEREST = ("erdc", "zrdc", "brdc")
import modelSpecification as ms  # noqa: E402
import ppc_lever1_review as plv  # noqa: E402
# Canonical (prop × sharp) grouping that matches the paper's R² convention
# (identical to r2_ladder.py / parsimony_frontier.py).
plv.GROUP_COLS = ["relevant_property", "sharpness"]
from ppc_lever1_review import (  # noqa: E402
    load_dataset_dc, compute_model_proportions, compute_human_proportions,
)

NC = HERE / "inference_data" / (
    "mcmc_results_contextual_pcalpha_canon_parsimony_no_alphaF_"
    "speaker_hier_dc_warmup4000_samples2000_chains4.nc"
)
SUBSET = ("erdc", "zrdc", "brdc")
# Sampled sites in the recommended model (alpha_F dropped, beta_lm fixed).
SITES = ["alpha_D", "alpha_C", "lambda_suff", "lambda_form_mod",
         "lambda_noncanon", "gamma_base", "gamma_oneword", "gamma_sharp",
         "epsilon", "tau", "delta_raw"]
MAX_DRAWS = 240  # subsample posterior for speed; PPC R² is stable by ~200

# Current fixed values (the manuscript model's constants).
CUR = dict(csv=0.971, fsv=0.50, k=0.5, wf=ms.WF_FIXED_ITER11_MEDIAN)


def build_dc_inputs():
    """Rebuild the dc-subset model kwargs exactly as run_inference does, and
    hard-validate against the recommended NC's observed data."""
    data = helper.import_dataset_hier(min_proportion=0.0)
    df = data["df"]
    keep = np.where(df["conditions"].isin(SUBSET).to_numpy())[0]
    keep_j = jax.numpy.asarray(keep)
    for key in ("states_train", "empirical_seq_flat", "empirical_flat",
                "empirical_seq", "seq_mask", "sharpness_idx",
                "is_colour_sufficient", "sufficient_dim",
                "has_one_word_solution"):
        if key in data and data[key] is not None:
            data[key] = data[key][keep_j]
    data["df"] = df.iloc[keep].reset_index(drop=True)

    old_pid = np.asarray(data["participant_idx"])[keep]
    uniq_p = sorted(set(old_pid.tolist()))
    remap = {p: i for i, p in enumerate(uniq_p)}
    new_pid = np.array([remap[p] for p in old_pid], dtype=np.int32)
    data["participant_idx"] = jax.numpy.asarray(new_pid, dtype=jax.numpy.int32)
    data["n_participants"] = len(uniq_p)

    old_cid = np.asarray(data["condition_idx"])[keep]
    uniq_c = sorted(set(old_cid.tolist()))
    cremap = {c: i for i, c in enumerate(uniq_c)}
    new_cid = np.array([cremap[c] for c in old_cid], dtype=np.int32)
    data["condition_idx"] = jax.numpy.asarray(new_cid, dtype=jax.numpy.int32)
    data["n_conditions"] = len(uniq_c)

    kwargs = dict(
        states=data["states_train"],
        empirical=None,  # generate (posterior predictive)
        participant_idx=data["participant_idx"],
        n_participants=data["n_participants"],
        sufficient_dim=data["sufficient_dim"],
        has_one_word_solution=data["has_one_word_solution"],
        is_sharp=data["sharpness_idx"],
        condition_idx=data["condition_idx"],
        n_conditions=data["n_conditions"],
    )
    return kwargs, np.asarray(data["empirical_seq_flat"]).astype(int)


def load_posterior(idata):
    post = idata.posterior
    samples = {}
    for s in SITES:
        if s not in post.data_vars:
            raise SystemExit(f"site '{s}' missing from NC posterior")
        arr = post[s].stack(sample=("chain", "draw")).transpose("sample", ...).values
        samples[s] = np.asarray(arr)
    n = samples["alpha_D"].shape[0]
    idx = np.linspace(0, n - 1, min(MAX_DRAWS, n)).astype(int)
    return {k: jax.numpy.asarray(v[idx]) for k, v in samples.items()}, len(idx)


def r2_at(constants, kwargs, samples, df_dc, all_codes, emp):
    model = ms._make_contextual_pcalpha_canon_parsimony_model(
        color_semval=constants["csv"], form_semval=constants["fsv"],
        k=constants["k"], wf=constants["wf"], drop=("alpha_F",),
    )
    pp = Predictive(model, samples)(random.PRNGKey(0), **kwargs)["obs"]
    pp_flat = np.asarray(pp)  # (sample, item)
    mod = compute_model_proportions(pp_flat, df_dc, max_draws=MAX_DRAWS)
    merged = emp[["relevant_property", "sharpness", "utt_code",
                  "human_mean", "n"]].merge(
        mod[["relevant_property", "sharpness", "utt_code", "model_mean"]],
        on=["relevant_property", "sharpness", "utt_code"], how="inner",
    )
    r_all, _ = pearsonr(merged["human_mean"], merged["model_mean"])
    sub = merged[merged["human_mean"] >= 0.02]
    r_sub, _ = pearsonr(sub["human_mean"], sub["model_mean"])
    return r_all ** 2, r_sub ** 2


def main():
    if not NC.exists():
        raise SystemExit(f"NC not found: {NC}")
    print(f"Loading recommended NC: {NC.name}")
    idata = az.from_netcdf(str(NC))

    kwargs, obs_codes = build_dc_inputs()
    # Hard validation: rebuilt inputs must match the NC's observed data.
    nc_obs = idata.observed_data["obs"].values.astype(int)
    assert kwargs["states"].shape[0] == nc_obs.shape[0], (
        f"N mismatch: rebuilt {kwargs['states'].shape[0]} vs NC {nc_obs.shape[0]}")
    match = float(np.mean(obs_codes == nc_obs))
    assert match > 0.999, f"obs-code mismatch ({match:.3%}) — data-prep drift"
    assert int(kwargs["n_participants"]) == int(
        idata.posterior.sizes.get("participants", kwargs["n_participants"])), \
        "n_participants mismatch vs NC"
    print(f"  data-prep validated: N={nc_obs.shape[0]}, "
          f"obs match={match:.2%}, participants={kwargs['n_participants']}")

    samples, ndraw = load_posterior(idata)
    print(f"  posterior draws used: {ndraw}")

    df_dc, _ = load_dataset_dc()
    all_codes = sorted(df_dc["annotation_seq_flat"].unique().tolist())
    emp = compute_human_proportions(df_dc, all_codes, n_boot=1, seed=431)

    # Sanity: reproduce the stored-NC R² at the current constants.
    base_all, base_sub = r2_at(CUR, kwargs, samples, df_dc, all_codes, emp)
    print(f"\nReplay @ current constants csv=0.971 fsv=0.50 k=0.5 "
          f"wf={CUR['wf']:.4f}: R²(all)={base_all:.4f} "
          f"R²(emp≥.02)={base_sub:.4f}  (stored NC ≈ 0.9206 / 0.9209)")

    grids = {
        "csv": [0.80, 0.85, 0.90, 0.95, 0.971, 0.99],
        "fsv": [0.50, 0.55, 0.60, 0.65, 0.70, 0.80],
        "k":   [0.30, 0.40, 0.50, 0.60, 0.70],
        "wf":  [0.40, 0.55, CUR["wf"], 0.85, 1.00],
    }
    rows = []
    for name, vals in grids.items():
        print(f"\n=== 1-D sweep: {name} (others at current) ===")
        print(f"{name:>8} {'R²(all)':<9} {'R²(emp≥.02)':<13} {'Δall':<8}")
        for v in vals:
            c = dict(CUR)
            c[name] = v
            ra, rs = r2_at(c, kwargs, samples, df_dc, all_codes, emp)
            tag = "  <- current" if abs(v - CUR[name]) < 1e-6 else ""
            print(f"{v:>8.4f} {ra:<9.4f} {rs:<13.4f} {ra-base_all:+.4f}{tag}")
            rows.append(dict(swept=name, value=v, R2_all=ra,
                             R2_emp_ge_002=rs, dR2_all=ra - base_all,
                             is_current=abs(v - CUR[name]) < 1e-6))

    out = HERE / "results" / "contextual_dc" / "semval_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
