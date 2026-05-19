# Design: 2×2 (speaker × semantics) on the best contextual model

**Date:** 2026-05-18
**Status:** approved (brainstorming) → spec review pending
**Branch:** `feat/2x2-best-model`

## Problem

The paper's main model comparison is a 2×2 (speaker: incremental/global ×
semantics: recursive/static), historically run on the *simple* "reported-
style" models (single `alpha`/`log_beta`/`tau`/`delta`, R²≈0.4). The paper
should instead manipulate speaker×semantics **on the architecture it actually
advocates** — the best contextual model (`contextual_pcalpha_canon_parsimony`
with `drop=alpha_F`, `color_semval=0.59`; 10 named params, R²≈0.94, merged
PR #7). Reporting a 2×2 on a model that fits R²≈0.4 while the advocated model
fits R²≈0.94 is misleading.

## Goal

Construct the 2×2 factorial on the best contextual model: four cells sharing
an **identical 10-parameter inventory** and all fixed components (GPT-2 LM
prior with fixed β_lm, canonical-order penalty, formmod, anchored size
semantics, csv=0.59, per-trial α hierarchy), differing **only** in two
structural switches. Compare via LOO/ELPD + PPC R², and quantify the
speaker×semantics interaction with the §7.12 parity diagnostic.

## Factor operationalization (approved)

The contextual incremental speaker is a `lax.scan` over token positions
carrying `(log_scores, per_utt_posts)`: each step (a) computes the literal-
listener semantics under the *current* posterior, (b) accrues the chosen
token's log-prob into `log_scores`, (c) Bayesian-updates `per_utt_posts`.

**SEMANTICS factor = listener recursion:**
- *Incremental speaker:* recursive = posterior updated & carried across
  tokens (current best model); static = posterior frozen at the uniform
  prior every step (no cross-token belief accumulation).
- *Global speaker:* recursive = full RSA pragmatic-listener recursion over
  the whole utterance; static = literal listener only (no pragmatic
  recursion). (Approved: keeps "static = no belief recursion" coherent
  across both speakers, since the global speaker has no token sequence.)

**SPEAKER factor = utility accrual:**
- incremental = Σ per-token log P(chosen tokenₜ) (sequential).
- global = single joint term α·log P(referent | full utterance) at the
  utterance level (non-sequential).

| Cell | Listener posterior | Utility accrual |
|------|--------------------|-----------------|
| inc × rec (= current best model) | Bayesian-updated, carried | Σ per-token log P(tokenₜ) |
| inc × static | frozen at uniform prior | Σ per-token log P(tokenₜ) vs frozen prior |
| glob × rec | RSA pragmatic over full utterance | joint α·log P(ref \| full utt) |
| glob × static | literal-only over full utterance | joint α·log P(ref \| full utt) |

All four retain the identical inventory — `alpha_D`, `alpha_C` (`alpha_F`
dropped per §7.8), `lambda_suff`, `lambda_form_mod`, `lambda_noncanon`,
`gamma_base`, `gamma_oneword`, `gamma_sharp`, `epsilon`, `tau` (**10 named +
339 latents**, the per-trial α offsets `delta_raw` 113×3 governed by `tau`) —
plus fixed `beta_lm` (=exp(1.907718)),
`color_semval=0.59`, `form_semval=0.50`, `k=0.5`, `wf=0.6856`, `alpha_F`
dropped. `lambda_suff` (first-word boost for the sufficient dim) stays well-
defined for the global speaker because every utterance still has a word
order (the LM prior & canon penalty operate on it); only the *utility
accrual* is joint. Strict parameter-matching ⇒ the §7.12 clean-factorial
reviewer defense holds by construction.

## Implementation (Approach C, approved)

- **Incremental cells:** extend the contextual canon speaker with a
  `recursive: bool` closure flag. recursive=True → current behaviour
  (`scan` carries `new_per_utt_posts`); recursive=False → the `step` returns
  `init_posts` as the carried posterior every iteration (freeze). One
  compile-time branch; no traced control flow.
- **Global cells:** a new speaker function
  `global_speaker_contextual_anchored_..._canon` computing one joint
  literal/pragmatic listener over the full utterance, reusing the existing
  helpers (`_anchored_size_sem`, `LOG_LM_RAW_15`, color/form sem, canon /
  formmod / length / ε terms), with its own `recursive: bool` flag
  (True = RSA pragmatic recursion; False = literal only).
- **Exposure:** a `cell=` argument on
  `_make_contextual_pcalpha_canon_parsimony_model` (analogous to the
  existing `drop=` / `free=`), value in
  `{"inc_rec","inc_static","glob_rec","glob_static"}`, default `"inc_rec"`
  (== current best model, behaviour unchanged — existing NCs unaffected).
  `drop=("alpha_F",)`, `color_semval=0.59` fixed for all four 2×2 variants.
- Register the 4 variants in `run_inference.py`
  (`contextual_pcalpha_canon_parsimony_2x2_{inc_rec,inc_static,glob_rec,glob_static}`)
  in the import / HIER_MODELS / family-set / argparse spots.

## Run & evaluation plan

- dc-subset (`erdc,zrdc,brdc`), `--num_warmup 4000 --num_samples 2000
  --num_chains 4` on CSP A6000 (matches the best-model runs this session).
- 4 fresh runs under one server-side chain (single GPU, sequential). inc_rec
  must reproduce R²≈0.94 / the merged csv=0.59 posterior — a built-in
  correctness check.
- Metrics: **LOO/ELPD** primary (`az.compare`, pairwise dSE) + **PPC R²**
  (`scripts/r2_ladder.py` convention, continuity) + reuse
  `scripts/diag_2x2_parity.py` for the parameter-parity / p_loo / 2×2
  decomposition (speaker ME, semantics ME, speaker×semantics interaction).
- One branch (`feat/2x2-best-model`) + one PR; diagnostics committed to
  `scripts/`; memo/results/figures git-ignored. Memo gets a new section
  (best-model 2×2 results + interaction) and the 2×2 memory is updated.

## Risks / open points

- **Global contextual speaker is genuinely new code** — the joint
  literal/pragmatic listener over the full utterance must reuse the exact
  anchored-size-sem + LM + canon + formmod + length + ε machinery so the
  only difference vs the incremental speaker is the utility-accrual
  structure. Risk: subtle divergence in the semantics path → mitigated by
  the inc_rec-reproduces-0.94 sanity check and a small unit check that the
  global speaker's per-utterance probs sum to 1 and respond to α.
- **Sampling cost/health unknown for the new global speaker** — if the
  global cells mix poorly (cf. the simple global models collapsing to
  p_loo≈3), that is a *finding* (report p_loo), not a blocker.
- Out of scope: full 9-condition data (dc-subset chosen, consistent with the
  contextual ladder); a csv-only vs full-§7.10 sensitivity (separate question
  already noted in §7.11).

## Success criteria

1. Four variants register; `inc_rec` reproduces the merged csv=0.59 model
   (R²≈0.94) — proves the refactor is behaviour-preserving.
2. All 4 run on CSP with reported r̂/divergences; LOO + R² + the 2×2
   decomposition computed and committed via scripts.
3. Memo + 2×2 memory updated with the best-model 2×2 table, the
   speaker×semantics interaction, and how it compares to the simple-model
   2×2 finding (does incremental≫global / recursive>static hold on the
   advocated architecture?).
4. One PR opened.
