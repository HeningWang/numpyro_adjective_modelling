# Extension v5 — Condition-Gated Colour Salience + Saturating Length Bias

**Date**: 2026-04-13
**Status**: Design (not yet implemented)
**Memo destination**: new section in `10-writing/extended_production_model_memo.md` after results
**Predecessor**: extended-v1 (per-dim α, linear γ, ε, hierarchical τ/δ; ELPD = −6387, R² = 0.596)

---

## 1. Motivation

Extended-v1 leaves seven systematic residuals (memo §5.3). Decomposed by length, they split into two structurally separable problems:

- **1-word (C) residuals are context-dependent**: C under-predicted in colour-sufficient (Δ ≈ −.31 / −.23) and over-predicted in both-necessary (Δ ≈ +.18 / +.21). No global change to α_C can fix both directions.
- **Length residuals show saturation, not level shift**: model is short by ~.28 on 2-word (SF in size-sufficient/low) but long by ~.17 on 3-word (SFC in size-sufficient). A linear per-extra-word γ cannot reproduce this — it applies the same bonus to the second redundant adjective as the first.

Two targeted mechanisms address these independently.

## 2. Model spec

Built on extended-v1. Speaker = incremental recursive. Per-dim α, hierarchical τ/δ, ε, form_sem = 0.50, color_sem = 0.971 all retained.

### 2.1 Condition-gated colour salience (mechanism B)

Step-level mention logit for colour receives an additive boost active **only** in colour-sufficient trials:

```text
logit_C(step) += λ_C · I(condition ∈ colour-sufficient)
```

- `λ_C ~ Normal(0, 1)` — sign-free; theory predicts positive but data decides.
- Condition gate makes λ_C identifiable against α_C: α_C scales utility in every trial; λ_C lifts C-mention only in 1/3 of trials, giving a distinct likelihood signature. (This avoids the v3 failure, where a globally-applied μ_C traded off 1-to-1 with α_C.)

### 2.2 Saturating length bias (mechanism P2)

Final utterance utility receives:

```text
bonus(utt) = γ_1 · I(k_extra ≥ 1) + γ_2 · I(k_extra ≥ 2)
```

- `k_extra` = redundant adjectives beyond the minimal discriminating set (existing definition).
- `γ_1 ~ Normal(0, 1)`, `γ_2 ~ Normal(0, 1)` — both sign-free.
- Replaces current `γ · k_extra`. Nests current model (γ_1 = γ_2 = γ) and indicator-only model (γ_2 = 0).

### 2.3 Implementation surface

- **Helper**: `helper.py::import_dataset` returns an additional per-trial `is_colour_sufficient: jnp.ndarray (N,) float32` derived from `df["conditions"]` (the `*cf` family).
- **Jitted speaker**: new `jitted_speaker_v5_hier(states, is_colour_sufficient, alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial, lambda_C, color_semval, form_semval, k, wf, beta, gamma_1, gamma_2, epsilon)`. Forks from `jitted_speaker_hier`; adds λ_C to the C-mention logit at each step (gated by per-trial flag), and replaces the linear γ term with the saturating two-step bonus in the utility computation.
- **Likelihood factories**: three new factories (`_make_v5a_model`, `_make_v5b_model`, `_make_v5_model`) following `_make_extended_v1_model` pattern.
- **No changes** to: state encoding (sizes/colors/forms), L0/L1 listener, ε mixing, hierarchical structure, MCMC config (warmup=2000, samples=1000, chains=4).

## 3. Comparison plan

LOO-CV across five models, all incremental recursive, all hierarchical:

| Tag | α | λ_C | Length | ε | Notes |
|---|---|---|---|---|---|
| baseline (reported) | single | — | — | — | `likelihood_function_reported_hier` (modelSpecification.py:1381) |
| ext-v1 | per-dim | — | linear γ | ✓ | `likelihood_function_incremental_speaker_hier` (modelSpecification.py:1425) |
| v5a | per-dim | ✓ | linear γ | ✓ | new |
| v5b | per-dim | — | γ_1, γ_2 | ✓ | new |
| **v5** | per-dim | ✓ | γ_1, γ_2 | ✓ | new |

**Headline**: baseline → v5 (full extension story for the memo).
**Ablations**: v5a, v5b isolate marginal contribution of each mechanism on top of the per-dim-α + ε scaffold.

After v5 fits, re-run the 2×2 (speaker × semantics) under v5 mechanisms to check that the semantic-regime effect still survives (cf. memo §5.1).

## 4. Identifiability pre-checks

Before full inference, run prior-predictive simulations:

1. **λ_C vs α_C**: fix α_C at v1 posterior mean; sweep λ_C ∈ {0, 1, 2, 3}; check predicted C-proportion moves in colour-sufficient while staying flat in size-sufficient and both-necessary.
2. **γ_1 vs γ_2**: inspect 2D posterior; report Pearson correlation. Ridge structure (|r| > 0.9) means data isn't separating them — fall back to v5a (drop γ_2).
3. **γ_1 vs ε**: both can lift rare long utterances; report posterior correlation and check ε posterior mean shifts.

## 5. Success criteria

- v5 ELPD − baseline ELPD ≥ +900 (ext-v1 already at +793)
- v5 ELPD − ext-v1 ELPD ≥ +20
- v5a − ext-v1 ≥ +10 **and** v5b − ext-v1 ≥ +10 (both mechanisms pull weight independently)
- ≥ 5 of the 7 listed residuals (memo §5.3) resolved (|Δ| < .10)
- ε posterior mean drops meaningfully below 0.23
- No divergences; r̂ ≤ 1.05 for all parameters
- 2×2 semantic-regime effect (Δ/dSE) still significant under v5 mechanisms

## 6. Risks and fallbacks

| Risk | Diagnostic | Fallback |
|---|---|---|
| λ_C absorbed by α_C despite condition gate | identifiability check #1 fails; r̂(λ_C) > 1.05 | tighten λ_C prior to Normal(0, 0.5); if still degenerate, drop λ_C |
| γ_1/γ_2 ridge | identifiability check #2 fails | fall back to v5a (linear γ + λ_C) |
| Semantic-regime effect collapses under v5 | 2×2 re-run shows Δ/dSE < 2 | report in memo; flag for paper narrative; do not adopt v5 as the new baseline for paper claims |
| ε stays high (≥ 0.20) | inspect posterior | confirms strategy heterogeneity hypothesis; motivates mixture-model direction (memo §6.2) |
| SFC overestimate persists in size-sufficient | residual table | indicates saturating length alone is insufficient; consider form-specific penalty as a follow-up |

## 7. Out of scope

- Mixture-of-strategies model (memo §6.2) — separate spec.
- Re-fitting input semantic representations (memo §6.3) — separate spec.
- Discriminability-gated γ_1 — only if v5 leaves the size-sufficient/low SF residual unresolved.
- Paper revision — this is memo work; paper integration is a separate decision after v5 results.

## 8. Artefacts

- Code: new functions in `05-modelling-production-data/modelSpecification.py`; new field in `helper.py::import_dataset`.
- Inference data: `05-modelling-production-data/inference_data/mcmc_results_v5{,_a,_b}_warmup2000_samples1000_chains4.nc` (run on remote server per `server-workflow` skill).
- Diagnostics: posterior summaries, identifiability checks, residual tables, PPC barplots, correlation plots — saved under `05-modelling-production-data/figures/v5/`.
- Memo update: new section in `10-writing/extended_production_model_memo.md` summarising mechanism, results, and remaining gaps.
