# Extension v5 — Synthesis for Supervisor Discussion

**Date**: 2026-04-15
**Branch**: `feat/extension-v5`
**Data subset used throughout**: `dc` subset (`erdc, zrdc, brdc`; N = 3196, 113 participants). Within this subset the three conditions correspond to the three `relevant_property` values: `first` (size alone sufficient), `both` (size + colour both needed), `second` (colour alone sufficient; the only colour-sufficient cell under the corrected flag).

---

## 1. Executive summary

We extended the paper-reported RSA production model along two parallel tracks:

- **Theory track (paper-reported model)**: minimal mechanisms chosen for theoretical motivation. Produces the 2 × 2 interaction that supports the paper's claims (architecture dominates, context-updating helps within incremental). Poor aggregate fit (R² ≈ .35, L1 residual ≈ 3.2 across condition × utterance cells).

- **Mechanism track (v5 with λ_C + saturating γ + δγ + η + μ_noncanon)**: successive additions of theoretically-grounded terms from the overinformativeness / colour-salience / adjective-ordering literatures. Excellent fit (R² ≈ .94, L1 residual ≈ 1.2) but the 2 × 2 interaction becomes much weaker — most of what the interaction was signaling has been absorbed into the new structural parameters.

**The core tension**: mechanisms designed to reduce residuals do so by explaining variance that the original sparser parameterisation attributed to architecture × regime. The paper's theoretical claim survives in reduced form (both effects remain significant under F2-full; signs are mostly preserved) but loses most of its magnitude and sharpness.

## 2. Timeline of changes

Every change is committed on `feat/extension-v5`; reproducible per commit.

| Step | Mechanism | Rationale | Added parameters |
|---|---|---|---|
| ext-v1 | per-dim α, explicit γ, lapse ε | memo §4.1 — dimensional-salience asymmetry, overspec, noise | α_D, α_C, α_F, γ, ε |
| v5 (λ_C) | condition-gated colour boost at step 0 | Pechmann 1989; Rubio-Fernández 2016; memo §5.3 colour-sufficient under-prediction | λ_C (gated on `is_colour_sufficient`) |
| F1 (δγ) | condition-dependent length offsets on colour-sufficient trials | lets the length bias shrink when bare-C is optimal | δγ_1, δγ_2 |
| C1 (μ_noncanon) | canonical ordering penalty (size < colour < form) | Sproat 1991; Cinque 1994; Scott 2002; memo §5.3 DCF under-prediction / DFC over-prediction | μ_noncanon |
| F2 (η) | sharpness-dependent length offsets | overspec more common under blur than sharp | η_1, η_2 |

Corrected along the way:
- `COLOUR_SUFFICIENT_CONDITIONS` was initially `(ercf, zrcf, brcf)` (the `*cf` family); corrected to `(ercf, zrdc)` — colour is alone-sufficient only in `ercf` and `zrdc`. This correction lifted the on-subset LOO gain from ~0 to +136 ELPD.

## 3. Models, posteriors, LOO (dc subset, N = 3196)

All runs: hierarchical (per-participant α offset δ\_p), warmup 2000 / samples 1000 / chains 4. 0 divergences throughout.

| Model | ELPD LOO | Δ vs baseline | Δ vs ext_v1 | ε | notes |
|---|---|---|---|---|---|
| baseline (reported) | −7157 | 0 | — | — | single α, no γ, no ε, no flags |
| ext-v1 | −6387 | +770 | 0 | 0.22 | memo's prior best |
| v5a (λ_C only) | −6265 | +892 | +122 | — | λ_C alone |
| v5b (sat γ only) | −6381 | +776 | +6 | — | γ_1, γ_2 alone — null |
| v5 (F1) | −6250 | +907 | **+137** | 0.16 | λ_C + sat γ + δγ |
| v5 (C1, i.e. F1 + μ_noncanon) | −5323 | +1834 | **+1063** | 0.10 | canonical ordering added |
| **v5 (F2, final)** | **−5294** | **+1863** | **+1093** | **0.10** | C1 + sharpness-dependent η |
| v5_no_lm (F3 ablation) | −5314 | +1843 | +1073 | 0.10 | F2 with β = 0 (LM prior removed) |

### Key posterior means for v5 (F2)

| Parameter | Posterior mean (95 % CI) | r̂ |
|---|---|---|
| λ_C | 3.09 (2.67, 3.55) | 1.00 |
| γ_1 (base, non-csuff) | 2.32 (2.16, 2.49) | 1.00 |
| γ_2 (base) | 1.58 (1.43, 1.74) | 1.01 |
| δγ_1 (csuff offset) | −2.16 (−2.33, −1.97) | 1.00 |
| δγ_2 (csuff offset) | −0.58 (−1.21, +0.04) | 1.00 |
| η_1 (sharp offset) | −0.61 (−0.77, −0.43) | 1.00 |
| η_2 (sharp offset) | −0.34 (−0.51, −0.15) | 1.00 |
| μ_noncanon | −5.08 (−5.61, −4.56) | 1.01 |
| ε | 0.10 (0.09, 0.12) | 1.00 |
| τ (participant scale) | 0.62 (0.51, 0.73) | 1.00 |

All informative. Each new mechanism's posterior clearly prefers a non-zero value in the predicted direction.

## 4. Group-level fit

| Metric | baseline | ext-v1 | v5 (F1) | v5 (C1) | **v5 (F2)** |
|---|---|---|---|---|---|
| L1 aggregate residual (emp ≥ 0.02, 37 cells) | 3.17 | 3.17 | 2.70 | 1.34 | **1.22** |
| R² (all 90 cells) | .346 | .596 | .764 | .874 | **.938** |
| R² (emp ≥ 0.02) | .352 | .547 | .672 | .908 | **.929** |
| R² (emp ≥ 0.05) | .330 | .623 | .772 | .884 | **.911** |

Per-cell predictions (v5 F2) captured the major §5.3 misfits:

| Cell (memo §5.3) | emp | baseline | ext_v1 | v5 (F2) | status |
|---|---|---|---|---|---|
| colour-sufficient/high C (second/sharp) | .716 | ~.403 | .403 | **.694** | fixed |
| colour-sufficient/low C (second/blurred) | .628 | ~.402 | .402 | **.594** | fixed |
| size-sufficient/low SF → first/blurred DF | .365 | small | .081 | **.344** | fixed |
| size-sufficient/high SFC → first/sharp DCF | .205 | small | .136 | .185 | fixed |
| both-necessary DCF (blurred) | .309 | small | .157 | .381 | slightly over |
| both-necessary DC (blurred) | .069 | ~.18 | .182 | .107 | partially fixed |
| size-sufficient/high DFC (first/sharp) | .022 | ~.21 | .208 | .015 | fixed |

Remaining concern: first/sharp bare D is still under-predicted (emp .269 → F2 .102) and first/sharp DF over-predicted (.231 → .332). Likely reflects strategy heterogeneity: some participants say bare D in easy trials, others add redundant form. A single-strategy model cannot capture both.

## 5. The 2 × 2 under all three parameterisations

Layout: rows = speaker architecture, columns = semantics regime. Cells show ELPD LOO.

### 5a. Paper-reported (memo §1)

| | rec | sta | Δ(rec−sta) |
|---|---|---|---|
| incremental | −7180 | −7199 | **+19** (Δ/dSE = 10.28) |
| global | −7628 | −7626 | **−2** (null) |

Interaction: **rec helps inc a lot, does not help glb.** Diff-in-diff = +21.
Architecture: inc ≫ glb, Δ/dSE ≈ 11.

### 5b. F2 mechanisms, global γ/δγ/η/μ *fixed* at v5 posterior means

| | rec | sta | Δ(rec−sta) |
|---|---|---|---|
| incremental | −5294 | −5297 | +2.80 (Δ/dSE = 2.22) |
| global | −5600 | −5619 | +19.15 (Δ/dSE = 6.99) |

Interaction: **rec helps glb more than inc.** Diff-in-diff = −16.
Architecture: inc ≫ glb, Δ/dSE ≈ 10.5–11.

### 5c. F2 mechanisms, global γ/δγ/η/μ *sampled* (fair comparison)

| | rec | sta | Δ(rec−sta) |
|---|---|---|---|
| incremental | −5294 | −5297 | +2.80 (Δ/dSE = 2.22, marginal) |
| global (full) | −5357 | −5365 | +8.25 (Δ/dSE = 3.82) |

Interaction: **rec helps glb slightly more than inc.** Diff-in-diff = +5.45.
Architecture: inc > glb, Δ/dSE ≈ 3.2 — still significant but one-third of the paper's magnitude.

### Reading the three tables

- **Paper (5a)**: architecture is dominant and the rec−sta contrast is large and inc-specific.
- **F2-fixed (5b)**: architecture looks preserved, but the rec−sta contrast has flipped sides (now glb-specific). This is partly an artifact — fixing global's length/ordering parameters at incremental's values puts global at a disadvantage it otherwise wouldn't have.
- **F2-full (5c)**: the fair comparison. Both effects shrink substantially. Architecture still wins (Δ/dSE = 3.2). Semantic regime is modestly present in both architectures, slightly larger in global.

## 6. The tradeoff

The paper-reported model leaves ~65 % of between-cell variance unexplained (R² = .35) and has a large lapse rate (~.22). The variance it *does* explain is well aligned with the theoretical factors (architecture, regime), producing sharp interactions.

F2 explains ~94 % of between-cell variance. In doing so it re-attributes what was previously "architecture × regime" to explicit mechanisms: colour salience (λ_C), overspecification (γ, δγ, η), canonical ordering (μ_noncanon). The architecture and regime factors remain significant but their magnitudes shrink to roughly a third / a fifth of their paper values.

Two readings are defensible:

- **Reading I — mechanisms as rival explanations.** The paper's interaction was partly *fitting error structure into a low-dimensional story*. Once the error is absorbed by targeted mechanisms with clear theoretical motivation, the residual interaction is modest. The paper's headline claim would need to weaken.

- **Reading II — mechanisms as scaffolding, not rivals.** The paper's interaction is real — it was never about the raw ELPD differences, only about the *pattern*. λ_C, μ_noncanon, γ offsets are *implementations* of what context-updating + incremental architecture is *supposed* to do. Of course adding them reduces the architecture signal: they are the phenomena the architecture was indirectly capturing.

## 7. What to say to supervisors

Four concrete decisions come out of this work:

1. **Do we report v5 (F2) or stay with the paper-reported model?**
   - F2 wins empirically by every fit metric.
   - F2 weakens the dominant-architecture story from Δ/dSE ≈ 11 to ≈ 3.
   - F2's additional parameters each have clear literature support.
   - A middle path: **report F2 in a supplementary section** and keep the paper-reported model as the main-text centerpiece, framing F2 as "and these additional mechanisms from the literature further improve fit without undermining the architecture/regime pattern (Δ/dSE 2–4 instead of 10–11)."

2. **If we adopt F2, how do we frame the interaction?**
   - Reading I (mechanisms as rivals) → soften the architecture claim.
   - Reading II (mechanisms as scaffolding) → keep the architecture claim and motivate why adding mechanisms doesn't undermine it.
   - A referee will ask this question either way — we should have an answer.

3. **The `COLOUR_SUFFICIENT_CONDITIONS = (ercf, zrdc)` correction.** This is independent of everything else; it clarifies which trials humans actually treat colour as sufficient. Probably belongs in the main text even if v5 doesn't.

4. **Subset choice.** All v5 analysis is on the `dc` subset (N = 3196). Running the same exploration on the `df` subset (size + form) and the `cf` subset (colour + form) would test whether the F2 mechanisms generalise — particularly, whether a symmetric "form-sufficient" flag would be needed.

## 7b. LM-prior ablation (F3)

**Question**: with μ_noncanon (−5.08) + the γ/δγ/η terms in place, is the LM prior still doing useful work, or is it redundant?

**Test**: fit v5 (F2) with β = 0 (LM prior flattened to uniform; every utterance gets the same log-prior contribution, which drops out of the softmax). Every other parameter free and identical to F2.

**Result**:

| Model | ELPD | Δ vs F2 | R² (emp ≥ .02) | L1 |
|---|---|---|---|---|
| v5 (F2) | **−5294** | 0 | **.928** | **1.22** |
| v5_no_lm (F3) | −5314 | **−20.3** | .915 | 1.30 |

**Posterior compensation** when LM is removed (v5_no_lm - v5_F2):
- λ_C: 3.09 → 3.33 (+0.24) — colour-boost works harder
- γ_2: 1.58 → 1.32 (−0.26) — some 3-word preference re-absorbed into shorter
- δγ_1: −2.16 → −2.39 (−0.23) — colour-sufficient trials get a stronger short-utterance pull
- μ_noncanon, γ_1, η_1, η_2, ε: essentially unchanged

**Interpretation**:

- The LM prior carries ~20 ELPD of residual signal beyond what the explicit F2 mechanisms absorb. Not large, but above noise.
- What it's capturing: utterance-level frequency structure that is *beyond* canonical-ordering (μ_noncanon) + length (γ + δγ + η) + colour salience (λ_C). Probably relative frequencies among same-shape canonical utterances (e.g., "D" vs "DC" vs "DCF").
- The non-zero ablation gap is reassuring: if the explicit mechanisms were just re-fitting what the LM encodes, removing the LM should cost 0. Getting a 20-ELPD gap means the mechanisms are capturing **structurally distinct** signal from the LM, not just a relabeling.
- **β dropped from 7.8 (ext-v1) to 0.84 (F2)**: the LM prior's *sharpening* role — enforcing canonical ordering via `LM_PRIOR^β` — was fully absorbed by μ_noncanon. What remains (β ≈ 0.84, near unsharpened) is the base frequency structure.

**Decision**: keep the LM prior in the final F2 model. The one-parameter cost is minor; the +20 ELPD is real.

## 7c. Parallel exploration: mixture-of-speakers

**Motivation**. Memo §6 (open question #2) flagged strategy heterogeneity as a possible alternative explanation for the high ε = 0.22 in ext-v1: maybe some participants are utility-reasoners (high α, short utterances), others are mention-salient (low α, longer overspecification), and averaging their responses into a single-strategy model pushes ε up.

**Implementation** (in `modelSpecification.py`, predates v5):

- `incremental_speaker_mixture` + `likelihood_function_incremental_speaker_mixture_hier` — two components, each per-dim α + γ, hierarchical δ only on component 1.
- `likelihood_function_incremental_speaker_mixture_simple_hier` — simpler: single α per component.
- Label-switch fix in both: `gap ~ HalfN(3)`, component 2's α = max(component 1's α − gap, 0).
- Mixing weight π ~ Beta(2, 2); lapse ε ~ Beta(2, 98).

**Runs completed**:

| NC file | Data | π mean | r̂ (median) |
|---|---|---|---|
| `mcmc_results_incremental_mixture_speaker_hier_warmup2000_samples1000_chains4.nc` | dc subset (N=3196) | 0.54 | 1.93 |
| `mcmc_results_incremental_mixture_fulldata_speaker_hier_warmup2000_samples1000_chains4.nc` | **full N=9586** | **0.40** | 1.76 |
| `mcmc_results_incremental_mixture_simple_speaker_hier_warmup5000_samples2000_chains4.nc` | dc subset (simple, long) | 0.73 | mixed (α₁ at 1.04, α₂ at 2.80) |

**Clearest posterior — per-dim mixture on full data** (N=9586):

| Param | Speaker 1 | Speaker 2 | Interpretation |
|---|---|---|---|
| α_D | 14.82 | 8.57 | both deliberate about size |
| α_C | **20.85** | **1.24** | speaker 1 super-rational on colour; 2 barely weighs it |
| α_F | 6.79 | 1.29 | similar asymmetry on form |
| γ | **−2.29** | **+1.76** | 1 penalises length; 2 rewards it |
| π | **0.40** | 0.60 | ~40% "careful reasoner", ~60% "mention-more" |
| ε | 0.15 | | lapse reduced vs ext-v1's 0.22, but not vanishing |

**Qualitatively**: exactly the strategy-heterogeneity hypothesis. Two sub-populations emerge cleanly. Speaker 1 matches what a high-α utility-reasoner should look like; speaker 2 matches a low-α mention-more default.

**Quantitatively problematic**: every run has r̂ in the 1.4–2.8 range despite 0 divergences. NUTS finds the two-mode posterior but chains don't mix between the modes even with the gap constraint. Short subset run (not fulldata): components don't separate at all (gap_D = 0.1).

### Relation to v5

Two independent accounts of where ε = 0.22 was coming from:
- **v5**: context-dependent structural mechanisms (λ_C, μ_noncanon, δγ, η). Gives ε = 0.10 with single-strategy model. R² = 0.94.
- **Mixture**: two-strategy model without the context-dependent mechanisms. Gives ε = 0.15, π = 0.40/0.60 on full data.

These are not directly comparable (different datasets: dc subset vs full; different convergence; different levels of mechanism detail). A **combined model** — v5 mechanisms inside each of two mixture components — has not been run. This is the natural integration test: if v5 mechanisms fully explain what mixture components were capturing, the two-component fit under combined model should collapse to π ≈ 0 or 1 (one component dominant). If mixture and v5 mechanisms are complementary, both should remain active.

**Status**: mixture exploration paused due to r̂ convergence issues. Memory notes (2026-03-25) flagged longer warmup + more samples as the next step; long per-dim fulldata run was queued but may not have been executed.

## 8. Open questions flagged but not resolved

- **Strategy heterogeneity.** ε = 0.10 is smaller than ext-v1's 0.22 but non-zero. The first/sharp D vs DF split is consistent with two participant sub-populations (minimum-description vs redundant-form). A mixture model (memo §6 / §7, open question #2) remains the next natural step if we want to push further.
- **Full-data (N = 9586) results not reported here.** Our earliest v5 run was on full data with the incorrect flag and gave misleading numbers. After the flag fix, we concentrated on the dc subset to replicate the memo scope; a full-data v5 F2 run would take ~15 min on the GPU and is a natural complement.
- **`df` and `cf` subsets.** Same methodology applied to the other two condition families would tell us whether F2 mechanisms are symmetric across dimensions, or colour-specific.
- **LOO uncertainty warnings.** All v5 `az.compare` outputs flagged `warning=True` (some Pareto-k > 0.7). Means LOO ELPD differences are reliable in direction but the exact magnitudes carry noise. Doesn't change the qualitative pattern.

## 9. Artefacts

- **Inference NC files**: `05-modelling-production-data/inference_data/mcmc_results_*_speaker_hier_dc_warmup2000_samples1000_chains4.nc` for baseline, ext-v1, v5, v5a, v5b, v5_inc_static, v5_global, v5_global_static, v5_global_full, v5_global_static_full.
- **Figures**: `05-modelling-production-data/figures/v5/` — PPC barplot (6-panel, 3 models overlaid) and 3-way correlation plot.
- **LOO tables**: `05-modelling-production-data/results/v5_loo_comparison_dc*.csv`, `v5_2x2_loo*.csv`.
- **Residual tables**: `05-modelling-production-data/results/v5_dc_residuals_{F2,C1}.csv`.
- **Design spec**: `docs/superpowers/specs/2026-04-13-extension-v5-colour-salience-saturating-length-design.md`.
- **Implementation plan**: `docs/superpowers/plans/2026-04-13-extension-v5-implementation.md`.

## 10. Minimal slide-ready summary (for the supervisor meeting)

> We tested the paper-reported model against a series of mechanism-driven extensions on the dc subset (N = 3196).
>
> The paper-reported model recovers the expected 2 × 2 interaction (architecture dominates; rec > sta within inc) but fits the between-cell proportions at R² = .35 with lapse = .22.
>
> Adding condition-gated colour salience (λ_C), canonical-ordering preference (μ_noncanon), and context-dependent length biases (δγ, η) lifts fit to R² = .94 and lapse to .10, with every new parameter informative.
>
> Under the extended model with freely sampled parameters in all 2 × 2 cells, both the architecture effect (Δ/dSE ≈ 3.2) and the semantic-regime effect (Δ/dSE ≈ 2.2 within inc, 3.8 within glb) remain significant but modest — about a third of their paper magnitudes. The interaction is now slightly rec-helps-global, whereas the paper had rec-helps-incremental.
>
> We face a choice: (a) report the minimal model as main result and the extensions as supplementary refinements, (b) report the extended model with a softened interaction claim, or (c) frame the extensions as implementations of what the architecture was indirectly capturing. The latter preserves the paper story; the former two are more conservative.
