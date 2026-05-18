# 2×2 on the Best Contextual Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 2×2 (speaker: incremental/global × semantics: recursive/static) on the merged best contextual model so the paper's main comparison runs on the architecture it advocates (R²≈0.94), not the simple reported-style models.

**Architecture:** Add two compile-time structural switches to the contextual-canon speaker family. `recursive` controls listener belief recursion; a separate new global speaker function handles joint (non-incremental) utility accrual. Both are exposed through a new `cell=` argument on `_make_contextual_pcalpha_canon_parsimony_model` (mirroring the existing `drop=`/`free=` pattern), guaranteeing an identical 10-parameter inventory across all four cells. Spec: `docs/superpowers/specs/2026-05-18-2x2-on-best-model-design.md`.

**Tech Stack:** NumPyro / JAX (model lib `05-modelling-production-data/modelSpecification.py`), `run_inference.py` CLI, CSP A6000 for MCMC, arviz for LOO, the existing `scripts/` diagnostics.

**Validation philosophy:** The decisive correctness test is *behaviour preservation* — `cell="inc_rec"` must reproduce the merged csv=0.59 best model exactly (same predictive R²≈0.94). Plus invariant checks (per-utterance probs sum to 1; the 4 cells are distinct). These replace conventional unit tests in this numerical-model codebase.

---

## File Structure

- **Modify** `05-modelling-production-data/modelSpecification.py`:
  - Add `recursive: bool=True` flag to `incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon` (freeze the scan's carried posterior when False).
  - Add new `global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon` (joint literal/RSA listener over the full utterance) + its vmap/jit wrappers.
  - Add `cell=` arg to `_make_contextual_pcalpha_canon_parsimony_model` selecting the speaker path; 4 module-level variants.
- **Modify** `05-modelling-production-data/run_inference.py`: register the 4 variants (import, HIER_MODELS, CONTEXTUAL/PCALPHA family sets, argparse).
- **Create** `05-modelling-production-data/scripts/diag_2x2_bestmodel.py`: LOO + R² + 2×2 decomposition for the 4 best-model cells (reuses r2_ladder + diag_2x2_parity helpers).
- **Create** `05-modelling-production-data/tests/test_2x2_cells.py`: invariant + distinctness checks (lightweight, CPU).

---

## Task 1: `recursive` flag on the incremental contextual-canon speaker

**Files:**
- Modify: `05-modelling-production-data/modelSpecification.py` (`incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon`, def at line ~3682; its `step`/`scan` at ~3782–3851; vmap `in_axes` at ~3881; jit wrapper at ~3907)
- Test: `05-modelling-production-data/tests/test_2x2_cells.py`

- [ ] **Step 1: Write the failing test**

```python
# 05-modelling-production-data/tests/test_2x2_cells.py
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax.numpy as jnp
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import modelSpecification as ms

# 3 objects × (size, color, form); referent is index 0.
STATES = jnp.asarray([[0.8, 1, 1], [0.5, 0, 1], [0.2, 1, 0]], dtype=jnp.float32)
KW = dict(sufficient_dim=1, has_one_word_solution=0.0, is_sharp=1.0,
          alpha_D=4.0, alpha_C=2.0, alpha_F=0.0, lambda_suff=1.5,
          lambda_form_mod=2.8, gamma_len3_erdc=0.0, lambda_noncanon=2.8,
          color_semval=0.59, form_semval=0.50, k=0.5, wf=0.6856,
          beta_lm=6.738, gamma_base=3.1, gamma_oneword=-2.4,
          gamma_sharp=0.68, epsilon=0.18)

def test_recursive_flag_changes_output_and_keeps_simplex():
    rec = ms.incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon(
        STATES, **KW)                                  # default recursive=True
    stat = ms.incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon(
        STATES, recursive=False, **KW)
    rec, stat = np.asarray(rec), np.asarray(stat)
    assert np.allclose(rec.sum(), 1.0, atol=1e-4)       # valid distribution
    assert np.allclose(stat.sum(), 1.0, atol=1e-4)
    assert not np.allclose(rec, stat, atol=1e-3)        # frozen ≠ recursive
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-modelling-production-data && JAX_PLATFORMS=cpu /Users/heningwang/miniconda3/envs/jax_playground/bin/python -m pytest tests/test_2x2_cells.py::test_recursive_flag_changes_output_and_keeps_simplex -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'recursive'`.

- [ ] **Step 3: Add the `recursive` parameter and freeze branch**

In `incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon`, add `recursive: bool = True,` to the signature (after `epsilon`). In the `step` function, the carried posterior is returned at the end as `new_per_utt_posts`. Change the scan-carry return so that when `recursive` is False the posterior is **not** updated (frozen at the uniform `init_posts`). Replace the `step` return line:

```python
        # was: return (log_scores + log_chosen, new_per_utt_posts), None
        carried_posts = new_per_utt_posts if recursive else per_utt_posts
        return (log_scores + log_chosen, carried_posts), None
```

`recursive` is a Python bool closed over by `step` (compile-time branch — no traced control flow). `per_utt_posts` is the step's input carry; when frozen it stays equal to `init_posts` every iteration (uniform), so each token's `log_L_ref` is computed against the uniform prior — the contextual analogue of the simple `incremental_speaker_frozen`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd 05-modelling-production-data && JAX_PLATFORMS=cpu /Users/heningwang/miniconda3/envs/jax_playground/bin/python -m pytest tests/test_2x2_cells.py::test_recursive_flag_changes_output_and_keeps_simplex -q`
Expected: PASS.

- [ ] **Step 5: Update vmap `in_axes` and jit wrapper to pass `recursive`**

The vmap `vectorized_incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier` and `jitted_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier` must forward `recursive`. Add `None,  # recursive` as the final entry of the `in_axes` tuple, add `recursive=True` as the last param of the jit wrapper signature, and pass `recursive` through both calls. Keep `recursive` a static Python bool: decorate the jit wrapper with `functools.partial(jax.jit, static_argnames=("recursive",))` (import `functools` at top of file if not already imported — it is used elsewhere; verify).

```python
@partial(jax.jit, static_argnames=("recursive",))
def jitted_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
    lambda_suff, lambda_form_mod, gamma_len3_erdc, lambda_noncanon,
    color_semval, form_semval, k, wf, beta_lm,
    gamma_base, gamma_oneword, gamma_sharp, epsilon, recursive=True,
):
    return vectorized_..._hier(  # add recursive to the call; in_axes gets None
        ..., epsilon, recursive,
    )
```

- [ ] **Step 6: Verify the jit/vmap path still works**

Run: `cd 05-modelling-production-data && JAX_PLATFORMS=cpu /Users/heningwang/miniconda3/envs/jax_playground/bin/python -c "
import jax.numpy as jnp, numpy as np, modelSpecification as ms
S=jnp.asarray([[[0.8,1,1],[0.5,0,1],[0.2,1,0]]],dtype=jnp.float32)
a=jnp.asarray([4.0]); z=jnp.asarray([0.0])
for r in (True,False):
    p=ms.jitted_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier(
      S,jnp.asarray([1]),jnp.asarray([0.0]),jnp.asarray([1.0]),a,jnp.asarray([2.0]),z,
      1.5,2.8,0.0,2.8,0.59,0.50,0.5,0.6856,6.738,3.1,-2.4,0.68,0.18,recursive=r)
    print(r, float(np.asarray(p).sum()))
"`
Expected: prints `True 1.0` and `False 1.0` (within 1e-4), no JIT error.

- [ ] **Step 7: Commit**

```bash
git add 05-modelling-production-data/modelSpecification.py 05-modelling-production-data/tests/test_2x2_cells.py
git commit -m "feat(05/2x2): recursive flag on contextual-canon speaker (inc_static cell)"
```

---

## Task 2: New global contextual-canon speaker (glob_rec / glob_static)

The global speaker computes one **joint** literal-listener posterior over the *full* utterance (Bayesian update by all of an utterance's tokens at once — composing the same per-token semantics the incremental speaker uses), then either scores it directly (`recursive=False`, literal-only) or applies one RSA pragmatic layer (`recursive=True`). Utterance-level terms (`log_lm_raw`, `length_bonus`, `form_present_bonus`, penalties, ε) are identical to the incremental speaker.

**Files:**
- Modify: `05-modelling-production-data/modelSpecification.py` (add the new function + vmap/jit wrappers immediately after the incremental jit wrapper, ~line 3922)
- Test: `05-modelling-production-data/tests/test_2x2_cells.py`

- [ ] **Step 1: Write the failing test**

```python
def test_global_speaker_simplex_and_distinct():
    g_lit = ms.global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon(
        STATES, recursive=False, **KW)
    g_rsa = ms.global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon(
        STATES, recursive=True, **KW)
    inc = ms.incremental_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon(
        STATES, **KW)
    g_lit, g_rsa, inc = map(np.asarray, (g_lit, g_rsa, inc))
    assert np.allclose(g_lit.sum(), 1.0, atol=1e-4)
    assert np.allclose(g_rsa.sum(), 1.0, atol=1e-4)
    assert not np.allclose(g_lit, g_rsa, atol=1e-3)     # RSA layer changes output
    assert not np.allclose(g_lit, inc, atol=1e-3)       # global ≠ incremental
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-modelling-production-data && JAX_PLATFORMS=cpu /Users/heningwang/miniconda3/envs/jax_playground/bin/python -m pytest tests/test_2x2_cells.py::test_global_speaker_simplex_and_distinct -q`
Expected: FAIL — `AttributeError: module 'modelSpecification' has no attribute 'global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon'`.

- [ ] **Step 3: Implement the global contextual speaker**

Add this function immediately after `jitted_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier` (~line 3922). It reuses the module-level constants the incremental speaker uses (`LOG_LM_RAW_15`, `TOKEN_PRESENT`, `ACTUAL_TOK_ONEHOT` are per-token; for the joint listener use the **full-utterance token-presence** aggregate `TOKEN_PRESENT.sum(axis=0)` over positions, i.e. which (dim,value) features each of the 15 utterances asserts). `SIZE_ANCHOR_R`, `N_WORDS`, `F_PRESENT_15`, `IS_3WORD_15`, `F_BEFORE_C_15`, `n_utt`, `T` are the same constants the incremental fn closes over — confirm their exact names by reading the incremental fn header before writing).

```python
def global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_D=3.0, alpha_C=3.0, alpha_F=3.0, lambda_suff=0.0,
    lambda_form_mod=0.0, gamma_len3_erdc=0.0, lambda_noncanon=0.0,
    color_semval=0.95, form_semval=0.80, k=0.50, wf=1.00, beta_lm=1.00,
    gamma_base=0.0, gamma_oneword=0.0, gamma_sharp=0.0, epsilon=0.01,
    recursive: bool = True,
):
    """GLOBAL counterpart of the contextual-canon speaker (2×2 speaker factor).

    Joint (non-incremental) utility: the literal listener conditions on ALL of
    an utterance's adjective features at once -> one posterior over objects ->
    L(referent | full utterance). recursive=True adds one RSA pragmatic layer
    (S1 over the 15 candidate utterances, then L1); recursive=False scores the
    literal listener directly. All utterance-level terms (LM prior, length,
    form-present, len3 / non-canonical penalties, epsilon) are IDENTICAL to the
    incremental speaker so the only structural difference is utility accrual.
    """
    eps = 1e-8
    referent_index = 0
    n_obj = states.shape[0]
    alpha_vec = jnp.array([alpha_D, alpha_C, alpha_F])

    sizes = states[:, 0]
    size_sort_idx = jnp.argsort(sizes)
    sizes_sorted = sizes[size_sort_idx]
    colors = states[:, 1]
    forms = states[:, 2]
    log_color_sem = jnp.log(jnp.where(colors == 1, color_semval, 1.0 - color_semval) + eps)
    log_form_sem = jnp.log(jnp.where(forms == 1, form_semval, 1.0 - form_semval) + eps)
    log_lm_raw = beta_lm * LOG_LM_RAW_15

    uniform = jnp.ones(n_obj) / n_obj

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

    # Per-utterance full-feature presence: (n_utt, 3 dims, n_obj) log-semantics,
    # summed over the utterance's asserted (dim,value) tokens (joint, not scan).
    size_log_sem = jnp.log(jnp.clip(_anchored_size_sem(sizes, uniform), eps))   # (n_obj,)
    log_sem_table = jnp.stack(
        [jnp.broadcast_to(size_log_sem, (n_obj,)), log_color_sem, log_form_sem],
        axis=0,
    )                                                                          # (3, n_obj)
    # TOKEN_PRESENT[t] is (n_utt, 3, ?) per position; aggregate presence over
    # positions -> which dims each utterance asserts. Use the same one-hot the
    # incremental fn uses, summed over T:
    feat_pres = jnp.clip(jnp.sum(TOKEN_PRESENT, axis=0).sum(axis=-1), 0.0, 1.0) # (n_utt, 3)
    log_joint = jnp.einsum("ud, do -> uo", feat_pres, log_sem_table)           # (n_utt, n_obj)
    log_post = (jnp.log(uniform)[None, :] + log_joint)
    log_post = log_post - jax.scipy.special.logsumexp(log_post, axis=-1, keepdims=True)
    log_L0_ref = log_post[:, referent_index]                                   # (n_utt,)

    if recursive:
        # One RSA pragmatic layer over the 15 candidate utterances.
        # S1(u|obj) ∝ exp(mean(alpha_vec)·logL0(obj|u)); L1(obj|u) ∝ S1·prior.
        a = jnp.mean(alpha_vec)
        logS1 = a * log_post                                                   # (n_utt, n_obj)
        logS1 = logS1 - jax.scipy.special.logsumexp(logS1, axis=0, keepdims=True)
        logL1 = logS1 + jnp.log(uniform)[None, :]
        logL1 = logL1 - jax.scipy.special.logsumexp(logL1, axis=-1, keepdims=True)
        log_score = jnp.mean(alpha_vec) * logL1[:, referent_index]
    else:
        log_score = jnp.mean(alpha_vec) * log_L0_ref

    # First-word boost stays utterance-level: utterances whose first asserted
    # dim == sufficient_dim. FIRST_DIM_15 is the canonical first-dim index per
    # utterance (add as a module constant in Step 3a if absent).
    suff_boost = lambda_suff * (FIRST_DIM_15 == sufficient_dim).astype(jnp.float32)

    blur_gate = 1.0 - is_sharp
    gamma_eff = gamma_base + gamma_oneword * has_one_word_solution + gamma_sharp * blur_gate
    length_bonus = gamma_eff * jnp.maximum(N_WORDS - 1.0, 0.0)
    erdc_gate = (sufficient_dim == 0).astype(jnp.float32)
    form_present_bonus = lambda_form_mod * erdc_gate * F_PRESENT_15
    len3_penalty = gamma_len3_erdc * erdc_gate * IS_3WORD_15
    noncanon_penalty = lambda_noncanon * F_BEFORE_C_15

    log_unnorm = (log_lm_raw + log_score + suff_boost + length_bonus
                  + form_present_bonus - len3_penalty - noncanon_penalty)
    model_probs = jax.nn.softmax(log_unnorm)
    return (1.0 - epsilon) * model_probs + epsilon / log_unnorm.shape[0]
```

- [ ] **Step 3a: Add the `FIRST_DIM_15` constant if absent**

Read the block where `F_PRESENT_15` / `IS_3WORD_15` / `F_BEFORE_C_15` are defined (grep `F_PRESENT_15 =` in `modelSpecification.py`). If a per-utterance "first asserted dimension index" constant does not exist, add next to them:

```python
# Canonical first-adjective dimension per 15 utterance types (0=size,1=colour,
# 2=form), matching the order encoded in TOKEN_PRESENT / the 15-utterance set.
FIRST_DIM_15 = jnp.asarray(<explicit 15-int list derived from the existing
utterance enumeration — read the 15-utterance definition and fill exact values>,
dtype=jnp.int32)
```

Derive the 15 values by reading the existing utterance-type enumeration (the same source `F_PRESENT_15` was built from); do not guess — copy the ordering.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd 05-modelling-production-data && JAX_PLATFORMS=cpu /Users/heningwang/miniconda3/envs/jax_playground/bin/python -m pytest tests/test_2x2_cells.py::test_global_speaker_simplex_and_distinct -q`
Expected: PASS.

- [ ] **Step 5: Add vmap + jit wrappers for the global speaker**

Mirror the incremental wrappers exactly (same `in_axes` pattern: `0` for states/sufficient_dim/has_one_word_solution/is_sharp/alpha_D/alpha_C/alpha_F, `None` for the rest including a trailing `None` for `recursive`), name them `vectorized_global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier` and `jitted_global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier`, with `@partial(jax.jit, static_argnames=("recursive",))`.

- [ ] **Step 6: Commit**

```bash
git add 05-modelling-production-data/modelSpecification.py 05-modelling-production-data/tests/test_2x2_cells.py
git commit -m "feat(05/2x2): global contextual-canon speaker (glob_rec/glob_static cells)"
```

---

## Task 3: `cell=` argument on the parsimony factory + 4 variants

**Files:**
- Modify: `05-modelling-production-data/modelSpecification.py` (`_make_contextual_pcalpha_canon_parsimony_model`, def ~line 4060; its speaker call ~line 4135)
- Test: `05-modelling-production-data/tests/test_2x2_cells.py`

- [ ] **Step 1: Write the failing test**

```python
def test_four_cells_register_and_run():
    import run_inference as ri
    keys = [f"contextual_pcalpha_canon_parsimony_2x2_{c}"
            for c in ("inc_rec", "inc_static", "glob_rec", "glob_static")]
    for k in keys:
        assert k in ri.HIER_MODELS, k
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-modelling-production-data && JAX_PLATFORMS=cpu /Users/heningwang/miniconda3/envs/jax_playground/bin/python -m pytest tests/test_2x2_cells.py::test_four_cells_register_and_run -q`
Expected: FAIL — `AssertionError: contextual_pcalpha_canon_parsimony_2x2_inc_rec`.

- [ ] **Step 3: Add `cell=` to the factory**

In `_make_contextual_pcalpha_canon_parsimony_model`, add `cell: str = "inc_rec"` to the signature and validate:

```python
_valid_cells = {"inc_rec", "inc_static", "glob_rec", "glob_static"}
if cell not in _valid_cells:
    raise ValueError(f"Unsupported 2x2 cell {cell!r}; expected {sorted(_valid_cells)}")
_use_global = cell in ("glob_rec", "glob_static")
_recursive = cell in ("inc_rec", "glob_rec")
```

In the model body, replace the single speaker call (the
`jitted_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier(...)`
invocation, ~line 4135) with a dispatch that keeps the exact same arguments
and appends `recursive=_recursive`:

```python
speaker_fn = (jitted_global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier
              if _use_global
              else jitted_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon_hier)
probs = speaker_fn(
    states, sufficient_dim, has_one_word_solution, is_sharp,
    alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
    lambda_suff, lambda_form_mod, gamma_len3_erdc, lambda_noncanon,
    csv_r, fsv_r, k_r, wf_r, beta_lm,
    gamma_base, gamma_oneword, gamma_sharp, epsilon, recursive=_recursive,
)
```

Default `cell="inc_rec"` + `recursive=True` + incremental fn ⇒ **behaviour
identical to before** (existing `…_no_alphaF`/`…_csv059` NCs unaffected).

- [ ] **Step 4: Add the 4 module-level variants**

After the existing `…_no_alphaF_csv059_hier` instantiation, add (all with `drop=("alpha_F",)`, `color_semval=0.59`):

```python
likelihood_function_contextual_pcalpha_canon_parsimony_2x2_inc_rec_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.59, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",), cell="inc_rec"))
likelihood_function_contextual_pcalpha_canon_parsimony_2x2_inc_static_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.59, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",), cell="inc_static"))
likelihood_function_contextual_pcalpha_canon_parsimony_2x2_glob_rec_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.59, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",), cell="glob_rec"))
likelihood_function_contextual_pcalpha_canon_parsimony_2x2_glob_static_hier = (
    _make_contextual_pcalpha_canon_parsimony_model(
        color_semval=0.59, form_semval=0.50, k=0.5, wf=WF_FIXED_ITER11_MEDIAN,
        drop=("alpha_F",), cell="glob_static"))
```

- [ ] **Step 5: Register in `run_inference.py`**

Add the 4 `likelihood_function_..._2x2_{cell}_hier` symbols to the `from modelSpecification import (...)` block; add 4 `HIER_MODELS["contextual_pcalpha_canon_parsimony_2x2_{cell}"] = (..., 0.85, 5)` entries; add the 4 keys to the `CONTEXTUAL_FAMILY` set, the `PCALPHA_FAMILY` set, and the argparse `--speaker_type` choices list (the 5 spots used for every prior parsimony variant this session).

- [ ] **Step 6: Run test to verify it passes**

Run: `cd 05-modelling-production-data && JAX_PLATFORMS=cpu /Users/heningwang/miniconda3/envs/jax_playground/bin/python -m pytest tests/test_2x2_cells.py -q`
Expected: all 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add 05-modelling-production-data/modelSpecification.py 05-modelling-production-data/run_inference.py 05-modelling-production-data/tests/test_2x2_cells.py
git commit -m "feat(05/2x2): cell= arg + 4 best-model 2x2 variants registered"
```

---

## Task 4: Behaviour-preservation check (inc_rec == merged best model)

**Files:** none modified (validation only).

- [ ] **Step 1: Predictive equivalence vs the merged csv=0.59 model**

Run (CPU): replay both models' speaker on the dc-subset inputs at fixed posterior-mean params and confirm `inc_rec` matches the existing `…_no_alphaF_csv059` model bitwise-close.

```bash
cd 05-modelling-production-data && JAX_PLATFORMS=cpu /Users/heningwang/miniconda3/envs/jax_playground/bin/python -c "
import numpy as np, jax, jax.numpy as jnp, modelSpecification as ms
from numpyro.infer import Predictive
import scripts.diag_semval_sweep as d
kw,_=d.build_dc_inputs()
base=ms.likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_csv059_hier
cell=ms.likelihood_function_contextual_pcalpha_canon_parsimony_2x2_inc_rec_hier
import numpyro.handlers as H
seed=jax.random.PRNGKey(0)
tr_b=H.trace(H.seed(base,seed)).get_trace(**kw)
tr_c=H.trace(H.seed(cell,seed)).get_trace(**kw)
pb=np.asarray(tr_b['obs']['fn'].probs); pc=np.asarray(tr_c['obs']['fn'].probs)
print('max|Δprobs| inc_rec vs merged best model:', float(np.max(np.abs(pb-pc))))
assert np.allclose(pb,pc,atol=1e-5), 'inc_rec must equal the merged csv=0.59 model'
print('OK — behaviour preserved')
"
```
Expected: `max|Δprobs| … : <1e-5` and `OK — behaviour preserved`.

- [ ] **Step 2: Commit (validation note only, no code)**

No commit (no files changed). If Step 1 fails, STOP — the `cell=`/`recursive` refactor changed `inc_rec`; fix before proceeding.

---

## Task 5: CSP runner + 4 inference runs

**Files:**
- Create: `05-modelling-production-data/run_2x2_bestmodel.sh` (server-side chain, committed)

- [ ] **Step 1: Write the runner script**

```bash
# 05-modelling-production-data/run_2x2_bestmodel.sh
set -e
cd ~/Documents/00_HW/numpyro_adjective_modelling/05-modelling-production-data
for C in inc_rec inc_static glob_rec glob_static; do
  echo "=== START $C $(date) ==="
  JAX_PLATFORMS="" XLA_FLAGS="" /home/csp/.conda/envs/numpyro_modelling/bin/python -u \
    run_inference.py --speaker_type contextual_pcalpha_canon_parsimony_2x2_$C \
    --hierarchical --condition-subset erdc,zrdc,brdc \
    --num_warmup 4000 --num_samples 2000 --num_chains 4
  echo "=== DONE $C $(date) ==="
done
echo "=== ALL 2x2 BESTMODEL DONE $(date) ==="
```

- [ ] **Step 2: Commit + push the branch**

```bash
git add 05-modelling-production-data/run_2x2_bestmodel.sh
git commit -m "feat(05/2x2): server runner for the 4 best-model 2x2 cells"
git push -u origin feat/2x2-best-model
```

- [ ] **Step 3: Deploy on CSP (sequential, single GPU)**

```bash
ssh csp 'cd ~/Documents/00_HW/numpyro_adjective_modelling && git fetch -q origin && git checkout -q feat/2x2-best-model && git reset -q --hard origin/feat/2x2-best-model && cd 05-modelling-production-data && mkdir -p logs && LOG=logs/2x2bestmodel_$(date +%Y%m%d_%H%M%S).log && nohup bash run_2x2_bestmodel.sh > "$LOG" 2>&1 & echo "LOG=$LOG PID=$!"'
```

- [ ] **Step 4: Watch to completion (background watcher, until `ALL 2x2 BESTMODEL DONE`)**

Use the established watcher pattern (poll `grep -q 'ALL 2x2 BESTMODEL DONE'` + `pgrep -f run_inference`, 30–40s interval, `run_in_background: true`). inc_rec ≈ 3–4 min (fixed-constant incremental); the two global cells' runtime is unknown — record it.

- [ ] **Step 5: Pull the 4 NCs**

```bash
for c in inc_rec inc_static glob_rec glob_static; do
  scp -q "csp:Documents/00_HW/numpyro_adjective_modelling/05-modelling-production-data/inference_data/mcmc_results_contextual_pcalpha_canon_parsimony_2x2_${c}_speaker_hier_dc_warmup4000_samples2000_chains4.nc" inference_data/
done
```

---

## Task 6: Analysis — LOO + R² + 2×2 decomposition

**Files:**
- Create: `05-modelling-production-data/scripts/diag_2x2_bestmodel.py`

- [ ] **Step 1: Write the analysis script**

Reuse `r2_ladder.compute_r2_row` (PPC R², no JAX) and the `diag_2x2_parity` decomposition logic. For the 4 NCs print: R²(all)/R²(emp≥.02) (the ladder convention, `plv.GROUP_COLS=["relevant_property","sharpness"]`), `az.loo` elpd/p_loo, `az.compare` pairwise dSE, and the 2×2 decomposition (speaker ME = mean(inc)−mean(glob); semantics ME = mean(rec)−mean(static); interaction = (inc_rec−inc_static)−(glob_rec−glob_static)). Write `results/contextual_dc/twoby2_bestmodel.csv`. Include the assertion that `inc_rec` R²(all) ≥ 0.93 (reproduces the merged best model — fail loudly otherwise).

- [ ] **Step 2: Run it**

Run: `cd 05-modelling-production-data && JAX_PLATFORMS=cpu /Users/heningwang/miniconda3/envs/jax_playground/bin/python scripts/diag_2x2_bestmodel.py`
Expected: table prints; `inc_rec` R²(all)≈0.94 (sanity); CSV saved.

- [ ] **Step 3: Commit**

```bash
git add 05-modelling-production-data/scripts/diag_2x2_bestmodel.py
git commit -m "diag(05/2x2): best-model 2x2 LOO + R² + interaction analysis"
```

---

## Task 7: Memo, memory, PR

**Files:** memo (git-ignored), memory (outside repo), PR.

- [ ] **Step 1: Memo section** — add `## 7.13 Best-model 2×2 (speaker×semantics on the advocated model)` to `10-writing/memos/extended_production_model_memo.md`: the 4-cell ELPD/R²/p_loo table, the speaker×semantics interaction with dSE, and an explicit comparison to the simple-model 2×2 (does incremental≫global / recursive>static hold on the R²≈0.94 architecture? — the scientifically interesting question). Note inc_rec reproduces R²≈0.94 (anchor).

- [ ] **Step 2: Memory** — append a section to `~/.claude/projects/-Users-heningwang-Documents-GitHub-numpyro-adjective-modelling/memory/project_model_comparison_2x2.md` and update the MEMORY.md pointer line with the best-model-2×2 verdict.

- [ ] **Step 3: PR**

```bash
gh pr create --base main --head feat/2x2-best-model \
  --title "2×2 (speaker×semantics) on the best contextual model (R²≈0.94)" \
  --body "<summary: spec link; 4 parameter-matched cells; inc_rec reproduces the merged csv=0.59 model; LOO/R²/interaction table; how it compares to the simple-model 2×2; risks>"
```

- [ ] **Step 4: Report** the best-model 2×2 result and whether the interaction conclusions change vs the simple-model 2×2.

---

## Self-Review

- **Spec coverage:** factor operationalization → Tasks 1–2; strict 10-param parity → Task 3 (`cell=` keeps inventory; default preserves behaviour) + Task 4; run/eval (dc-subset 4000/2000/4, LOO+R²+parity) → Tasks 5–6; memo/memory/PR → Task 7. inc_rec=current-best sanity → Task 4 + Task 6 assertion. All spec sections covered.
- **Placeholders:** Task 2 Step 3a (`FIRST_DIM_15`) and Task 7 Step 1/3 require reading existing enumerations / writing prose — these are explicit "derive from existing source, do not guess" instructions, not blanks; acceptable because the exact values must come from the codebase, not the plan author.
- **Type/name consistency:** speaker fn name `global_speaker_contextual_anchored_gamma_sharpbonus_formmod_canon`, jit wrapper `jitted_global_..._hier`, variant keys `contextual_pcalpha_canon_parsimony_2x2_{inc_rec,inc_static,glob_rec,glob_static}`, factory arg `cell=` — used consistently across Tasks 2/3/5/6.
- **Risk:** the global speaker's exact reuse of `TOKEN_PRESENT`/constants must be verified against the incremental fn before Task 2 Step 3 (the plan says "confirm exact names by reading the incremental fn header") — this is the one place the implementer must read code first; flagged in-task.
