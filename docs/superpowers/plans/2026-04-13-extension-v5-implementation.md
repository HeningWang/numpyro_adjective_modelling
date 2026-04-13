# Extension v5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add condition-gated colour salience (λ_C) and saturating length bias (γ_1, γ_2) to the production-data RSA model; run LOO-CV against baseline and ext-v1.

**Architecture:** Fork the existing `incremental_speaker` into a v5 variant that adds (a) a per-trial `is_colour_sufficient` flag threaded through vmap and applied as an additive logit boost to the C-mention candidate at every step, and (b) a two-step length bonus replacing the linear γ term in the final utility softmax. Three new likelihood factories (`v5a`, `v5b`, `v5`) follow the existing `_make_extended_v1_model` pattern. Inference runs hierarchically (per-participant α offsets) on the remote GPU server via `run_inference.py`. Posterior analysis reuses existing tooling under `posterior_analysis.py` / `analyse_inference.ipynb`.

**Tech Stack:** NumPyro, JAX (vmap + jit), ArviZ (LOO-CV), pandas, Python 3.11. Remote A6000 GPU per the `server-workflow` skill.

**Spec:** [`docs/superpowers/specs/2026-04-13-extension-v5-colour-salience-saturating-length-design.md`](../specs/2026-04-13-extension-v5-colour-salience-saturating-length-design.md)

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `05-modelling-production-data/helper.py` | modify | Derive per-trial `is_colour_sufficient` and `participant_idx` (already present); return alongside states. |
| `05-modelling-production-data/modelSpecification.py` | modify | Add `incremental_speaker_v5`, vmaps + jitted versions, three new factories (`_make_v5a_model`, `_make_v5b_model`, `_make_v5_model`), and three exported likelihood functions. |
| `05-modelling-production-data/scripts/check_v5_sanity.py` | create | Small script: build a 2-trial synthetic state, run v5 with λ_C=0 and γ_2=γ_1 against ext-v1; assert numerical equivalence. |
| `05-modelling-production-data/scripts/prior_predictive_v5.py` | create | Identifiability checks (memo §4): sweep λ_C and inspect predicted C-proportion per condition; sweep γ_1, γ_2 grid. |
| `05-modelling-production-data/run_inference.py` | modify | Add `v5a`, `v5b`, `v5` to the model registry. |
| `05-modelling-production-data/run_inference.sh` | modify | Add invocations for the three new variants. |
| `05-modelling-production-data/posterior_analysis.py` | modify | Add v5 NC-file loaders; LOO-CV across 5 models (baseline, ext-v1, v5a, v5b, v5); residual diff table vs memo §5.3. |
| `10-writing/extended_production_model_memo.md` | modify | New §6 documenting v5 mechanism, results, identifiability findings, residual resolution. |

---

## Phase 1 — Helper: per-trial colour-sufficient flag

### Task 1: Derive `is_colour_sufficient` in helper

**Files:**
- Modify: `05-modelling-production-data/helper.py` (around the `import_dataset` return block)

- [ ] **Step 1: Inspect `CONDITIONS_OF_INTEREST` and confirm the colour-sufficient family**

Run: `grep -n "CONDITIONS_OF_INTEREST\|cf\"" 05-modelling-production-data/helper.py`
Expected: confirm `ercf`, `zrcf`, `brcf` are the colour-sufficient (C+F) conditions per memo §1 (third+fourth chars = the two distinguishing dimensions; `cf` = colour and form distinguish, so size is non-distinguishing → colour-sufficient when colour alone identifies the target). Cross-check by grouping `df.groupby("conditions")["annotation"].value_counts()` and confirming the C-dominant conditions are the `*cf` family.

- [ ] **Step 2: Add the flag derivation in `import_dataset`**

In `helper.py::import_dataset`, after `df = df[df["conditions"].isin(CONDITIONS_OF_INTEREST)].copy()`:

```python
COLOUR_SUFFICIENT_CONDITIONS = ("ercf", "zrcf", "brcf")
is_colour_sufficient_np = df["conditions"].isin(COLOUR_SUFFICIENT_CONDITIONS).to_numpy(dtype=np.float32)
is_colour_sufficient = jnp.array(is_colour_sufficient_np, dtype=jnp.float32)
```

Add `COLOUR_SUFFICIENT_CONDITIONS` to the module-level constants block (next to `CONDITIONS_OF_INTEREST`).

- [ ] **Step 3: Return the flag**

Locate the return statement of `import_dataset` and add `is_colour_sufficient` to the returned dict / tuple. Per memo, `import_dataset` for production returns a **dict** — add key `"is_colour_sufficient"`.

- [ ] **Step 4: Verify the flag**

Add a tiny check in a Python REPL or scratch script:

```python
from helper import import_dataset
data = import_dataset()
flag = data["is_colour_sufficient"]
assert flag.shape == data["states_train"].shape[:1]
assert float(flag.sum()) > 0  # some trials are colour-sufficient
assert float(flag.sum()) < flag.shape[0]  # not all trials
print(f"Colour-sufficient: {int(flag.sum())} / {flag.shape[0]} trials")
```

Expected: roughly 1/3 of trials flagged (3 of 9 conditions are `*cf`).

- [ ] **Step 5: Commit**

```bash
git add 05-modelling-production-data/helper.py
git commit -m "feat(helper): expose per-trial is_colour_sufficient flag for v5 model"
```

---

## Phase 2 — Speaker forward function (v5)

### Task 2: Implement `incremental_speaker_v5`

**Files:**
- Modify: `05-modelling-production-data/modelSpecification.py` (insert after `incremental_speaker_extended` at line ~1130)

- [ ] **Step 1: Add the new function**

Fork `incremental_speaker` (modelSpecification.py:566). The two changes are inside the `step` scan and in the final utility computation.

```python
def incremental_speaker_v5(
    states:                jnp.ndarray,
    is_colour_sufficient:  float,           # 0.0 or 1.0 (per-trial scalar after vmap)
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
    epsilon:               float = 0.01,
) -> jnp.ndarray:
    """Extension v5: per-dim alpha + condition-gated colour salience boost
    + saturating two-step length bias + lapse rate. Forks incremental_speaker.

    is_colour_sufficient: per-trial scalar flag (0/1), vmapped over the leading axis.
    """
    # Body identical to incremental_speaker until the masked-softmax in `step`
    # and the final utility computation. Two surgical changes below.

    # CHANGE 1 — inside `step`, after computing logits but before adding gamma:
    #   add lambda_C * is_colour_sufficient to the C-mention column (dim index 1)
    # CHANGE 2 — replace `gamma * k_extra` with the two-step bonus.
```

The full body should mirror `incremental_speaker` (modelSpecification.py:566–700). Inside the `step` function, modify the logits computation:

```python
# Inside step(), after `logits = jnp.where(cand_mask_t, alpha_vec[None, :] * log_L_ref, -1e9)`:
# Add condition-gated colour boost on the C-mention column (index 1).
boost_vec = jnp.array([0.0, lambda_C * is_colour_sufficient, 0.0])  # (3,)
logits = jnp.where(cand_mask_t, logits + boost_vec[None, :], -1e9)
```

For the saturating length bias, locate the final utility step in `incremental_speaker` where `gamma * (length - minimal_length)` (or equivalent `k_extra` term) is added to log-scores. Replace with:

```python
k_extra = jnp.maximum(UTT_LENGTHS - MINIMAL_LENGTH, 0)   # (n_utt,)
length_bonus = (
    gamma_1 * (k_extra >= 1).astype(jnp.float32)
    + gamma_2 * (k_extra >= 2).astype(jnp.float32)
)
final_log_scores = log_scores + length_bonus + log_P_beta
```

If `MINIMAL_LENGTH` or `UTT_LENGTHS` are not already module-level constants, locate the existing `gamma`-application block in `incremental_speaker` to see how `k_extra` is currently computed and reuse the same definition.

- [ ] **Step 2: Add vmaps and jitted wrappers**

```python
# v5: hierarchical (per-trial alpha + per-trial condition flag)
vectorized_incremental_speaker_v5_hier = jax.vmap(
    incremental_speaker_v5,
    in_axes=(0,    # states
             0,    # is_colour_sufficient ← per-trial
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
             None, # epsilon
             ),
)

@jax.jit
def jitted_speaker_v5_hier(
    states, is_colour_sufficient,
    alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
    lambda_C, color_semval, form_semval, k, wf, beta,
    gamma_1, gamma_2, epsilon,
):
    return vectorized_incremental_speaker_v5_hier(
        states, is_colour_sufficient,
        alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
        lambda_C, color_semval, form_semval, k, wf, beta,
        gamma_1, gamma_2, epsilon,
    )
```

Place after the existing `jitted_speaker_extended_hier` block (around line 1199).

- [ ] **Step 3: Commit (intermediate, before sanity check)**

```bash
git add 05-modelling-production-data/modelSpecification.py
git commit -m "feat(model): add incremental_speaker_v5 with lambda_C and saturating length bias"
```

### Task 3: Sanity-check v5 against ext-v1

**Files:**
- Create: `05-modelling-production-data/scripts/check_v5_sanity.py`

- [ ] **Step 1: Write the sanity script**

```python
"""Sanity check: with lambda_C=0 and gamma_2=gamma_1, v5 must equal ext-v1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp
import numpy as np
from helper import import_dataset
from modelSpecification import jitted_speaker_hier, jitted_speaker_v5_hier

data = import_dataset()
states = data["states_train"][:8]                     # 8-trial subset
flag   = data["is_colour_sufficient"][:8]
N      = states.shape[0]

# Identical alpha per trial for both models.
alpha_D = jnp.full((N,), 6.94)
alpha_C = jnp.full((N,), 1.84)
alpha_F = jnp.full((N,), 1.88)

color_semval, form_semval, k, wf, beta, gamma, epsilon = 0.971, 0.50, 0.5, 1.0, jnp.exp(2.05), 2.32, 0.23

probs_v1 = jitted_speaker_hier(
    states, alpha_D, alpha_C, alpha_F,
    color_semval, form_semval, k, wf, beta, gamma, epsilon,
)

# v5 with lambda_C=0 and gamma_2=gamma_1 should reproduce v1 exactly.
probs_v5 = jitted_speaker_v5_hier(
    states, flag,
    alpha_D, alpha_C, alpha_F,
    0.0, color_semval, form_semval, k, wf, beta,
    gamma, gamma, epsilon,
)

max_diff = float(jnp.max(jnp.abs(probs_v1 - probs_v5)))
print(f"Max abs diff (v1 vs v5 with neutral params): {max_diff:.2e}")
assert max_diff < 1e-5, f"v5 does not reduce to v1: max diff {max_diff:.2e}"

# Now turn lambda_C on; only colour-sufficient trials should change.
probs_v5_boosted = jitted_speaker_v5_hier(
    states, flag,
    alpha_D, alpha_C, alpha_F,
    2.0, color_semval, form_semval, k, wf, beta,
    gamma, gamma, epsilon,
)
diff_per_trial = jnp.max(jnp.abs(probs_v5 - probs_v5_boosted), axis=-1)
print("Per-trial max diff with lambda_C=2:")
for i in range(N):
    print(f"  trial {i}: flag={float(flag[i]):.0f}, diff={float(diff_per_trial[i]):.4f}")
assert jnp.all(jnp.where(flag == 0, diff_per_trial < 1e-5, True)), \
    "lambda_C affected non-colour-sufficient trials"
print("OK: lambda_C only affects colour-sufficient trials.")
```

- [ ] **Step 2: Run the sanity check**

Run: `cd 05-modelling-production-data && python scripts/check_v5_sanity.py`
Expected: both assertions pass; max diff in equivalence check < 1e-5; only flag=1 trials show changes under lambda_C=2.

- [ ] **Step 3: If sanity check fails, debug**

Common failures:
- `gamma_1`/`gamma_2` definition uses different `k_extra` baseline than ext-v1's `gamma` — re-inspect how `k_extra` is computed in `incremental_speaker` and align.
- `boost_vec` applied at wrong index (C is dim 1; D=0, C=1, F=2 per `SYMBOL_TO_INDEX`).
- `is_colour_sufficient` not properly vmapped — check `in_axes`.

- [ ] **Step 4: Commit**

```bash
git add 05-modelling-production-data/scripts/check_v5_sanity.py
git commit -m "test(v5): add sanity check verifying v5 reduces to ext-v1 with neutral params"
```

---

## Phase 3 — Likelihood factories

### Task 4: Add v5, v5a, v5b model factories

**Files:**
- Modify: `05-modelling-production-data/modelSpecification.py` (insert after `_make_extended_v1_model` at line ~1422)

- [ ] **Step 1: Add the v5 factory**

```python
def _make_v5_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0):
    """Factory for v5: ext-v1 + condition-gated lambda_C + saturating gamma_1, gamma_2."""
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              is_colour_sufficient=None):
        log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 0.5))
        beta     = jnp.exp(log_beta)

        alpha_D  = numpyro.sample("alpha_D",  dist.HalfNormal(5.0))
        alpha_C  = numpyro.sample("alpha_C",  dist.HalfNormal(5.0))
        alpha_F  = numpyro.sample("alpha_F",  dist.HalfNormal(5.0))
        lambda_C = numpyro.sample("lambda_C", dist.Normal(0.0, 1.0))
        gamma_1  = numpyro.sample("gamma_1",  dist.Normal(0.0, 1.0))
        gamma_2  = numpyro.sample("gamma_2",  dist.Normal(0.0, 1.0))
        epsilon  = numpyro.sample("epsilon",  dist.Beta(1.0, 50.0))
        tau      = numpyro.sample("tau",      dist.HalfNormal(0.2))

        with numpyro.plate("participants", n_participants):
            delta = numpyro.sample("delta", dist.Normal(0.0, tau))

        alpha_D_per_trial = jnp.maximum(alpha_D + delta[participant_idx], 0.0)
        alpha_C_per_trial = jnp.maximum(alpha_C + delta[participant_idx], 0.0)
        alpha_F_per_trial = jnp.maximum(alpha_F + delta[participant_idx], 0.0)

        with numpyro.plate("data", len(states)):
            probs = jitted_speaker_v5_hier(
                states, is_colour_sufficient,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_C, color_semval, form_semval, k, wf, beta,
                gamma_1, gamma_2, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_v5_hier = _make_v5_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
)
```

- [ ] **Step 2: Add the v5a factory (lambda_C only, linear gamma)**

```python
def _make_v5a_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0):
    """v5a: lambda_C added on top of ext-v1; linear gamma retained (gamma_2 := gamma_1)."""
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              is_colour_sufficient=None):
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
                states, is_colour_sufficient,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                lambda_C, color_semval, form_semval, k, wf, beta,
                gamma, gamma, epsilon,    # gamma_1 = gamma_2 = gamma → linear
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_v5a_hier = _make_v5a_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
)
```

- [ ] **Step 3: Add the v5b factory (saturating gamma only, no lambda_C)**

```python
def _make_v5b_model(color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0):
    """v5b: saturating gamma_1, gamma_2 added on top of ext-v1; no lambda_C."""
    def model(states=None, empirical=None,
              participant_idx=None, n_participants=None,
              is_colour_sufficient=None):
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
                states, is_colour_sufficient,
                alpha_D_per_trial, alpha_C_per_trial, alpha_F_per_trial,
                0.0, color_semval, form_semval, k, wf, beta,    # lambda_C = 0
                gamma_1, gamma_2, epsilon,
            )
            if empirical is None:
                numpyro.sample("obs", dist.Categorical(probs=probs))
            else:
                numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)
    return model


likelihood_function_v5b_hier = _make_v5b_model(
    color_semval=0.971, form_semval=0.50, k=0.5, wf=1.0,
)
```

- [ ] **Step 4: Smoke-test that the models load and sample one trace step**

```python
import jax
import numpyro
from numpyro.infer import NUTS, MCMC
from helper import import_dataset
from modelSpecification import likelihood_function_v5_hier

data = import_dataset()
mcmc = MCMC(NUTS(likelihood_function_v5_hier), num_warmup=2, num_samples=2, num_chains=1, progress_bar=False)
mcmc.run(
    jax.random.PRNGKey(0),
    states=data["states_train"][:32],
    empirical=data["empirical_train"][:32],
    participant_idx=data["participant_idx"][:32],
    n_participants=int(data["participant_idx"].max()) + 1,
    is_colour_sufficient=data["is_colour_sufficient"][:32],
)
print(mcmc.get_samples().keys())
```

Expected: keys include `alpha_D`, `alpha_C`, `alpha_F`, `lambda_C`, `gamma_1`, `gamma_2`, `epsilon`, `tau`, `delta`. No exceptions.

- [ ] **Step 5: Commit**

```bash
git add 05-modelling-production-data/modelSpecification.py
git commit -m "feat(model): add v5, v5a, v5b likelihood factories"
```

---

## Phase 4 — Identifiability prior-predictive checks

### Task 5: Prior-predictive sweep for λ_C

**Files:**
- Create: `05-modelling-production-data/scripts/prior_predictive_v5.py`

- [ ] **Step 1: Write the sweep script**

```python
"""Identifiability pre-checks for v5 (memo §4)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp
import numpy as np
import pandas as pd
from helper import import_dataset, FLAT_TO_CATEGORIES
from modelSpecification import jitted_speaker_v5_hier

data = import_dataset()
states  = data["states_train"]
flag    = data["is_colour_sufficient"]
df      = data["df"] if "df" in data else None  # adapt to actual return
N       = states.shape[0]

# Posterior means from ext-v1 (memo Table §4.1).
alpha_D = jnp.full((N,), 6.94)
alpha_C = jnp.full((N,), 1.84)
alpha_F = jnp.full((N,), 1.88)
gamma   = 2.32
epsilon = 0.23
beta    = jnp.exp(2.05)

# Sweep lambda_C; check C-proportion per condition family.
lambda_grid = [0.0, 1.0, 2.0, 3.0]
rows = []
for lam in lambda_grid:
    probs = jitted_speaker_v5_hier(
        states, flag, alpha_D, alpha_C, alpha_F,
        lam, 0.971, 0.50, 0.5, 1.0, beta, gamma, gamma, epsilon,
    )
    # C is utterance code 5 in FLAT_TO_CATEGORIES.
    p_C = np.asarray(probs[:, 5])
    rows.append({
        "lambda_C": lam,
        "p_C_in_csuff": p_C[np.asarray(flag) == 1].mean(),
        "p_C_in_other": p_C[np.asarray(flag) == 0].mean(),
    })
print(pd.DataFrame(rows).to_string(index=False))
```

- [ ] **Step 2: Run and inspect**

Run: `cd 05-modelling-production-data && python scripts/prior_predictive_v5.py`
Expected output: monotonically increasing `p_C_in_csuff` with `lambda_C`; `p_C_in_other` flat across the grid (within ~0.001).

If `p_C_in_other` moves: the condition gate is not isolating the boost — investigate `is_colour_sufficient` propagation through the vmap.

- [ ] **Step 3: Add a γ_1 × γ_2 sweep block to the same script**

Append to the script:

```python
print("\n--- gamma_1 × gamma_2 sweep ---")
gamma_grid = [-1.0, 0.0, 1.0, 2.0]
rows = []
for g1 in gamma_grid:
    for g2 in gamma_grid:
        probs = jitted_speaker_v5_hier(
            states, flag, alpha_D, alpha_C, alpha_F,
            0.0, 0.971, 0.50, 0.5, 1.0, beta, g1, g2, epsilon,
        )
        # Mean utterance length = sum_i length(utt_i) * p(utt_i).
        UTT_LEN = jnp.array([len(s) for s in FLAT_TO_CATEGORIES.values()])
        mean_len = float(jnp.mean(jnp.sum(probs * UTT_LEN[None, :], axis=1)))
        rows.append({"gamma_1": g1, "gamma_2": g2, "mean_len": mean_len})
print(pd.DataFrame(rows).pivot(index="gamma_1", columns="gamma_2", values="mean_len").to_string())
```

- [ ] **Step 4: Run and inspect**

Run: `cd 05-modelling-production-data && python scripts/prior_predictive_v5.py`
Expected: the (γ_1, γ_2) → mean-length surface should not be a single ridge — moving γ_1 with γ_2 fixed should change mean length differently from moving γ_2 with γ_1 fixed. If the table shows the surface depends only on `γ_1 + γ_2`, the parameters are not separately identifiable on this metric — but data carries more information than mean length, so this is a soft check only. Document the surface in the memo.

- [ ] **Step 5: Commit**

```bash
git add 05-modelling-production-data/scripts/prior_predictive_v5.py
git commit -m "test(v5): add prior-predictive identifiability checks for lambda_C and gamma_1, gamma_2"
```

---

## Phase 5 — Inference

### Task 6: Wire v5 variants into `run_inference.py`

**Files:**
- Modify: `05-modelling-production-data/run_inference.py`
- Modify: `05-modelling-production-data/run_inference.sh`

- [ ] **Step 1: Inspect the existing model registry**

Run: `grep -n "model_name\|MODEL_REGISTRY\|likelihood_function" 05-modelling-production-data/run_inference.py | head -20`
Expected: identify how models are dispatched (likely a dict mapping a CLI string to a likelihood callable).

- [ ] **Step 2: Add the three v5 entries**

Pattern follows whatever the file already does. If there's a dict like `MODEL_REGISTRY = {"reported": likelihood_function_reported_hier, "ext_v1": likelihood_function_incremental_speaker_hier, ...}`, add:

```python
"v5":  likelihood_function_v5_hier,
"v5a": likelihood_function_v5a_hier,
"v5b": likelihood_function_v5b_hier,
```

Also locate the `mcmc.run(...)` call site and ensure `is_colour_sufficient=data["is_colour_sufficient"]` is passed when the selected model is one of v5/v5a/v5b. If the runner already passes `**data` or similar, no change needed beyond adding the dataset key.

- [ ] **Step 3: Add invocations to `run_inference.sh`**

Append three blocks following the existing pattern (model name + warmup + samples + chains + output filename). Output NC files:

- `inference_data/mcmc_results_v5_warmup2000_samples1000_chains4.nc`
- `inference_data/mcmc_results_v5a_warmup2000_samples1000_chains4.nc`
- `inference_data/mcmc_results_v5b_warmup2000_samples1000_chains4.nc`

- [ ] **Step 4: Local smoke run (CPU, tiny config)**

Run on local machine with `num_warmup=10, num_samples=10, num_chains=1` for the `v5` model only. Expected: completes in < 5 min, produces an NC file, no NaNs in the trace.

- [ ] **Step 5: Commit**

```bash
git add 05-modelling-production-data/run_inference.py 05-modelling-production-data/run_inference.sh
git commit -m "feat(inference): wire v5, v5a, v5b into model registry and run script"
```

### Task 7: Remote inference

**Files:** none modified locally; runs on remote server per the `server-workflow` skill.

- [ ] **Step 1: Push commits to remote**

```bash
git push origin main
```

- [ ] **Step 2: SSH and deploy**

Follow `.github/skills/server-workflow/SKILL.md`. Pull latest, activate env, ensure no other GPU job is running (`nvidia-smi`).

- [ ] **Step 3: Run v5 inference**

Per memo, use `JAX_PLATFORMS='' XLA_FLAGS=''` for GPU mode. Run:

```bash
JAX_PLATFORMS='' XLA_FLAGS='' python run_inference.py --model v5 --warmup 2000 --samples 1000 --chains 4 \
    > logs/v5_$(date +%Y%m%d_%H%M).log 2>&1 &
```

Wait for completion (~3–5 hours at 5 it/s on full data per memo). Monitor with `tail -f logs/v5_*.log`.

- [ ] **Step 4: Run v5a and v5b sequentially**

Same pattern. Do not parallelise on the same GPU (memo: "don't run 2 jobs simultaneously").

- [ ] **Step 5: Pull NC files back**

```bash
scp server:~/.../inference_data/mcmc_results_v5*_warmup2000_samples1000_chains4.nc \
    05-modelling-production-data/inference_data/
```

- [ ] **Step 6: Verify samples**

In a Python REPL:

```python
import arviz as az
for tag in ["v5", "v5a", "v5b"]:
    idata = az.from_netcdf(f"05-modelling-production-data/inference_data/mcmc_results_{tag}_warmup2000_samples1000_chains4.nc")
    summary = az.summary(idata, var_names=["alpha_D", "alpha_C", "alpha_F", "lambda_C", "gamma_1", "gamma_2", "epsilon", "tau"], filter_vars="like")
    print(f"\n=== {tag} ===")
    print(summary)
    n_div = int(idata.sample_stats["diverging"].sum())
    print(f"Divergences: {n_div}")
```

Expected: `r_hat ≤ 1.05` for all parameters; divergences = 0 (or very few).

If divergences > 10 or r_hat > 1.1: investigate — likely identifiability problem (e.g., γ_1/γ_2 ridge). Tighten priors on the offending parameter (Normal(0, 0.5)) and re-run.

- [ ] **Step 7: Commit NC files**

```bash
git add 05-modelling-production-data/inference_data/mcmc_results_v5*_warmup2000_samples1000_chains4.nc
git commit -m "data(v5): add MCMC traces for v5, v5a, v5b inference"
```

---

## Phase 6 — Posterior analysis and memo update

### Task 8: LOO-CV across five models

**Files:**
- Modify: `05-modelling-production-data/posterior_analysis.py`

- [ ] **Step 1: Add a 5-model LOO comparison block**

```python
import arviz as az

NC_FILES = {
    "baseline":  "inference_data/mcmc_results_reported_top_warmup2000_samples1000_chains4.nc",  # confirm exact filename
    "ext_v1":    "inference_data/mcmc_results_incremental_speaker_warmup2000_samples1000_chains4.nc",  # confirm exact filename
    "v5a":       "inference_data/mcmc_results_v5a_warmup2000_samples1000_chains4.nc",
    "v5b":       "inference_data/mcmc_results_v5b_warmup2000_samples1000_chains4.nc",
    "v5":        "inference_data/mcmc_results_v5_warmup2000_samples1000_chains4.nc",
}

idatas = {tag: az.from_netcdf(path) for tag, path in NC_FILES.items()}
comparison = az.compare(idatas, ic="loo", method="stacking")
print(comparison)
comparison.to_csv("results/v5_loo_comparison.csv")
```

Verify NC filenames against `inference_data/` listing first; the `baseline` and `ext_v1` filenames in the dict are placeholders to confirm.

- [ ] **Step 2: Run and verify against success criteria**

Run: `cd 05-modelling-production-data && python -c "exec(open('posterior_analysis.py').read())"` (or whatever the file's existing entry pattern is).

Check (spec §5):
- `loo[v5] - loo[baseline] ≥ +900`
- `loo[v5] - loo[ext_v1] ≥ +20`
- `loo[v5a] - loo[ext_v1] ≥ +10` and `loo[v5b] - loo[ext_v1] ≥ +10`

If the +20 ablation criterion fails: report and discuss in the memo; do not adopt v5 over the simpler ablation.

- [ ] **Step 3: Commit results**

```bash
git add 05-modelling-production-data/posterior_analysis.py 05-modelling-production-data/results/v5_loo_comparison.csv
git commit -m "feat(analysis): add 5-model LOO comparison for v5 ablation"
```

### Task 9: Residual diff table vs memo §5.3

**Files:**
- Modify: `05-modelling-production-data/posterior_analysis.py`

- [ ] **Step 1: Generate posterior predictive proportions per condition**

Compute `mean_pred[condition, utterance_type]` from the v5 posterior using the same approach as the existing PPC code (search for `posterior_predictive` or `ppc` in `posterior_analysis.py`). Compare against the empirical proportions in the same conditions.

- [ ] **Step 2: Print the 7-row residual table**

For the 7 rows in memo §5.3:

```python
RESIDUALS_TO_CHECK = [
    ("colour-sufficient", "high", "C",   0.716),
    ("size-sufficient",   "low",  "SF",  0.365),
    ("colour-sufficient", "low",  "C",   0.628),
    ("both-necessary",    "low",  "C",   0.009),
    ("size-sufficient",   "high", "SFC", 0.022),
    ("both-necessary",    "high", "C",   0.000),
    ("size-sufficient",   "low",  "SFC", 0.032),
]
# For each: print empirical vs ext_v1 vs v5 prediction, and Δ.
```

Success: at least 5 of 7 rows have `|Δ_v5| < 0.10`.

- [ ] **Step 3: Save residual table CSV**

```bash
results/v5_residuals_vs_memo.csv
```

- [ ] **Step 4: Commit**

```bash
git add 05-modelling-production-data/posterior_analysis.py 05-modelling-production-data/results/v5_residuals_vs_memo.csv
git commit -m "feat(analysis): add residual-resolution check for v5 against memo §5.3 misfits"
```

### Task 10: PPC barplot and correlation plot for v5

**Files:**
- Modify: `05-modelling-production-data/posterior_analysis.py`
- Create: `05-modelling-production-data/figures/v5/production_ppc_barplot_v5.pdf`
- Create: `05-modelling-production-data/figures/v5/production_correlation_v5.pdf`

- [ ] **Step 1: Reuse existing PPC plotting code**

The existing memo has `production_ppc_barplot_extended_v1.png` and `production_correlation_extended_v1.png`. Locate the function that produced them in `posterior_analysis.py`, parameterise on the model tag, and produce the v5 versions.

- [ ] **Step 2: Save figures**

PDF and PNG variants under `figures/v5/`.

- [ ] **Step 3: Commit**

```bash
git add 05-modelling-production-data/figures/v5/
git commit -m "feat(figures): add v5 PPC barplot and correlation plot"
```

### Task 11: Update memo with v5 results

**Files:**
- Modify: `10-writing/extended_production_model_memo.md`

- [ ] **Step 1: Add new section §6 "Extension v5 — Condition-gated colour salience + saturating length bias"**

Structure (~300 words, mirroring §4.1):

1. **Mechanism**: λ_C (condition-gated, only in colour-sufficient) and (γ_1, γ_2) saturating length bias. Why these targets (residual decomposition by length).
2. **Parameter table**: priors, posterior means, ESS, r̂.
3. **LOO comparison table**: 5-model LOO results.
4. **Residual resolution table**: 7 misfits before/after.
5. **PPC figure** referenced.
6. **Identifiability findings**: from the prior-predictive checks.
7. **Interpretation**: what does the data say about λ_C sign and magnitude? What does γ_2 vs γ_1 indicate about saturation?
8. **Remaining gaps**: which of the 7 misfits are not resolved; what this implies for the mixture-model direction (existing §6).

- [ ] **Step 2: Renumber existing §6 (Open questions) → §7**

- [ ] **Step 3: Compile draft.tex if cited there**

If `extended_production_model_memo.md` is referenced from `draft.tex`, no recompile needed (memo is internal). Otherwise nothing to do.

- [ ] **Step 4: Commit**

```bash
git add 10-writing/extended_production_model_memo.md
git commit -m "docs(memo): add §6 documenting extension v5 results"
```

---

## Self-review checklist

After implementing, verify before declaring complete:

- [ ] All 5 LOO comparisons run; CSV saved.
- [ ] At least 5 of 7 memo §5.3 misfits resolved (|Δ| < 0.10).
- [ ] ε posterior mean < 0.20 (drop from 0.23).
- [ ] No divergences in any of v5, v5a, v5b traces.
- [ ] r̂ ≤ 1.05 for all sampled parameters.
- [ ] Sanity check (Task 3) passes: v5 with neutral params == ext-v1.
- [ ] Identifiability: λ_C only affects colour-sufficient predictions; γ_1, γ_2 not on a strict ridge.
- [ ] Memo §6 written; references match figure paths.
- [ ] Spec §6 risks reviewed against actual outcomes; if a risk fired, the corresponding fallback was applied and documented.
