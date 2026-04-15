"""Sanity check: with lambda_C=0 and gamma_2=gamma_1, v5 must equal ext-v1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp
import numpy as np
from helper import import_dataset
from modelSpecification import jitted_speaker_hier, jitted_speaker_v5_hier

data   = import_dataset()
states = data["states_train"][:8]                     # 8-trial subset
flag   = data["is_colour_sufficient"][:8]
N      = states.shape[0]

# Identical alpha per trial for both models (ext-v1 posterior means).
alpha_D = jnp.full((N,), 6.94)
alpha_C = jnp.full((N,), 1.84)
alpha_F = jnp.full((N,), 1.88)

color_semval = 0.971
form_semval  = 0.50
k            = 0.5
wf           = 1.0
beta         = jnp.exp(2.05)
gamma        = 2.32
epsilon      = 0.23

probs_v1 = jitted_speaker_hier(
    states, alpha_D, alpha_C, alpha_F,
    color_semval, form_semval, k, wf, beta, gamma, epsilon,
)

# v5 with lambda_C=0 and gamma_2=gamma_1 and deltas=0 should match v1's softmax output exactly.
probs_v5 = jitted_speaker_v5_hier(
    states, flag,
    alpha_D, alpha_C, alpha_F,
    0.0, color_semval, form_semval, k, wf, beta,
    gamma, gamma, 0.0, 0.0, 0.0, epsilon,
)

max_diff = float(jnp.max(jnp.abs(probs_v1 - probs_v5)))
print(f"Max abs diff (v1 vs v5 with neutral params): {max_diff:.2e}")
assert max_diff < 1e-5, f"v5 does not reduce to v1: max diff {max_diff:.2e}"

# Now turn lambda_C on; only colour-sufficient trials should change.
probs_v5_boosted = jitted_speaker_v5_hier(
    states, flag,
    alpha_D, alpha_C, alpha_F,
    2.0, color_semval, form_semval, k, wf, beta,
    gamma, gamma, 0.0, 0.0, 0.0, epsilon,
)
diff_per_trial = jnp.max(jnp.abs(probs_v5 - probs_v5_boosted), axis=-1)
print("Per-trial max diff with lambda_C=2:")
for i in range(N):
    print(f"  trial {i}: flag={float(flag[i]):.0f}, diff={float(diff_per_trial[i]):.4f}")

flag_np = np.asarray(flag)
diff_np = np.asarray(diff_per_trial)
non_csuff_max = float(diff_np[flag_np == 0].max()) if (flag_np == 0).any() else 0.0
assert non_csuff_max < 1e-5, f"lambda_C affected non-colour-sufficient trials: max diff {non_csuff_max:.2e}"
print("OK: lambda_C only affects colour-sufficient trials.")
