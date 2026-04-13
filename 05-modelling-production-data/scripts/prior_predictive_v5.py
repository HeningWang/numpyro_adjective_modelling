"""Identifiability pre-checks for v5 (memo §4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp
import numpy as np
import pandas as pd

from helper import import_dataset, FLAT_TO_CATEGORIES
from modelSpecification import jitted_speaker_v5_hier


data    = import_dataset()
states  = data["states_train"]
flag    = data["is_colour_sufficient"]
N       = states.shape[0]

# Posterior means from ext-v1 (memo §4.1 table).
alpha_D = jnp.full((N,), 6.94)
alpha_C = jnp.full((N,), 1.84)
alpha_F = jnp.full((N,), 1.88)
gamma   = 2.32
epsilon = 0.23
beta    = float(jnp.exp(2.05))

print("=== Sweep 1: lambda_C identifiability ===")
print("With alpha_C fixed at v1 posterior mean, sweep lambda_C and check")
print("that p(C) increases in colour-sufficient trials but stays flat elsewhere.\n")

lambda_grid = [0.0, 1.0, 2.0, 3.0]
flag_np = np.asarray(flag)
rows = []
for lam in lambda_grid:
    probs = jitted_speaker_v5_hier(
        states, flag, alpha_D, alpha_C, alpha_F,
        lam, 0.971, 0.50, 0.5, 1.0, beta, gamma, gamma, epsilon,
    )
    p_C = np.asarray(probs[:, 5])
    rows.append({
        "lambda_C": lam,
        "p_C_in_csuff": float(p_C[flag_np == 1].mean()),
        "p_C_in_other": float(p_C[flag_np == 0].mean()),
    })
df_lam = pd.DataFrame(rows)
print(df_lam.to_string(index=False))

# Sanity: csuff must increase monotonically with lambda_C; other must stay flat (< 1e-3 spread).
csuff_diffs = np.diff(df_lam["p_C_in_csuff"].values)
other_spread = df_lam["p_C_in_other"].max() - df_lam["p_C_in_other"].min()
print(f"\nMonotonicity (csuff): all_increasing = {bool((csuff_diffs >= 0).all())}")
print(f"Other-condition spread across lambda_C grid: {other_spread:.2e}")
if other_spread > 1e-3:
    print("WARNING: lambda_C is bleeding into non-colour-sufficient trials. Check the gate.")

print("\n=== Sweep 2: gamma_1 x gamma_2 surface ===")
print("Mean utterance length over the (gamma_1, gamma_2) grid.")
print("If the surface depends only on (gamma_1 + gamma_2), parameters are not separately identifiable on this metric.\n")

UTT_LEN = jnp.array(
    [len(s) for s in FLAT_TO_CATEGORIES.values()],
    dtype=jnp.float32,
)

gamma_grid = [-1.0, 0.0, 1.0, 2.0]
rows2 = []
for g1 in gamma_grid:
    for g2 in gamma_grid:
        probs = jitted_speaker_v5_hier(
            states, flag, alpha_D, alpha_C, alpha_F,
            0.0, 0.971, 0.50, 0.5, 1.0, beta, g1, g2, epsilon,
        )
        mean_len = float(jnp.mean(jnp.sum(probs * UTT_LEN[None, :], axis=1)))
        rows2.append({"gamma_1": g1, "gamma_2": g2, "mean_len": mean_len})

df_g = pd.DataFrame(rows2)
print(df_g.pivot(index="gamma_1", columns="gamma_2", values="mean_len").to_string())

print("\nNote: mean length is a soft summary — full posterior identifiability ")
print("depends on the full categorical likelihood, which carries more information.")
