import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax.numpy as jnp
try:
    import pytest  # noqa: F401  (optional — analysis env may lack it)
except ModuleNotFoundError:
    pytest = None
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import modelSpecification as ms

# 3 objects x (size, color, form); referent is index 0.
STATES = jnp.asarray([[0.8, 1, 1], [0.5, 0, 1], [0.2, 1, 0]], dtype=jnp.float32)
# The inner speaker fn is normally called under vmap; the per-trial-axis
# args (sufficient_dim, has_one_word_solution, is_sharp, alpha_*) must be
# jnp scalars (the fn does e.g. (sufficient_dim == 0).astype(...)).
KW = dict(sufficient_dim=jnp.int32(1),
          has_one_word_solution=jnp.float32(0.0), is_sharp=jnp.float32(1.0),
          alpha_D=jnp.float32(4.0), alpha_C=jnp.float32(2.0),
          alpha_F=jnp.float32(0.0), lambda_suff=1.5,
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
    assert not np.allclose(rec, stat, atol=1e-3)        # frozen != recursive


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
    assert not np.allclose(g_lit, inc, atol=1e-3)       # global != incremental


def test_four_cells_register_and_run():
    import run_inference as ri
    keys = [f"contextual_pcalpha_canon_parsimony_2x2_{c}"
            for c in ("inc_rec", "inc_static", "glob_rec", "glob_static")]
    for k in keys:
        assert k in ri.HIER_MODELS, k


if __name__ == "__main__":
    # Runnable without pytest (the analysis env lacks it). Each test is run
    # independently so a NameError/AttributeError in a not-yet-implemented
    # task does not mask an earlier passing test.
    import traceback
    selected = sys.argv[1:] or [
        "test_recursive_flag_changes_output_and_keeps_simplex",
        "test_global_speaker_simplex_and_distinct",
        "test_four_cells_register_and_run",
    ]
    rc = 0
    for name in selected:
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception:
            rc = 1
            print(f"FAIL {name}")
            traceback.print_exc()
    sys.exit(rc)
