import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import modelSpecification as ms


STATES = jnp.asarray(
    [[0.8, 1, 1], [0.5, 0, 1], [0.2, 1, 0]],
    dtype=jnp.float32,
)


BASE_KW = dict(
    sufficient_dim=jnp.int32(1),
    has_one_word_solution=jnp.float32(0.0),
    is_sharp=jnp.float32(0.0),
    alpha=jnp.float32(3.0),
    beta_order=jnp.float32(1.0),
    lambda_frontload=jnp.float32(0.8),
    gamma_uncertainty_len=jnp.float32(0.8),
    color_semval=0.59,
    form_semval=0.50,
    k=0.5,
    wf=0.6856,
    epsilon=0.01,
)

PRINCIPLED_KW = dict(
    sufficient_dim=jnp.int32(1),
    has_one_word_solution=jnp.float32(0.0),
    is_sharp=jnp.float32(0.0),
    alpha=jnp.float32(3.0),
    beta_order=jnp.float32(1.0),
    lambda_salience=jnp.float32(0.8),
    gamma_uncertainty_len=jnp.float32(4.0),
    color_semval=0.59,
    form_semval=0.50,
    k=0.5,
    wf=0.6856,
    epsilon=0.01,
)


def test_simplified_speaker_is_simplex_and_mechanisms_change_output():
    base = np.asarray(ms.incremental_speaker_simplified(STATES, **BASE_KW))
    no_frontload = np.asarray(
        ms.incremental_speaker_simplified(
            STATES,
            **{**BASE_KW, "lambda_frontload": jnp.float32(0.0)},
        )
    )
    no_length = np.asarray(
        ms.incremental_speaker_simplified(
            STATES,
            **{**BASE_KW, "gamma_uncertainty_len": jnp.float32(0.0)},
        )
    )

    assert np.all(base >= 0.0)
    assert np.allclose(base.sum(), 1.0, atol=1e-4)
    assert not np.allclose(base, no_frontload, atol=1e-4)
    assert not np.allclose(base, no_length, atol=1e-4)


def test_simplified_models_register_for_hierarchical_inference():
    import run_inference as ri

    for key in (
        "simplified_lm_resid",
        "simplified_lm_raw",
        "simplified_hand_order",
        "simplified_no_frontload",
        "simplified_no_uncertainty_len",
        "simplified_no_order",
    ):
        assert key in ri.HIER_MODELS


def test_principled_speaker_is_simplex_and_uses_soft_features():
    base = np.asarray(ms.incremental_speaker_principled(STATES, **PRINCIPLED_KW))
    no_salience = np.asarray(
        ms.incremental_speaker_principled(
            STATES,
            **{**PRINCIPLED_KW, "lambda_salience": jnp.float32(0.0)},
        )
    )
    no_uncertainty = np.asarray(
        ms.incremental_speaker_principled(
            STATES,
            **{**PRINCIPLED_KW, "gamma_uncertainty_len": jnp.float32(0.0)},
        )
    )

    assert np.all(base >= 0.0)
    assert np.allclose(base.sum(), 1.0, atol=1e-4)
    assert not np.allclose(base, no_salience, atol=1e-4)
    assert not np.allclose(base, no_uncertainty, atol=1e-4)


def test_principled_models_register_for_hierarchical_inference():
    import run_inference as ri

    for key in (
        "principled",
        "principled_no_order",
        "principled_no_salience",
        "principled_no_uncertainty_len",
    ):
        assert key in ri.HIER_MODELS


if __name__ == "__main__":
    test_simplified_speaker_is_simplex_and_mechanisms_change_output()
    test_simplified_models_register_for_hierarchical_inference()
    test_principled_speaker_is_simplex_and_uses_soft_features()
    test_principled_models_register_for_hierarchical_inference()
    print("PASS simplified model tests")
