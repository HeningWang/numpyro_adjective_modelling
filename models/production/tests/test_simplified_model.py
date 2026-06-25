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

COLOR_SALIENT_STATES = jnp.asarray(
    [
        [0.8, 1, 1],
        [0.7, 0, 1],
        [0.6, 0, 1],
        [0.5, 0, 0],
        [0.4, 0, 1],
        [0.3, 0, 0],
    ],
    dtype=jnp.float32,
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


def test_size_sharp_models_register_for_hierarchical_inference():
    import posterior_analysis as pa
    import run_inference as ri

    for key in (
        "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_static_fixedeps",
    ):
        assert key in ri.HIER_MODELS
        assert key in pa.SIMPLIFIED_MODELS


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


def test_principled_salience_stop_favors_single_salient_adjective():
    base_kw = {
        **PRINCIPLED_KW,
        "gamma_uncertainty_len": jnp.float32(0.0),
        "order_scores": jnp.zeros(ms.n_utt),
    }
    no_stop = np.asarray(
        ms.incremental_speaker_principled(
            COLOR_SALIENT_STATES,
            **{**base_kw, "rho_salience_stop": jnp.float32(0.0)},
        )
    )
    with_stop = np.asarray(
        ms.incremental_speaker_principled(
            COLOR_SALIENT_STATES,
            **{**base_kw, "rho_salience_stop": jnp.float32(2.0)},
        )
    )

    assert with_stop[5] > no_stop[5]  # C
    assert with_stop[[6, 7, 8, 9]].sum() < no_stop[[6, 7, 8, 9]].sum()


def test_principled_planned_prefix_zero_scale_recovers_base_and_can_shift_output():
    base_kw = {
        **PRINCIPLED_KW,
        "has_one_word_solution": jnp.float32(1.0),
        "gamma_uncertainty_len": jnp.float32(0.0),
        "rho_salience_stop": jnp.float32(0.5),
    }
    base = np.asarray(ms.incremental_speaker_principled(COLOR_SALIENT_STATES, **base_kw))
    planned_zero = np.asarray(
        ms.incremental_speaker_principled_planned_prefix(
            COLOR_SALIENT_STATES,
            **{**base_kw, "planning_scale": jnp.float32(0.0)},
        )
    )
    planned = np.asarray(
        ms.incremental_speaker_principled_planned_prefix(
            COLOR_SALIENT_STATES,
            **{**base_kw, "planning_scale": jnp.float32(1.0)},
        )
    )

    assert np.all(planned >= 0.0)
    assert np.allclose(planned.sum(), 1.0, atol=1e-4)
    assert np.allclose(base, planned_zero, atol=1e-5)
    assert not np.allclose(base, planned, atol=1e-4)


def test_principled_response_policy_zero_recovers_base_and_shifts_output():
    base_kw = {
        **PRINCIPLED_KW,
        "has_one_word_solution": jnp.float32(1.0),
        "gamma_uncertainty_len": jnp.float32(0.0),
        "rho_salience_stop": jnp.float32(0.5),
    }
    base = np.asarray(ms.incremental_speaker_principled(COLOR_SALIENT_STATES, **base_kw))
    response_zero = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "is_colour_sufficient": jnp.float32(1.0),
                "lambda_sufficient_single": jnp.float32(0.0),
                "lambda_reliability_form": jnp.float32(0.0),
            },
        )
    )
    sufficient_single = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "is_colour_sufficient": jnp.float32(1.0),
                "lambda_sufficient_single": jnp.float32(1.5),
                "lambda_reliability_form": jnp.float32(0.0),
            },
        )
    )
    reliability_form = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "sufficient_dim": jnp.int32(-1),
                "has_one_word_solution": jnp.float32(0.0),
                "is_colour_sufficient": jnp.float32(0.0),
                "lambda_sufficient_single": jnp.float32(0.0),
                "lambda_reliability_form": jnp.float32(1.5),
            },
        )
    )
    sufficient_form_pair = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "is_colour_sufficient": jnp.float32(1.0),
                "lambda_sufficient_single": jnp.float32(0.0),
                "lambda_reliability_form": jnp.float32(0.0),
                "lambda_sufficient_form_pair": jnp.float32(1.5),
            },
        )
    )
    three_word_penalty = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "is_colour_sufficient": jnp.float32(1.0),
                "lambda_sufficient_single": jnp.float32(0.0),
                "lambda_reliability_form": jnp.float32(0.0),
                "lambda_three_word_penalty": jnp.float32(1.5),
            },
        )
    )

    f_present = np.asarray(ms.F_PRESENT_15, dtype=bool)
    three_words = np.asarray(ms.N_WORDS == 3.0, dtype=bool)
    assert np.all(response_zero >= 0.0)
    assert np.allclose(response_zero.sum(), 1.0, atol=1e-4)
    assert np.allclose(base, response_zero, atol=1e-5)
    assert sufficient_single[5] > base[5]  # C is sufficient in base_kw.
    assert reliability_form[f_present].sum() > base[f_present].sum()
    assert sufficient_form_pair[8] > base[8]  # CF adds form to sufficient C.
    assert three_word_penalty[three_words].sum() < base[three_words].sum()


def test_principled_response_policy_can_suppress_sharp_one_word_form():
    base_kw = {
        **PRINCIPLED_KW,
        "sufficient_dim": jnp.int32(0),
        "has_one_word_solution": jnp.float32(1.0),
        "gamma_uncertainty_len": jnp.float32(0.0),
        "rho_salience_stop": jnp.float32(0.5),
        "is_colour_sufficient": jnp.float32(0.0),
        "lambda_sufficient_single": jnp.float32(0.0),
        "lambda_reliability_form": jnp.float32(1.5),
        "lambda_sufficient_form_pair": jnp.float32(1.5),
        "lambda_three_word_penalty": jnp.float32(0.0),
    }
    sharp_without_suppression = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "is_sharp": jnp.float32(1.0),
                "lambda_sharp_form_suppression": jnp.float32(0.0),
            },
        )
    )
    sharp_with_suppression = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "is_sharp": jnp.float32(1.0),
                "lambda_sharp_form_suppression": jnp.float32(2.0),
            },
        )
    )
    blurred_with_suppression = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "is_sharp": jnp.float32(0.0),
                "lambda_sharp_form_suppression": jnp.float32(2.0),
            },
        )
    )
    blurred_without_suppression = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "is_sharp": jnp.float32(0.0),
                "lambda_sharp_form_suppression": jnp.float32(0.0),
            },
        )
    )

    f_present = np.asarray(ms.F_PRESENT_15, dtype=bool)
    assert np.allclose(sharp_with_suppression.sum(), 1.0, atol=1e-4)
    assert sharp_with_suppression[f_present].sum() < sharp_without_suppression[f_present].sum()
    assert np.allclose(blurred_with_suppression, blurred_without_suppression, atol=1e-5)


def test_principled_response_policy_size_sharp_targets_d_not_all_form():
    base_kw = {
        **PRINCIPLED_KW,
        "sufficient_dim": jnp.int32(0),
        "has_one_word_solution": jnp.float32(1.0),
        "is_sharp": jnp.float32(1.0),
        "gamma_uncertainty_len": jnp.float32(0.0),
        "rho_salience_stop": jnp.float32(0.5),
        "is_colour_sufficient": jnp.float32(0.0),
        "lambda_sufficient_single": jnp.float32(0.0),
        "lambda_reliability_form": jnp.float32(1.5),
        "lambda_sufficient_form_pair": jnp.float32(1.5),
        "lambda_three_word_penalty": jnp.float32(0.0),
        "lambda_sharp_form_suppression": jnp.float32(0.0),
    }
    baseline = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "lambda_size_sharp_single_bonus": jnp.float32(0.0),
                "lambda_size_sharp_form_pair_penalty": jnp.float32(0.0),
            },
        )
    )
    size_sharp = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "lambda_size_sharp_single_bonus": jnp.float32(2.0),
                "lambda_size_sharp_form_pair_penalty": jnp.float32(2.0),
            },
        )
    )
    blurred = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "is_sharp": jnp.float32(0.0),
                "lambda_size_sharp_single_bonus": jnp.float32(2.0),
                "lambda_size_sharp_form_pair_penalty": jnp.float32(2.0),
            },
        )
    )
    blurred_base = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "is_sharp": jnp.float32(0.0),
                "lambda_size_sharp_single_bonus": jnp.float32(0.0),
                "lambda_size_sharp_form_pair_penalty": jnp.float32(0.0),
            },
        )
    )
    colour_sufficient = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "sufficient_dim": jnp.int32(1),
                "is_colour_sufficient": jnp.float32(1.0),
                "lambda_size_sharp_single_bonus": jnp.float32(2.0),
                "lambda_size_sharp_form_pair_penalty": jnp.float32(2.0),
            },
        )
    )
    colour_sufficient_base = np.asarray(
        ms.incremental_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "sufficient_dim": jnp.int32(1),
                "is_colour_sufficient": jnp.float32(1.0),
                "lambda_size_sharp_single_bonus": jnp.float32(0.0),
                "lambda_size_sharp_form_pair_penalty": jnp.float32(0.0),
            },
        )
    )

    assert np.allclose(size_sharp.sum(), 1.0, atol=1e-4)
    assert np.allclose(blurred, blurred_base, atol=1e-5)
    assert np.allclose(colour_sufficient, colour_sufficient_base, atol=1e-5)
    assert size_sharp[0] > baseline[0]  # D
    assert size_sharp[3] < baseline[3]  # DF
    assert size_sharp[11] < baseline[11]  # FD
    assert np.isclose(size_sharp[8] / baseline[8], size_sharp[2] / baseline[2], rtol=1e-5)


def test_global_principled_response_policy_zero_recovers_base_and_shifts_output():
    base_kw = {
        **PRINCIPLED_KW,
        "has_one_word_solution": jnp.float32(1.0),
        "gamma_uncertainty_len": jnp.float32(0.0),
        "rho_salience_stop": jnp.float32(0.5),
    }
    base = np.asarray(ms.global_speaker_principled(COLOR_SALIENT_STATES, **base_kw))
    response_zero = np.asarray(
        ms.global_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "is_colour_sufficient": jnp.float32(1.0),
                "lambda_sufficient_single": jnp.float32(0.0),
                "lambda_reliability_form": jnp.float32(0.0),
            },
        )
    )
    sufficient_single = np.asarray(
        ms.global_speaker_principled_response_policy(
            COLOR_SALIENT_STATES,
            **{
                **base_kw,
                "is_colour_sufficient": jnp.float32(1.0),
                "lambda_sufficient_single": jnp.float32(1.5),
                "lambda_reliability_form": jnp.float32(0.0),
            },
        )
    )

    assert np.all(response_zero >= 0.0)
    assert np.allclose(response_zero.sum(), 1.0, atol=1e-4)
    assert np.allclose(base, response_zero, atol=1e-5)
    assert sufficient_single[5] > base[5]  # C is sufficient in base_kw.


def test_global_principled_response_policy_jitted_batch_is_simplex():
    probs = np.asarray(
        ms.jitted_global_speaker_principled_response_policy_hier(
            jnp.stack([COLOR_SALIENT_STATES, COLOR_SALIENT_STATES]),
            jnp.asarray([1, 1], dtype=jnp.int32),
            jnp.asarray([1.0, 1.0], dtype=jnp.float32),
            jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            jnp.asarray([1.0, 0.0], dtype=jnp.float32),
            jnp.asarray([3.0, 2.5], dtype=jnp.float32),
            jnp.float32(1.0),
            jnp.float32(0.8),
            jnp.float32(0.5),
            jnp.float32(1.5),
            jnp.float32(1.5),
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
            0.59,
            0.50,
            0.50,
            0.6856,
            0.01,
            ms.LOG_LM_ORDER_ONLY_15,
            ms.BASE_VISUAL_SALIENCE,
            recursive=True,
        )
    )

    assert probs.shape == (2, ms.n_utt)
    assert np.all(probs >= 0.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_principled_base_salience_constants_are_fixed_sweep_inputs():
    default = np.asarray(ms.incremental_speaker_principled(STATES, **PRINCIPLED_KW))
    higher_color = np.asarray(
        ms.incremental_speaker_principled(
            STATES,
            **{
                **PRINCIPLED_KW,
                "base_visual_salience": jnp.asarray([0.0, 1.6, 0.25], dtype=jnp.float32),
            },
        )
    )

    assert np.allclose(default.sum(), 1.0, atol=1e-4)
    assert np.allclose(higher_color.sum(), 1.0, atol=1e-4)
    assert not np.allclose(default, higher_color, atol=1e-4)


def test_principled_2x2_architectures_are_simplex_and_distinct():
    inc_rec = np.asarray(ms.incremental_speaker_principled(STATES, **PRINCIPLED_KW))
    inc_static = np.asarray(
        ms.incremental_speaker_principled(
            STATES,
            **{**PRINCIPLED_KW, "recursive": False},
        )
    )
    glob_rec = np.asarray(ms.global_speaker_principled(COLOR_SALIENT_STATES, **PRINCIPLED_KW))
    glob_static = np.asarray(
        ms.global_speaker_principled(
            COLOR_SALIENT_STATES,
            **{**PRINCIPLED_KW, "recursive": False},
        )
    )

    for probs in (inc_rec, inc_static, glob_rec, glob_static):
        assert np.all(probs >= 0.0)
        assert np.allclose(probs.sum(), 1.0, atol=1e-4)

    assert not np.allclose(inc_rec, inc_static, atol=1e-4)
    assert not np.allclose(inc_rec, glob_rec, atol=1e-4)
    assert not np.allclose(glob_rec, glob_static, atol=1e-4)


def test_principled_models_register_for_hierarchical_inference():
    import run_inference as ri

    for key in (
        "principled",
        "principled_no_order",
        "principled_no_salience",
        "principled_no_uncertainty_len",
        "principled_salience_stop",
        "principled_salience_stop_regularized",
        "principled_salience_stop_strong_regularized",
        "principled_salience_stop_regularized_2x2_inc_rec",
        "principled_salience_stop_regularized_2x2_inc_static",
        "principled_salience_stop_regularized_plannedprefix_2x2_inc_rec",
        "principled_salience_stop_regularized_plannedprefix_2x2_inc_static",
        "principled_salience_stop_regularized_responsepolicy_2x2_inc_rec",
        "principled_salience_stop_regularized_responsepolicy_2x2_inc_static",
        "principled_salience_stop_regularized_responsepolicy_2x2_glob_rec",
        "principled_salience_stop_regularized_responsepolicy_2x2_glob_static",
        "principled_salience_stop_regularized_responsepolicy_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_2x2_inc_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_2x2_glob_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_2x2_glob_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static_fixedeps",
        "principled_salience_stop_regularized_2x2_glob_rec",
        "principled_salience_stop_regularized_2x2_glob_static",
        "principled_salience_stop_regularized_2x2_glob_rec_fixedeps",
        "principled_salience_stop_regularized_2x2_glob_static_fixedeps",
    ):
        assert key in ri.HIER_MODELS


if __name__ == "__main__":
    test_simplified_speaker_is_simplex_and_mechanisms_change_output()
    test_simplified_models_register_for_hierarchical_inference()
    test_size_sharp_models_register_for_hierarchical_inference()
    test_principled_speaker_is_simplex_and_uses_soft_features()
    test_principled_salience_stop_favors_single_salient_adjective()
    test_principled_planned_prefix_zero_scale_recovers_base_and_can_shift_output()
    test_principled_response_policy_zero_recovers_base_and_shifts_output()
    test_principled_response_policy_can_suppress_sharp_one_word_form()
    test_principled_response_policy_size_sharp_targets_d_not_all_form()
    test_global_principled_response_policy_zero_recovers_base_and_shifts_output()
    test_global_principled_response_policy_jitted_batch_is_simplex()
    test_principled_base_salience_constants_are_fixed_sweep_inputs()
    test_principled_2x2_architectures_are_simplex_and_distinct()
    test_principled_models_register_for_hierarchical_inference()
    print("PASS simplified model tests")
