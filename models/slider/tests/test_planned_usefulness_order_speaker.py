import os
import sys
from pathlib import Path

import numpy as np


os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "models" / "slider"))

import jax.numpy as jnp  # noqa: E402
from jax import random  # noqa: E402
from numpyro import handlers  # noqa: E402
import modelSpecification as ms  # noqa: E402
import run_inference  # noqa: E402


def make_listener_arrays(size=0.25, color=0.75, dc=0.55, cd=0.55):
    l1 = jnp.array(
        [
            [size, 0.15, 0.10],
            [color, 0.20, 0.10],
        ],
        dtype=jnp.float32,
    )
    l2 = jnp.array(
        [
            [dc, 0.20, 0.10],
            [cd, 0.20, 0.10],
        ],
        dtype=jnp.float32,
    )
    return l1, l2


def test_usefulness_adjusted_order_bias_drops_for_color_advantage():
    l1, _ = make_listener_arrays(size=0.25, color=0.75)

    adjusted = ms._usefulness_adjusted_order_bias(l1, bias=2.0, usefulness_order_scale=4.0)

    assert float(adjusted) == 0.0


def test_planned_usefulness_order_returns_valid_probability():
    l1, l2 = make_listener_arrays()

    pred = ms._planned_usefulness_order_from_listeners(
        l1,
        l2,
        alpha=1.0,
        bias=2.0,
        usefulness_order_scale=4.0,
    )

    assert np.isfinite(float(pred))
    assert 0.0 <= float(pred) <= 1.0


def test_vectorized_planned_usefulness_order_returns_simplex_component():
    l1, l2 = make_listener_arrays()
    pred = ms.jitted_planned_usefulness_order_speaker_fast(
        jnp.stack([l1, l1]),
        jnp.stack([l2, l2]),
        1.0,
        2.0,
        4.0,
    )

    assert pred.shape == (2,)
    assert np.all(np.isfinite(np.asarray(pred)))
    assert np.all((np.asarray(pred) >= 0.0) & (np.asarray(pred) <= 1.0))


def test_signed_usefulness_adjusted_order_bias_can_reward_color_initial():
    l1, _ = make_listener_arrays(size=0.25, color=0.75)

    adjusted = ms._signed_usefulness_adjusted_order_bias(l1, bias=2.0, signed_order_scale=8.0)

    assert float(adjusted) < 0.0


def test_planned_signed_usefulness_order_returns_valid_probability():
    l1, l2 = make_listener_arrays()

    pred = ms._planned_signed_usefulness_order_from_listeners(
        l1,
        l2,
        alpha=1.0,
        bias=2.0,
        signed_order_scale=4.0,
    )

    assert np.isfinite(float(pred))
    assert 0.0 <= float(pred) <= 1.0


def test_planned_usefulness_mixture_respects_endpoints():
    l1, l2 = make_listener_arrays()
    greedy = ms._inc_speaker_from_listeners(l1, l2, alpha=1.0, bias=2.0)
    planned = ms._planned_usefulness_order_from_listeners(
        l1,
        l2,
        alpha=1.0,
        bias=2.0,
        usefulness_order_scale=4.0,
    )

    pred_greedy = ms._planned_usefulness_mixture_from_listeners(
        l1, l2, 1.0, 2.0, 4.0, 0.0
    )
    pred_planned = ms._planned_usefulness_mixture_from_listeners(
        l1, l2, 1.0, 2.0, 4.0, 1.0
    )

    assert np.isclose(float(pred_greedy), float(greedy))
    assert np.isclose(float(pred_planned), float(planned))


def test_anchored_mixture_cli_registration():
    assert "planned_usefulness_mixture_anchored" in run_inference.SPEAKER_CHOICES
    assert "planned_usefulness_mixture_anchored_static" in run_inference.SPEAKER_CHOICES
    assert (
        run_inference.get_hier_model("planned_usefulness_mixture_anchored")
        is ms.likelihood_planned_usefulness_mixture_anchored_speaker_hier
    )
    assert (
        run_inference.get_hier_model("planned_usefulness_mixture_anchored_static")
        is ms.likelihood_planned_usefulness_mixture_anchored_speaker_static_hier
    )


def test_anchored_mixture_hier_smoke_trace():
    l1, l2 = make_listener_arrays()
    model = handlers.seed(
        ms.likelihood_planned_usefulness_mixture_anchored_speaker_hier,
        random.PRNGKey(0),
    )
    trace = handlers.trace(model).get_trace(
        states=None,
        data=jnp.array([0.4, 0.6]),
        pi0=0.01,
        pi1=0.01,
        participant_idx=jnp.array([0, 1]),
        n_participants=2,
        L1_all=jnp.stack([l1, l1]),
        L2_all=jnp.stack([l2, l2]),
    )

    assert "usefulness_order_scale" in trace
    assert "planned_mixture_weight" in trace
    assert "obs" in trace


def test_production_anchor_cli_registration():
    expected = {
        "production_anchor_sizesharp_2x2_inc_rec",
        "production_anchor_sizesharp_2x2_inc_static",
        "production_anchor_sizesharp_2x2_glob_rec",
        "production_anchor_sizesharp_2x2_glob_static",
        "production_anchor_reliabilitybackup_2x2_inc_rec",
        "production_anchor_reliabilitybackup_2x2_inc_static",
        "production_anchor_reliabilitybackup_2x2_glob_rec",
        "production_anchor_reliabilitybackup_2x2_glob_static",
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_rec",
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_static",
        "production_anchor_reliabilitybackup_logalpha_2x2_glob_rec",
        "production_anchor_reliabilitybackup_logalpha_2x2_glob_static",
        "production_anchor_reliabilitybackup_orderplan_2x2_inc_rec",
        "production_anchor_reliabilitybackup_orderplan_2x2_inc_static",
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_rec",
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_static",
    }

    assert expected.issubset(set(run_inference.SPEAKER_CHOICES))
    assert (
        run_inference.get_hier_model("production_anchor_sizesharp_2x2_inc_rec")
        is ms.likelihood_production_anchor_inc_speaker_hier
    )
    assert (
        run_inference.get_hier_model("production_anchor_sizesharp_2x2_inc_static")
        is ms.likelihood_production_anchor_inc_speaker_hier
    )
    assert (
        run_inference.get_hier_model("production_anchor_sizesharp_2x2_glob_rec")
        is ms.likelihood_production_anchor_global_speaker_hier
    )
    assert (
        run_inference.get_hier_model("production_anchor_sizesharp_2x2_glob_static")
        is ms.likelihood_production_anchor_global_speaker_hier
    )
    assert (
        run_inference.get_hier_model("production_anchor_reliabilitybackup_2x2_inc_rec")
        is ms.likelihood_production_anchor_inc_speaker_hier
    )
    assert (
        run_inference.get_hier_model("production_anchor_reliabilitybackup_2x2_glob_static")
        is ms.likelihood_production_anchor_global_speaker_hier
    )
    assert (
        run_inference.get_hier_model("production_anchor_reliabilitybackup_logalpha_2x2_inc_rec")
        is ms.likelihood_production_anchor_inc_speaker_logalpha_hier
    )
    assert (
        run_inference.get_hier_model("production_anchor_reliabilitybackup_logalpha_2x2_glob_static")
        is ms.likelihood_production_anchor_global_speaker_logalpha_hier
    )
    assert (
        run_inference.get_hier_model("production_anchor_reliabilitybackup_orderplan_2x2_inc_rec")
        is ms.likelihood_production_anchor_orderplan_inc_speaker_hier
    )
    assert (
        run_inference.get_hier_model("production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_static")
        is ms.likelihood_production_anchor_orderplan_inc_speaker_logalpha_hier
    )


def test_production_anchor_speakers_return_valid_probabilities():
    states = jnp.array(
        [
            [10.0, 1.0, 1.0],
            [8.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    L1, L2 = ms.precompute_listeners_production_anchor(jnp.stack([states]), recursive=True)

    inc = ms.jitted_production_anchor_inc_speaker_fast(
        L1,
        L2,
        jnp.stack([states]),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([2.0], dtype=jnp.float32),
        1.0,
        0.5,
        0.1,
        ms.PRODUCTION_ANCHOR_EPSILON,
    )
    glob = ms.jitted_production_anchor_global_speaker_fast(
        L2,
        jnp.array([2.0], dtype=jnp.float32),
        1.0,
        ms.PRODUCTION_ANCHOR_EPSILON,
    )

    assert inc.shape == (1,)
    assert glob.shape == (1,)
    assert np.all(np.isfinite(np.asarray(inc)))
    assert np.all(np.isfinite(np.asarray(glob)))
    assert 0.0 <= float(inc[0]) <= 1.0
    assert 0.0 <= float(glob[0]) <= 1.0


def test_production_anchor_order_planning_zero_recovers_anchor_and_shifts_order():
    states = jnp.array(
        [
            [10.0, 1.0, 1.0],
            [8.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    L1, L2 = ms.precompute_listeners_production_anchor(jnp.stack([states]), recursive=True)
    base = ms.jitted_production_anchor_inc_speaker_fast(
        L1,
        L2,
        jnp.stack([states]),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([2.0], dtype=jnp.float32),
        1.0,
        0.5,
        0.1,
        ms.PRODUCTION_ANCHOR_EPSILON,
    )
    planned_zero = ms.jitted_production_anchor_orderplan_inc_speaker_fast(
        L1,
        L2,
        jnp.stack([states]),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([-1], dtype=jnp.int32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([2.0], dtype=jnp.float32),
        1.0,
        0.5,
        0.1,
        0.0,
        ms.PRODUCTION_ANCHOR_EPSILON,
    )
    planned = ms.jitted_production_anchor_orderplan_inc_speaker_fast(
        L1,
        L2,
        jnp.stack([states]),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([-1], dtype=jnp.int32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([2.0], dtype=jnp.float32),
        1.0,
        0.5,
        0.1,
        2.0,
        ms.PRODUCTION_ANCHOR_EPSILON,
    )

    assert np.allclose(np.asarray(base), np.asarray(planned_zero), atol=1e-6)
    assert np.all(np.isfinite(np.asarray(planned)))
    assert 0.0 <= float(planned[0]) <= 1.0
    assert not np.allclose(np.asarray(base), np.asarray(planned), atol=1e-4)


def test_production_anchor_hier_smoke_trace():
    l1, l2 = make_listener_arrays()
    model = handlers.seed(
        ms.likelihood_production_anchor_inc_speaker_hier,
        random.PRNGKey(0),
    )
    trace = handlers.trace(model).get_trace(
        states=jnp.stack([
            jnp.array(
                [
                    [10.0, 1.0, 1.0],
                    [8.0, 1.0, 0.0],
                    [3.0, 0.0, 1.0],
                ],
                dtype=jnp.float32,
            ),
            jnp.array(
                [
                    [9.0, 1.0, 1.0],
                    [8.0, 0.0, 0.0],
                    [4.0, 0.0, 1.0],
                ],
                dtype=jnp.float32,
            ),
        ]),
        data=jnp.array([0.4, 0.6]),
        pi0=0.01,
        pi1=0.01,
        participant_idx=jnp.array([0, 1]),
        n_participants=2,
        L1_all=jnp.stack([l1, l1]),
        L2_all=jnp.stack([l2, l2]),
        is_sharp_all=jnp.array([1.0, 0.0], dtype=jnp.float32),
    )

    assert "log_beta_order" in trace
    assert "lambda_salience" in trace
    assert "rho_salience_stop" in trace
    assert "delta_raw" in trace
    assert "delta" in trace
    assert "epsilon" in trace
    assert "obs" in trace


def test_production_anchor_logalpha_hier_smoke_trace():
    l1, l2 = make_listener_arrays()
    model = handlers.seed(
        ms.likelihood_production_anchor_inc_speaker_logalpha_hier,
        random.PRNGKey(0),
    )
    trace = handlers.trace(model).get_trace(
        states=jnp.stack([
            jnp.array(
                [
                    [10.0, 1.0, 1.0],
                    [8.0, 1.0, 0.0],
                    [3.0, 0.0, 1.0],
                ],
                dtype=jnp.float32,
            ),
            jnp.array(
                [
                    [9.0, 1.0, 1.0],
                    [8.0, 0.0, 0.0],
                    [4.0, 0.0, 1.0],
                ],
                dtype=jnp.float32,
            ),
        ]),
        data=jnp.array([0.4, 0.6]),
        pi0=0.01,
        pi1=0.01,
        participant_idx=jnp.array([0, 1]),
        n_participants=2,
        L1_all=jnp.stack([l1, l1]),
        L2_all=jnp.stack([l2, l2]),
        is_sharp_all=jnp.array([1.0, 0.0], dtype=jnp.float32),
    )

    assert "log_alpha" in trace
    assert "alpha" in trace
    assert "alpha_tau" in trace
    assert "alpha_offset_raw" in trace
    assert "alpha_participant" in trace
    assert "delta" not in trace
    alpha_participant = np.asarray(trace["alpha_participant"]["value"])
    assert np.all(np.isfinite(alpha_participant))
    assert np.all(alpha_participant > 0.0)
    assert "obs" in trace


def test_production_anchor_orderplan_logalpha_hier_smoke_trace():
    l1, l2 = make_listener_arrays()
    states = jnp.stack([
        jnp.array(
            [
                [10.0, 1.0, 1.0],
                [8.0, 1.0, 0.0],
                [3.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        ),
        jnp.array(
            [
                [9.0, 1.0, 1.0],
                [8.0, 0.0, 0.0],
                [4.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        ),
    ])
    model = handlers.seed(
        ms.likelihood_production_anchor_orderplan_inc_speaker_logalpha_hier,
        random.PRNGKey(0),
    )
    trace = handlers.trace(model).get_trace(
        states=states,
        data=jnp.array([0.4, 0.6]),
        pi0=0.01,
        pi1=0.01,
        participant_idx=jnp.array([0, 1]),
        n_participants=2,
        L1_all=jnp.stack([l1, l1]),
        L2_all=jnp.stack([l2, l2]),
        is_sharp_all=jnp.array([1.0, 0.0], dtype=jnp.float32),
        sufficient_dim_all=jnp.array([0, 1], dtype=jnp.int32),
        has_one_word_solution_all=jnp.array([1.0, 1.0], dtype=jnp.float32),
        is_colour_sufficient_all=jnp.array([0.0, 1.0], dtype=jnp.float32),
    )

    assert "lambda_order_planning" in trace
    assert "log_alpha" in trace
    assert "alpha_participant" in trace
    assert "obs" in trace


if __name__ == "__main__":
    test_usefulness_adjusted_order_bias_drops_for_color_advantage()
    test_planned_usefulness_order_returns_valid_probability()
    test_vectorized_planned_usefulness_order_returns_simplex_component()
    test_signed_usefulness_adjusted_order_bias_can_reward_color_initial()
    test_planned_signed_usefulness_order_returns_valid_probability()
    test_planned_usefulness_mixture_respects_endpoints()
    test_anchored_mixture_cli_registration()
    test_anchored_mixture_hier_smoke_trace()
    test_production_anchor_cli_registration()
    test_production_anchor_speakers_return_valid_probabilities()
    test_production_anchor_order_planning_zero_recovers_anchor_and_shifts_order()
    test_production_anchor_hier_smoke_trace()
    test_production_anchor_logalpha_hier_smoke_trace()
    test_production_anchor_orderplan_logalpha_hier_smoke_trace()
    print("PASS planned usefulness-order speaker tests")
