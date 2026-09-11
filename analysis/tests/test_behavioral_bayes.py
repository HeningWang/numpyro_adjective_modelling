from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.behavioral_bayes import (
    CELL_LABELS,
    _mixed_model,
    hypothesis_rows,
    load_analysis_dataset,
    posterior_cell_draws,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_slider_datasets_match_reported_samples_and_design() -> None:
    study1a = load_analysis_dataset(REPO_ROOT, "study1a_slider")
    study1b = load_analysis_dataset(REPO_ROOT, "study1b_slider")

    assert study1a.n_observations == 3485
    assert study1a.n_participants == 134
    assert study1b.n_observations == 2786
    assert study1b.n_participants == 104
    assert study1a.fixed_design.shape == (3485, 6)
    assert study1b.random_design.shape == (2786, 3)
    assert study1a.item_index is None
    assert study1b.item_index is None


def test_production_uses_canonical_size_colour_trials() -> None:
    size_initial = load_analysis_dataset(REPO_ROOT, "production_size_initial")
    overinformative = load_analysis_dataset(
        REPO_ROOT, "production_overinformative"
    )

    assert size_initial.n_observations == 3034
    assert size_initial.n_participants == 113
    assert size_initial.n_items == 27
    assert size_initial.item_index is not None
    assert np.array_equal(size_initial.fixed_design, overinformative.fixed_design)
    assert set(np.unique(size_initial.outcome)) == {0.0, 1.0}
    assert set(np.unique(overinformative.outcome)) == {0.0, 1.0}


def test_binary_cell_predictions_marginalize_random_effect_variance() -> None:
    draws = 5
    samples = {
        "beta": np.zeros((1, draws, 6)),
        "participant_scale": np.zeros((1, draws, 3)),
        "participant_corr_cholesky": np.broadcast_to(
            np.eye(3), (1, draws, 3, 3)
        ).copy(),
        "item_scale": np.zeros((1, draws)),
    }

    predictions = posterior_cell_draws(samples, likelihood="bernoulli")

    assert predictions.shape == (draws, 6)
    assert np.allclose(predictions, 0.5)


def test_population_vectors_are_declared_as_joint_events() -> None:
    import jax.numpy as jnp
    from numpyro import handlers

    prepared = load_analysis_dataset(REPO_ROOT, "study1a_slider")
    trace = handlers.trace(handlers.seed(_mixed_model(prepared), 0)).get_trace(
        fixed_design=jnp.asarray(prepared.fixed_design[:10]),
        random_design=jnp.asarray(prepared.random_design[:10]),
        participant_index=jnp.asarray(prepared.participant_index[:10]),
        outcome=jnp.asarray(prepared.outcome[:10]),
        item_index=None,
    )

    assert trace["beta"]["fn"].event_shape == (6,)
    assert trace["participant_scale"]["fn"].event_shape == (3,)


def test_slider_hypotheses_encode_context_gradient_and_residual_preference() -> None:
    cells = np.array(
        [
            [0.30, 0.28, 0.20, 0.18, 0.06, 0.04],
            [0.32, 0.30, 0.22, 0.20, 0.05, 0.03],
        ]
    )
    rows = hypothesis_rows(cells, "study1a_slider")

    assert tuple(CELL_LABELS) == (
        "size_sufficient:low",
        "size_sufficient:high",
        "both_necessary:low",
        "both_necessary:high",
        "colour_sufficient:low",
        "colour_sufficient:high",
    )
    gradient = rows.loc[rows.hypothesis == "graded_context_order"].iloc[0]
    residual = rows.loc[
        rows.hypothesis == "colour_sufficient_above_neutral"
    ].iloc[0]
    assert gradient.posterior_probability == 1.0
    assert residual.posterior_probability == 1.0
    assert gradient["median"] > 0
    assert residual["median"] > 0


def test_production_hypotheses_capture_the_two_interactions() -> None:
    size_cells = np.array(
        [[0.80, 0.75, 0.50, 0.65, 0.04, 0.02]] * 4
    )
    over_cells = np.array(
        [[0.95, 0.70, 0.38, 0.34, 0.21, 0.08]] * 4
    )

    size_rows = hypothesis_rows(size_cells, "production_size_initial")
    over_rows = hypothesis_rows(over_cells, "production_overinformative")

    size_interaction = size_rows.loc[
        size_rows.hypothesis == "both_necessary_high_discriminability_advantage"
    ].iloc[0]
    over_interaction = over_rows.loc[
        over_rows.hypothesis == "size_sufficient_low_discriminability_advantage"
    ].iloc[0]
    assert size_interaction.posterior_probability == 1.0
    assert over_interaction.posterior_probability == 1.0
    assert size_interaction["median"] > 0
    assert over_interaction["median"] > 0
