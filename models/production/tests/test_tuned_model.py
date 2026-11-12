"""Forward-pass checks of the theory-mapped model variants."""
from pathlib import Path
import sys

import numpy as np
import numpyro.handlers as handlers
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helper import import_dataset_hier
import modelSpecification as ms
import tuned_model as tm

ROOT = Path(__file__).resolve().parents[3]
N_ROWS = 24
DEAD_SITES = {"rho_salience_stop", "lambda_size_reliability_single_bonus",
              "lambda_size_reliability_form_pair_tradeoff"}


@pytest.fixture(scope="module")
def data():
    full = import_dataset_hier(file_path=ROOT / "data/production_model_input.csv",
                               state_encoding="target_match")
    # Spread rows over conditions and both discriminability groups.
    rows = np.linspace(0, len(full["states_train"]) - 1, N_ROWS).astype(int)
    return dict(
        states=full["states_train"][rows], empirical=full["empirical_seq_flat"][rows],
        participant_idx=np.arange(N_ROWS) % 3, n_participants=3,
        sufficient_dim=full["sufficient_dim"][rows],
        has_one_word_solution=full["has_one_word_solution"][rows],
        is_sharp=full["sharpness_idx"][rows],
        is_colour_sufficient=full["is_colour_sufficient"][rows])


def _trace(model, kwargs, seed=17):
    return handlers.trace(handlers.seed(model, seed)).get_trace(**kwargs)


def _latents(trace):
    return {k: v["value"] for k, v in trace.items()
            if v["type"] == "sample" and not v["is_observed"]}


def _probs(model, kwargs, values):
    replay = handlers.trace(handlers.substitute(handlers.seed(model, 0), data=values))
    return np.asarray(replay.get_trace(**kwargs)["obs"]["fn"].probs)


def _tuned_kwargs(data, name, spec):
    features = tm.precompute_tuned_features(
        data["states"], data["is_sharp"], recursive="UPD" in name, spec=spec)
    return dict(data, precomputed_features=features)


@pytest.mark.parametrize("name", ["G-HO", "K-HO", "I-HO", "K-UPD-HO"])
def test_default_spec_reproduces_primary_model(data, name):
    primary_kwargs = dict(data, precomputed_features=ms.precompute_principled_discovery_features(
        data["states"], data["is_sharp"], recursive="UPD" in name))
    primary = {**ms.V10_ORDER_MODELS, **ms.V10_ORDER_UPD_MODELS}[name]
    values = _latents(_trace(primary, primary_kwargs))
    spec = tm.TuningSpec()
    tuned = tm.make_tuned_model(name, spec)
    assert set(_latents(_trace(tuned, _tuned_kwargs(data, name, spec)))) == set(values)
    np.testing.assert_allclose(
        _probs(tuned, _tuned_kwargs(data, name, spec), values),
        _probs(primary, primary_kwargs, values), atol=1e-10)


def test_dead_terms_are_removed(data):
    spec = tm.TuningSpec(drop_dead_terms=True)
    default = _latents(_trace(tm.make_tuned_model("K-HO", tm.TuningSpec()),
                              _tuned_kwargs(data, "K-HO", tm.TuningSpec())))
    tuned = _latents(_trace(tm.make_tuned_model("K-HO", spec), _tuned_kwargs(data, "K-HO", spec)))
    assert set(tuned) == set(default) - DEAD_SITES


def test_salience_split_is_additive(data):
    whole = tm.TuningSpec(drop_dead_terms=True)
    split = tm.TuningSpec(drop_dead_terms=True, split_salience=True)
    features = tm.precompute_tuned_features(data["states"], data["is_sharp"],
                                            recursive=False, spec=split)
    np.testing.assert_allclose(features["salience_base"] + features["salience_contrast"],
                               features["salience"], atol=1e-6)
    np.testing.assert_allclose(features["salience_base"].sum(axis=1), 0, atol=1e-6)
    np.testing.assert_allclose(features["salience_contrast"].sum(axis=1), 0, atol=1e-6)
    whole_model = tm.make_tuned_model("K-HO", whole)
    values = _latents(_trace(whole_model, _tuned_kwargs(data, "K-HO", whole)))
    split_values = {k: v for k, v in values.items() if k != "lambda_salience"}
    split_values.update(lambda_salience_base=values["lambda_salience"],
                        lambda_salience_contrast=values["lambda_salience"])
    split_model = tm.make_tuned_model("K-HO", split)
    split_trace = _latents(_trace(split_model, _tuned_kwargs(data, "K-HO", split)))
    assert "lambda_salience" not in split_trace
    assert {"lambda_salience_base", "lambda_salience_contrast"} <= set(split_trace)
    np.testing.assert_allclose(
        _probs(split_model, _tuned_kwargs(data, "K-HO", split), split_values),
        _probs(whole_model, _tuned_kwargs(data, "K-HO", whole), values), atol=1e-10)


def test_size_contrast_scaling_and_group_blur(data):
    base = tm.TuningSpec()
    tuned = tm.TuningSpec(unscaled_size_contrast=True, size_blur_multiplier=2.0)
    f_base = tm.precompute_tuned_features(data["states"], data["is_sharp"],
                                          recursive=False, spec=base)
    f_tuned = tm.precompute_tuned_features(data["states"], data["is_sharp"],
                                           recursive=False, spec=tuned)
    sharp = np.asarray(data["is_sharp"]) == 1
    assert sharp.any() and (~sharp).any()
    # High-discriminability rows are untouched by both changes.
    np.testing.assert_allclose(f_tuned["log_target"][sharp], f_base["log_target"][sharp],
                               atol=1e-12)
    np.testing.assert_allclose(f_tuned["salience"][sharp], f_base["salience"][sharp], atol=1e-6)
    assert not np.allclose(f_tuned["log_target"][~sharp], f_base["log_target"][~sharp])


def test_display_sufficiency_mask():
    # Rows: target first; columns: size, colour match, form match.
    colour_only = np.array([[5, 1, 1], [9, 0, 1], [9, 0, 1], [3, 0, 0], [3, 0, 1], [9, 0, 0]], float)
    size_only = np.array([[9, 1, 1], [3, 1, 1], [3, 1, 0], [2, 0, 1], [2, 1, 1], [3, 0, 1]], float)
    none = np.array([[9, 1, 1], [9, 1, 0], [3, 0, 1], [2, 1, 1], [9, 0, 0], [3, 0, 1]], float)
    mask = tm.display_sufficiency_mask(np.stack([colour_only, size_only, none]))
    np.testing.assert_array_equal(mask, [[0, 1, 0], [1, 0, 0], [0, 0, 0]])


def test_display_sufficiency_agrees_with_design_for_colour_and_form(data):
    mask = tm.display_sufficiency_mask(np.asarray(data["states"]))
    labels = np.asarray(data["sufficient_dim"])
    for row, label in enumerate(labels):
        if label in (1, 2):
            assert mask[row, label] == 1


def test_display_sufficiency_and_unconditional_form_change_only_their_terms(data):
    reference = tm.TuningSpec(drop_dead_terms=True)
    model = tm.make_tuned_model("K-HO", reference)
    values = _latents(_trace(model, _tuned_kwargs(data, "K-HO", reference)))
    p_reference = _probs(model, _tuned_kwargs(data, "K-HO", reference), values)

    form = tm.TuningSpec(drop_dead_terms=True, unconditional_form_term=True)
    p_form = _probs(tm.make_tuned_model("K-HO", form), _tuned_kwargs(data, "K-HO", form), values)
    colour_sufficient = np.asarray(data["is_colour_sufficient"]) == 1
    assert colour_sufficient.any()
    np.testing.assert_allclose(p_form[~colour_sufficient], p_reference[~colour_sufficient],
                               atol=1e-12)
    assert not np.allclose(p_form[colour_sufficient], p_reference[colour_sufficient])

    display = tm.TuningSpec(drop_dead_terms=True, display_sufficiency=True)
    p_display = _probs(tm.make_tuned_model("K-HO", display),
                       _tuned_kwargs(data, "K-HO", display), values)
    np.testing.assert_allclose(p_display.sum(axis=1), 1, atol=1e-12)
    mask = tm.display_sufficiency_mask(np.asarray(data["states"]))
    labels = np.asarray(data["sufficient_dim"])
    label_mask = np.zeros_like(mask)
    label_mask[labels >= 0, labels[labels >= 0]] = 1
    same = (mask == label_mask).all(axis=1)
    np.testing.assert_allclose(p_display[same], p_reference[same], atol=1e-12)


def test_switches_remove_their_sites(data):
    full = tm.TuningSpec(drop_dead_terms=True, split_salience=True)
    sites = set(_latents(_trace(tm.make_tuned_model("K-HO", full),
                                _tuned_kwargs(data, "K-HO", full))))
    for switch, removed in [
        ("drop_order_term", {"log_beta_order", "tau_order", "z_order"}),
        ("drop_informativeness", {"alpha", "tau_log_alpha", "z_alpha"}),
        ("drop_salience_base", {"lambda_salience_base"}),
        ("drop_salience_contrast", {"lambda_salience_contrast"}),
        ("drop_sufficiency", {"lambda_sufficient_single"}),
    ]:
        spec = tm.TuningSpec(drop_dead_terms=True, split_salience=True, **{switch: True})
        trace = _trace(tm.make_tuned_model("K-HO", spec), _tuned_kwargs(data, "K-HO", spec))
        assert set(_latents(trace)) == sites - removed
        np.testing.assert_allclose(np.asarray(trace["obs"]["fn"].probs).sum(axis=1), 1, atol=1e-12)


def test_free_salience_base_generalises_the_fixed_baseline(data):
    fixed = tm.TuningSpec(split_salience=True, drop_salience_contrast=True)
    free = tm.TuningSpec(split_salience=True, drop_salience_contrast=True,
                         free_salience_base=True)
    fixed_model = tm.make_tuned_model("K-HO", fixed)
    values = _latents(_trace(fixed_model, _tuned_kwargs(data, "K-HO", fixed)))
    weight = values["lambda_salience_base"]
    free_values = {k: v for k, v in values.items() if k != "lambda_salience_base"}
    free_values.update(salience_base_colour=weight * 1.0, salience_base_form=weight * 0.25)
    free_model = tm.make_tuned_model("K-HO", free)
    sites = set(_latents(_trace(free_model, _tuned_kwargs(data, "K-HO", free))))
    assert sites == set(free_values)
    np.testing.assert_allclose(
        _probs(free_model, _tuned_kwargs(data, "K-HO", free), free_values),
        _probs(fixed_model, _tuned_kwargs(data, "K-HO", fixed), values), atol=1e-10)


def test_invalid_specs_are_rejected():
    with pytest.raises(ValueError):
        tm.TuningSpec(display_sufficiency=True)  # needs the label-gated terms removed
    with pytest.raises(ValueError):
        tm.TuningSpec(drop_salience_base=True)  # needs the split
    with pytest.raises(ValueError):
        tm.TuningSpec(size_blur_multiplier=0.5)
    with pytest.raises(ValueError):
        tm.TuningSpec(free_salience_base=True)  # needs the split
    with pytest.raises(ValueError):
        tm.make_tuned_model("K-HKO", tm.TuningSpec())
