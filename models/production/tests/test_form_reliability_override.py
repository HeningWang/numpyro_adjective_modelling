"""Forward-pass checks for the form-reliability (nu_F) override."""
from pathlib import Path
import sys

import numpy as np
import numpyro.handlers as handlers
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modelSpecification as ms
from helper import import_dataset_hier

ROOT = Path(__file__).resolve().parents[3]
MODELS = ["G-HO", "K-HO", "I-HO", "G-UPD-HO", "K-UPD-HO", "I-UPD-HO"]
REGISTRY = {**ms.V10_ORDER_MODELS, **ms.V10_ORDER_UPD_MODELS}


@pytest.fixture(scope="module")
def args():
    data = import_dataset_hier(file_path=ROOT/"data/production_model_input.csv",
                               state_encoding="target_match")
    rows = np.r_[0:4, 4000:4004]  # include colour-form and size-form trials
    return dict(states=data["states_train"][rows], empirical=data["empirical_seq_flat"][rows],
                participant_idx=np.zeros(len(rows), dtype=int), n_participants=1,
                sufficient_dim=data["sufficient_dim"][rows],
                has_one_word_solution=data["has_one_word_solution"][rows],
                is_sharp=data["sharpness_idx"][rows],
                is_colour_sufficient=data["is_colour_sufficient"][rows])


def forward(model, **kwargs):
    return np.asarray(handlers.trace(handlers.seed(model, 17)).get_trace(**kwargs)["obs"]["fn"].probs)


def features(args, name, form_semval):
    return ms.precompute_principled_discovery_features(
        args["states"], args["is_sharp"], recursive="UPD" in name, form_semval=form_semval)


@pytest.mark.parametrize("name", MODELS)
def test_default_matches_registry(args, name):
    baseline = forward(REGISTRY[name], **args)
    default = forward(ms.form_reliability_model(name, .50), **args)
    np.testing.assert_allclose(baseline, default, rtol=0, atol=1e-12)
    pre = forward(ms.form_reliability_model(name, .50), **args,
                  precomputed_features=features(args, name, .50))
    np.testing.assert_allclose(baseline, pre, rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", MODELS)
def test_informative_form_paths_agree_and_change(args, name):
    model = ms.form_reliability_model(name, .75)
    direct = forward(model, **args)
    pre = forward(model, **args, precomputed_features=features(args, name, .75))
    np.testing.assert_allclose(direct, pre, rtol=0, atol=1e-10)
    np.testing.assert_allclose(pre.sum(axis=1), 1, atol=1e-12)
    assert np.max(np.abs(pre - forward(REGISTRY[name], **args))) > 1e-5


def test_nu_f_site_records_override(args):
    tr = handlers.trace(handlers.seed(ms.form_reliability_model("K-HO", .75), 0)).get_trace(**args)
    assert float(tr["nu_F"]["value"]) == pytest.approx(.75)


@pytest.mark.parametrize("bad", [.49, 1.01, float("nan")])
def test_rejects_invalid_reliability(bad):
    with pytest.raises(ValueError):
        ms.form_reliability_model("K-HO", bad)
