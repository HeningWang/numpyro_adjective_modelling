"""One forward-pass check of the runner's optional condition-term ablation."""
from pathlib import Path
import sys

import numpy as np
import numpyro.handlers as handlers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helper import import_dataset_hier
import modelSpecification as ms
from run_corrected_primary import CONDITION_TERMS, MODELS, model_for_run

ROOT = Path(__file__).resolve().parents[3]


def test_condition_term_ablation_preserves_default_and_removes_latents():
    data = import_dataset_hier(file_path=ROOT/'data/production_model_input.csv',
                               state_encoding='target_match')
    args = dict(states=data['states_train'][:4], empirical=data['empirical_seq_flat'][:4],
        participant_idx=np.zeros(4, dtype=int), n_participants=1,
        sufficient_dim=data['sufficient_dim'][:4],
        has_one_word_solution=data['has_one_word_solution'][:4],
        is_sharp=data['sharpness_idx'][:4], is_colour_sufficient=data['is_colour_sufficient'][:4])
    for name, registered in MODELS.items():
        kwargs = dict(args, precomputed_features=ms.precompute_principled_discovery_features(
            args['states'], args['is_sharp'], recursive='UPD' in name))
        def trace(model):
            return handlers.trace(handlers.seed(model, 17)).get_trace(**kwargs)
        original = trace(registered)
        default = trace(model_for_run(name))
        np.testing.assert_array_equal(original['obs']['fn'].probs, default['obs']['fn'].probs)
        ablated = trace(model_for_run(name, drop_condition_terms=True))
        def latents(result):
            return {k for k, v in result.items() if v['type'] == 'sample' and not v['is_observed']}
        assert set(CONDITION_TERMS) <= latents(default)
        assert latents(ablated) == latents(default) - set(CONDITION_TERMS)
        for site in CONDITION_TERMS:
            assert ablated[site]['is_observed']
            np.testing.assert_array_equal(ablated[site]['value'], 0.)
        assert np.isfinite(ablated['obs']['fn'].probs).all()
        np.testing.assert_allclose(ablated['obs']['fn'].probs.sum(axis=1), 1, atol=1e-12)
