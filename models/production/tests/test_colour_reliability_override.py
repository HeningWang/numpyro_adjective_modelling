"""One forward-pass check that the reliability override preserves the default."""
from pathlib import Path
import sys

import numpy as np
import numpyro.handlers as handlers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modelSpecification as ms
from helper import import_dataset_hier

ROOT = Path(__file__).resolve().parents[3]


def test_colour_override_preserves_default():
    data = import_dataset_hier(file_path=ROOT/'data/production_model_input.csv',
                               state_encoding='target_match')
    args = dict(states=data['states_train'][:4], empirical=data['empirical_seq_flat'][:4],
        participant_idx=np.zeros(4, dtype=int), n_participants=1,
        sufficient_dim=data['sufficient_dim'][:4],
        has_one_word_solution=data['has_one_word_solution'][:4],
        is_sharp=data['sharpness_idx'][:4], is_colour_sufficient=data['is_colour_sufficient'][:4])
    for name, registry in [('K-HO', ms.V10_ORDER_MODELS), ('K-UPD-HO', ms.V10_ORDER_UPD_MODELS)]:
        def forward(model, **kwargs):
            return handlers.trace(handlers.seed(model, 17)).get_trace(**kwargs)['obs']['fn'].probs
        baseline = np.asarray(forward(registry[name], **args))
        default = np.asarray(forward(ms.colour_reliability_model(name, .59), **args))
        np.testing.assert_allclose(baseline, default, rtol=0, atol=1e-12)
        # This also checks the precomputed feature path used by the fitting CLI.
        changed = np.asarray(forward(ms.colour_reliability_model(name, .75), **args,
            precomputed_features=ms.precompute_principled_discovery_features(
                args['states'], args['is_sharp'], recursive='UPD' in name, color_semval=.75)))
        assert np.isfinite(changed).all()
        np.testing.assert_allclose(changed.sum(axis=1), 1, atol=1e-12)
        assert np.max(np.abs(changed-baseline)) > 1e-5
