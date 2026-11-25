"""Forward-pass checks of the terminal-utility and order-term ablations."""
from pathlib import Path
import sys

import numpy as np
import numpyro.handlers as handlers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helper import import_dataset_hier
import modelSpecification as ms
from run_corrected_primary import model_for_run

ROOT = Path(__file__).resolve().parents[3]
ORDER_SITES = {'log_beta_order', 'tau_order', 'z_order'}
ALPHA_SITES = {'alpha', 'tau_log_alpha', 'z_alpha'}


def test_utility_ablations_change_only_their_component():
    data = import_dataset_hier(file_path=ROOT/'data/production_model_input.csv',
                               state_encoding='target_match')
    args = dict(states=data['states_train'][:6], empirical=data['empirical_seq_flat'][:6],
        participant_idx=np.zeros(6, dtype=int), n_participants=1,
        sufficient_dim=data['sufficient_dim'][:6],
        has_one_word_solution=data['has_one_word_solution'][:6],
        is_sharp=data['sharpness_idx'][:6], is_colour_sufficient=data['is_colour_sufficient'][:6])
    features = ms.precompute_principled_discovery_features(
        args['states'], args['is_sharp'], recursive=False)
    single = np.asarray(ms.N_WORDS) == 1
    for log_target, salience in zip(features['log_target'], features['salience']):
        prefix = ms.principled_path_components_from_features(log_target, salience, 2., .7)
        terminal = ms.principled_terminal_path_components_from_features(log_target, salience, 2., .7)
        np.testing.assert_allclose(terminal[0][single], prefix[0][single], atol=1e-12)
        np.testing.assert_array_equal(terminal[1], prefix[1])
        no_salience = ms.principled_terminal_path_components_from_features(log_target, salience, 2., 0.)
        np.testing.assert_allclose(no_salience[0], 2. * log_target, atol=1e-12)
    for name in ('G-HO', 'K-HO', 'K-UPD-HO'):
        kwargs = dict(args, precomputed_features=ms.precompute_principled_discovery_features(
            args['states'], args['is_sharp'], recursive='UPD' in name))
        def trace(model):
            return handlers.trace(handlers.seed(model, 17)).get_trace(**kwargs)
        def latents(result):
            return {k for k, v in result.items() if v['type'] == 'sample' and not v['is_observed']}
        default = trace(model_for_run(name))
        terminal = trace(model_for_run(name, terminal_utility=True))
        no_order = trace(model_for_run(name, drop_order_term=True))
        assert latents(terminal) == latents(default)
        assert latents(no_order) == latents(default) - ORDER_SITES
        assert not np.allclose(terminal['obs']['fn'].probs, default['obs']['fn'].probs)
        for result in (terminal, no_order):
            np.testing.assert_allclose(result['obs']['fn'].probs.sum(axis=1), 1, atol=1e-12)


def test_informativeness_ablation_removes_alpha_and_semantics():
    data = import_dataset_hier(file_path=ROOT/'data/production_model_input.csv',
                               state_encoding='target_match')
    args = dict(states=data['states_train'][:6], empirical=data['empirical_seq_flat'][:6],
        participant_idx=np.zeros(6, dtype=int), n_participants=1,
        sufficient_dim=data['sufficient_dim'][:6],
        has_one_word_solution=data['has_one_word_solution'][:6],
        is_sharp=data['sharpness_idx'][:6], is_colour_sufficient=data['is_colour_sufficient'][:6])
    probs = {}
    for name in ('K-HO', 'K-UPD-HO', 'G-HO'):
        kwargs = dict(args, precomputed_features=ms.precompute_principled_discovery_features(
            args['states'], args['is_sharp'], recursive='UPD' in name))
        def trace(model):
            return handlers.trace(handlers.seed(model, 17)).get_trace(**kwargs)
        latents = lambda r: {k for k, v in r.items() if v['type'] == 'sample' and not v['is_observed']}
        default = trace(model_for_run(name))
        ablated = trace(model_for_run(name, drop_informativeness=True))
        assert latents(ablated) == latents(default) - ALPHA_SITES
        np.testing.assert_allclose(ablated['obs']['fn'].probs.sum(axis=1), 1, atol=1e-12)
        probs[name] = np.asarray(ablated['obs']['fn'].probs)
    # Without alpha the semantic regime has no route into the likelihood.
    np.testing.assert_allclose(probs['K-HO'], probs['K-UPD-HO'], atol=1e-12)
    try:
        model_for_run('K-HO', terminal_utility=True, drop_informativeness=True)
    except ValueError:
        pass
    else:
        raise AssertionError('terminal utility without informativeness must be rejected')


def test_prefix_salience_ablation_removes_only_lambda_salience():
    data = import_dataset_hier(file_path=ROOT/'data/production_model_input.csv',
                               state_encoding='target_match')
    args = dict(states=data['states_train'][:6], empirical=data['empirical_seq_flat'][:6],
        participant_idx=np.zeros(6, dtype=int), n_participants=1,
        sufficient_dim=data['sufficient_dim'][:6],
        has_one_word_solution=data['has_one_word_solution'][:6],
        is_sharp=data['sharpness_idx'][:6], is_colour_sufficient=data['is_colour_sufficient'][:6])
    for name in ('G-HO', 'K-HO', 'I-HO'):
        kwargs = dict(args, precomputed_features=ms.precompute_principled_discovery_features(
            args['states'], args['is_sharp'], recursive=False))
        def trace(model):
            return handlers.trace(handlers.seed(model, 17)).get_trace(**kwargs)
        latents = lambda r: {k for k, v in r.items() if v['type'] == 'sample' and not v['is_observed']}
        default = trace(model_for_run(name))
        ablated = trace(model_for_run(name, drop_prefix_salience=True))
        assert latents(ablated) == latents(default) - {'lambda_salience'}
        np.testing.assert_allclose(ablated['obs']['fn'].probs.sum(axis=1), 1, atol=1e-12)


def test_ablations_extend_to_participant_kappa_cells():
    data = import_dataset_hier(file_path=ROOT/'data/production_model_input.csv',
                               state_encoding='target_match')
    args = dict(states=data['states_train'][:6], empirical=data['empirical_seq_flat'][:6],
        participant_idx=np.zeros(6, dtype=int), n_participants=1,
        sufficient_dim=data['sufficient_dim'][:6],
        has_one_word_solution=data['has_one_word_solution'][:6],
        is_sharp=data['sharpness_idx'][:6], is_colour_sufficient=data['is_colour_sufficient'][:6])
    for name in ('K-HKO', 'K-UPD-HKO'):
        kwargs = dict(args, precomputed_features=ms.precompute_principled_discovery_features(
            args['states'], args['is_sharp'], recursive='UPD' in name))
        def trace(model):
            return handlers.trace(handlers.seed(model, 17)).get_trace(**kwargs)
        def latents(result):
            return {k for k, v in result.items() if v['type'] == 'sample' and not v['is_observed']}
        default = trace(model_for_run(name))
        no_order = trace(model_for_run(name, drop_order_term=True))
        no_info = trace(model_for_run(name, drop_informativeness=True))
        no_salience = trace(model_for_run(name, drop_prefix_salience=True))
        kappa_sites = {k for k in latents(default) if 'kappa' in k}
        assert len(kappa_sites) > 1  # participant-specific kappa hierarchy
        assert latents(no_order) == latents(default) - ORDER_SITES
        assert latents(no_info) == latents(default) - ALPHA_SITES
        for result in (no_order, no_info, no_salience):
            assert kappa_sites <= latents(result)
            assert not np.allclose(result['obs']['fn'].probs, default['obs']['fn'].probs)
            np.testing.assert_allclose(result['obs']['fn'].probs.sum(axis=1), 1, atol=1e-12)
