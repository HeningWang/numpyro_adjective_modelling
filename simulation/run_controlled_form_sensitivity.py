"""Sensitivity analysis: the controlled simulation with informative form adjectives.

The production models fit form reliability at nu_F = .50, so form adjectives apply
equally to all objects and the controlled simulation inherits this. Here form
membership follows the preserved scenes' form feature with a chosen nu_F > .5;
scenes, rules, supports and all other parameters are unchanged.

Usage: python run_controlled_form_sensitivity.py --nu-f .75 --scenes DIR --output DIR
"""
from __future__ import annotations
import os
os.environ.setdefault('JAX_PLATFORMS', 'cuda')
os.environ['JAX_ENABLE_X64'] = 'true'
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_controlled_random_scenes as base  # noqa: E402

ms = base.ms
NU_F = .5


def literal_all(scene, params, recursive):
    """Same lexical kernel as the base script, with form membership nu_F / 1 - nu_F."""
    colour, k, wf = params
    prior = jnp.ones(scene.shape[0], dtype=jnp.float64) / scene.shape[0]
    sizes = scene[:, 0]
    idx = jnp.argsort(sizes)

    def size(post):
        return jnp.clip(ms.compute_size_semantics_fast_presorted(sizes, idx, sizes[idx], post, k, wf), 1e-8)

    fixed = size(prior)
    color = jnp.where(scene[:, 1] == 1, colour, 1 - colour) + 1e-8
    form = jnp.where(scene[:, 2] == 1, NU_F, 1 - NU_F) + 1e-8

    def interpret(tokens):
        def step(post, token):
            m = jnp.stack([size(post) if recursive else fixed, color, form])[jnp.maximum(token, 0)]
            updated = post * m
            updated = updated / updated.sum()
            return jnp.where(token >= 0, updated, post), None
        return jax.lax.scan(step, prior, tokens[::-1])[0]

    return jax.vmap(interpret)(ms.utterance_list)


def validate(scene, params):
    """Check the modified kernel and speakers against the production code at the same nu_F."""
    assert (scene[0, 1] == 1) and (scene[0, 2] == 1), 'target must be blue and of form 1'
    errors = {}
    for rec in [False, True]:
        L = np.asarray(literal_all(jnp.asarray(scene), jnp.asarray(params), rec))
        np.testing.assert_allclose(L.sum(axis=1), 1, atol=1e-12)
        ref = np.asarray(ms.principled_prefix_log_listeners(jnp.asarray(scene), params[0], NU_F, params[1], params[2], rec))
        err = float(np.max(np.abs(np.log(np.clip(L[:, 0], 1e-8, None)) - ref)))
        assert err < 1e-11, err
        errors[f'lexical_{rec}'] = err
        for rule, kappa in base.RULES[1:]:
            actual = np.asarray(base.structural_probs(jnp.asarray(L), 'fifteen', rule, kappa, 0.))[0]
            expected = np.asarray(ms.incremental_speaker_principled_discovery(
                states=jnp.asarray(scene), sufficient_dim=-1, has_one_word_solution=0., is_sharp=0.,
                is_colour_sufficient=0., alpha=1., beta_order=0., kappa=kappa, nu_F=NU_F,
                color_semval=params[0], size_threshold_k=params[1], wf=params[2], epsilon=0.,
                recursive=rec, prefix_mode='B'))
            err = float(np.max(np.abs(actual - expected)))
            assert err < 1e-10, (rec, rule, err)
            errors[f'forward_{rec}_{rule}'] = err
    return errors


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nu-f', type=float, required=True)
    p.add_argument('--scenes', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch', type=int, default=25)
    args = p.parse_args()
    if not .5 <= args.nu_f < 1:
        raise ValueError('nu_F must lie in [.5, 1)')
    NU_F = args.nu_f
    base.literal_all = literal_all
    base.validate = validate
    base.main(args)
