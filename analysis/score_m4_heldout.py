"""Joint new-participant predictive densities using the production model forward pass."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cuda')
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp
import xarray as xr

from m4_common import (DATA, FIELDS, MODELS, SEED, encoded, folds, job_path, production,
                       read_fit, sha256)


def joint_log_score(log_likelihood):
    """Input axes (posterior draw, effect draw, trial); effects persist over trials."""
    values = np.asarray(log_likelihood, dtype=float)
    if values.ndim != 3 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError('Expected finite (S, M, trials) log likelihoods')
    return log_mean_joint(values.sum(axis=-1))


def log_mean_joint(joint):
    joint = np.asarray(joint, dtype=float)
    if not joint.size or not np.isfinite(joint).all():
        raise ValueError('Empty/nonfinite joint log likelihoods')
    return float(logsumexp(joint)-np.log(joint.size))


def population_draws(path, manifest, count):
    with xr.open_dataset(path / 'posterior.nc', group='posterior', engine='h5netcdf') as posterior:
        total = posterior.sizes['chain']*posterior.sizes['draw']
        if not 1 <= count <= total:
            raise ValueError(f'Requested S={count}, available posterior draws={total}')
        indices = np.linspace(0, total-1, count, dtype=int)
        result = {}
        for site in manifest['population_sites']:
            variable = posterior[site].transpose('chain', 'draw').values
            result[site] = variable.reshape(-1)[indices]
    return result, indices


def new_participant_draws(population, participant_sites, count, fold, participant, seed):
    """Discard all fitted participant effects; draw new standard-normal z's."""
    s_count = len(next(iter(population.values())))
    result = {k: np.repeat(v, count) for k, v in population.items()}
    # Per-S streams make M=50 an exact prefix of M=100, also across separate calls.
    normal = np.stack([np.random.default_rng(np.random.SeedSequence(
        [seed, fold, int(participant), s])).normal(size=(count, 3)) for s in range(s_count)])
    columns = {'z_alpha': 0, 'z_order': 1, 'z_kappa': 2}
    if not set(participant_sites) <= set(columns):
        raise ValueError('Unrecognized participant effect; review new-participant scoring')
    for site in participant_sites:
        result[site] = normal[:, :, columns[site]].reshape(-1, 1)
    return result


def score(args):
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import log_likelihood
    if not jax.config.x64_enabled or jax.default_backend() != ('cpu' if args.cpu_smoke else 'gpu'):
        raise RuntimeError('Scoring requires x64 and CUDA, or explicit --cpu-smoke on CPU')
    if min(args.posterior_draws, args.effect_draws, args.batch) < 1:
        raise ValueError('S, M and batch must be positive')
    if args.compare_m is not None and not 1 <= args.compare_m < args.effect_draws:
        raise ValueError('--compare-m must be positive and smaller than M')
    path = job_path(args.root, args.model, args.fold)
    manifest = read_fit(path, allow_smoke=args.cpu_smoke)
    assignment_path = args.folds or args.root / 'fold_assignment.csv'
    assignment = folds(assignment_path, args.data)
    if (manifest['dataset_sha256'] != sha256(args.data) or
            manifest['fold_assignment_sha256'] != sha256(assignment_path) or
            manifest['model'] != args.model or manifest['fold'] != args.fold):
        raise ValueError('Fit does not match model/fold/data')
    if args.cpu_smoke and (args.posterior_draws > 20 or args.effect_draws > 20 or
                           manifest['artifact_role'] != 'm4_smoke'):
        raise ValueError('CPU scoring smoke requires a smoke fit, S<=20 and M<=20')
    heldout = manifest['heldout_participant_ids']
    if (set(manifest['train_participant_ids']) & set(heldout) or
            not set(heldout) <= set(assignment.loc[assignment.fold.eq(args.fold), 'id'])):
        raise ValueError('Held-out participant overlap or fold mismatch')
    output = args.root / 'scores' / f'fold_{args.fold:02d}' / f'{args.model}.csv'
    receipt_path = output.with_name(f'{args.model}_manifest.json')
    config = dict(model=args.model, fold=args.fold, posterior_draws=args.posterior_draws,
        effect_draws=args.effect_draws, compare_m=args.compare_m, seed=args.seed,
        fit_artifact_sha256=manifest['artifact_sha256'],
        scorer_sha256=sha256(Path(__file__)), cpu_smoke=args.cpu_smoke)
    if output.exists() or receipt_path.exists():
        if args.skip_completed and output.exists() and receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            if (old.get('complete') and all(old.get(k) == v for k, v in config.items())
                    and old.get('scores_sha256') == sha256(output)):
                print(f'Skip completed scores: {args.model} fold {args.fold}', flush=True)
                return
        raise FileExistsError(f'Use a fresh root; existing/incomplete scores: {output}')
    start = time.perf_counter()
    population, indices = population_draws(path, manifest, args.posterior_draws)
    rows, data = encoded(args.data)
    ms, models = production()
    model = models[args.model]
    fold_positions = np.flatnonzero(rows.id.isin(heldout))
    features = ms.precompute_principled_discovery_features(
        data['states_train'][fold_positions], data['sharpness_idx'][fold_positions],
        recursive='UPD' in args.model)
    max_trials = int(rows.iloc[fold_positions].groupby('id').size().max())

    @jax.jit
    def evaluate(samples, kwargs, mask):
        ll = log_likelihood(model, samples, parallel=True, n_participants=1, **kwargs)['obs']
        return jnp.sum(ll*mask, axis=-1)

    results = []
    for participant in heldout:
        local = np.flatnonzero(rows.iloc[fold_positions].id.to_numpy() == participant)
        n_trials = len(local)
        # Equal trial/batch shapes reuse compilation; repeated padding rows are masked out.
        padded = np.pad(local, (0, max_trials-n_trials), mode='edge')
        positions = fold_positions[padded]
        kwargs = {key: jnp.asarray(data[field][positions]) for key, field in FIELDS.items()}
        kwargs['participant_idx'] = jnp.zeros(max_trials, dtype=jnp.int32)
        kwargs['precomputed_features'] = {k: v[padded] for k, v in features.items()}
        mask = jnp.asarray(np.arange(max_trials) < n_trials)
        draws = new_participant_draws(population, manifest['participant_sites'], args.effect_draws,
                                      args.fold, participant, args.seed)
        total = args.posterior_draws*args.effect_draws
        chunks = []
        for begin in range(0, total, args.batch):
            take = np.minimum(np.arange(begin, begin+args.batch), total-1)
            batch = {k: jnp.asarray(v[take]) for k, v in draws.items()}
            values = np.asarray(evaluate(batch, kwargs, mask))
            chunks.append(values[:min(args.batch, total-begin)])
        joint = np.concatenate(chunks).reshape(args.posterior_draws, args.effect_draws)
        value = log_mean_joint(joint)
        result = dict(model=args.model, fold=args.fold, id=participant, n_observations=n_trials,
            log_marginal_predictive=value, posterior_draws=args.posterior_draws,
            effect_draws=args.effect_draws,
            integration_weight_ess=float(np.exp(2*logsumexp(joint)-logsumexp(2*joint))))
        if args.compare_m is not None:
            smaller = log_mean_joint(joint[:, :args.compare_m])
            result.update(compare_m=args.compare_m, compare_log_marginal_predictive=smaller,
                          full_minus_compare=value-smaller)
        results.append(result)
        print(json.dumps({k: int(v) if isinstance(v, np.integer) else v for k, v in result.items()}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output, index=False)
    receipt = dict(**config, complete=True, scores_sha256=sha256(output),
                   posterior_flat_indices=indices.tolist(), total_seconds=time.perf_counter()-start)
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('tmp/robustness/m4'))
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--folds', type=Path)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--fold', type=int, choices=range(10), required=True)
    p.add_argument('--posterior-draws', '--S', type=int, default=400)
    p.add_argument('--effect-draws', '--M', type=int, default=100)
    p.add_argument('--compare-m', type=int)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--cpu-smoke', action='store_true')
    p.add_argument('--skip-completed', action='store_true')
    score(p.parse_args())
