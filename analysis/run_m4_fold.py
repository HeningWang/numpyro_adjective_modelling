"""Fit one M4 (model, participant fold), independently of primary artifact contracts."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cuda')
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import argparse
import json
from pathlib import Path
import time

import arviz as az
import jax
import numpy as np
import pandas as pd
import numpyro
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

from m4_common import (DATA, MODELS, arguments, encoded, folds, job_path, production,
                       read_fit, sha256)


def run(args):
    fold_path = args.folds or args.root / 'fold_assignment.csv'
    assignment = folds(fold_path, args.data)
    if args.make_folds_only:
        print(assignment.groupby(['fold', 'sharpness']).size().to_string())
        return
    if args.model is None or args.fold is None:
        raise ValueError('--model and --fold are required for a fit')
    if not jax.config.x64_enabled or jax.default_backend() != ('cpu' if args.cpu_smoke else 'gpu'):
        raise RuntimeError('M4 requires x64 and CUDA, or explicit --cpu-smoke on CPU')
    if args.cpu_smoke:
        if not (1 <= args.warmup <= 20 and 4 <= args.samples <= 20 and 1 <= args.max_tree_depth <= 3):
            raise ValueError('CPU smoke is capped at 20 warmup/draws and depth 3')
    elif (args.warmup, args.samples, args.max_tree_depth) != (1000, 1000, 6):
        raise ValueError('M4 full-fit contract is 1000 warmup, 1000 draws, depth 6')
    path = job_path(args.root, args.model, args.fold)
    config = dict(model=args.model, fold=args.fold, dataset_sha256=sha256(args.data),
                  fold_assignment_sha256=sha256(fold_path), warmup=args.warmup,
                  samples=args.samples, max_tree_depth=args.max_tree_depth,
                  artifact_role='m4_smoke' if args.cpu_smoke else 'm4_fold')
    if path.exists() and any(path.iterdir()):
        if args.skip_completed:
            old = read_fit(path, allow_smoke=args.cpu_smoke)
            if all(old.get(k) == v for k, v in config.items()):
                print(f'Skip completed {args.model} fold {args.fold}', flush=True)
                return
        raise FileExistsError(f'Existing/incomplete M4 output; preserve it and use a fresh root: {path}')
    start = time.perf_counter()
    rows, data = encoded(args.data)
    train_ids = assignment.loc[assignment.fold.ne(args.fold), 'id'].to_numpy()
    heldout_ids = assignment.loc[assignment.fold.eq(args.fold), 'id'].to_numpy()
    if args.cpu_smoke:
        train_ids = train_ids[:2]
        heldout_ids = heldout_ids[:1]
    positions = np.flatnonzero(rows.id.isin(train_ids))
    kwargs, ids = arguments(rows, data, positions)
    ms, models = production()
    model = models[args.model]
    kwargs['precomputed_features'] = ms.precompute_principled_discovery_features(
        kwargs['states'], kwargs['is_sharp'], recursive='UPD' in args.model)
    trace = handlers.trace(handlers.seed(model, 0)).get_trace(**kwargs)
    sampled = [k for k, site in trace.items() if site['type'] == 'sample' and not site['is_observed']]
    population = [k for k in sampled if not k.startswith('z_')]
    path.mkdir(parents=True, exist_ok=True)
    manifest = dict(**config, train_participant_ids=ids.tolist(), heldout_participant_ids=heldout_ids.tolist(),
        train_canonical_rows=rows.iloc[positions].canonical_row_position.tolist(),
        chains=4, seed=4711, target_accept=.9, chain_method='vectorized',
        population_sites=population, participant_sites=[k for k in sampled if k.startswith('z_')],
        jax=jax.__version__, numpyro=numpyro.__version__, backend=jax.default_backend(),
        source_hashes={p.name: sha256(p)
                       for p in (Path(__file__), Path(ms.__file__), Path(__file__).with_name('m4_common.py'))},
        complete=False)
    manifest_path = path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    kernel = NUTS(model, target_accept_prob=.9, max_tree_depth=args.max_tree_depth)
    sampler = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples,
                   num_chains=4, chain_method='vectorized', progress_bar=False)
    _, key = jax.random.split(jax.random.PRNGKey(4711))
    sampler.run(key, extra_fields=('num_steps', 'accept_prob'), **kwargs)
    posterior = {k: np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()}
    if not all(np.isfinite(v).all() for v in posterior.values()):
        raise ValueError('Nonfinite posterior values')
    stats = {k: np.asarray(v) for k, v in sampler.get_extra_fields(group_by_chain=True).items()}
    stats['n_steps'] = stats.pop('num_steps')
    dims = {k: ['participants'] for k, v in posterior.items() if v.ndim == 3 and v.shape[-1] == len(ids)}
    idata = az.from_dict(posterior=posterior, sample_stats=stats,
                         coords={'participants': ids}, dims=dims)
    idata.attrs.update(model=args.model, fold=args.fold, artifact_role=config['artifact_role'])
    idata.to_netcdf(path / 'posterior.nc')
    diagnostics = az.summary(idata, var_names=sampled, kind='diagnostics', round_to='none')
    diagnostics.to_csv(path / 'parameter_diagnostics.csv', index_label='parameter')
    row = dict(model=args.model, fold=args.fold, max_rhat=float(diagnostics.r_hat.max()),
        divergences=int(stats['diverging'].sum()), train_participants=len(ids),
        heldout_participants=len(heldout_ids), train_rows=len(positions),
        warmup=args.warmup, samples=args.samples, max_tree_depth=args.max_tree_depth,
        total_seconds=time.perf_counter()-start)
    pd.DataFrame([row]).to_csv(path / 'diagnostics.csv', index=False)
    manifest.update(artifact_sha256=sha256(path / 'posterior.nc'), complete=True, **{'diagnostics': row})
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('tmp/robustness/m4'))
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--folds', type=Path)
    p.add_argument('--make-folds-only', action='store_true')
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--fold', type=int, choices=range(10))
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--max-tree-depth', type=int, default=6)
    p.add_argument('--cpu-smoke', action='store_true')
    p.add_argument('--skip-completed', action='store_true')
    run(p.parse_args())
