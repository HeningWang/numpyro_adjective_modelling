"""Paired participant Bayesian bootstrap and strict robustness input readers."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/production_model_input.csv'
SEED = 20260906
DRAWS = 200_000
REGIMES = {
    'fixed': ('G-HO', 'I-HO', 'K-HO', 'K-HKO'),
    'updating': ('G-UPD-HO', 'I-UPD-HO', 'K-UPD-HO', 'K-UPD-HKO'),
}


def bootstrap_totals(values, *, draws=DRAWS, seed=SEED, batch=5000):
    """Dirichlet(1,...,1) weights on participant sums, scaled by N participants."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite, nonempty participant totals')
    if draws < 1 or batch < 1:
        raise ValueError('Draws and batch must be positive')
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, draws, batch):
        weights = rng.dirichlet(np.ones(len(values)), size=min(batch, draws-start))
        samples.append(len(values) * (weights @ values))
    samples = np.concatenate(samples)
    return dict(estimate_elpd=float(values.sum()), posterior_mean=float(samples.mean()),
                credible_lower_95=float(np.quantile(samples, .025)),
                credible_upper_95=float(np.quantile(samples, .975)),
                probability_positive=float((samples > 0).mean()),
                participant_count=len(values), posterior_draws=draws, seed=seed)


def load_rows(data, pointwise_dir, models):
    """Align by canonical key; reject missing/extra/duplicate rows and nonfinite LOO."""
    rows = pd.read_csv(data)
    key = 'canonical_row_position'
    if rows[key].isna().any() or rows[key].duplicated().any():
        raise ValueError('Dataset canonical keys must be unique and nonmissing')
    if rows[['id', 'combination', 'annotation']].isna().any().any():
        raise ValueError('Missing participant, family or response')
    for model in models:
        path = pointwise_dir / f'{model}_pointwise_loo.csv'
        if not path.exists():
            path = pointwise_dir / 'pointwise.csv'
        scores = pd.read_csv(path)[[key, model]]
        if scores[key].duplicated().any() or scores[key].isna().any():
            raise ValueError(f'{model}: duplicate or missing canonical key')
        if set(scores[key]) != set(rows[key]):
            raise ValueError(f'{model}: pointwise rows do not match dataset')
        if not np.isfinite(scores[model].to_numpy(dtype=float)).all():
            raise ValueError(f'{model}: nonfinite pointwise ELPD')
        rows = rows.merge(scores, on=key, validate='one_to_one')
    return rows.sort_values(key).reset_index(drop=True)


def participant_kappa(artifact, ids):
    """Runner encodes participants in sorted ID order, with zero-based coordinates."""
    ids = np.sort(np.asarray(ids))
    with xr.open_dataset(artifact, group='posterior', engine='h5netcdf') as posterior:
        k = posterior.kappa_by_participant
        dim, = [d for d in k.dims if d not in ('chain', 'draw')]
        if k.sizes[dim] != len(ids) or not np.array_equal(k[dim], np.arange(len(ids))):
            raise ValueError('Posterior participant coordinates do not match sorted dataset IDs')
        v = k.transpose('chain', 'draw', dim).values.reshape(-1, len(ids))
    if not np.isfinite(v).all() or ((v < 0) | (v > 1)).any():
        raise ValueError('Invalid participant kappa draws')
    return pd.DataFrame(dict(id=ids, kappa_mean=v.mean(axis=0),
        p_kappa_interior=((v > .1) & (v < .9)).mean(axis=0),
        p_kappa_low=(v < .1).mean(axis=0), p_kappa_high=(v > .9).mean(axis=0)))


def fresh_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Use an empty output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_readme(path, tables, notes):
    # Plain CSV blocks keep the report independent of optional table-format packages.
    text = '\n'.join(notes) + '\n'
    for title, table in tables.items():
        text += f'\n{title}\n\n```csv\n{table.to_csv(index=False)}```\n'
    (path / 'README.md').write_text(text)
