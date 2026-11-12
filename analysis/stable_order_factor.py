"""Stable order term as a third factor: architecture x semantics x order term.

Crosses the six shared-kappa models with the stable order term (present or
removed, beta = 0) and adds an informativeness ablation (alpha = 0) per
architecture. All contrasts are paired participant Bayesian bootstraps.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from robustness_utils import (DATA, DRAWS, SEED, bootstrap_totals, fresh_output,
                              load_rows, write_readme)

ARCHITECTURES = {'G': 'global', 'K': 'plan-guided', 'I': 'fully incremental'}
REGIMES = {'fixed': '{a}-HO', 'updating': '{a}-UPD-HO'}
CONDITIONS = ('order', 'no_order', 'no_info')


def cell_name(architecture: str, regime: str) -> str:
    return REGIMES[regime].format(a=architecture)


def load_condition(data: Path, pointwise_dirs: list[Path], models: list[str],
                   condition: str) -> pd.DataFrame:
    """Load each model from the first directory that holds it, tagged by condition."""
    rows = None
    for model in models:
        source = next((d for d in pointwise_dirs
                       if (d / f'{model}_pointwise_loo.csv').exists()
                       or (d / 'pointwise.csv').exists()
                       and model in pd.read_csv(d / 'pointwise.csv', nrows=0).columns), None)
        if source is None:
            raise FileNotFoundError(f'{condition}: no pointwise LOO for {model}')
        loaded = load_rows(data, source, [model]).rename(columns={model: f'{model}|{condition}'})
        rows = loaded if rows is None else rows.merge(
            loaded[['canonical_row_position', f'{model}|{condition}']],
            on='canonical_row_position', validate='one_to_one')
    return rows


def checked_diagnostics(dirs: list[Path], models: list[str], flag: str) -> pd.DataFrame:
    frames = []
    for model in models:
        path = next((d / f'{model}_diagnostics.csv' for d in dirs
                     if (d / f'{model}_diagnostics.csv').exists()), None)
        if path is None:
            raise FileNotFoundError(f'{model}: missing {flag} diagnostics')
        frame = pd.read_csv(path)
        if not frame[flag].astype(bool).all() or not frame.artifact_role.eq('ablation').all():
            raise ValueError(f'{model}: diagnostics are not a {flag} ablation')
        frames.append(frame.assign(condition=flag))
    return pd.concat(frames, ignore_index=True)


def shared_kappa(artifact: Path) -> dict:
    with xr.open_dataset(artifact, group='posterior', engine='h5netcdf') as posterior:
        values = posterior.kappa.values.ravel()
    return dict(mean=float(values.mean()), q025=float(np.quantile(values, .025)),
                q975=float(np.quantile(values, .975)))


def analyse(primary_dir: Path, no_order_dirs: list[Path], no_info_dirs: list[Path],
            kappa_artifacts: dict[str, Path], output: Path, *, data: Path = DATA,
            draws: int = DRAWS, seed: int = SEED) -> dict[str, pd.DataFrame]:
    six = [cell_name(a, r) for a in ARCHITECTURES for r in REGIMES]
    no_info = [cell_name(a, 'fixed') for a in ARCHITECTURES]
    rows = load_condition(data, [primary_dir], six, 'order')
    for frame in (load_condition(data, no_order_dirs, six, 'no_order'),
                  load_condition(data, no_info_dirs, no_info, 'no_info')):
        scores = [c for c in frame.columns if '|' in c]
        rows = rows.merge(frame[['canonical_row_position', *scores]],
                          on='canonical_row_position', validate='one_to_one')

    def contrast(label: str, a: str, b: str) -> dict:
        totals = rows.assign(delta=rows[a] - rows[b]).groupby('id').delta.sum()
        return dict(contrast=label, minuend=a, subtrahend=b,
                    **bootstrap_totals(totals, draws=draws, seed=seed))

    landscape = []
    for column in [c for c in rows.columns if '|' in c]:
        model, condition = column.split('|')
        architecture = model.split('-')[0]
        landscape.append(dict(model=model, architecture=ARCHITECTURES[architecture],
                              semantics='updating' if 'UPD' in model else 'fixed',
                              condition=condition, elpd_loo=float(rows[column].sum())))

    contrasts = []
    for condition in ('order', 'no_order'):
        for regime in REGIMES:
            k, g, i = (f'{cell_name(a, regime)}|{condition}' for a in 'KGI')
            contrasts += [contrast(f'{condition}, {regime}: plan-guided minus global', k, g),
                          contrast(f'{condition}, {regime}: plan-guided minus incremental', k, i)]
        for a in ARCHITECTURES:
            contrasts.append(contrast(
                f'{condition}, {ARCHITECTURES[a]}: updating minus fixed',
                f'{cell_name(a, "updating")}|{condition}', f'{cell_name(a, "fixed")}|{condition}'))
    for model in six:
        contrasts.append(contrast(f'{model}: with minus without order term',
                                  f'{model}|order', f'{model}|no_order'))
    for a in ARCHITECTURES:
        model = cell_name(a, 'fixed')
        contrasts.append(contrast(f'{model}: with minus without informativeness',
                                  f'{model}|order', f'{model}|no_info'))
    contrasts += [contrast('no_info, fixed: plan-guided minus global', 'K-HO|no_info', 'G-HO|no_info'),
                  contrast('no_info, fixed: plan-guided minus incremental',
                           'K-HO|no_info', 'I-HO|no_info')]

    kappas = pd.DataFrame([dict(cell=label, **shared_kappa(path))
                           for label, path in kappa_artifacts.items()])
    diagnostics = pd.concat([checked_diagnostics(no_order_dirs, six, 'drop_order_term'),
                             checked_diagnostics(no_info_dirs, no_info, 'drop_informativeness')],
                            ignore_index=True)
    tables = {'landscape': pd.DataFrame(landscape), 'contrasts': pd.DataFrame(contrasts),
              'shared_kappa': kappas, 'diagnostics': diagnostics}
    output = fresh_output(output)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    failed = diagnostics.loc[~diagnostics.sampler_gate_pass.eq(True), 'model']
    write_readme(output, tables, [
        'order: paper primary fits; no_order: beta = 0 with its participant hierarchy removed;',
        'no_info: alpha = 0 with its participant hierarchy removed (semantics has no effect).',
        'Sampler gate failures: ' + (', '.join(failed.tolist()) if len(failed) else 'none') + '.',
        f'Participant Dirichlet(1,...,1) bootstrap, N-scaled totals: {draws} draws, seed {seed}.'])
    return tables


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary-pointwise-dir', type=Path, required=True)
    p.add_argument('--no-order-dir', type=Path, action='append', required=True)
    p.add_argument('--no-info-dir', type=Path, action='append', required=True)
    p.add_argument('--kappa-artifact', action='append', default=[],
                   help='label=path to a K-cell .nc for the shared kappa summary')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=DATA)
    args = p.parse_args()
    kappas = dict(item.split('=', 1) for item in args.kappa_artifact)
    analyse(args.primary_pointwise_dir, args.no_order_dir, args.no_info_dir,
            {k: Path(v) for k, v in kappas.items()}, args.output, data=args.data)


if __name__ == '__main__':
    main()
