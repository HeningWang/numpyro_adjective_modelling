"""Terminal-utility (C0) and stable-order-term (C1) ablations: paired LOO contrasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from robustness_utils import (DATA, DRAWS, SEED, bootstrap_totals, fresh_output,
                              load_rows, write_readme)

ABLATIONS = {'terminal': ('terminal_utility', ('G-HO', 'K-HO', 'G-UPD-HO', 'K-UPD-HO')),
             'no_order': ('drop_order_term', ('K-HO', 'K-UPD-HO'))}


def checked_diagnostics(root, flag, models):
    rows = []
    for model in models:
        manifest = json.loads((root / 'primary' / f'{model}_manifest.json').read_text())
        if (not manifest['complete'] or manifest['artifact_role'] != 'ablation'
                or manifest.get(flag) is not True or manifest.get('drop_condition_terms')
                or tuple(manifest[k] for k in ('warmup', 'samples', 'max_tree_depth', 'seed'))
                != (1000, 1000, 6, 4711)):
            raise ValueError(f'{model}: expected a completed {flag} ablation')
        diagnostic = pd.read_csv(root / 'summaries/corrected_inputs' / f'{model}_diagnostics.csv')
        if not diagnostic.artifact_sha256.eq(manifest['artifact_sha256']).all():
            raise ValueError(f'{model}: diagnostics do not match the artifact')
        rows.append(diagnostic.assign(ablation=flag))
    return pd.concat(rows, ignore_index=True)


def shared_kappa(artifact):
    with xr.open_dataset(artifact, group='posterior', engine='h5netcdf') as posterior:
        values = posterior.kappa.values.ravel()
    return dict(mean=float(values.mean()), q025=float(np.quantile(values, .025)),
                q975=float(np.quantile(values, .975)))


def analyse(roots, primary_pointwise_dir, output, *, data=DATA, draws=DRAWS, seed=SEED):
    def summarize(rows, delta):
        totals = rows.assign(delta=np.asarray(delta)).groupby('id').delta.sum()
        return bootstrap_totals(totals, draws=draws, seed=seed)
    contrasts, kappas, diagnostics = [], [], []
    for name, (flag, models) in ABLATIONS.items():
        root = roots[name]
        diagnostics.append(checked_diagnostics(root, flag, models))
        rows = load_rows(data, root / 'summaries/corrected_inputs', models)
        primary = load_rows(data, primary_pointwise_dir, models)
        for model in models:
            contrasts.append(dict(ablation=name, contrast=f'primary {model} minus ablated',
                                  **summarize(rows, primary[model] - rows[model])))
        if name == 'terminal':
            for regime, (g, k) in {'fixed': ('G-HO', 'K-HO'),
                                   'updating': ('G-UPD-HO', 'K-UPD-HO')}.items():
                contrasts.append(dict(ablation=name, contrast=f'{k} minus {g} ({regime})',
                                      **summarize(rows, rows[k] - rows[g])))
            pairs = (('G-HO', 'G-UPD-HO'), ('K-HO', 'K-UPD-HO'))
        else:
            pairs = (('K-HO', 'K-UPD-HO'),)
        for fixed, updating in pairs:
            contrasts.append(dict(ablation=name, contrast=f'{updating} minus {fixed}',
                                  **summarize(rows, rows[updating] - rows[fixed])))
        for model in models:
            if model.startswith('K-'):
                kappas.append(dict(ablation=name, model=model,
                                   **shared_kappa(root / 'primary' / f'{model}.nc')))
    tables = {'contrasts': pd.DataFrame(contrasts), 'shared_kappa': pd.DataFrame(kappas),
              'diagnostics': pd.concat(diagnostics, ignore_index=True)}
    output = fresh_output(output)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    failed = tables['diagnostics'].loc[~tables['diagnostics'].sampler_gate_pass.eq(True), 'model']
    write_readme(output, tables, [
        'terminal: completed utterances scored by final target probability; local competition unchanged.',
        'no_order: stable order term removed (beta = 0) together with its participant hierarchy.',
        'Positive "primary minus ablated" values indicate predictive loss after ablation.',
        'Sampler gate failures: ' + (', '.join(failed.tolist()) if len(failed) else 'none') + '.',
        f'Participant Dirichlet(1,...,1) bootstrap, N-scaled totals: {draws} draws, seed {seed}.'])
    return tables


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--terminal-root', type=Path, required=True)
    p.add_argument('--no-order-root', type=Path, required=True)
    p.add_argument('--primary-pointwise-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=DATA)
    args = p.parse_args()
    analyse({'terminal': args.terminal_root, 'no_order': args.no_order_root},
            args.primary_pointwise_dir, args.output, data=args.data)


if __name__ == '__main__':
    main()
