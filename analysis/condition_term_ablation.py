"""M5 paired pointwise-LOO contrasts, primary losses and ablated shared kappa."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from robustness_utils import (DATA, DRAWS, SEED, bootstrap_totals, fresh_output,
                              load_rows, write_readme)

REGIMES = {'fixed': ('G-HO', 'I-HO', 'K-HO'),
           'updating': ('G-UPD-HO', 'I-UPD-HO', 'K-UPD-HO')}
MODELS = tuple(model for models in REGIMES.values() for model in models)


def ablation_diagnostics(campaign, data):
    diagnostics = []
    for model in MODELS:
        artifact = campaign / 'primary' / f'{model}.nc'
        manifest = json.loads(artifact.with_name(f'{model}_manifest.json').read_text())
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if (not manifest['complete'] or manifest['artifact_sha256'] != digest
                or manifest['dataset_sha256'] != hashlib.sha256(Path(data).read_bytes()).hexdigest()
                or manifest['model'] != model or manifest['artifact_role'] != 'ablation'
                or manifest.get('drop_condition_terms') is not True
                or manifest.get('colour_reliability', .59) != .59
                or manifest.get('colour_reliability_override') is not None):
            raise ValueError(f'{model}: expected a completed default-colour M5 ablation')
        if tuple(manifest[k] for k in ('warmup', 'samples', 'max_tree_depth', 'chains', 'seed')) != (
                1000, 1000, 6, 4, 4711):
            raise ValueError(f'{model}: ablation sampler contract mismatch')
        diagnostic = pd.read_csv(campaign / 'summaries/corrected_inputs' / f'{model}_diagnostics.csv')
        if (len(diagnostic) != 1 or not diagnostic.model.eq(model).all()
                or not diagnostic.artifact_role.eq('ablation').all()
                or not diagnostic.artifact_sha256.eq(digest).all()):
            raise ValueError(f'{model}: missing or mismatched ablation diagnostics')
        diagnostics.append(diagnostic)
    return pd.concat(diagnostics, ignore_index=True)


def analyse(campaign, primary, output, *, primary_pointwise_dir=None,
            data=DATA, draws=DRAWS, seed=SEED):
    diagnostics = ablation_diagnostics(campaign, data)
    rows = load_rows(data, campaign / 'summaries/corrected_inputs', MODELS)
    baseline = load_rows(data, primary_pointwise_dir if primary_pointwise_dir is not None else
                         primary / 'summaries/corrected_inputs', MODELS)
    contrasts, losses, kappas = [], [], []
    def summarize(delta):
        totals = rows.assign(delta=np.asarray(delta)).groupby('id').delta.sum()
        return bootstrap_totals(totals, draws=draws, seed=seed)
    for endpoint in (0, 1):
        deltas = []
        for regime, models in REGIMES.items():
            delta = rows[models[2]] - rows[models[endpoint]]
            deltas.append(delta)
            contrasts.append(dict(regime=regime, contrast=f'{models[2]} minus {models[endpoint]}',
                                  **summarize(delta)))
        contrasts.append(dict(regime='average', contrast='plan-guided minus ' +
                              ('global' if endpoint == 0 else 'fully incremental'),
                              **summarize((deltas[0] + deltas[1]) / 2)))
    for fixed, updating in zip(REGIMES['fixed'], REGIMES['updating']):
        contrasts.append(dict(regime='updating minus fixed', contrast=f'{updating} minus {fixed}',
                              **summarize(rows[updating] - rows[fixed])))
    for model in MODELS:
        losses.append(dict(model=model, contrast='primary minus ablated',
                           **summarize(baseline[model] - rows[model])))
    for regime, models in REGIMES.items():
        model = models[2]
        with xr.open_dataset(campaign / 'primary' / f'{model}.nc',
                             group='posterior', engine='h5netcdf') as posterior:
            values = posterior.kappa.values.ravel()
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f'{model}: invalid shared kappa')
        kappas.append(dict(regime=regime, model=model, mean=float(values.mean()),
                           q025=float(np.quantile(values, .025)), q975=float(np.quantile(values, .975))))
    tables = {'contrasts': pd.DataFrame(contrasts), 'primary_losses': pd.DataFrame(losses),
              'shared_kappa': pd.DataFrame(kappas), 'diagnostics': diagnostics}
    output = fresh_output(output)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    failed = diagnostics.loc[~diagnostics.sampler_gate_pass.eq(True), 'model'].tolist()
    write_readme(output, tables, [
        'Both condition-specific auxiliary coefficients are fixed at exactly zero.',
        'Sampler gate failures: ' + (', '.join(failed) if failed else 'none') + '.',
        'Inspect failed gates before interpreting contrasts; all six fits use the standard contract.',
        'Regime averages are arithmetic means of paired trial differences before participant aggregation.',
        'Positive primary-minus-ablated values indicate predictive loss after ablation.',
        f'Participant Dirichlet(1,...,1) bootstrap, N-scaled totals: {draws} draws, seed {seed}.',
        'Shared kappa summaries use ablated posterior draws.'])
    print('Sampler gate failures: ' + (', '.join(failed) if failed else 'none'))
    return tables


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--primary', type=Path, required=True)
    p.add_argument('--primary-pointwise-dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--draws', type=int, default=DRAWS)
    p.add_argument('--seed', type=int, default=SEED)
    args = p.parse_args()
    analyse(args.campaign, args.primary, args.output, primary_pointwise_dir=args.primary_pointwise_dir,
            data=args.data, draws=args.draws, seed=args.seed)


if __name__ == '__main__':
    main()
