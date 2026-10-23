"""M2 updating-minus-fixed predictive contrasts and shared kappa by fixed nu_C."""
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


def shared_kappa(campaign, model, colour, data=DATA):
    artifact = campaign / 'primary' / f'{model}.nc'
    manifest_path = artifact.with_name(f'{model}_manifest.json')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (not manifest['complete'] or manifest['artifact_sha256'] !=
                hashlib.sha256(artifact.read_bytes()).hexdigest()):
            raise ValueError('Incomplete artifact or hash mismatch')
        if manifest['dataset_sha256'] != hashlib.sha256(Path(data).read_bytes()).hexdigest():
            raise ValueError('Artifact dataset hash mismatch')
        if manifest.get('colour_reliability', .59) != colour:
            raise ValueError('Artifact colour reliability mismatch')
        if colour != .59 and manifest.get('artifact_role') != 'sensitivity':
            raise ValueError('Expected full sensitivity fit')
    elif colour != .59:
        raise ValueError('Sensitivity fits require a manifest recording the override')
    with xr.open_dataset(artifact, group='posterior', engine='h5netcdf') as posterior:
        values = posterior.kappa.values.ravel()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid shared kappa')
    return dict(mean=float(values.mean()), q025=float(np.quantile(values, .025)),
                q975=float(np.quantile(values, .975)))


def analyse(campaigns, output, *, data=DATA, draws=DRAWS, seed=SEED):
    results = []
    for colour, campaign in campaigns:
        rows = load_rows(data, campaign / 'summaries/corrected_inputs', ('K-HO', 'K-UPD-HO'))
        delta = rows.assign(delta=rows['K-UPD-HO']-rows['K-HO']).groupby('id').delta.sum()
        result = dict(colour_reliability=colour, contrast='K-UPD-HO minus K-HO',
                      **bootstrap_totals(delta, draws=draws, seed=seed))
        for regime, model in (('fixed', 'K-HO'), ('updating', 'K-UPD-HO')):
            result.update({f'{regime}_kappa_{k}': v for k, v in
                           shared_kappa(campaign, model, colour, data).items()})
        results.append(result)
    result = pd.DataFrame(results)
    output = fresh_output(output)
    result.to_csv(output / 'colour_reliability_sensitivity.csv', index=False)
    write_readme(output, {'Sensitivity': result}, [
        'Updating-minus-fixed paired pointwise ELPD, summed within participant.',
        f'Bayesian bootstrap: participant Dirichlet(1,...,1), N-scaled totals, {draws} draws, seed {seed}.',
        'Shared kappa intervals use the saved posterior draws for each fit.',
        'Legacy baseline artifacts without manifests use the original fixed nu_C=.59 contract.'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--nu075', type=Path, required=True)
    p.add_argument('--nu090', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--draws', type=int, default=DRAWS)
    p.add_argument('--seed', type=int, default=SEED)
    args = p.parse_args()
    analyse([(.59, args.baseline), (.75, args.nu075), (.90, args.nu090)],
            args.output, data=args.data, draws=args.draws, seed=args.seed)


if __name__ == '__main__':
    main()
