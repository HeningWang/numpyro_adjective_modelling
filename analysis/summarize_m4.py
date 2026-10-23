"""Assemble 50 M4 fold scores, sampler diagnostics and participant-bootstrap contrasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from m4_common import DATA, MODELS, SEED, folds, job_path, read_fit, sha256
from robustness_utils import bootstrap_totals, write_readme

CONTRASTS = (('K-HO', 'G-HO'), ('K-HO', 'I-HO'), ('K-HKO', 'K-HO'), ('K-UPD-HO', 'K-HO'))


def summarize(args):
    fold_path = args.root / 'fold_assignment.csv'
    assignment = folds(fold_path, args.data)
    observed = pd.read_csv(args.data).groupby('id').size()
    scores, diagnostics, receipts = [], [], {}
    for fold in range(10):
        expected_ids = set(assignment.loc[assignment.fold.eq(fold), 'id'])
        for model in MODELS:
            path = job_path(args.root, model, fold)
            manifest = read_fit(path)
            if (manifest['dataset_sha256'] != sha256(args.data) or
                    manifest['fold_assignment_sha256'] != sha256(fold_path) or
                    set(manifest['heldout_participant_ids']) != expected_ids or
                    set(manifest['train_participant_ids']) != set(assignment.id)-expected_ids or
                    (manifest['warmup'], manifest['samples'], manifest['max_tree_depth'],
                     manifest['chains'], manifest['seed']) != (1000, 1000, 6, 4, 4711)):
                raise ValueError(f'M4 full-fold contract mismatch: {path}')
            score_path = args.root / 'scores' / f'fold_{fold:02d}' / f'{model}.csv'
            receipt_path = score_path.with_name(f'{model}_manifest.json')
            receipt = json.loads(receipt_path.read_text())
            if (not receipt.get('complete') or receipt.get('cpu_smoke') or
                    receipt['fit_artifact_sha256'] != manifest['artifact_sha256'] or
                    receipt['scores_sha256'] != sha256(score_path) or
                    receipt['posterior_draws'] != args.posterior_draws or
                    receipt['effect_draws'] != args.effect_draws or receipt['seed'] != SEED):
                raise ValueError(f'Incomplete/mismatched held-out scores: {score_path}')
            table = pd.read_csv(score_path)
            if (len(table) != len(expected_ids) or set(table.id) != expected_ids or
                    not table.model.eq(model).all() or not table.fold.eq(fold).all() or
                    not np.isfinite(table.log_marginal_predictive).all() or
                    not table.set_index('id').n_observations.eq(observed.loc[table.id].to_numpy()).all()):
                raise ValueError(f'Invalid participant scores: {score_path}')
            if fold == args.stability_fold and receipt['compare_m'] != args.compare_m:
                raise ValueError('Missing M=50/M=100 stability comparison on designated fold')
            scores.append(table)
            diag_path = path / 'diagnostics.csv'
            diag = pd.read_csv(diag_path)
            if len(diag) != 1 or not diag.model.eq(model).all() or not diag.fold.eq(fold).all():
                raise ValueError(f'Invalid fold diagnostics: {diag_path}')
            diagnostics.append(diag)
            receipts[str(score_path)] = sha256(score_path)
            receipts[str(receipt_path)] = sha256(receipt_path)
            receipts[str(diag_path)] = sha256(diag_path)
    output = args.output or args.root
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / 'summary_manifest.json'
    config = dict(inputs=receipts, fold_assignment_sha256=sha256(fold_path),
                  source_sha256=sha256(Path(__file__)), bootstrap_draws=200_000, seed=SEED)
    names = ('heldout_participant_scores.csv', 'cv_contrasts.csv', 'fold_diagnostics.csv',
             'monte_carlo_stability.csv', 'README.md')
    if receipt_path.exists() or any((output / n).exists() for n in names):
        if args.skip_completed and receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            if (old.get('complete') and all(old.get(k) == v for k, v in config.items()) and
                    all((output / n).exists() and old['outputs'].get(n) == sha256(output / n) for n in names)):
                print('Skip completed M4 summary')
                return
        raise FileExistsError('Existing/incomplete M4 summary; choose a fresh --output directory')
    scores = pd.concat(scores, ignore_index=True).sort_values(['id', 'model'])
    wide = scores.pivot(index='id', columns='model', values='log_marginal_predictive')
    if wide.isna().any().any() or len(wide) != len(assignment):
        raise ValueError('Every participant must have exactly one score for every model')
    contrasts = pd.DataFrame([dict(contrast=f'{a} minus {b}',
        **bootstrap_totals(wide[a]-wide[b])) for a, b in CONTRASTS])
    diagnostics = pd.concat(diagnostics, ignore_index=True)
    subset = scores.loc[scores.fold.eq(args.stability_fold)]
    stability = subset.groupby('model', as_index=False).agg(
        participants=('id', 'size'), total_full=('log_marginal_predictive', 'sum'),
        total_compare=('compare_log_marginal_predictive', 'sum'),
        full_minus_compare=('full_minus_compare', 'sum'),
        max_abs_participant_difference=('full_minus_compare', lambda v: v.abs().max()))
    stability = stability.assign(fold=args.stability_fold, posterior_draws=args.posterior_draws,
                                 effect_draws=args.effect_draws, compare_m=args.compare_m)
    scores.to_csv(output / names[0], index=False)
    contrasts.to_csv(output / names[1], index=False)
    diagnostics.to_csv(output / names[2], index=False)
    stability.to_csv(output / names[3], index=False)
    write_readme(output, {'Held-out contrasts': contrasts, 'Monte Carlo stability': stability,
                         'Fold diagnostics': diagnostics}, [
        'Ten participant folds stratified by between-participant size discriminability; seed 20260906.',
        'Each participant score jointly marginalizes all responses over posterior and new-participant effects.',
        'Fresh standard-normal participant effects use the existing model non-centred transformations.',
        f'S={args.posterior_draws}, M={args.effect_draws}; fold {args.stability_fold} also uses M={args.compare_m}.',
        'Bootstrap: N-scaled participant Dirichlet(1,...,1), 200000 draws, seed 20260906.',
        f'Maximum fold R-hat: {diagnostics.max_rhat.max():.6g}; total divergences: {diagnostics.divergences.sum()}.',
        'Inspect fold diagnostics and M stability before scientific interpretation; no primary artifact gate was applied.'])
    config.update(complete=True, outputs={name: sha256(output / name) for name in names})
    receipt_path.write_text(json.dumps(config, indent=2)+'\n')
    print(contrasts.to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('tmp/robustness/m4'))
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--output', type=Path)
    p.add_argument('--posterior-draws', type=int, default=400)
    p.add_argument('--effect-draws', type=int, default=100)
    p.add_argument('--stability-fold', type=int, choices=range(10), default=0)
    p.add_argument('--compare-m', type=int, default=50)
    p.add_argument('--skip-completed', action='store_true')
    summarize(p.parse_args())
