"""M1 participant architecture checks and M3 family uncertainty from saved fits."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from robustness_utils import (DATA, DRAWS, REGIMES, SEED, bootstrap_totals, fresh_output,
                              load_rows, participant_kappa, write_readme)


def participant_architecture(rows, models):
    g, i, k, ki = models
    totals = rows.groupby('id', sort=True)[list(models)].sum().rename(
        columns={g: 'elpd_G', i: 'elpd_I', k: 'elpd_K', ki: 'elpd_Ki'})
    totals['elpd_oracle'] = totals[['elpd_G', 'elpd_I']].max(axis=1)
    totals['Ki_minus_G'] = totals.elpd_Ki-totals.elpd_G
    totals['Ki_minus_I'] = totals.elpd_Ki-totals.elpd_I
    for endpoint, name in ((g, 'G'), (i, 'I')):
        delta = rows.assign(delta=rows[ki]-rows[endpoint]).groupby('id').delta
        # SE of a participant's summed paired trial differences. Undefined for N=1.
        totals[f'se_Ki_minus_{name}'] = np.sqrt(delta.count()) * delta.std(ddof=1)
    totals['Ki_exceeds_both'] = (totals.Ki_minus_G > 0) & (totals.Ki_minus_I > 0)
    totals['Ki_exceeds_both_2se'] = (
        (totals.Ki_minus_G > 2*totals.se_Ki_minus_G) &
        (totals.Ki_minus_I > 2*totals.se_Ki_minus_I))
    return totals.reset_index()


def m1(rows, campaign, output, *, draws=DRAWS, seed=SEED):
    participants, summaries, counts, totals = [], [], [], []
    for regime, models in REGIMES.items():
        table = participant_architecture(rows, models)
        artifact = campaign / 'primary' / f'{models[3]}.nc'
        posterior_available = artifact.exists()
        if posterior_available:
            table = table.merge(participant_kappa(artifact, rows.id.unique()),
                                on='id', validate='one_to_one')
        else:
            for column in ('kappa_mean', 'p_kappa_interior', 'p_kappa_low', 'p_kappa_high'):
                table[column] = np.nan
        table['posterior_available'] = posterior_available
        table['regime'] = regime
        table['near_endpoint'] = np.select(
            [table.kappa_mean < .1, table.kappa_mean > .9], ['global', 'incremental'], 'interior' if posterior_available else 'unavailable')
        # Positive gap means the participant model predicts better than the relevant endpoint.
        table['Ki_minus_near_endpoint'] = np.where(table.kappa_mean < .1,
            table.Ki_minus_G, np.where(table.kappa_mean > .9, table.Ki_minus_I, np.nan))
        participants.append(table)
        for label, left, right in (
            ('Ki_minus_oracle', 'elpd_Ki', 'elpd_oracle'),
            ('K_minus_oracle', 'elpd_K', 'elpd_oracle'),
            ('Ki_minus_K', 'elpd_Ki', 'elpd_K')):
            summaries.append(dict(regime=regime, contrast=label,
                **bootstrap_totals(table[left]-table[right], draws=draws, seed=seed)))
        totals.append(dict(regime=regime, **table.filter(regex='^elpd_').sum().to_dict()))
        row = dict(regime=regime, participant_count=len(table),
            Ki_exceeds_both=int(table.Ki_exceeds_both.sum()),
            Ki_exceeds_both_2se=int(table.Ki_exceeds_both_2se.sum()),
            p_interior_gt_half=int((table.p_kappa_interior > .5).sum()) if posterior_available else np.nan)
        for label in ('global', 'incremental'):
            gaps = table.loc[table.near_endpoint.eq(label), 'Ki_minus_near_endpoint']
            row.update({f'{label}_count': len(gaps) if posterior_available else np.nan,
                        f'{label}_gap_sum': gaps.sum() if posterior_available else np.nan,
                        f'{label}_gap_mean': gaps.mean(), f'{label}_gap_median': gaps.median()})
        counts.append(row)
    output = fresh_output(output)
    summary, counts, totals = pd.DataFrame(summaries), pd.DataFrame(counts), pd.DataFrame(totals)
    summary.to_csv(output / 'participant_architecture_summary.csv', index=False)
    pd.concat(participants).to_csv(output / 'participant_architecture.csv', index=False)
    counts.to_csv(output / 'participant_counts.csv', index=False)
    totals.to_csv(output / 'architecture_totals.csv', index=False)
    write_readme(output, {'Contrasts': summary, 'Totals': totals, 'Counts and endpoint gaps': counts}, [
        'Oracle = max(sum G trials, sum I trials) within participant; it has an in-sample selection advantage.',
        'SE = sqrt(N trials) * sample SD of paired pointwise differences within participant.',
        'The >2 SE criterion must hold against both endpoints; SE is undefined for one trial.',
        'Kappa participant order follows the runner: sorted dataset IDs.',
        'Posterior probabilities and endpoint-kappa summaries are unavailable when primary/<model>.nc is absent; CSV fields are missing, not zero.',
        f'Bayesian bootstrap: participant Dirichlet(1,...,1), N-scaled totals, {draws} draws, seed {seed}.'])
    return summary


def family_contrasts(rows, *, draws=DRAWS, seed=SEED):
    groups = [(family, 'all', group) for family, group in rows.groupby('combination')]
    dc = rows.loc[rows.combination.eq('dimension_color')]
    for contains in (False, True):
        groups.append(('dimension_color', 'contains_form' if contains else 'no_form',
                       dc.loc[dc.annotation.str.contains('F', regex=False).eq(contains)]))
    summaries = []
    for family, subset, group in groups:
        if group.empty:
            continue
        for model in ('K-HO', 'K-HKO'):
            for endpoint, reference in (
                ('average_endpoints', (group['G-HO']+group['I-HO'])/2),
                ('G-HO', group['G-HO']), ('I-HO', group['I-HO'])):
                values = group.assign(delta=group[model]-reference).groupby('id').delta.sum()
                summaries.append(dict(combination=family, response_subset=subset, model=model,
                    reference=endpoint, n_observations=len(group),
                    **bootstrap_totals(values, draws=draws, seed=seed)))
    return pd.DataFrame(summaries)


def check_family_totals(summary, reference, tolerance):
    ref = pd.read_csv(reference).loc[lambda t: t.grouping_type.eq('adjective_family')]
    checks = []
    for endpoint, col in (
        ('average_endpoints', 'total_plan_vs_average_endpoints'),
        ('G-HO', 'total_plan_vs_global'), ('I-HO', 'total_plan_vs_fully_incremental')):
        for _, row in ref.iterrows():
            selected = summary.loc[summary.model.eq('K-HO') &
                summary.response_subset.eq('all') & summary.combination.eq(row.combination) &
                summary.reference.eq(endpoint)]
            if len(selected) != 1:
                raise ValueError('Missing or duplicate family contrast')
            actual = selected.estimate_elpd.iloc[0]
            checks.append(dict(combination=row.combination, reference=endpoint,
                expected=row[col], observed=actual, error=actual-row[col], tolerance=tolerance,
                passed=abs(actual-row[col]) <= tolerance))
    return pd.DataFrame(checks)


def m3(rows, output, *, reference=None, tolerance=.02, draws=DRAWS, seed=SEED):
    result = family_contrasts(rows, draws=draws, seed=seed)
    output = fresh_output(output)
    result.to_csv(output / 'family_contrasts.csv', index=False)
    tables = {'Family contrasts': result}
    checks = None
    if reference is not None:
        checks = check_family_totals(result, reference, tolerance)
        checks.to_csv(output / 'acceptance_checks.csv', index=False)
        tables['Point-total acceptance checks'] = checks
    write_readme(output, tables, [
        'Paired trial differences summed within participant and family/subset.',
        'Each subset bootstraps participants observed in that subset; counts are in the CSV.',
        f'Bayesian bootstrap: participant Dirichlet(1,...,1), N-scaled totals, {draws} draws, seed {seed}.',
        'Acceptance tolerance is absolute ELPD; a refit requires a documented Monte Carlo tolerance.'])
    if checks is not None and not checks.passed.all():
        raise ValueError(f'Family acceptance check failed; see {output}/acceptance_checks.csv')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('m1', 'm3'), required=True)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--pointwise-dir', type=Path)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--draws', type=int, default=DRAWS)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--reference', type=Path,
                   help='M3 frozen localization CSV; mismatches save evidence then fail')
    p.add_argument('--tolerance', type=float, default=.02)
    args = p.parse_args()
    if args.tolerance < 0 or not np.isfinite(args.tolerance):
        p.error('Tolerance must be finite and nonnegative')
    models = list(sum(REGIMES.values(), ())) if args.stage == 'm1' else list(REGIMES['fixed'])
    rows = load_rows(args.data, args.pointwise_dir or
                     args.campaign / 'summaries/corrected_inputs', models)
    if args.stage == 'm1':
        m1(rows, args.campaign, args.output, draws=args.draws, seed=args.seed)
    else:
        m3(rows, args.output, reference=args.reference, tolerance=args.tolerance,
           draws=args.draws, seed=args.seed)


if __name__ == '__main__':
    main()
