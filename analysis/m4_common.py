"""Shared M4 fold/data handling; existing production models remain unchanged."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/production_model_input.csv'
MODELS = ('G-HO', 'I-HO', 'K-HO', 'K-HKO', 'K-UPD-HO')
SEED = 20260906
FIELDS = dict(states='states_train', empirical='empirical_seq_flat',
              sufficient_dim='sufficient_dim', has_one_word_solution='has_one_word_solution',
              is_sharp='sharpness_idx', is_colour_sufficient='is_colour_sufficient')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def folds(path, data=DATA):
    """Create once, then require the same stratified, seeded participant assignment."""
    from sklearn.model_selection import StratifiedKFold
    rows = pd.read_csv(data)
    if not rows.groupby('id').sharpness.nunique().eq(1).all():
        raise ValueError('Size discriminability must be between participants')
    table = rows.groupby('id', as_index=False, sort=True).sharpness.first()
    table['fold'] = -1
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for fold, (_, heldout) in enumerate(splitter.split(table.id, table.sharpness)):
        table.loc[heldout, 'fold'] = fold
    table['seed'] = SEED
    path = Path(path)
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), table)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
    return table


def production():
    sys.path.insert(0, str(ROOT / 'models/production'))
    import modelSpecification as ms
    models = {**ms.V10_ORDER_MODELS, **ms.V11_JOINT_PARTICIPANT_MODELS,
              'K-UPD-HO': ms.V10_ORDER_UPD_MODELS['K-UPD-HO']}
    return ms, models


def encoded(data=DATA):
    sys.path.insert(0, str(ROOT / 'models/production'))
    from helper import import_dataset_hier
    rows = pd.read_csv(data)
    d = import_dataset_hier(file_path=data, state_encoding='target_match')
    if len(d['states_train']) != len(rows):
        raise ValueError('Encoded data do not preserve the canonical input rows')
    np.testing.assert_array_equal(rows.canonical_row_position, np.arange(len(rows)))
    return rows, d


def arguments(rows, data, positions):
    import jax.numpy as jnp
    positions = np.asarray(positions)
    ids, index = np.unique(rows.iloc[positions].id, return_inverse=True)
    kwargs = {key: jnp.asarray(data[field][positions]) for key, field in FIELDS.items()}
    kwargs.update(participant_idx=jnp.asarray(index), n_participants=len(ids))
    return kwargs, ids


def job_path(root, model, fold):
    return Path(root) / 'fits' / f'fold_{fold:02d}' / model


def read_fit(path, *, allow_smoke=False):
    manifest = json.loads((path / 'manifest.json').read_text())
    if (not manifest.get('complete') or
            manifest.get('artifact_role') not in (('m4_fold', 'm4_smoke') if allow_smoke else ('m4_fold',))):
        raise ValueError(f'Incomplete or inappropriate M4 fit: {path}')
    if sha256(path / 'posterior.nc') != manifest['artifact_sha256']:
        raise ValueError(f'Artifact hash mismatch: {path}')
    return manifest
