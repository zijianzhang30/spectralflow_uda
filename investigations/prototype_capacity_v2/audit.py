"""Three-seed target audit after every new arm/selection has been frozen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import hdf5storage
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (1341, 1174, 1370)
SELECTIONS = ('last', 'best_source_val')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def score(probabilities, labels, full_n):
    pred = probabilities.argmax(1)
    cm = confusion_matrix(labels, pred, labels=np.arange(7))
    recall = np.diag(cm) / cm.sum(1)
    return dict(oa=float((pred == labels).sum() / full_n), aa=float(recall.mean()),
                kappa=float(cohen_kappa_score(labels, pred)),
                recall=recall.tolist(), confusion_matrix=cm.tolist(),
                evaluated_n=len(labels), denominator=full_n)


def main():
    destination = HERE / 'AUDIT.json'
    assert not destination.exists()
    frozen = {}
    # Validate all prediction/checkpoint hashes and center order before opening target GT.
    for seed in SEEDS:
        for mode in ('prototype', 'global'):
            run = ((ROOT / 'investigations/class_relation_v1' / f'formal_prototype_{seed}')
                   if seed == 1341 and mode == 'prototype' else
                   HERE / f'formal_{mode}_{seed}')
            for selection in SELECTIONS:
                manifest_path = run / f'{selection}_freeze.json'
                prediction_path = run / f'{selection}_predictions_before_gt.npz'
                checkpoint_path = run / f'{selection}.pth'
                manifest = json.loads(manifest_path.read_text())
                assert manifest['target_gt_accessed'] is False
                assert manifest['seed'] == seed and manifest['mode'] == mode
                assert manifest['selection'] == selection
                assert sha256(prediction_path) == manifest['prediction_sha256']
                assert sha256(checkpoint_path) == manifest['checkpoint_sha256']
                with np.load(prediction_path) as z:
                    frozen[(seed, mode, selection)] = (
                        z['centers'].copy(), z['probabilities'].copy(), manifest)

    baseline_fixed = json.loads((ROOT / 'investigations/mluda_fixed100_v1/AUDIT.json').read_text())['seeds']
    result = {'scope': 'Houston official three-seed follow-up; all new target predictions frozen before audit',
              'seeds': {}, 'summary': {}}
    for seed in SEEDS:
        centers = frozen[(seed, 'prototype', 'last')][0]
        assert centers.shape == (53200, 2)
        assert all(np.array_equal(centers, frozen[(seed, mode, selection)][0])
                   for mode in ('prototype', 'global') for selection in SELECTIONS)
        cfg_path = (ROOT / 'investigations/class_relation_v1' / f'formal_prototype_{seed}' /
                    'config.json') if seed == 1341 else HERE / f'formal_prototype_{seed}' / 'config.json'
        cfg = json.loads(cfg_path.read_text())
        target_gt = hdf5storage.loadmat(str(Path(cfg['data']) / 'Houston18_7gt.mat'))['map']
        y_full = target_gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
        y = y_full[:53184]
        assert y.min() == 0 and y.max() == 6
        official_best = json.loads((ROOT / 'official_aligned/runs/round1/formal' /
                                    str(seed) / 'MLUDA_full/final_target.json').read_text())
        result['seeds'][str(seed)] = {
            'official_mluda': {
                'last': {key: baseline_fixed[str(seed)][key] for key in ('oa', 'aa', 'kappa', 'recall')},
                'best_source_val': {key: official_best[key] for key in ('oa', 'aa', 'kappa')},
            }, 'prototype': {}, 'global': {}}
        for mode in ('prototype', 'global'):
            for selection in SELECTIONS:
                _, probabilities, manifest = frozen[(seed, mode, selection)]
                assert probabilities.shape == (53184, 7)
                assert np.isfinite(probabilities).all()
                assert np.max(np.abs(probabilities.sum(1) - 1)) < 1e-5
                result['seeds'][str(seed)][mode][selection] = {
                    'metrics': score(probabilities, y, len(y_full)), 'freeze': manifest}

    for selection in SELECTIONS:
        rows = {}
        for mode in ('official_mluda', 'prototype', 'global'):
            rows[mode] = {}
            for metric in ('oa', 'aa', 'kappa'):
                values = np.asarray([
                    (result['seeds'][str(seed)][mode][selection][metric]
                     if mode == 'official_mluda' else
                     result['seeds'][str(seed)][mode][selection]['metrics'][metric])
                    for seed in SEEDS])
                rows[mode][metric] = {'mean': float(values.mean()),
                                      'sd': float(values.std(ddof=1)),
                                      'values': values.tolist()}
        rows['prototype_minus_global'] = {
            metric: {'mean': float((np.array(rows['prototype'][metric]['values']) -
                                    np.array(rows['global'][metric]['values'])).mean()),
                     'sd': float((np.array(rows['prototype'][metric]['values']) -
                                  np.array(rows['global'][metric]['values'])).std(ddof=1))}
            for metric in ('oa', 'aa', 'kappa')}
        result['summary'][selection] = rows
    destination.write_text(json.dumps(result, indent=2))
    for selection in SELECTIONS:
        print(selection)
        for seed in SEEDS:
            row = result['seeds'][str(seed)]
            print(seed, ' '.join(
                f"{mode}: {((row[mode][selection]['oa'], row[mode][selection]['aa']) if mode == 'official_mluda' else (row[mode][selection]['metrics']['oa'], row[mode][selection]['metrics']['aa']))[0]*100:.2f}/{((row[mode][selection]['oa'], row[mode][selection]['aa']) if mode == 'official_mluda' else (row[mode][selection]['metrics']['oa'], row[mode][selection]['metrics']['aa']))[1]*100:.2f}"
                for mode in ('official_mluda', 'prototype', 'global')))
        print('paired prototype - global', result['summary'][selection]['prototype_minus_global'])


if __name__ == '__main__':
    main()
