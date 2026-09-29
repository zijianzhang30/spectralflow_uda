"""Score both frozen prototype arms only after all predictions are frozen."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import hdf5storage
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 1341


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def score(probabilities, labels, denominator):
    pred = probabilities.argmax(1)
    cm = confusion_matrix(labels, pred, labels=np.arange(7))
    recall = np.diag(cm) / cm.sum(1)
    return {
        'oa': float((pred == labels).sum() / denominator),
        'aa': float(recall.mean()),
        'kappa': float(cohen_kappa_score(labels, pred)),
        'recall': recall.tolist(), 'confusion_matrix': cm.tolist(),
        'evaluated_n': len(labels), 'denominator': denominator,
    }


def main():
    assert not (HERE / 'AUDIT_1341.json').exists()
    frozen = {}
    for mode in ('prototype', 'class_token'):
        run = HERE / f'formal_{mode}_{SEED}'
        for selection in ('last', 'best_source_val'):
            manifest = json.loads((run / f'{selection}_freeze.json').read_text())
            assert manifest['target_gt_accessed'] is False
            assert manifest['mode'] == mode and manifest['selection'] == selection
            assert sha256(run / f'{selection}_predictions_before_gt.npz') == manifest['prediction_sha256']
            assert sha256(run / f'{selection}.pth') == manifest['checkpoint_sha256']
            with np.load(run / f'{selection}_predictions_before_gt.npz') as z:
                frozen[(mode, selection)] = (z['centers'].copy(), z['probabilities'].copy(), manifest)
    centers = frozen[('prototype', 'last')][0]
    assert centers.shape == (53200, 2)
    assert all(np.array_equal(centers, item[0]) for item in frozen.values())
    with np.load(ROOT / 'investigations/mluda_fixed100_v1/seed_1341_before_gt.npz') as baseline:
        assert np.array_equal(centers, baseline['centers'])
    cfg = json.loads((HERE / f'formal_prototype_{SEED}' / 'config.json').read_text())
    target_gt = hdf5storage.loadmat(str(Path(cfg['data']) / 'Houston18_7gt.mat'))['map']
    y_full = target_gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    y = y_full[:53184]
    assert y.min() == 0 and y.max() == 6
    result = {'seed': SEED, 'scope': 'one-seed development pilot; target GT post-freeze only',
              'arms': {}, 'official_mluda': {}}
    for mode in ('prototype', 'class_token'):
        result['arms'][mode] = {}
        for selection in ('last', 'best_source_val'):
            _, probabilities, manifest = frozen[(mode, selection)]
            assert probabilities.shape == (53184, 7)
            assert np.isfinite(probabilities).all()
            assert np.max(np.abs(probabilities.sum(1) - 1)) < 1e-5
            result['arms'][mode][selection] = {
                'metrics': score(probabilities, y, len(y_full)), 'freeze': manifest}
    baseline_fixed = json.loads((ROOT / 'investigations/mluda_fixed100_v1/AUDIT.json').read_text())['seeds']['1341']
    baseline_best = json.loads((ROOT / 'official_aligned/runs/round1/formal/1341/MLUDA_full/final_target.json').read_text())
    result['official_mluda'] = {
        'last': {'oa': baseline_fixed['oa'], 'aa': baseline_fixed['aa'],
                 'kappa': baseline_fixed['kappa'], 'recall': baseline_fixed['recall']},
        'best_source_val': {'oa': baseline_best['oa'], 'aa': baseline_best['aa'],
                            'kappa': baseline_best['kappa'],
                            'recall': baseline_best['per_class_accuracy']},
    }
    (HERE / 'AUDIT_1341.json').write_text(json.dumps(result, indent=2))
    for selection in ('last', 'best_source_val'):
        print(selection)
        for mode in ('prototype', 'class_token'):
            row = result['arms'][mode][selection]['metrics']
            print(f"  {mode}: OA {row['oa']*100:.2f}% AA {row['aa']*100:.2f}% "
                  f"Kappa {row['kappa']:.4f}; recall {[round(x*100,2) for x in row['recall']]}")
        base = result['official_mluda'][selection]
        print(f"  official MLUDA: OA {base['oa']*100:.2f}% "
              f"AA {base['aa']*100:.2f}% Kappa {base['kappa']:.4f}")


if __name__ == '__main__':
    main()
