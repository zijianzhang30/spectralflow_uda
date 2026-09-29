"""Score both frozen relation-token checkpoint predictions after training."""
from __future__ import annotations

import json
from pathlib import Path

import hdf5storage
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 1341


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
    run = HERE / f'formal_{SEED}'
    assert not (HERE / 'AUDIT_1341.json').exists()
    frozen = {}
    for selection in ('last', 'best_source_val'):
        manifest = json.loads((run / f'{selection}_freeze.json').read_text())
        assert manifest['target_gt_accessed'] is False
        with np.load(run / f'{selection}_predictions_before_gt.npz') as z:
            frozen[selection] = (z['centers'].copy(), z['probabilities'].copy(), manifest)
    centers = frozen['last'][0]
    assert centers.shape == (53200, 2)
    assert np.array_equal(centers, frozen['best_source_val'][0])
    with np.load(ROOT / 'investigations/mluda_fixed100_v1/seed_1341_before_gt.npz') as baseline:
        assert np.array_equal(centers, baseline['centers'])
    target_gt = hdf5storage.loadmat(
        '/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/Houston18_7gt.mat')['map']
    y_full = target_gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    y = y_full[:53184]
    assert y.min() == 0 and y.max() == 6
    result = {'seed': SEED, 'scope': 'one-seed development pilot; target GT post-freeze only',
              'relation_tokens': {}, 'official_mluda': {}}
    for selection in ('last', 'best_source_val'):
        _, probabilities, manifest = frozen[selection]
        assert probabilities.shape == (53184, 7)
        assert np.isfinite(probabilities).all()
        assert np.max(np.abs(probabilities.sum(1) - 1)) < 1e-5
        result['relation_tokens'][selection] = {
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
        ours = result['relation_tokens'][selection]['metrics']
        base = result['official_mluda'][selection]
        print(f"{selection}: ours OA {ours['oa']*100:.2f}% AA {ours['aa']*100:.2f}% "
              f"Kappa {ours['kappa']:.4f}; MLUDA OA {base['oa']*100:.2f}% "
              f"AA {base['aa']*100:.2f}% Kappa {base['kappa']:.4f}")


if __name__ == '__main__':
    main()
