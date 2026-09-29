"""Independently score already frozen official SH2HZ three-seed predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
DATA = Path('/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Shanghai-Hangzhou/DataCube.mat')
SEEDS = (1341, 1535, 1631)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    gt = sio.loadmat(str(DATA), variable_names=['gt2'])['gt2']
    labeled_centers = set(map(tuple, np.argwhere(gt > 0)))
    report = {'input_sha256': sha256(DATA), 'seeds': {}}
    for seed in SEEDS:
        run = HERE / 'official_three_runs' / f'seed_{seed}'
        with np.load(run / 'predictions_before_score.npz') as saved:
            probabilities = saved['probabilities']
            centers = saved['centers']
        assert probabilities.shape == (len(centers) // 32 * 32, 3)
        assert len(set(map(tuple, centers))) == len(centers)
        assert set(map(tuple, centers)) == labeled_centers
        y = gt[centers[:len(probabilities), 0], centers[:len(probabilities), 1]].astype(int) - 1
        pred = probabilities.argmax(axis=1)
        cm = confusion_matrix(y, pred, labels=np.arange(3))
        recalls = np.diag(cm) / cm.sum(axis=1)
        metrics = {'oa_pct': 100 * float((pred == y).sum()) / len(centers),
                   'aa_pct': 100 * float(recalls.mean()),
                   'kappa': float(cohen_kappa_score(y, pred)),
                   'recall_pct': (100 * recalls).tolist(),
                   'confusion_matrix': cm.tolist(),
                   'evaluated_n': len(probabilities), 'official_n': len(centers),
                   'prediction_sha256': sha256(run / 'predictions_before_score.npz'),
                   'checkpoint_sha256': sha256(run / 'last.pth'),
                   'source_split_sha256': sha256(run / 'source_split.npz')}
        upstream = json.loads((run / 'metrics.json').read_text())
        assert np.isclose(metrics['oa_pct'], upstream['oa'])
        assert np.isclose(metrics['aa_pct'] / 100, upstream['aa'])
        assert np.isclose(metrics['kappa'], upstream['kappa'])
        report['seeds'][str(seed)] = metrics
    (HERE / 'BASELINE_AUDIT.json').write_text(json.dumps(report, indent=2))
    for seed, result in report['seeds'].items():
        print(seed, result['oa_pct'], result['aa_pct'], result['kappa'])


if __name__ == '__main__':
    main()
