"""Score a frozen Hangzhou seed; never use this output to choose a method."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
GT_PATH = Path('/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Shanghai-Hangzhou/DataCube.mat')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def score(probabilities: np.ndarray, labels: np.ndarray, official_n: int):
    pred = probabilities.argmax(axis=1)
    cm = confusion_matrix(labels, pred, labels=np.arange(3))
    recall = np.diag(cm) / cm.sum(axis=1)
    return {'oa_pct': 100 * float((pred == labels).sum()) / official_n,
            'aa_pct': 100 * float(recall.mean()),
            'kappa': float(cohen_kappa_score(labels, pred)),
            'recall_pct': (100 * recall).tolist(),
            'evaluated_n': len(labels), 'official_n': official_n,
            'confusion_matrix': cm.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1341, 1535, 1631))
    args = parser.parse_args()
    output_path = HERE / f'AUDIT_{args.seed}.json'
    assert not output_path.exists(), output_path
    paths = {
        'teacher': HERE / 'hypersigma_runs' / f'seed_{args.seed}' / 'target_predictions_before_gt.npz',
        'dcrn': HERE / 'prior_controls' / f'seed_{args.seed}_dcrn.npz',
        'mluda': HERE / 'prior_controls' / f'seed_{args.seed}_mluda.npz',
    }
    with np.load(paths['teacher']) as saved:
        centers = saved['centers'].copy()
        teacher = saved['probabilities'].copy()
        prior = saved['scene_prior'].copy()
    assert centers.shape == (135700, 2) and teacher.shape == (135700, 3)
    gt = sio.loadmat(str(GT_PATH), variable_names=['gt2'])['gt2']
    y_full = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    assert y_full.shape == (135700,) and y_full.min() == 0 and y_full.max() == 2
    y = y_full[:len(y_full) // 32 * 32]
    result = {'seed': args.seed, 'scope': 'single-seed descriptive audit; no method selection',
              'target_gt_sha256': sha256(GT_PATH),
              'prediction_sha256': {key: sha256(path) for key, path in paths.items()},
              'teacher': score(teacher, y_full, len(y_full)),
              'true_prior': (np.bincount(y_full, minlength=3) / len(y_full)).tolist(),
              'teacher_prior': prior.tolist(), 'models': {}}
    result['teacher_prior_tv'] = float(np.abs(
        np.asarray(result['true_prior']) - prior).sum() / 2)
    for model_name in ('dcrn', 'mluda'):
        with np.load(paths[model_name]) as saved:
            assert np.array_equal(centers, saved['centers'])
            assert np.allclose(prior, saved['prior'])
            variants = {name: saved[name].copy() for name in ('raw', 'soft_kl', 'hard', 'v1')}
        for name, probabilities in variants.items():
            assert probabilities.shape == (len(y), 3)
            result['models'][f'{model_name}_{name}'] = score(probabilities, y, len(y_full))
    baseline = json.loads((HERE / 'official_three_runs' / f'seed_{args.seed}' / 'metrics.json').read_text())
    original = result['models']['mluda_raw']
    assert np.isclose(original['oa_pct'], baseline['oa'])
    assert np.isclose(original['aa_pct'] / 100, baseline['aa'])
    assert np.isclose(original['kappa'], baseline['kappa'])
    output_path.write_text(json.dumps(result, indent=2))
    for name, metrics in result['models'].items():
        print(f"{name:17s} OA {metrics['oa_pct']:6.2f}%  AA {metrics['aa_pct']:6.2f}%  Kappa {metrics['kappa']:.4f}")
    metrics = result['teacher']
    print(f"{'hypersigma':17s} OA {metrics['oa_pct']:6.2f}%  AA {metrics['aa_pct']:6.2f}%  Kappa {metrics['kappa']:.4f}")


if __name__ == '__main__':
    main()
