"""Audit the structurally correct original DCRN after freezing predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from investigations.pavia_global_prior_v1.audit_one_seed import score, sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1622, 1322, 1256))
    args = parser.parse_args()
    control_path = HERE / 'prior_controls' / f'seed_{args.seed}_original_dcrn.npz'
    output_path = HERE / f'AUDIT_ORIGINAL_DCRN_{args.seed}.json'
    assert not output_path.exists(), output_path
    with np.load(control_path) as saved:
        centers = saved['centers'].copy()
        variants = {name: saved[name].copy() for name in ('raw', 'soft_kl', 'hard')}
    with np.load(HERE / 'official_three_runs' / f'seed_{args.seed}' /
                 'predictions_before_score.npz') as saved:
        assert np.array_equal(centers, saved['centers'])
    gt_path = Path('/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Pavia/pavia_gt_7.mat')
    gt = sio.loadmat(str(gt_path))['pavia_gt_7']
    labels = gt[centers[:39328, 0], centers[:39328, 1]].astype(np.int64) - 1
    results = {name: score(q, labels, len(centers)) for name, q in variants.items()}
    history = json.loads((HERE / 'original_dcrn_runs' / f'seed_{args.seed}' /
                          'history.json').read_text())
    output = {'seed': args.seed, 'scope': 'post-hoc structural-correction control',
              'source_val_epoch100': history[-1]['source_val_acc'],
              'prediction_sha256': sha256(control_path),
              'target_gt_sha256': sha256(gt_path), 'results': results}
    output_path.write_text(json.dumps(output, indent=2))
    for name, result in results.items():
        print(f"{args.seed} original_dcrn_{name}: OA {result['oa_pct']:.2f}%, "
              f"AA {result['aa_pct']:.2f}%, Kappa {result['kappa']:.4f}")


if __name__ == '__main__':
    main()
