"""Score a frozen target-only BN diagnostic; no parameter selection."""
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
    parser.add_argument('--architecture', choices=('conv_trunk', 'original_dcrn'), required=True)
    args = parser.parse_args()
    input_path = HERE / f'BN_RECAL_{args.architecture}_{args.seed}.npz'
    output_path = HERE / f'BN_RECAL_AUDIT_{args.architecture}_{args.seed}.json'
    assert not output_path.exists(), output_path
    with np.load(input_path) as saved:
        centers = saved['centers'].copy()
        probabilities = saved['probabilities'].copy()
    gt_path = Path('/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Pavia/pavia_gt_7.mat')
    gt = sio.loadmat(str(gt_path))['pavia_gt_7']
    labels = gt[centers[:39328, 0], centers[:39328, 1]].astype(np.int64) - 1
    result = score(probabilities, labels, len(centers))
    if args.architecture == 'original_dcrn':
        baseline = json.loads((HERE / f'AUDIT_ORIGINAL_DCRN_{args.seed}.json').read_text())['results']['raw']
    else:
        baseline = json.loads((HERE / f'AUDIT_{args.seed}.json').read_text())['models']['dcrn_raw']
    output_path.write_text(json.dumps({
        'seed': args.seed, 'architecture': args.architecture,
        'scope': 'post-hoc target-only BN diagnostic; not a method result',
        'rule': 'reset all BN running buffers, cumulative-mean target pass over 39,328 official centers, then eval',
        'target_gt_used_for_recalibration': False,
        'recal_prediction_sha256': sha256(input_path),
        'target_gt_sha256': sha256(gt_path),
        'original': baseline, 'target_bn_recal': result,
    }, indent=2))
    print(f"{args.seed} {args.architecture}: OA {baseline['oa_pct']:.2f}→{result['oa_pct']:.2f}%, "
          f"AA {baseline['aa_pct']:.2f}→{result['aa_pct']:.2f}%, "
          f"Kappa {baseline['kappa']:.4f}→{result['kappa']:.4f}")


if __name__ == '__main__':
    main()
