"""Freeze predeclared Pavia prior controls without opening target GT."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from investigations.distribution_correction_validation.projection import project_kl  # noqa: E402
from investigations.distribution_correction_validation.soft_projection import soft_project  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1622, 1322, 1256))
    parser.add_argument('--model', choices=('all', 'dcrn', 'mluda', 'original_dcrn'),
                        default='all')
    args = parser.parse_args()
    teacher_path = HERE / 'hypersigma_runs' / f'seed_{args.seed}' / 'target_predictions_before_gt.npz'
    with np.load(teacher_path) as teacher:
        centers = teacher['centers'].copy()
        probabilities = teacher['probabilities'].copy()
        prior = teacher['scene_prior'].astype(np.float64)
    assert probabilities.shape == (39355, 7)
    assert np.allclose(probabilities.mean(axis=0), prior, atol=1e-6)
    prior_safe = np.maximum(prior, 1e-12)
    prior_safe /= prior_safe.sum()
    output_dir = HERE / 'prior_controls'
    output_dir.mkdir(exist_ok=True)
    paths = {
        'dcrn': HERE / 'dcrn_runs' / f'seed_{args.seed}' / 'predictions_before_score.npz',
        'mluda': HERE / 'official_three_runs' / f'seed_{args.seed}' / 'predictions_before_score.npz',
        'original_dcrn': HERE / 'original_dcrn_runs' / f'seed_{args.seed}' / 'predictions_before_score.npz',
    }
    names = ('dcrn', 'mluda') if args.model == 'all' else (args.model,)
    for model_name in names:
        raw_path = paths[model_name]
        with np.load(raw_path) as source:
            assert np.array_equal(centers, source['centers'])
            raw = source['probabilities'].copy()
        assert raw.shape == (39328, 7)
        safe = np.maximum(raw.astype(np.float64), 1e-12)
        safe /= safe.sum(axis=1, keepdims=True)
        soft, soft_solver = soft_project(safe, prior_safe, strength=1.0)
        hard, hard_solver = project_kl(safe, prior_safe)
        output_path = output_dir / f'seed_{args.seed}_{model_name}.npz'
        assert not output_path.exists(), output_path
        np.savez_compressed(output_path, centers=centers, raw=raw,
                            soft_kl=soft, hard=hard, prior=prior)
        manifest = {
            'seed': args.seed, 'local_model': model_name,
            'teacher_prediction_sha256': sha256(teacher_path),
            'local_prediction_sha256': sha256(raw_path),
            'control_prediction_sha256': sha256(output_path),
            'raw_probability_floor_count': int((raw <= 1e-12).sum()),
            'prior_probability_floor_count': int((prior <= 1e-12).sum()),
            'numerical_floor': 1e-12, 'soft_kl_lambda': 1.0,
            'soft_solver': soft_solver, 'hard_solver': hard_solver,
            'target_gt_accessed': False,
        }
        (output_dir / f'seed_{args.seed}_{model_name}.json').write_text(
            json.dumps(manifest, indent=2))
        print(json.dumps({'seed': args.seed, 'local_model': model_name,
                          'soft_residual': soft_solver['stationarity_max_abs_residual'],
                          'hard_residual': hard_solver['prior_max_abs_error']}), flush=True)


if __name__ == '__main__':
    main()
