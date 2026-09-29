"""Freeze predeclared Hangzhou prior controls without opening target GT."""
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
from investigations.global_local_allocation_v1.freeze import supported_prior, allocate  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1341, 1535, 1631))
    args = parser.parse_args()
    teacher_path = HERE / 'hypersigma_runs' / f'seed_{args.seed}' / 'target_predictions_before_gt.npz'
    with np.load(teacher_path) as teacher:
        centers = teacher['centers'].copy()
        probabilities = teacher['probabilities'].copy()
        prior = teacher['scene_prior'].astype(np.float64)
    assert probabilities.shape == (len(centers), 3)
    assert np.allclose(probabilities.mean(axis=0), prior, atol=1e-6)
    prior_safe = np.maximum(prior, 1e-12)
    prior_safe /= prior_safe.sum()
    output_dir = HERE / 'prior_controls'
    output_dir.mkdir(exist_ok=True)
    paths = {
        'mluda': HERE / 'official_three_runs' / f'seed_{args.seed}' /
                 'predictions_before_score.npz',
        'dcrn': HERE / 'original_dcrn_runs' / f'seed_{args.seed}' /
                'predictions_before_score.npz',
    }
    names = ('mluda', 'dcrn')
    for model_name in names:
        raw_path = paths[model_name]
        with np.load(raw_path) as source:
            assert np.array_equal(centers, source['centers'])
            raw = source['probabilities'].copy()
        assert raw.shape == (len(centers) // 32 * 32, 3)
        safe = np.maximum(raw.astype(np.float64), 1e-12)
        safe /= safe.sum(axis=1, keepdims=True)
        soft, soft_solver = soft_project(safe, prior_safe, strength=1.0)
        hard, hard_solver = project_kl(safe, prior_safe)
        effective, floor = supported_prior(safe, prior_safe)
        sorted_safe = np.sort(safe, axis=1)
        weight = 1 + sorted_safe[:, -1] - sorted_safe[:, -2]
        v1, v1_solver = allocate(safe, effective, weight)
        output_path = output_dir / f'seed_{args.seed}_{model_name}.npz'
        assert not output_path.exists(), output_path
        np.savez_compressed(output_path, centers=centers, raw=raw,
                            soft_kl=soft, hard=hard, v1=v1, prior=prior,
                            effective_prior=effective, support_floor=floor)
        manifest = {
            'seed': args.seed, 'local_model': model_name,
            'teacher_prediction_sha256': sha256(teacher_path),
            'local_prediction_sha256': sha256(raw_path),
            'control_prediction_sha256': sha256(output_path),
            'raw_probability_floor_count': int((raw <= 1e-12).sum()),
            'prior_probability_floor_count': int((prior <= 1e-12).sum()),
            'numerical_floor': 1e-12, 'soft_kl_lambda': 1.0,
            'soft_solver': soft_solver, 'hard_solver': hard_solver,
            'v1_solver': v1_solver,
            'v1_support_floor': floor.tolist(),
            'v1_effective_prior': effective.tolist(),
            'target_gt_accessed': False,
        }
        (output_dir / f'seed_{args.seed}_{model_name}.json').write_text(
            json.dumps(manifest, indent=2))
        print(json.dumps({'seed': args.seed, 'local_model': model_name,
                          'soft_residual': soft_solver['stationarity_max_abs_residual'],
                          'hard_residual': hard_solver['prior_max_abs_error']}), flush=True)


if __name__ == '__main__':
    main()
