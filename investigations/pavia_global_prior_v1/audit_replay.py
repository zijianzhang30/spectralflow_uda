"""Independently replay saved local-model predictions from final checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing

from train_dcrn_local import DATA, ILDA, Backbone, OriginalDCRNLocal, make_patches, mluda_utils

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    source, _ = mluda_utils.load_data_pavia(str(DATA / 'paviaU.mat'),
                                             str(DATA / 'paviaU_gt_7.mat'))
    raw = sio.loadmat(str(DATA / 'pavia.mat'))['pavia']
    target = preprocessing.scale(raw.reshape(-1, 102)).reshape(raw.shape)
    del raw
    _, target = ILDA(source, target, 2, 0.00009)
    for name, constructor in (
        ('dcrn_runs', lambda: Backbone(bands=102, classes=7)),
        ('original_dcrn_runs', OriginalDCRNLocal),
    ):
        run = HERE / name / f'seed_{args.seed}'
        with np.load(run / 'predictions_before_score.npz') as saved:
            n = len(saved['probabilities']) if args.full else 64
            centers = saved['centers'][:n].copy()
            expected = saved['probabilities'][:n].copy()
        model = constructor().to(args.device)
        checkpoint = torch.load(run / 'last.pth', map_location='cpu', weights_only=False)
        assert checkpoint['seed'] == args.seed and checkpoint['epoch'] == 100
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(centers), 32):
                x = torch.from_numpy(make_patches(target, centers[start:start + 32])).to(args.device)
                outputs.append(torch.softmax(model(x)[1], 1).cpu().numpy())
        actual = np.concatenate(outputs)
        max_abs = float(np.max(np.abs(actual - expected)))
        agreement = float(np.mean(actual.argmax(1) == expected.argmax(1)))
        mismatches = int(np.sum(actual.argmax(1) != expected.argmax(1)))
        print(f'{name}: n={len(centers)}, max_abs={max_abs:.9g}, '
              f'argmax_mismatches={mismatches}, agreement={agreement:.6f}', flush=True)
        assert max_abs < 0.002 and mismatches == 0


if __name__ == '__main__':
    main()
