"""Replay frozen SH2HZ DCRN probabilities from epoch-100 checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing

from train_dcrn_local import DATA, ILDA, OriginalDCRNLocal, make_patches

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--samples', type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(4)
    mat = sio.loadmat(str(DATA), variable_names=['DataCube1', 'DataCube2'])
    raw_s, raw_t = mat['DataCube1'], mat['DataCube2']
    source = preprocessing.scale(raw_s.reshape(-1, 198)).reshape(raw_s.shape)
    target = preprocessing.scale(raw_t.reshape(-1, 198)).reshape(raw_t.shape)
    del mat, raw_s, raw_t
    _, target = ILDA(source, target, 2, 0.00009)
    for seed in (1341, 1535, 1631):
        run = HERE / 'original_dcrn_runs' / f'seed_{seed}'
        with np.load(run / 'predictions_before_score.npz') as saved:
            centers = saved['centers'][:args.samples].copy()
            expected = saved['probabilities'][:args.samples].copy()
        checkpoint = torch.load(run / 'last.pth', map_location='cpu', weights_only=False)
        assert checkpoint['seed'] == seed and checkpoint['epoch'] == 100
        model = OriginalDCRNLocal().to(args.device).eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(centers), 32):
                x = torch.from_numpy(make_patches(target, centers[start:start + 32])).to(args.device)
                outputs.append(torch.softmax(model(x)[1], dim=1).cpu().numpy())
        actual = np.concatenate(outputs)
        max_abs = float(np.max(np.abs(actual - expected)))
        disagreements = int(np.count_nonzero(actual.argmax(1) != expected.argmax(1)))
        print(seed, 'n', len(centers), 'max_abs', max_abs,
              'argmax_disagreements', disagreements, flush=True)
        assert max_abs < 0.002 and disagreements == 0


if __name__ == '__main__':
    main()
