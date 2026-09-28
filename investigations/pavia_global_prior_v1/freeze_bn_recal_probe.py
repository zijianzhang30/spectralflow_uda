"""Post-hoc unlabeled target BN reset/recalibration probe, not a paper method."""
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
    parser.add_argument('--seed', type=int, required=True, choices=(1622, 1322, 1256))
    parser.add_argument('--architecture', choices=('conv_trunk', 'original_dcrn'), required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    directory = 'dcrn_runs' if args.architecture == 'conv_trunk' else 'original_dcrn_runs'
    run = HERE / directory / f'seed_{args.seed}'
    source, _ = mluda_utils.load_data_pavia(str(DATA / 'paviaU.mat'),
                                             str(DATA / 'paviaU_gt_7.mat'))
    raw = sio.loadmat(str(DATA / 'pavia.mat'))['pavia']
    target = preprocessing.scale(raw.reshape(-1, 102)).reshape(raw.shape)
    del raw
    _, target = ILDA(source, target, 2, 0.00009)
    with np.load(run / 'predictions_before_score.npz') as saved:
        centers = saved['centers'].copy()
    model = (Backbone(bands=102, classes=7) if args.architecture == 'conv_trunk'
             else OriginalDCRNLocal()).to(args.device)
    checkpoint = torch.load(run / 'last.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    bn_layers = [layer for layer in model.modules()
                 if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)]
    assert bn_layers
    for layer in bn_layers:
        layer.reset_running_stats()
        layer.momentum = None
    model.train()
    with torch.inference_mode():
        for start in range(0, 39328, 32):
            x = torch.from_numpy(make_patches(target, centers[start:start + 32])).to(args.device)
            model(x)
            if start % 8192 == 0:
                print(f'BN calibration {start + len(x)}/39328', flush=True)
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, 39328, 32):
            x = torch.from_numpy(make_patches(target, centers[start:start + 32])).to(args.device)
            chunks.append(torch.softmax(model(x)[1], 1).cpu().numpy())
    probabilities = np.concatenate(chunks).astype(np.float32)
    assert probabilities.shape == (39328, 7)
    output_path = HERE / f'BN_RECAL_{args.architecture}_{args.seed}.npz'
    assert not output_path.exists(), output_path
    np.savez_compressed(output_path, probabilities=probabilities, centers=centers)
    print(f'frozen {output_path}', flush=True)


if __name__ == '__main__':
    main()
