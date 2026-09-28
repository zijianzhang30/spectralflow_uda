"""Freeze Pavia HyperSIGMA target probabilities without reading target labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from investigations.pavia_global_prior_v1.train_hypersigma import SSFusionFramework, patches  # noqa: E402

TARGET = Path('/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Pavia/pavia.mat')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1622, 1322, 1256))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = HERE / 'hypersigma_runs' / f'seed_{args.seed}'
    selection = json.loads((run / 'selected.json').read_text())
    checkpoint_path = Path(selection['selected_checkpoint'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = SSFusionFramework(img_size=33, in_channels=102, patch_size=2,
                              classes=7, model_size='base')
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(args.device).eval()

    raw = sio.loadmat(str(TARGET))['pavia']
    target = preprocessing.scale(raw.reshape(-1, 102)).reshape(raw.shape)
    del raw
    with np.load(HERE / 'official_three_runs' / f'seed_{args.seed}' /
                 'predictions_before_score.npz') as official:
        centers = official['centers'].copy()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(centers), 32):
            x = torch.from_numpy(patches(target, centers[start:start + 32], 102)).to(args.device)
            chunks.append(torch.softmax(model(x), 1).cpu().numpy())
            if start % 4096 == 0:
                print(json.dumps({'seed': args.seed, 'predicted': start + len(x),
                                  'total': len(centers)}), flush=True)
    probabilities = np.concatenate(chunks).astype(np.float32)
    assert probabilities.shape == (len(centers), 7)
    pi_h = probabilities.mean(axis=0)
    np.savez_compressed(run / 'target_predictions_before_gt.npz',
                        probabilities=probabilities, centers=centers, scene_prior=pi_h)
    (run / 'prior.json').write_text(json.dumps({
        'seed': args.seed, 'selected_stage': selection['selected_stage'],
        'source_val_acc': checkpoint['best']['val_acc'],
        'target_count': len(centers), 'scene_prior': pi_h.tolist(),
        'target_gt_accessed': False,
    }, indent=2))
    print(json.dumps({'seed': args.seed, 'complete': True,
                      'target_count': len(centers), 'prior': pi_h.tolist()}), flush=True)


if __name__ == '__main__':
    main()
