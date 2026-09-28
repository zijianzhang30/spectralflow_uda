"""Pavia local DCRN baseline on the paired official source split.

This is the existing convolution-only DCRN A recipe (source CE with mixed-domain
BN forwards), transplanted to official Pavia image preprocessing and schedule.
Target labels are never opened. The fixed epoch-100 prediction is frozen before
any target-GT audit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch import nn
from sklearn import preprocessing
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = Path('/home/zhangzj26/TGRS_MLUDA-2024')
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(ROOT))
from UtilsCMS import ILDA  # noqa: E402
from net2 import DCRN  # noqa: E402
from experiments.round9.model import Backbone  # noqa: E402
from experiments.round9.augmentation import radiation_noise, flip_augmentation  # noqa: E402

utils_spec = importlib.util.spec_from_file_location('mluda_pavia_utils', LEGACY / 'utils.py')
assert utils_spec is not None and utils_spec.loader is not None
mluda_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(mluda_utils)
DATA = Path('/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Pavia')
SEEDS = (1622, 1322, 1256)


class OriginalDCRNLocal(nn.Module):
    """Original single-input DCRN with its own 152-dimensional FC head."""

    def __init__(self):
        super().__init__()
        self.backbone = DCRN(102, 11, 7)

    def forward(self, x):
        features = self.backbone(x, x)  # DCRN's second argument is unused.
        return features, self.backbone.fc1(features)


def make_patches(cube: np.ndarray, centers: np.ndarray, half: int = 5):
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode='constant')
    width = 2 * half + 1
    result = np.empty((len(centers), cube.shape[2], width, width), dtype=np.float32)
    for i, (r, c) in enumerate(centers):
        result[i] = padded[r:r + width, c:c + width].transpose(2, 0, 1)
    return result


@torch.inference_mode()
def source_validation(model, loader, device):
    model.eval()
    correct = count = 0
    for x, y in loader:
        logits = model(x.to(device))[1]
        correct += int((logits.argmax(1).cpu() == y).sum())
        count += len(y)
    return correct / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=SEEDS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--architecture', choices=('conv_trunk', 'original_dcrn'),
                        default='conv_trunk')
    args = parser.parse_args()
    torch.set_num_threads(4)
    # Upstream seeds are set after deterministic image preprocessing/ILDA.
    source, source_gt = mluda_utils.load_data_pavia(
        str(DATA / 'paviaU.mat'), str(DATA / 'paviaU_gt_7.mat'))
    target_raw = sio.loadmat(str(DATA / 'pavia.mat'))['pavia']
    target = preprocessing.scale(target_raw.reshape(-1, 102)).reshape(target_raw.shape)
    del target_raw
    source, target = ILDA(source, target, 2, 0.00009)
    mluda_utils.set_seed(args.seed)

    teacher_split = HERE / 'hypersigma_runs' / f'seed_{args.seed}' / 'source_split.npz'
    with np.load(teacher_split) as split:
        train_centers = split['train_centers'].copy()
        val_centers = split['val_centers'].copy()
        train_y = split['train_labels'].copy()
        val_y = split['val_labels'].copy()
    assert np.array_equal(source_gt[train_centers[:, 0], train_centers[:, 1]] - 1, train_y)
    assert np.array_equal(source_gt[val_centers[:, 0], val_centers[:, 1]] - 1, val_y)
    official_predictions = HERE / 'official_three_runs' / f'seed_{args.seed}' / 'predictions_before_score.npz'
    with np.load(official_predictions) as prior:
        target_centers = prior['centers'].copy()
    out_root = 'dcrn_runs' if args.architecture == 'conv_trunk' else 'original_dcrn_runs'
    out = HERE / out_root / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps({
        'seed': args.seed, 'source_train_n': len(train_centers),
        'source_val_n': len(val_centers), 'target_n': len(target_centers),
        'input': 'official Pavia per-scene z-score + ILDA PCA2 radius=0.00009',
        'model': ('round9 truncated convolutional trunk Backbone(bands=102)'
                  if args.architecture == 'conv_trunk' else
                  'original net2.DCRN(102,11,7) with its own fc1 classifier'),
        'training': 'source CE; original+noise+flip source/target mixed-BN forwards',
        'schedule': '100 epochs, 38 steps/epoch, SGD LR=0.001 constant, momentum=0.9, weight decay=0.0005, optimizer reset per epoch',
        'selection': 'fixed epoch 100; source-val tracked for diagnosis only',
        'target_labels_used': False,
        'evaluation': 'official center order, batch32 drop_last',
    }, indent=2))

    train_x = make_patches(source, train_centers)
    val_x = make_patches(source, val_centers)
    target_x = make_patches(target, target_centers)
    del source, target
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                              batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
                            batch_size=32, shuffle=False)
    target_dataset = TensorDataset(torch.from_numpy(target_x))
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True, drop_last=True)
    eval_loader = DataLoader(target_dataset, batch_size=32, shuffle=False, drop_last=True)
    model = (Backbone(bands=102, classes=7) if args.architecture == 'conv_trunk'
             else OriginalDCRNLocal()).to(args.device)
    history = []
    for epoch in range(1, 101):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                                    weight_decay=0.0005)
        model.train()
        source_iter, target_iter = iter(train_loader), iter(target_loader)
        ce_sum = 0.0
        for _ in range(1, len(train_loader)):
            x, y = next(source_iter)
            (target_batch,) = next(target_iter)
            source_noise = radiation_noise(x).float().to(args.device)
            source_flip = flip_augmentation(x).to(args.device)
            target_noise = radiation_noise(target_batch).float().to(args.device)
            target_flip = flip_augmentation(target_batch).to(args.device)
            x, y = x.to(args.device), y.to(args.device)
            target_batch = target_batch.to(args.device)
            logits = model(x)[1]
            ce = F.cross_entropy(logits, y)
            with torch.no_grad():
                model(target_batch)
                model(source_noise)
                model(target_noise)
                model(source_flip)
                model(target_flip)
            optimizer.zero_grad(set_to_none=True)
            ce.backward()
            optimizer.step()
            ce_sum += float(ce.detach())
        row = {'epoch': epoch, 'source_ce': ce_sum / (len(train_loader)-1),
               'source_val_acc': source_validation(model, val_loader, args.device)}
        history.append(row)
        print(json.dumps(row), flush=True)
    torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'seed': args.seed, 'epoch': 100}, out / 'last.pth')
    (out / 'history.json').write_text(json.dumps(history, indent=2))
    model.eval()
    chunks = []
    with torch.inference_mode():
        for (x,) in eval_loader:
            chunks.append(torch.softmax(model(x.to(args.device))[1], 1).cpu().numpy())
    probabilities = np.concatenate(chunks).astype(np.float32)
    assert len(probabilities) == 32 * len(eval_loader)
    np.savez_compressed(out / 'predictions_before_score.npz',
                        probabilities=probabilities, centers=target_centers)
    print(json.dumps({'seed': args.seed, 'complete': True,
                      'predicted_n': len(probabilities)}), flush=True)


if __name__ == '__main__':
    main()
