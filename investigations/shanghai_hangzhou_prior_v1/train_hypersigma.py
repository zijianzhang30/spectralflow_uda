"""Train the Shanghai 198-band HyperSIGMA prior provider on official source splits.

Target images and target labels are not opened during training or selection.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = Path('/home/zhangzj26/TGRS_MLUDA-2024')
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LEGACY / 'third_party/HyperSIGMA/ImageClassification'))
hyper_model = types.ModuleType('model')
hyper_model.__path__ = [str(LEGACY / 'third_party/HyperSIGMA/ImageClassification/model')]
sys.modules['model'] = hyper_model
from model.ss_fusion_cls import SSFusionFramework  # noqa: E402
from investigations.hypersigma_teacher_gate.train_matched import (  # noqa: E402
    load_pretrained, patches, seed_all, sha256, train_stage,
)

utils_spec = importlib.util.spec_from_file_location('mluda_pavia_utils', LEGACY / 'utils.py')
assert utils_spec is not None and utils_spec.loader is not None
mluda_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(mluda_utils)

DATA = Path('/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Shanghai-Hangzhou/DataCube.mat')
SEEDS = (1341, 1535, 1631)


def official_source_centers(gt: np.ndarray, seed: int):
    """Replay the official per-class shuffles; keep first 30 remaining/class."""
    rng = np.random.RandomState(seed)
    padded = gt
    row, col = np.nonzero(padded)
    labels = padded[row, col]
    train_indices, validation_indices = [], []
    for cls in range(1, 4):
        indices = np.flatnonzero(labels == cls).tolist()
        rng.shuffle(indices)
        train_indices.extend(indices[:180])
        validation_indices.extend(indices[180:210])
    rng.shuffle(train_indices)
    train = np.stack((row[train_indices], col[train_indices]), axis=1)
    val = np.stack((row[validation_indices], col[validation_indices]), axis=1)
    return train.astype(np.int64), val.astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=SEEDS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stage', choices=('both', 'full'), default='both')
    args = parser.parse_args()
    seed_all(args.seed)
    torch.set_num_threads(4)

    out = HERE / 'hypersigma_runs' / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    split_path = HERE / 'official_three_runs' / f'seed_{args.seed}' / 'source_split.npz'
    contents = sio.loadmat(str(DATA), variable_names=['DataCube1', 'gt1'])
    raw, gt = contents['DataCube1'], contents['gt1']
    source = preprocessing.scale(raw.reshape(-1, 198)).reshape(raw.shape)
    del raw, contents
    train_centers, val_centers = official_source_centers(gt, args.seed)
    if split_path.exists():
        with np.load(split_path) as split:
            assert np.array_equal(train_centers, split['train_centers'])
            official_val = set(map(tuple, split['val_centers']))
            assert set(map(tuple, val_centers)).issubset(official_val)
    train_y = gt[train_centers[:, 0], train_centers[:, 1]].astype(np.int64) - 1
    val_y = gt[val_centers[:, 0], val_centers[:, 1]].astype(np.int64) - 1
    assert np.array_equal(np.bincount(train_y, minlength=3), np.full(3, 180))
    assert np.array_equal(np.bincount(val_y, minlength=3), np.full(3, 30))
    np.savez_compressed(out / 'source_split.npz', train_centers=train_centers,
                        train_labels=train_y, val_centers=val_centers, val_labels=val_y)
    (out / 'config.json').write_text(json.dumps({
        'seed': args.seed,
        'data_sha256': sha256(DATA),
        'input': 'Shanghai 198-band per-scene z-score, pre-ILDA, 33x33 constant-padded patches',
        'train': 'official MLUDA 180 source centers/class',
        'validation': 'first 30 remaining official source centers/class',
        'batch_size': 32, 'stage1_epochs': 20, 'full_epochs': 20,
        'optimizer': 'Houston HyperSIGMA teacher protocol: AdamW and cosine schedule',
        'selection': 'first maximum source-validation accuracy in each stage; better stage wins, tie stage1',
        'target_used_for_training_or_selection': False,
    }, indent=2))

    train_loader = DataLoader(TensorDataset(torch.from_numpy(patches(source, train_centers, 198)),
                                            torch.from_numpy(train_y)),
                              batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(patches(source, val_centers, 198)),
                                          torch.from_numpy(val_y)),
                            batch_size=32, shuffle=False, num_workers=0)
    del source
    model = SSFusionFramework(img_size=33, in_channels=198, patch_size=2,
                              classes=3, model_size='base')
    if args.stage == 'both':
        load_pretrained(model)
        stage1 = train_stage(model, 'stage1', train_loader, val_loader,
                             torch.device(args.device), 20, out / 'stage1')
    else:
        stage1 = json.loads((out / 'stage1' / 'history.json').read_text())
    checkpoint = torch.load(out / 'stage1' / 'best.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    seed_all(args.seed + 10000)
    full = train_stage(model, 'full', train_loader, val_loader,
                       torch.device(args.device), 20, out / 'full')
    first = max(stage1, key=lambda r: r['val_acc'])
    second = max(full, key=lambda r: r['val_acc'])
    selected = 'stage1' if first['val_acc'] >= second['val_acc'] else 'full'
    (out / 'selected.json').write_text(json.dumps({
        'seed': args.seed, 'selected_stage': selected,
        'selected_checkpoint': str(out / selected / 'best.pth'),
        'stage1_best': first, 'full_best': second,
    }, indent=2))
    print(json.dumps({'seed': args.seed, 'selected_stage': selected,
                      'source_val_acc': max(first['val_acc'], second['val_acc'])}), flush=True)


if __name__ == '__main__':
    main()
