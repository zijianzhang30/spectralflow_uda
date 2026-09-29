"""Paired official Houston MLUDA runs: class prototype versus global source anchor."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import hdf5storage
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFFICIAL = ROOT / 'official_aligned'
sys.path.insert(0, str(OFFICIAL))
sys.path.insert(0, str(OFFICIAL / 'reference'))
from data import load_images, loaders, Patches  # noqa: E402
from runtime import seed_everything, evaluation, atomic_json, save_checkpoint  # noqa: E402
from full_mluda_objective import objective  # noqa: E402
sys.path.insert(0, str(HERE))
from model import make_model  # noqa: E402


def validate(model, loader, device):
    correct = count = 0
    total_loss = 0.0
    with evaluation(model):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, x)[3]
            correct += int((logits.argmax(1) == y).sum())
            count += len(y)
            total_loss += float(F.cross_entropy(logits, y, reduction='sum'))
    return {'source_val_accuracy': correct / count,
            'source_val_loss': total_loss / count, 'source_val_n': count}


def refresh_source_bank(model, source, split, device):
    """Use only labeled source training centers, preserving training RNG state."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    numpy_rng = np.random.get_state()
    captured = []

    def pre_mbca(_module, inputs):
        captured.append(inputs[0].detach())

    hook = model.feature_layers.atten.register_forward_pre_hook(pre_mbca)
    try:
        loader = DataLoader(Patches(source, split['train_centers'], split['train_labels']),
                            batch_size=32, shuffle=False, drop_last=False, num_workers=0)
        all_features = []
        with evaluation(model), torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                model(x, x)
                assert len(captured) == 1
                all_features.append(captured.pop().cpu())
        features = torch.cat(all_features, dim=0)
        labels = torch.as_tensor(split['train_labels'])
        model.feature_layers.atten.set_bank(features, labels)
    finally:
        hook.remove()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1341, choices=(1341, 1174, 1370))
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', choices=('prototype', 'global'), required=True)
    args = parser.parse_args()
    assert 1 <= args.epochs <= 100
    torch.set_num_threads(2)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    original_run = OFFICIAL / 'runs/round1/formal' / str(args.seed) / 'MLUDA_full'
    cfg = json.loads((original_run / 'config.json').read_text())
    data_path = Path(cfg['data'])
    source, target = load_images(data_path, 'official_ilda', Path(cfg['ilda_cache']))
    source_gt = hdf5storage.loadmat(str(data_path / 'Houston13_7gt.mat'))['map']
    target_gt = hdf5storage.loadmat(str(data_path / 'Houston18_7gt.mat'))['map']
    source_loader, target_loader, val_loader, split = loaders(
        source, target, source_gt, args.seed, target_gt)
    with np.load(original_run / 'source_split.npz') as official:
        for key, value in split.items():
            assert np.array_equal(value, official[key]), key
    np.savez_compressed(out / 'source_split.npz', **split)
    model = make_model(args.mode == 'global', args.device)
    added = sum(p.numel() for p in model.feature_layers.atten.parameters()) - sum(
        p.numel() for p in model.feature_layers.atten.base.parameters())
    config = {
        'method': f'MLUDA_source_anchor_{args.mode}', 'seed': args.seed,
        'epochs': args.epochs, 'lr_horizon': 100, 'data': str(data_path),
        'ilda_cache': cfg['ilda_cache'], 'batch_size': 32, 'patch_size': 7,
        'steps_per_epoch': len(source_loader) - 1,
        'class_anchors': 7, 'relation_dim': 288,
        'bank': 'all labeled source-train centers; refreshed before epoch 1 and after each epoch',
        'class_reliability': 'source-train leave-one-out nearest-prototype recall; floor 0.25',
        'retrieval': 'cosine softmax, temperature 0.2',
        'anchor_mode': ('one global source mean replicated seven times' if args.mode == 'global'
                        else 'seven labeled source class means'),
        'target_residual_gate_initial': 0.0200,
        'source_branch_updated': False,
        'added_parameters': added,
        'training_objective': 'official full MLUDA objective, unchanged',
        'optimizer': 'official epoch-wise SGD; feature_layers includes relation module',
        'selection': 'source_validation_best and fixed_epoch_100',
        'target_labels_used_for_training': False,
        'target_gt_usage': 'official center-order reproduction only',
        'paired_official_run': str(original_run),
    }
    atomic_json(out / 'config.json', config)
    refresh_source_bank(model, source, split, args.device)
    history = []
    best = -1.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        lr = .01 / (1 + 10 * (epoch - 1) / 100) ** .75
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.fc1.parameters(), 'lr': lr},
            {'params': model.fc2.parameters(), 'lr': lr},
            {'params': model.head1.parameters(), 'lr': lr},
            {'params': model.head2.parameters(), 'lr': lr},
        ], lr=lr, momentum=.9, weight_decay=5e-4)
        model.train()
        source_iter, target_iter = iter(source_loader), iter(target_loader)
        sums = {}
        module_grad_norm = 0.0
        for step in range(1, len(source_loader)):
            x, y = next(source_iter)
            target_x = next(target_iter)
            loss, parts = objective(model, x, target_x, y, epoch, 100)
            if not torch.isfinite(loss):
                raise FloatingPointError((epoch, step, parts))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            monitor = model.feature_layers.atten.delta[0].weight
            grad = monitor.grad
            assert grad is not None and torch.isfinite(grad).all()
            module_grad_norm += float(grad.norm())
            optimizer.step()
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
        refresh_source_bank(model, source, split, args.device)
        row = {'epoch': epoch, 'lr': lr,
               **{key: value / (len(source_loader) - 1) for key, value in sums.items()},
               'relation_gradient_norm_sum': module_grad_norm,
               'source_class_reliability': model.feature_layers.atten.reliability.cpu().tolist(),
               **validate(model, val_loader, args.device)}
        assert np.isfinite(module_grad_norm) and module_grad_norm > 0
        history.append(row)
        checkpoint = {
            'model': model.state_dict(), 'epoch': epoch, 'metrics': row,
            'config': config, 'last_source_batch': x.clone(),
            'cpu_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(),
        }
        if row['source_val_accuracy'] > best:
            best = row['source_val_accuracy']
            save_checkpoint(out / 'best_source_val.pth', checkpoint)
            atomic_json(out / 'best_source_val.json', row)
        save_checkpoint(out / 'last.pth', checkpoint)
        atomic_json(out / 'history.json', history)
        print(json.dumps(row), flush=True)
    atomic_json(out / 'training_complete.json',
                {'epochs': args.epochs, 'seconds': time.time() - start_time})
    print(json.dumps({'complete': True, 'seed': args.seed, 'mode': args.mode,
                      'epochs': args.epochs, 'best_source_val_accuracy': best}), flush=True)


if __name__ == '__main__':
    main()
