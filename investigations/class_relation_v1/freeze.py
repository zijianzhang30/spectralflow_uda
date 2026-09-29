"""Freeze class-relation Houston target predictions before reading target labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFFICIAL = ROOT / 'official_aligned'
sys.path.insert(0, str(OFFICIAL))
from data import Patches, load_images  # noqa: E402
from runtime import evaluation  # noqa: E402
sys.path.insert(0, str(HERE))
from model import make_model  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1341)
    parser.add_argument('--selection', choices=('last', 'best_source_val'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', choices=('prototype', 'class_token'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = HERE / f'formal_{args.mode}_{args.seed}'
    complete = json.loads((run / 'training_complete.json').read_text())
    assert complete['epochs'] == 100
    checkpoint_path = run / f'{args.selection}.pth'
    output = run / f'{args.selection}_predictions_before_gt.npz'
    manifest_path = run / f'{args.selection}_freeze.json'
    assert not output.exists() and not manifest_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if args.selection == 'last':
        assert checkpoint['epoch'] == 100
    cfg = json.loads((run / 'config.json').read_text())
    with np.load(run / 'source_split.npz') as split:
        centers = split['target_centers'].copy()
    _, target = load_images(Path(cfg['data']), 'official_ilda',
                                 Path(cfg['ilda_cache']))
    loader = DataLoader(Patches(target, centers), batch_size=32, shuffle=False,
                        drop_last=True, num_workers=0)
    model = make_model(args.mode == 'class_token', args.device)
    model.load_state_dict(checkpoint['model'], strict=True)
    reference = checkpoint['last_source_batch'].to(args.device)
    probabilities = []
    with evaluation(model):
        for x in loader:
            logits = model(reference, x.to(args.device))[8]
            probabilities.append(torch.softmax(logits, 1).cpu().numpy())
    probabilities = np.concatenate(probabilities).astype(np.float32)
    assert probabilities.shape == (53184, 7)
    np.savez_compressed(output, probabilities=probabilities, centers=centers)

    relation = model.feature_layers.atten
    manifest = {
        'seed': args.seed, 'mode': args.mode, 'selection': args.selection,
        'checkpoint_epoch': checkpoint['epoch'],
        'source_val_accuracy': checkpoint['metrics']['source_val_accuracy'],
        'checkpoint_sha256': sha256(checkpoint_path),
        'prediction_sha256': sha256(output),
        'target_count': len(centers), 'evaluated_n': len(probabilities),
        'target_gt_accessed': False,
        'class_token_norm': (float(relation.class_tokens.detach().norm())
                             if args.mode == 'class_token' else None),
        'source_class_reliability': relation.reliability.cpu().tolist(),
        'target_gate_weight_norm': float(relation.gate.weight.detach().norm()),
        'model_code_sha256': sha256(HERE / 'model.py'),
        'training_code_sha256': sha256(HERE / 'train.py'),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
