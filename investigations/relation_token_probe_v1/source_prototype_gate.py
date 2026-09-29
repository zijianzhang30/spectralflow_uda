"""Source-label-only feasibility check for class prototypes before MLUDA MBCA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFFICIAL = ROOT / 'official_aligned'
sys.path.insert(0, str(OFFICIAL))
sys.path.insert(0, str(OFFICIAL / 'reference'))
from data import Patches, load_images  # noqa: E402
from runtime import evaluation  # noqa: E402
from baseline_pool import install_deterministic_pool  # noqa: E402
from net2 import DSANSS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1341, 1174, 1370))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = OFFICIAL / 'runs/round1/formal' / str(args.seed) / 'MLUDA_full'
    cfg = json.loads((run / 'config.json').read_text())
    cp = torch.load(run / 'best_source_val.pth', map_location='cpu', weights_only=False)
    with np.load(run / 'source_split.npz') as z:
        train_centers = z['train_centers'].copy()
        train_labels = z['train_labels'].copy()
        val_centers = z['val_centers'].copy()
        val_labels = z['val_labels'].copy()
    source, _ = load_images(Path(cfg['data']), 'official_ilda', Path(cfg['ilda_cache']))
    model = install_deterministic_pool(DSANSS(48, 7, 7).to(args.device))
    model.load_state_dict(cp['model'], strict=True)
    captured = []

    def save_pre_mbca(_module, inputs):
        captured.append(inputs[0].detach().cpu())

    hook = model.feature_layers.atten.register_forward_pre_hook(save_pre_mbca)

    def extract(centers, labels):
        loader = DataLoader(Patches(source, centers, labels), batch_size=32,
                            shuffle=False, drop_last=False, num_workers=0)
        features, predictions = [], []
        with evaluation(model):
            for x, _ in loader:
                x = x.to(args.device)
                logits = model(x, x)[3]
                predictions.append(logits.argmax(1).cpu().numpy())
                features.append(captured.pop().numpy())
        return np.concatenate(features), np.concatenate(predictions)

    train_features, _ = extract(train_centers, train_labels)
    val_features, classifier_predictions = extract(val_centers, val_labels)
    hook.remove()
    prototypes = np.stack([train_features[train_labels == c].mean(0) for c in range(7)])
    proto = F.normalize(torch.from_numpy(prototypes), dim=1).numpy()
    val = F.normalize(torch.from_numpy(val_features), dim=1).numpy()
    similarity = val @ proto.T
    proto_predictions = similarity.argmax(1)
    top2 = np.sort(similarity, axis=1)[:, -2:]
    result = {
        'seed': args.seed, 'selected_epoch': cp['epoch'],
        'train_n': len(train_labels), 'val_n': len(val_labels),
        'train_per_class': np.bincount(train_labels, minlength=7).tolist(),
        'val_per_class': np.bincount(val_labels, minlength=7).tolist(),
        'feature_location': '288-dimensional pooled features immediately before MBCA',
        'classifier_val_accuracy': float(np.mean(classifier_predictions == val_labels)),
        'classifier_recall': [float(np.mean(classifier_predictions[val_labels == c] == c)) for c in range(7)],
        'nearest_prototype_val_accuracy': float(np.mean(proto_predictions == val_labels)),
        'nearest_prototype_recall': [float(np.mean(proto_predictions[val_labels == c] == c)) for c in range(7)],
        'prototype_cosine': (proto @ proto.T).tolist(),
        'nearest_prototype_margin_mean': float(np.mean(top2[:, 1] - top2[:, 0])),
        'target_gt_accessed': False,
    }
    (HERE / f'SOURCE_PROTOTYPE_{args.seed}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'prototype_cosine'}, indent=2))


if __name__ == '__main__':
    main()
