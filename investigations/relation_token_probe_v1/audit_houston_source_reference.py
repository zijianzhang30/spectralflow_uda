"""Label-free source-reference sensitivity of frozen Houston full MLUDA."""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(OFFICIAL / 'reference'))
from data import Patches, load_images  # noqa: E402
from runtime import evaluation  # noqa: E402
from baseline_pool import install_deterministic_pool  # noqa: E402
from net2 import DSANSS  # noqa: E402

COUNT = 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1341, 1174, 1370))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = OFFICIAL / 'runs/round1/formal' / str(args.seed) / 'MLUDA_full'
    cfg = json.loads((run / 'config.json').read_text())
    checkpoint = torch.load(run / 'last.pth', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 100
    with np.load(run / 'source_split.npz') as z:
        train_centers = z['train_centers'].copy()
        target_centers = z['target_centers'][:COUNT].copy()
    with np.load(ROOT / 'investigations/mluda_fixed100_v1' /
                 f'seed_{args.seed}_before_gt.npz') as z:
        frozen = z['predictions'][:COUNT].copy()
        assert np.array_equal(z['centers'][:COUNT], target_centers)
    source, target = load_images(Path(cfg['data']), 'official_ilda',
                                 Path(cfg['ilda_cache']))
    target_loader = DataLoader(Patches(target, target_centers), batch_size=32,
                               shuffle=False, drop_last=True, num_workers=0)
    source_dataset = Patches(source, train_centers)
    source_batches = []
    for start in (0, 32, 64, 96):
        source_batches.append(torch.stack([source_dataset[i] for i in range(start, start + 32)]))
    model = install_deterministic_pool(DSANSS(48, 7, 7).to(args.device))
    model.load_state_dict(checkpoint['model'], strict=True)
    source_ref = checkpoint['last_source_batch'].to(args.device)

    def predict(ref):
        chunks = []
        with evaluation(model), torch.inference_mode():
            for x in target_loader:
                logits = model(ref.to(args.device), x.to(args.device))[8]
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(chunks)

    baseline = predict(source_ref)
    replay_disagreements = int(np.sum(baseline.argmax(1) != frozen))
    assert replay_disagreements == 0, replay_disagreements
    rng = np.random.RandomState(20260928)
    rows = []
    for rep in range(8):
        p = predict(source_ref[rng.permutation(32)])
        rows.append({'kind': 'permutation', 'rep': rep,
                     'argmax_change_fraction': float(np.mean(p.argmax(1) != baseline.argmax(1))),
                     'mean_probability_l1': float(np.mean(np.abs(p - baseline).sum(1))),
                     'max_probability_abs': float(np.max(np.abs(p - baseline)))})
    for rep, ref in enumerate(source_batches):
        p = predict(ref)
        rows.append({'kind': 'replacement', 'rep': rep,
                     'argmax_change_fraction': float(np.mean(p.argmax(1) != baseline.argmax(1))),
                     'mean_probability_l1': float(np.mean(np.abs(p - baseline).sum(1))),
                     'max_probability_abs': float(np.max(np.abs(p - baseline)))})
    result = {'seed': args.seed, 'sample_count': COUNT,
              'checkpoint': str(run / 'last.pth'), 'epoch': 100,
              'frozen_prediction_replay_disagreements': replay_disagreements,
              'target_gt_accessed': False, 'rows': rows}
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / f'HOUSTON_{args.seed}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
