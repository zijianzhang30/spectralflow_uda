"""Check whether frozen MLUDA target predictions depend on arbitrary source order.

Uses no Hangzhou labels. The original final-epoch source batch, target centers,
checkpoint and preprocessing are replayed before permuting source positions.
"""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing

HERE = Path(__file__).resolve().parent
LEGACY = Path('/home/zhangzj26/TGRS_MLUDA-2024')
sys.path.insert(0, str(LEGACY))
from net2 import DSANSS  # noqa: E402
from UtilsCMS import ILDA  # noqa: E402

COUNT = 1024
PERMUTATIONS = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1341, 1535, 1631))
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = HERE / 'official_three_runs' / f'seed_{args.seed}'
    with np.load(run / 'predictions_before_score.npz') as z:
        centers = z['centers'][:COUNT].copy()
        reference = z['probabilities'][:COUNT].copy()
    checkpoint = torch.load(run / 'last.pth', map_location='cpu', weights_only=False)
    source_batch = checkpoint['last_source_batch'].to('cuda:0')
    assert source_batch.shape == (32, 198, 1, 1)
    model = DSANSS(198, 1, 3).to('cuda:0').eval()
    model.load_state_dict(checkpoint['model'], strict=True)

    data = sio.loadmat(str(LEGACY / 'datasets/Shanghai-Hangzhou/DataCube.mat'),
                       variable_names=['DataCube1', 'DataCube2'])
    source_raw, target_raw = data['DataCube1'], data['DataCube2']
    source = preprocessing.scale(source_raw.reshape(-1, 198)).reshape(source_raw.shape)
    target = preprocessing.scale(target_raw.reshape(-1, 198)).reshape(target_raw.shape)
    del data, source_raw, target_raw
    source, target = ILDA(source, target, 2, 0.00009)
    with np.load(run / 'source_split.npz') as split:
        train_centers = split['train_centers'].copy()
    alternate_batches = []
    for start in (0, 32, 64, 96):
        selected = train_centers[start:start + 32]
        z = source[selected[:, 0], selected[:, 1]].astype(np.float32)
        alternate_batches.append(torch.from_numpy(z[:, :, None, None]).to('cuda:0'))
    del source
    x = target[centers[:, 0], centers[:, 1]].astype(np.float32)
    x = torch.from_numpy(x[:, :, None, None]).to('cuda:0')
    del target

    def predict(s):
        chunks = []
        with torch.inference_mode():
            for start in range(0, COUNT, 32):
                output = model(s, x[start:start + 32])[8]
                chunks.append(torch.softmax(output, dim=1).cpu().numpy())
        return np.concatenate(chunks)

    baseline = predict(source_batch)
    replay_max_abs = float(np.max(np.abs(baseline - reference)))
    replay_argmax_disagreements = int(np.sum(baseline.argmax(1) != reference.argmax(1)))
    assert replay_max_abs < 1e-3 and replay_argmax_disagreements == 0, (
        replay_max_abs, replay_argmax_disagreements)
    rng = np.random.RandomState(20260928)
    rows = []
    for rep in range(PERMUTATIONS):
        order = rng.permutation(32)
        p = predict(source_batch[order])
        rows.append({
            'repetition': rep,
            'argmax_change_fraction': float(np.mean(p.argmax(1) != baseline.argmax(1))),
            'mean_probability_l1': float(np.mean(np.abs(p - baseline).sum(axis=1))),
            'max_probability_abs': float(np.max(np.abs(p - baseline))),
        })
    replacements = []
    for rep, alternate in enumerate(alternate_batches):
        p = predict(alternate)
        replacements.append({
            'source_batch_index': rep,
            'argmax_change_fraction': float(np.mean(p.argmax(1) != baseline.argmax(1))),
            'mean_probability_l1': float(np.mean(np.abs(p - baseline).sum(axis=1))),
            'max_probability_abs': float(np.max(np.abs(p - baseline))),
        })
    result = {
        'seed': args.seed, 'sample_count': COUNT, 'source_batch_count': 32,
        'source_order_permutations': PERMUTATIONS,
        'baseline_replay_max_abs': replay_max_abs,
        'baseline_replay_argmax_disagreements': replay_argmax_disagreements,
        'target_gt_accessed': False, 'rows': rows,
        'source_batch_replacements': replacements,
        'mean_argmax_change_fraction': float(np.mean([r['argmax_change_fraction'] for r in rows])),
        'mean_probability_l1': float(np.mean([r['mean_probability_l1'] for r in rows])),
    }
    (HERE / f'SOURCE_PAIRING_AUDIT_{args.seed}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
