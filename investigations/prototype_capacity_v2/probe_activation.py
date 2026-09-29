"""Label-free target inference intervention on all new source-anchor checkpoints."""
from __future__ import annotations

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


def main():
    torch.set_num_threads(2)
    cfg = json.loads((HERE / 'formal_global_1341/config.json').read_text())
    _, target = load_images(Path(cfg['data']), 'official_ilda', Path(cfg['ilda_cache']))
    results = {}
    for seed in (1341, 1174, 1370):
        results[str(seed)] = {}
        for mode in ('prototype', 'global'):
            if seed == 1341 and mode == 'prototype':
                continue  # Existing v1 probe covers this fixed-epoch checkpoint.
            run = HERE / f'formal_{mode}_{seed}'
            with np.load(run / 'source_split.npz') as z:
                centers = z['target_centers'][:1024].copy()
            loader = DataLoader(Patches(target, centers), batch_size=32, shuffle=False,
                                drop_last=True, num_workers=0)
            results[str(seed)][mode] = {}
            for selection in ('last', 'best_source_val'):
                cp = torch.load(run / f'{selection}.pth', map_location='cpu', weights_only=False)
                model = make_model(mode == 'global', 'cuda:0')
                model.load_state_dict(cp['model'], strict=True)
                ref = cp['last_source_batch'].cuda()

                def predict():
                    pieces = []
                    with evaluation(model):
                        for x in loader:
                            pieces.append(torch.softmax(model(ref, x.cuda())[8], 1).cpu().numpy())
                    return np.concatenate(pieces)

                active = predict()
                with np.load(run / f'{selection}_predictions_before_gt.npz') as z:
                    frozen = z['probabilities'][:1024]
                assert np.array_equal(active.argmax(1), frozen.argmax(1))
                assert np.max(np.abs(active - frozen)) < 1e-6
                model.feature_layers.atten.bank_ready.fill_(False)
                bypass = predict()
                results[str(seed)][mode][selection] = {
                    'argmax_changed_n_of_1024': int((active.argmax(1) != bypass.argmax(1)).sum()),
                    'mean_probability_l1': float(np.mean(np.abs(active - bypass).sum(1))),
                    'max_probability_abs': float(np.max(np.abs(active - bypass))),
                    'target_gt_accessed': False,
                }
    (HERE / 'ACTIVATION_PROBE.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
