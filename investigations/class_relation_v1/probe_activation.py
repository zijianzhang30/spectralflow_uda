"""Label-free inference intervention on the frozen epoch-100 checkpoints."""
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
    results = {}
    for mode in ('prototype', 'class_token'):
        run = HERE / f'formal_{mode}_1341'
        cfg = json.loads((run / 'config.json').read_text())
        cp = torch.load(run / 'last.pth', map_location='cpu', weights_only=False)
        with np.load(run / 'source_split.npz') as z:
            centers = z['target_centers'][:1024].copy()
        _, target = load_images(Path(cfg['data']), 'official_ilda', Path(cfg['ilda_cache']))
        loader = DataLoader(Patches(target, centers), batch_size=32, shuffle=False,
                            drop_last=True, num_workers=0)
        model = make_model(mode == 'class_token', 'cuda:0')
        model.load_state_dict(cp['model'], strict=True)
        ref = cp['last_source_batch'].cuda()

        def predict():
            out = []
            with evaluation(model):
                for x in loader:
                    out.append(torch.softmax(model(ref, x.cuda())[8], dim=1).cpu().numpy())
            return np.concatenate(out)

        active = predict()
        with np.load(run / 'last_predictions_before_gt.npz') as z:
            frozen = z['probabilities'][:1024]
        assert np.array_equal(active.argmax(1), frozen.argmax(1))
        assert np.max(np.abs(active - frozen)) < 1e-6
        relation = model.feature_layers.atten
        relation.bank_ready.fill_(False)
        bypass = predict()
        result = {
            'argmax_changed_n_of_1024': int((active.argmax(1) != bypass.argmax(1)).sum()),
            'mean_probability_l1': float(np.mean(np.abs(active - bypass).sum(1))),
            'max_probability_abs': float(np.max(np.abs(active - bypass))),
            'target_gt_accessed': False,
        }
        if mode == 'class_token':
            relation.bank_ready.fill_(True)
            with torch.no_grad():
                relation.class_tokens.zero_()
            zero_token = predict()
            result['zero_token_argmax_changed_n_of_1024'] = int(
                (active.argmax(1) != zero_token.argmax(1)).sum())
            result['zero_token_mean_probability_l1'] = float(
                np.mean(np.abs(active - zero_token).sum(1)))
        results[mode] = result
    (HERE / 'ACTIVATION_PROBE_1341.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
