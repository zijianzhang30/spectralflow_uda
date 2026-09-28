"""Post-hoc semantic purity of fixed diagnostic OT batches, not training replay."""
from __future__ import annotations

import json
from pathlib import Path

import hdf5storage
import numpy as np
import torch

from flow import ot_pairs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (202601, 202602, 202603)
BATCHES = 200


def main():
    torch.set_num_threads(2)
    out = {"scope": "post-hoc representative fixed random batches; not exact training replay",
           "batches": BATCHES, "seeds": {}}
    for seed in SEEDS:
        with np.load(HERE / f"formal_{seed}/frozen_features_before_gt.npz") as data:
            source_z = data["source_z"].copy()
            labels = data["source_labels"].copy()
            target_z = data["target_z"].copy()
            q = data["target_q"].copy()
            centers = data["target_centers"].copy()
        shift = ROOT / "investigations/scene_shift_v1" / f"formal_{seed}"
        cfg = json.loads((shift / "config.json").read_text())
        gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
        target_y = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
        rng = np.random.RandomState(seed + 87031)
        sampler = torch.Generator(device="cpu").manual_seed(seed + 99173)
        pair_n = correct_n = 0
        present = np.zeros(7, np.int64)
        class_pairs = np.zeros(7, np.int64)
        class_correct = np.zeros(7, np.int64)
        for _ in range(BATCHES):
            src = rng.choice(len(source_z), 32, replace=False)
            dst = rng.choice(len(target_z), 32, replace=False)
            si, ti = ot_pairs(torch.from_numpy(source_z[src]),
                              torch.from_numpy(target_z[dst]),
                              torch.from_numpy(labels[src]),
                              torch.from_numpy(q[dst]), sampler)
            present[np.unique(labels[src])] += 1
            if not len(si):
                continue
            picked_class = labels[src[si.numpy()]]
            picked_true = target_y[dst[ti.numpy()]]
            pair_n += len(si)
            correct_n += int((picked_class == picked_true).sum())
            for c in range(7):
                mask = picked_class == c
                class_pairs[c] += int(mask.sum())
                class_correct[c] += int((picked_true[mask] == c).sum())
        out["seeds"][str(seed)] = {
            "sampled_pairs": int(pair_n),
            "pair_semantic_purity": float(correct_n / pair_n),
            "class_pair_counts": class_pairs.tolist(),
            "class_pair_purity": [None if not class_pairs[c] else
                                  float(class_correct[c] / class_pairs[c]) for c in range(7)],
            "source_class_presence_batches": present.tolist(),
        }
    (HERE / "PAIR_AUDIT.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
