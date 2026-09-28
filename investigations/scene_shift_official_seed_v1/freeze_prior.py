"""Reuse source-trained HyperSIGMA; estimate a prior on all target centers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "investigations/hypersigma_teacher_gate"))
from evaluate_matched import CenterPatches, SSFusionFramework, probabilities


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(1341, 1174, 1370), required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    out = HERE / f"prior_{args.seed}_before_gt.npz"
    meta_path = HERE / f"prior_{args.seed}_freeze.json"
    assert not out.exists() and not meta_path.exists()
    teacher_run = ROOT / "investigations/hypersigma_teacher_gate/runs" / f"seed_{args.seed}"
    cfg = json.loads((teacher_run / "config.json").read_text())
    mluda_run = ROOT / "official_aligned/runs/round1/formal" / str(args.seed) / "MLUDA_full"
    with np.load(mluda_run / "source_split.npz") as split:
        centers = split["target_centers"].copy()
    with np.load(ROOT / "experiments/round9/runs" / f"formal_A_{args.seed}" / "source_split.npz") as split:
        assert np.array_equal(centers, split["target_centers"])
    assert centers.shape == (53200, 2)
    stage_paths = [teacher_run / "stage1/best.pth", teacher_run / "full/best.pth"]
    stage_meta = [torch.load(x, map_location="cpu", weights_only=False)["best"] for x in stage_paths]
    selected = int(stage_meta[1]["val_acc"] > stage_meta[0]["val_acc"])
    checkpoint_path = stage_paths[selected]
    model = SSFusionFramework(img_size=33, in_channels=48, patch_size=2,
                              classes=7, model_size="base")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"])
    model.to(args.device).eval()
    data_root = Path("/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Houston")
    raw = hdf5storage.loadmat(str(data_root / "Houston18.mat"))["ori_data"].astype(np.float32)
    with np.load(teacher_run / "target_probabilities.npz") as older:
        assert np.array_equal(centers[:53184], older["centers"])
        frozen_first = older["teacher"].copy()
    assert frozen_first.shape == (53184, 7)
    # The old run was evaluated with drop_last=True; only its final 16
    # official-order centers need fresh teacher inference.
    final_loader = DataLoader(CenterPatches(raw, centers[53184:], 33), batch_size=16,
                              shuffle=False, num_workers=0)
    final_q = probabilities(model, final_loader, args.device)
    assert final_q.shape == (16, 7)
    q = np.concatenate((frozen_first, final_q), axis=0)
    assert q.shape == (53200, 7) and np.allclose(q.sum(axis=1), 1, atol=1e-5)
    prior = q.astype(np.float64).mean(axis=0)
    np.savez_compressed(out, centers=centers, teacher=q, prior=prior)
    meta_path.write_text(json.dumps({"seed": args.seed, "selected_stage": "full" if selected else "stage1",
                                     "teacher_checkpoint_sha256": sha(checkpoint_path),
                                     "prior_prediction_sha256": sha(out),
                                     "frozen_first_53184_sha256": sha(teacher_run / "target_probabilities.npz"),
                                     "target_gt_opened": False, "prior_n": 53200,
                                     "prior": prior.tolist()}, indent=2))
    print(json.dumps({"seed": args.seed, "stage": "full" if selected else "stage1",
                      "prior": prior.tolist()}), flush=True)


if __name__ == "__main__":
    main()
