"""Candidate and first-200-batch OT audit for fixed100 Mean Teacher sensitivity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.utils.data import DataLoader

from run import (HERE, PROJECT, Backbone, Patches, load_images, empty_stats,
                 score_candidates, score_pairs, finish)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(1341, 1174, 1370), required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    out = HERE / "runs" / f"seed_{args.seed}"
    with np.load(out / "mean_teacher_fixed100_sensitivity_probabilities.npz") as f:
        centers, variants = f["centers"].copy(), {k: f[k].copy() for k in ("A", "B", "C")}
    assert centers.shape == (53184, 2)
    run = PROJECT / "experiments/round9/runs" / f"formal_B_{args.seed}"
    cfg = json.loads((run / "config.json").read_text())
    checkpoint = torch.load(run / "last.pth", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 100
    model = Backbone().to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    source, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    with np.load(run / "source_split.npz") as f:
        assert np.array_equal(f["target_centers"][:53184], centers)
        train_centers, train_labels = f["train_centers"].copy(), f["train_labels"].copy()
    gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
    y = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    support = np.bincount(y, minlength=7)
    stats = {name: empty_stats() for name in variants}
    for name, q in variants.items():
        score_candidates(stats[name], torch.from_numpy(q.max(1) >= 0.8),
                         torch.from_numpy(q.argmax(1)), torch.from_numpy(y))
    target_loader = DataLoader(Patches(target, centers[:6400]), batch_size=32,
                               shuffle=False, drop_last=True)
    source_loader = DataLoader(Patches(source, train_centers, train_labels),
                               batch_size=32, shuffle=False, drop_last=True)
    source_iter = iter(source_loader)
    with torch.inference_mode():
        for batch, tx in enumerate(target_loader):
            try:
                sx, sy = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                sx, sy = next(source_iter)
            zs, _ = model(sx.to(args.device))
            zt, _ = model(tx.to(args.device))
            ty = torch.from_numpy(y[batch * 32:(batch + 1) * 32])
            for name, q in variants.items():
                qb = q[batch * 32:(batch + 1) * 32]
                score_pairs(stats[name], zs, zt, sy, ty,
                            torch.from_numpy(qb.max(1) >= 0.8),
                            torch.from_numpy(qb.argmax(1)), torch.from_numpy(qb))
    result = {"seed": args.seed, "selection": "Mean Teacher B fixed epoch 100",
              "candidate_gate": 0.8, "pair_audit_batches": 200,
              "target_gt_use": "post-hoc candidate and pair-purity audit only",
              "methods": {name: finish(stats[name], support, 53184 // 32) for name in variants}}
    (out / "mean_teacher_fixed100_pair_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"seed": args.seed,
                      "outcomes": {name: {"coverage": result["methods"][name]["candidate_coverage_fraction"],
                                          "class7_candidate_recall": result["methods"][name]["true_class_candidate_recall"][6],
                                          "pair_expected_purity": result["methods"][name]["pair_expected_purity_overall"]}
                                   for name in variants}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
