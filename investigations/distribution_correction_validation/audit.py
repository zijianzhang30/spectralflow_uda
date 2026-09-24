"""GT-only final audit of saved, frozen A/B predictions and OT input quality."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "investigations/distribution_correction_v1"))
sys.path.insert(0, str(PROJECT / "investigations/correspondence_v1"))
sys.path.insert(0, str(PROJECT / "experiments/round9"))
from run import metric, hash_file
from screen import empty_stats, score_candidates, score_pairs, finish
from data import Patches, load_images
from model import Backbone


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(202601, 202602, 202603), required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    student_run = HERE / "runs" / f"student_{args.seed}"
    out = HERE / "runs" / f"correction_{args.seed}"
    audit_path = out / "audit.json"
    assert not audit_path.exists(), f"Refusing to overwrite {audit_path}"
    metadata = json.loads((out / "selection_and_projection.json").read_text())
    assert metadata["target_gt_opened_by_correction_script"] is False
    with np.load(out / "predictions_before_gt.npz") as data:
        centers = data["centers"].copy()
        q = {"A": data["A"].copy(), "B": data["B"].copy()}
        prior = data["prior"].copy()
        teacher = data["teacher"].copy()
    assert centers.shape == (53200, 2) and teacher.shape == (53200, 7)
    assert all(x.shape == (53184, 7) for x in q.values())
    assert np.allclose(teacher.astype(np.float64).mean(axis=0), prior, atol=1e-7)
    assert np.max(np.abs(q["B"].astype(np.float64).mean(axis=0) - prior)) < 1e-5
    with np.load(student_run / "source_split.npz") as split:
        assert np.array_equal(centers, split["target_centers"])
        train_centers, train_labels = split["train_centers"].copy(), split["train_labels"].copy()
    cfg = json.loads((student_run / "config.json").read_text())
    gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
    gt = hdf5storage.loadmat(str(gt_path))["map"]
    y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
    assert np.min(y) == 0 and np.max(y) == 6
    support = np.bincount(y, minlength=7)
    stats = {name: empty_stats() for name in q}
    for name, x in q.items():
        for start in range(0, 53184, 32):
            xb = x[start:start + 32]
            score_candidates(stats[name], torch.from_numpy(xb.max(axis=1) >= 0.8),
                             torch.from_numpy(xb.argmax(axis=1)),
                             torch.from_numpy(y[start:start + 32]))
    model = Backbone().to(args.device).eval()
    checkpoint = student_run / "best_source_val.pth"
    assert hash_file(checkpoint) == metadata["student_checkpoint_sha256"]
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    source, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    target_loader = DataLoader(Patches(target, centers[:200 * 32]), batch_size=32,
                               shuffle=False, drop_last=True)
    source_loader = DataLoader(Patches(source, train_centers, train_labels), batch_size=32,
                               shuffle=False, drop_last=True)
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
            for name, x in q.items():
                xb = x[batch * 32:(batch + 1) * 32]
                score_pairs(stats[name], zs, zt, sy, ty,
                            torch.from_numpy(xb.max(axis=1) >= 0.8),
                            torch.from_numpy(xb.argmax(axis=1)), torch.from_numpy(xb))
    variants = {}
    for name, x in q.items():
        variants[name] = {"classification": metric(y, x),
                          "correspondence": finish(stats[name], support, 53184 // 32)}
    official = json.loads((student_run / "final_target_source_val_best.json").read_text())
    assert variants["A"]["classification"]["confusion_matrix"] == official["metrics"]["confusion_matrix"]
    result = {"seed": args.seed, "prediction_file_sha256": hash_file(out / "predictions_before_gt.npz"),
              "target_gt_sha256": hash_file(gt_path), "source_val_best_checkpoint_sha256": hash_file(checkpoint),
              "prior": prior.tolist(), "variants": variants,
              "GT_role": "post-hoc classification and correspondence audit only"}
    audit_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"seed": args.seed, "A_OA": variants["A"]["classification"]["oa_official"],
                      "B_OA": variants["B"]["classification"]["oa_official"],
                      "A_AA": variants["A"]["classification"]["aa"],
                      "B_AA": variants["B"]["classification"]["aa"],
                      "A_class7": variants["A"]["classification"]["recall"][6],
                      "B_class7": variants["B"]["classification"]["recall"][6]}), flush=True)


if __name__ == "__main__":
    main()
