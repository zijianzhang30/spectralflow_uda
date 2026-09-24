"""Post-hoc classification and correspondence audit of the frozen KL curve."""
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
    old_run = HERE / "runs" / f"correction_{args.seed}"
    soft_run = HERE / "runs" / f"soft_{args.seed}"
    audit_path = soft_run / "audit.json"
    assert not audit_path.exists(), f"Refusing to overwrite {audit_path}"
    frozen = json.loads((soft_run / "solver.json").read_text())
    assert frozen["target_gt_opened_by_soft_script"] is False
    assert frozen["source_probabilities_sha256"] == hash_file(old_run / "predictions_before_gt.npz")
    with np.load(soft_run / "probabilities_before_gt.npz") as data:
        centers = data["centers"].copy()
        prior = data["prior"].copy()
        variants = {name: data[name].copy() for name in frozen["grid"]}
    with np.load(old_run / "predictions_before_gt.npz") as data:
        assert np.array_equal(centers, data["centers"])
        assert np.array_equal(prior, data["prior"])
        assert np.array_equal(variants["lambda_0"], data["A"])
        assert np.array_equal(variants["hard"], data["B"])
    assert all(v.shape == (53184, 7) for v in variants.values())
    with np.load(student_run / "source_split.npz") as split:
        assert np.array_equal(centers, split["target_centers"])
        source_centers = split["train_centers"].copy()
        source_labels = split["train_labels"].copy()
    cfg = json.loads((student_run / "config.json").read_text())
    gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
    gt = hdf5storage.loadmat(str(gt_path))["map"]
    y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
    support = np.bincount(y, minlength=7)
    stats = {name: empty_stats() for name in variants}
    for name, q in variants.items():
        pred = q.argmax(1)
        mask = q.max(1) >= 0.8
        for start in range(0, 53184, 32):
            score_candidates(stats[name], torch.from_numpy(mask[start:start + 32]),
                             torch.from_numpy(pred[start:start + 32]),
                             torch.from_numpy(y[start:start + 32]))
    checkpoint = student_run / "best_source_val.pth"
    model = Backbone().to(args.device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    source, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    target_loader = DataLoader(Patches(target, centers[:200 * 32]), batch_size=32,
                               shuffle=False, drop_last=True)
    source_loader = DataLoader(Patches(source, source_centers, source_labels),
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
    result_variants = {name: {"classification": metric(y, q),
                              "correspondence": finish(stats[name], support, 53184 // 32)}
                       for name, q in variants.items()}
    old_audit = json.loads((old_run / "audit.json").read_text())
    for name, old_name in (("lambda_0", "A"), ("hard", "B")):
        assert result_variants[name]["classification"] == old_audit["variants"][old_name]["classification"]
        old_corr = old_audit["variants"][old_name]["correspondence"]
        new_corr = result_variants[name]["correspondence"]
        assert new_corr["candidate_coverage_fraction"] == old_corr["candidate_coverage_fraction"]
        assert np.isclose(new_corr["pair_expected_purity_overall"], old_corr["pair_expected_purity_overall"], atol=1e-8)
    result = {"seed": args.seed, "grid": frozen["grid"],
              "prediction_file_sha256": hash_file(soft_run / "probabilities_before_gt.npz"),
              "checkpoint_sha256": hash_file(checkpoint),
              "target_gt_sha256": hash_file(gt_path),
              "GT_role": "post-hoc metrics only; no solver fitting or lambda selection",
              "variants": result_variants}
    audit_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"seed": args.seed,
                      "curve": {name: {"oa": result_variants[name]["classification"]["oa_official"],
                                       "aa": result_variants[name]["classification"]["aa"],
                                       "class7": result_variants[name]["classification"]["recall"][6]}
                                for name in frozen["grid"]}}), flush=True)


if __name__ == "__main__":
    main()
