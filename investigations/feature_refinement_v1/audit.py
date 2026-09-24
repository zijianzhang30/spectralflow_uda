"""Post-hoc official target scoring of the pre-frozen three-seed grid."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "investigations/distribution_correction_v1"))
from run import metric

SEEDS = (202601, 202602, 202603)
METHODS = ("A", "F1", "F2")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def details(q, y):
    p = q.argmax(axis=1)
    keep = q.max(axis=1) >= 0.8
    return {"metrics": metric(y, q),
            "class7_correct_candidate_recall": float(((p == 6) & keep & (y == 6)).sum() / (y == 6).sum()),
            "candidate_coverage": float(keep.mean()),
            "soft_expected_correct_fraction": float(q[np.arange(len(y)), y].mean())}


def main():
    out = {"seeds": list(SEEDS), "methods": list(METHODS),
           "scope": "source-val-best, official Houston18 protocol; post-hoc target GT only",
           "results": {}}
    for seed in SEEDS:
        baseline = PROJECT / "investigations/distribution_correction_validation/runs" / f"soft_{seed}"
        baseline_audit = json.loads((baseline / "audit.json").read_text())
        with np.load(baseline / "probabilities_before_gt.npz") as data:
            centers = data["centers"].copy()
            baseline_q = {"raw": data["lambda_0"].copy(),
                          "corrected": data["lambda_1"].copy()}
        student = PROJECT / "investigations/distribution_correction_validation/runs" / f"student_{seed}"
        cfg = json.loads((student / "config.json").read_text())
        gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
        gt = hdf5storage.loadmat(str(gt_path))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        row = {"target_gt_sha256": digest(gt_path), "methods": {}}
        for method in METHODS:
            if method == "A":
                q = baseline_q
                checkpoint = student / "best_source_val.pth"
                selected_epoch = json.loads((student / "best_source_val.json").read_text())["epoch"]
            else:
                run = HERE / "runs" / f"formal_{method}_{seed}"
                meta = json.loads((run / "prediction_metadata.json").read_text())
                path = run / "predictions_before_gt.npz"
                assert not meta["target_gt_opened"]
                assert meta["prediction_sha256"] == digest(path)
                checkpoint = run / "best_source_val.pth"
                assert meta["checkpoint_sha256"] == digest(checkpoint)
                with np.load(path) as data:
                    assert np.array_equal(centers, data["centers"])
                    q = {"raw": data["raw"].copy(), "corrected": data["corrected"].copy()}
                selected_epoch = meta["selected_epoch"]
            variants = {name: details(probs, y) for name, probs in q.items()}
            if method == "A":
                assert variants["raw"]["metrics"] == baseline_audit["variants"]["lambda_0"]["classification"]
                assert variants["corrected"]["metrics"] == baseline_audit["variants"]["lambda_1"]["classification"]
            row["methods"][method] = {"selected_epoch": selected_epoch,
                                      "checkpoint_sha256": digest(checkpoint),
                                      "variants": variants}
        out["results"][str(seed)] = row
    path = HERE / "AUDIT.json"
    assert not path.exists(), f"Refusing to overwrite {path}"
    path.write_text(json.dumps(out, indent=2))
    for seed in SEEDS:
        for method in METHODS:
            for variant in ("raw", "corrected"):
                metrics = out["results"][str(seed)]["methods"][method]["variants"][variant]["metrics"]
                print(seed, method, variant, "OA", round(metrics["oa_official"] * 100, 2),
                      "AA", round(metrics["aa"] * 100, 2),
                      "class7", round(metrics["recall"][6] * 100, 2))


if __name__ == "__main__":
    main()
