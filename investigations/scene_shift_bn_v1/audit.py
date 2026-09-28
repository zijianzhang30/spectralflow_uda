"""Post-hoc official scoring, only after all SHIFT_BN predictions are frozen."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "investigations/distribution_correction_v1"))
from run import metric

SEEDS = (202601, 202602, 202603)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    # Prevent a partial read of target GT while predictions are still being generated.
    for seed in SEEDS:
        run = HERE / f"formal_{seed}"
        for selection in ("best", "paired"):
            meta = json.loads((run / f"prediction_metadata_{selection}.json").read_text())
            assert meta["target_gt_opened"] is False
            assert meta["prediction_sha256"] == digest(
                run / f"predictions_before_gt_{selection}.npz")
    out = {"scope": "Post-hoc target-GT diagnosis; not checkpoint or hyperparameter selection",
           "methods": ("A", "SHIFT", "SHIFT_BN_best", "SHIFT_BN_paired"), "seeds": {}}
    for seed in SEEDS:
        base = ROOT / "investigations/distribution_correction_validation/runs"
        with np.load(base / f"soft_{seed}/probabilities_before_gt.npz") as data:
            centers = data["centers"].copy()
            probs = {"A": {"raw": data["lambda_0"].copy(),
                           "corrected": data["lambda_1"].copy()}}
        for method, path in (("SHIFT", ROOT / "investigations/scene_shift_v1" /
                              f"formal_{seed}/predictions_before_gt.npz"),
                             ("SHIFT_BN_best", HERE / f"formal_{seed}/predictions_before_gt_best.npz"),
                             ("SHIFT_BN_paired", HERE / f"formal_{seed}/predictions_before_gt_paired.npz")):
            with np.load(path) as data:
                assert np.array_equal(centers, data["centers"])
                probs[method] = {v: data[v].copy() for v in ("raw", "corrected")}
        cfg = json.loads((base / f"student_{seed}/config.json").read_text())
        gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        row = {}
        for method in ("A", "SHIFT", "SHIFT_BN_best", "SHIFT_BN_paired"):
            row[method] = {}
            for variant in ("raw", "corrected"):
                q = probs[method][variant]
                row[method][variant] = {
                    "metrics": metric(y, q),
                    "soft_mass": q.mean(axis=0).tolist(),
                    "candidate_coverage_at_0.8": float((q.max(axis=1) >= .8).mean()),
                    "class7_correct_candidate_recall_at_0.8":
                        float(((q.argmax(1) == 6) & (q.max(axis=1) >= .8) & (y == 6)).sum() /
                              (y == 6).sum()),
                }
        out["seeds"][str(seed)] = row
    (HERE / "AUDIT.json").write_text(json.dumps(out, indent=2))
    for variant in ("raw", "corrected"):
        for method in ("A", "SHIFT", "SHIFT_BN_best", "SHIFT_BN_paired"):
            items = [out["seeds"][str(s)][method][variant]["metrics"] for s in SEEDS]
            print(variant, method,
                  "OA", np.mean([x["oa_official"] for x in items]) * 100,
                  "AA", np.mean([x["aa"] for x in items]) * 100,
                  "Kappa", np.mean([x["kappa"] for x in items]),
                  "class7", np.mean([x["recall"][6] for x in items]) * 100)


if __name__ == "__main__":
    main()
