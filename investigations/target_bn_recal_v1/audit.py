"""Official post-hoc scoring after all six recalibrated predictions are frozen."""
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
    lock_hash = digest(HERE / "EXPERIMENT_LOCK.md")
    for seed in SEEDS:
        for method in ("A", "SHIFT"):
            run = HERE / f"{method}_{seed}"
            meta = json.loads((run / "metadata.json").read_text())
            assert meta["target_gt_opened"] is False
            assert meta["lock_sha256"] == lock_hash
            assert meta["parameter_hash_unchanged"] and meta["calibration_batches"] == 1663
            assert meta["bn_buffer_hash_before"] != meta["bn_buffer_hash_after"]
            assert meta["prediction_sha256"] == digest(run / "predictions_before_gt.npz")
            assert meta["calibrated_state_sha256"] == digest(run / "calibrated_state.pth")
    result = {"scope": "Post-hoc diagnostic; no target-based method or checkpoint selection",
              "seeds": {}}
    for seed in SEEDS:
        base = ROOT / "investigations/distribution_correction_validation/runs"
        with np.load(base / f"soft_{seed}/probabilities_before_gt.npz") as data:
            centers = data["centers"].copy()
            arrays = {"A_original": {"raw": data["lambda_0"].copy(),
                                     "corrected": data["lambda_1"].copy()}}
        for label, path in (
            ("SHIFT_original", ROOT / f"investigations/scene_shift_v1/formal_{seed}/predictions_before_gt.npz"),
            ("A_recal", HERE / f"A_{seed}/predictions_before_gt.npz"),
            ("SHIFT_recal", HERE / f"SHIFT_{seed}/predictions_before_gt.npz"),
        ):
            with np.load(path) as data:
                assert np.array_equal(centers, data["centers"])
                arrays[label] = {k: data[k].copy() for k in ("raw", "corrected")}
        cfg = json.loads((base / f"student_{seed}/config.json").read_text())
        gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        result["seeds"][str(seed)] = {
            method: {variant: metric(y, q) for variant, q in probs.items()}
            for method, probs in arrays.items()
        }
    (HERE / "AUDIT.json").write_text(json.dumps(result, indent=2))
    for variant in ("raw", "corrected"):
        for method in ("A_original", "A_recal", "SHIFT_original", "SHIFT_recal"):
            vals = [result["seeds"][str(s)][method][variant] for s in SEEDS]
            print(variant, method,
                  "OA", [round(100*v["oa_official"], 2) for v in vals],
                  "OA_mean", round(100*np.mean([v["oa_official"] for v in vals]), 2),
                  "AA_mean", round(100*np.mean([v["aa"] for v in vals]), 2),
                  "Kappa_mean", round(np.mean([v["kappa"] for v in vals]), 4),
                  "class7_mean", round(100*np.mean([v["recall"][6] for v in vals]), 2))


if __name__ == "__main__":
    main()
