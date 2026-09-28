"""Score only after all equal-weight target probabilities are frozen."""
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
    frozen = json.loads((HERE / "FREEZE_MANIFEST.json").read_text())
    assert frozen["target_gt_opened"] is False
    assert frozen["lock_sha256"] == digest(HERE / "EXPERIMENT_LOCK.md")
    for seed in SEEDS:
        paths = {
            "MLUDA": ROOT / f"investigations/mluda_prior_probe/seed_{seed}/predictions_before_gt.npz",
            "SHIFT": ROOT / f"investigations/scene_shift_v1/formal_{seed}/predictions_before_gt.npz",
            "equal_mean": HERE / f"seed_{seed}_fused_before_gt.npz",
        }
        rec = frozen["seeds"][str(seed)]
        assert digest(paths["MLUDA"]) == rec["mluda_prediction_sha256"]
        assert digest(paths["SHIFT"]) == rec["scene_shift_prediction_sha256"]
        assert digest(paths["equal_mean"]) == rec["fused_prediction_sha256"]

    output = {"scope": "Post-hoc GT diagnosis of fixed probability mean", "seeds": {}}
    for seed in SEEDS:
        arrays = {}
        for method in ("MLUDA", "SHIFT", "equal_mean"):
            path = (ROOT / f"investigations/mluda_prior_probe/seed_{seed}/predictions_before_gt.npz"
                    if method == "MLUDA" else
                    ROOT / f"investigations/scene_shift_v1/formal_{seed}/predictions_before_gt.npz"
                    if method == "SHIFT" else HERE / f"seed_{seed}_fused_before_gt.npz")
            with np.load(path) as data:
                arrays[method] = {k: data[k].copy() for k in ("centers", "raw", "corrected")}
        centers = arrays["MLUDA"]["centers"]
        assert all(np.array_equal(centers, arrays[m]["centers"]) for m in arrays)
        cfg_path = ROOT / f"investigations/distribution_correction_validation/runs/student_{seed}/config.json"
        cfg = json.loads(cfg_path.read_text())
        gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        row = {}
        for variant in ("raw", "corrected"):
            row[variant] = {m: metric(y, arrays[m][variant]) for m in arrays}
            a = arrays["MLUDA"][variant].argmax(1)
            b = arrays["SHIFT"][variant].argmax(1)
            row[variant]["complementarity"] = {
                "disagreement": float((a != b).mean()),
                "mluda_only_correct": float(((a == y) & (b != y)).mean()),
                "shift_only_correct": float(((b == y) & (a != y)).mean()),
                "both_wrong": float(((a != y) & (b != y)).mean()),
            }
        output["seeds"][str(seed)] = row
    (HERE / "AUDIT.json").write_text(json.dumps(output, indent=2))
    for variant in ("raw", "corrected"):
        for method in ("MLUDA", "SHIFT", "equal_mean"):
            vals = [output["seeds"][str(s)][variant][method] for s in SEEDS]
            print(variant, method, "OA", [round(100*v["oa_official"], 2) for v in vals],
                  "mean", round(100*np.mean([v["oa_official"] for v in vals]), 2),
                  "AA", round(100*np.mean([v["aa"] for v in vals]), 2),
                  "Kappa", round(np.mean([v["kappa"] for v in vals]), 4))


if __name__ == "__main__":
    main()
