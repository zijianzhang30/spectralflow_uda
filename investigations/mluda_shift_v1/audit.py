"""Post-hoc official scoring after all three prediction files are frozen."""
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
    for seed in SEEDS:
        run = HERE / f"formal_{seed}"
        meta = json.loads((run / "metadata.json").read_text())
        assert meta["target_gt_opened"] is False
        assert meta["prediction_sha256"] == digest(run / "predictions_before_gt.npz")
        assert meta["lock_sha256"] == digest(HERE / "EXPERIMENT_LOCK.md")
    out = {"scope": "Post-hoc benchmark; no training or target-based selection",
           "seeds": {}}
    for seed in SEEDS:
        run = HERE / f"formal_{seed}"
        cfg = json.loads((run / "config.json").read_text())
        prior = ROOT / "investigations/distribution_correction_validation/runs" / f"correction_{seed}"
        with np.load(run / "predictions_before_gt.npz") as data:
            centers, raw, corrected, p = (data[k].copy() for k in
                                          ("centers", "raw", "corrected", "prior"))
        with np.load(prior / "predictions_before_gt.npz") as data:
            assert np.array_equal(centers, data["centers"])
            assert np.array_equal(p, data["prior"])
        gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        metrics = {name: metric(y, q) for name, q in
                   (("raw", raw), ("corrected", corrected))}
        out["seeds"][str(seed)] = {
            name: {"metrics": metrics[name],
                   "soft_mass": q.mean(axis=0).tolist(),
                   "candidate_coverage_at_0.8": float((q.max(1) >= .8).mean()),
                   "class7_correct_candidate_recall_at_0.8":
                   float(((q.argmax(1) == 6) & (q.max(1) >= .8) & (y == 6)).sum() /
                         (y == 6).sum())}
            for name, q in (("raw", raw), ("corrected", corrected))}
    (HERE / "AUDIT.json").write_text(json.dumps(out, indent=2))
    for name in ("raw", "corrected"):
        for key in ("oa_official", "aa", "kappa"):
            arr = [out["seeds"][str(s)][name]["metrics"][key] for s in SEEDS]
            print(name, key, "mean", float(np.mean(arr)), "std", float(np.std(arr, ddof=1)),
                  "per_seed", arr)
        print(name, "class7", [out["seeds"][str(s)][name]["metrics"]["recall"][6]
                                for s in SEEDS])


if __name__ == "__main__":
    main()
