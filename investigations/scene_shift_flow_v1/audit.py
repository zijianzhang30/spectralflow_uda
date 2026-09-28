"""Post-hoc official paired evaluation after all Flow predictions are frozen."""
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
NAMES = ("base_raw", "base_corrected", "flow_raw", "flow_corrected")


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
        assert meta["flow_checkpoint_sha256"] == digest(run / "last.pth")
    out = {"scope": "post-hoc paired target-GT evaluation, no target-based selection",
           "seeds": {}}
    for seed in SEEDS:
        run = HERE / f"formal_{seed}"
        shift = ROOT / "investigations/scene_shift_v1" / f"formal_{seed}"
        with np.load(run / "predictions_before_gt.npz") as data:
            centers = data["centers"].copy()
            q = {name: data[name].copy() for name in NAMES}
        with np.load(shift / "predictions_before_gt.npz") as data:
            assert np.array_equal(centers, data["centers"])
            assert np.array_equal(q["base_raw"], data["raw"])
            assert np.array_equal(q["base_corrected"], data["corrected"])
        cfg = json.loads((shift / "config.json").read_text())
        gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        out["seeds"][str(seed)] = {
            name: {"metrics": metric(y, probs),
                   "soft_mass": probs.mean(axis=0).tolist(),
                   "candidate_coverage_at_0.8": float((probs.max(1) >= .8).mean()),
                   "class7_correct_candidate_recall_at_0.8":
                       float(((probs.argmax(1) == 6) & (probs.max(1) >= .8) &
                              (y == 6)).sum() / (y == 6).sum())}
            for name, probs in q.items()}
    (HERE / "AUDIT.json").write_text(json.dumps(out, indent=2))
    for name in NAMES:
        rows = [out["seeds"][str(s)][name]["metrics"] for s in SEEDS]
        print(name, "OA", round(np.mean([x["oa_official"] for x in rows]) * 100, 3),
              "AA", round(np.mean([x["aa"] for x in rows]) * 100, 3),
              "Kappa", round(np.mean([x["kappa"] for x in rows]), 4),
              "class6", round(np.mean([x["recall"][5] for x in rows]) * 100, 3),
              "class7", round(np.mean([x["recall"][6] for x in rows]) * 100, 3),
              "seed_OA", [round(x["oa_official"] * 100, 3) for x in rows])


if __name__ == "__main__":
    main()
