"""Post-hoc official scoring of the locked support-matched inference rule."""
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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    seeds = (202601, 202602, 202603)
    for seed in seeds:
        run = HERE / f"gated_{seed}"
        meta = json.loads((run / "metadata.json").read_text())
        assert meta["target_gt_opened"] is False
        assert meta["prediction_sha256"] == digest(run / "predictions_before_gt.npz")
    out = {"scope": "post-hoc exploratory, rule fixed after ungated Flow scores",
           "seeds": {}}
    for seed in seeds:
        run = HERE / f"gated_{seed}"
        shift = ROOT / "investigations/scene_shift_v1" / f"formal_{seed}"
        cfg = json.loads((shift / "config.json").read_text())
        with np.load(run / "predictions_before_gt.npz") as data:
            centers = data["centers"].copy()
            q = {v: data[v].copy() for v in ("gated_raw", "gated_corrected")}
            moved = data["moved_mask"].copy()
        with np.load(shift / "predictions_before_gt.npz") as data:
            assert np.array_equal(centers, data["centers"])
            q.update(base_raw=data["raw"].copy(),
                     base_corrected=data["corrected"].copy())
        gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        out["seeds"][str(seed)] = {name: {"metrics": metric(y, probs),
                                         "soft_mass": probs.mean(axis=0).tolist()}
                                   for name, probs in q.items()}
        out["seeds"][str(seed)]["moved_fraction"] = float(moved.mean())
    (HERE / "GATED_AUDIT.json").write_text(json.dumps(out, indent=2))
    for name in ("base_raw", "gated_raw", "base_corrected", "gated_corrected"):
        rows = [out["seeds"][str(seed)][name]["metrics"] for seed in seeds]
        print(name, "OA", round(np.mean([x["oa_official"] for x in rows]) * 100, 3),
              "AA", round(np.mean([x["aa"] for x in rows]) * 100, 3),
              "Kappa", round(np.mean([x["kappa"] for x in rows]), 4),
              "class6", round(np.mean([x["recall"][5] for x in rows]) * 100, 3),
              "class7", round(np.mean([x["recall"][6] for x in rows]) * 100, 3),
              "OA_by_seed", [round(x["oa_official"] * 100, 3) for x in rows])


if __name__ == "__main__":
    main()
