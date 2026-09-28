"""Score the five prelocked conditions after all predictions are frozen."""
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
    out_path = HERE / "AUDIT.json"
    if out_path.exists():
        raise FileExistsError(out_path)
    lock_hash = digest(HERE / "EXPERIMENT_LOCK.md")
    names = ("original", "global", "orthogonal", "parallel", "target_bn")
    # Verify that every new output was frozen before opening target GT.
    for seed in (202601, 202602, 202603):
        out = HERE / str(seed)
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["target_gt_opened"] is False and meta["lock_sha256"] == lock_hash
        assert meta["parameter_hash_unchanged"] and meta["buffer_hash_unchanged"]
        assert meta["original_reference_argmax_disagreement"] == 0
        assert meta["prediction_sha256"] == digest(out / "predictions_before_gt.npz")
        run = ROOT / f"investigations/distribution_correction_validation/runs/student_{seed}"
        assert meta["source_checkpoint_sha256"] == digest(run / "best_source_val.pth")
        assert meta["source_split_sha256"] == digest(run / "source_split.npz")
        target_bn_path = ROOT / f"investigations/target_bn_recal_v1/A_{seed}/predictions_before_gt.npz"
        assert meta["existing_target_bn_sha256"] == digest(target_bn_path)
    output = {"scope": "Frozen-model Houston development diagnostic; no target-based selection",
              "seeds": {}}
    for seed in (202601, 202602, 202603):
        run = ROOT / f"investigations/distribution_correction_validation/runs/student_{seed}"
        cfg = json.loads((run / "config.json").read_text())
        with np.load(HERE / str(seed) / "predictions_before_gt.npz") as contents:
            centers = contents["centers"].copy()
            q = {name: contents[name].copy() for name in names[:-1]}
        target_bn_path = ROOT / f"investigations/target_bn_recal_v1/A_{seed}/predictions_before_gt.npz"
        with np.load(target_bn_path) as contents:
            assert np.array_equal(centers, contents["centers"])
            q["target_bn"] = contents["raw"].copy()
        assert centers.shape == (53200, 2)
        assert all(array.shape == (53184, 7) for array in q.values())
        gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
        gt = hdf5storage.loadmat(str(gt_path))["map"]
        labels = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        assert len(labels) == 53184 and labels.min() >= 0 and labels.max() < 7
        output["seeds"][str(seed)] = {name: metric(labels, q[name]) for name in names}
    out_path.write_text(json.dumps(output, indent=2))
    for name in names:
        rows = [output["seeds"][str(seed)][name] for seed in (202601, 202602, 202603)]
        oa = np.array([100 * row["oa_official"] for row in rows])
        aa = np.array([100 * row["aa"] for row in rows])
        kappa = np.array([row["kappa"] for row in rows])
        recall = np.array([row["recall"] for row in rows]) * 100
        print(name, "OA", np.round(oa, 2).tolist(),
              "mean±sd", round(oa.mean(), 2), round(oa.std(ddof=1), 2),
              "AA", round(aa.mean(), 2), round(aa.std(ddof=1), 2),
              "Kappa", round(kappa.mean(), 4),
              "recall", np.round(recall.mean(0), 1).tolist())


if __name__ == "__main__":
    main()
