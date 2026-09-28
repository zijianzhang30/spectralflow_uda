"""Post-hoc metric comparison after all graph predictions are frozen."""
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
NAMES = ("A_raw", "A_prior", "MLUDA_raw", "MLUDA_prior")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    for seed in SEEDS:
        run = HERE / f"seed_{seed}"
        meta = json.loads((run / "metadata.json").read_text())
        assert meta["target_gt_opened"] is False
        assert meta["prediction_sha256"] == digest(run / "predictions_before_gt.npz")
    out = {"scope": "Post-hoc exploratory; no target-based rule selection", "seeds": {}}
    for seed in SEEDS:
        base = ROOT / "investigations/distribution_correction_validation/runs" / f"soft_{seed}"
        mluda = ROOT / "investigations/mluda_prior_probe" / f"seed_{seed}"
        graph = HERE / f"seed_{seed}"
        with np.load(base / "probabilities_before_gt.npz") as data:
            centers = data["centers"].copy()
            original = {"A_raw": data["lambda_0"].copy(),
                        "A_prior": data["lambda_1"].copy()}
        with np.load(mluda / "predictions_before_gt.npz") as data:
            assert np.array_equal(centers, data["centers"])
            original.update(MLUDA_raw=data["raw"].copy(),
                            MLUDA_prior=data["corrected"].copy())
        with np.load(graph / "predictions_before_gt.npz") as data:
            assert np.array_equal(centers[:53184], data["centers"])
            propagated = {name: data[name].copy() for name in NAMES}
        cfg = json.loads((ROOT / "investigations/distribution_correction_validation/runs" /
                          f"student_{seed}/config.json").read_text())
        gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        row = {}
        for name in NAMES:
            row[name] = {}
            for version, q in (("original", original[name]), ("graph", propagated[name])):
                row[name][version] = {"metrics": metric(y, q),
                                      "soft_mass": q.mean(axis=0).tolist(),
                                      "candidate_coverage_at_0.8": float((q.max(1) >= .8).mean()),
                                      "class7_correct_candidate_recall_at_0.8":
                                      float(((q.argmax(1) == 6) & (q.max(1) >= .8) &
                                             (y == 6)).sum() / (y == 6).sum())}
        out["seeds"][str(seed)] = row
    (HERE / "AUDIT.json").write_text(json.dumps(out, indent=2))
    for name in NAMES:
        for version in ("original", "graph"):
            rows = [out["seeds"][str(s)][name][version]["metrics"] for s in SEEDS]
            print(name, version,
                  "OA", round(np.mean([m["oa_official"] for m in rows]) * 100, 3),
                  "AA", round(np.mean([m["aa"] for m in rows]) * 100, 3),
                  "Kappa", round(np.mean([m["kappa"] for m in rows]), 4),
                  "class7", round(np.mean([m["recall"][6] for m in rows]) * 100, 3),
                  "per_seed_OA", [round(m["oa_official"] * 100, 3) for m in rows])


if __name__ == "__main__":
    main()
