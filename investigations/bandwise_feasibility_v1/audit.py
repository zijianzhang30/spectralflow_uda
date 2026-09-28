"""Post-freeze Houston target scoring of the complete feasibility curve."""
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

EPSILONS = (0.0, 0.01, 0.02, 0.05, 0.10)


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
    for seed in (202601, 202602, 202603):
        out = HERE / str(seed)
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["target_gt_opened"] is False
        assert meta["source_val_used_for_selection"] is False
        assert meta["lock_sha256"] == lock_hash
        assert meta["original_params_unchanged"] and meta["original_buffers_unchanged"]
        assert meta["prediction_sha256"] == digest(out / "predictions_before_gt.npz")
        assert meta["source_train_search_sha256"] == digest(out / "source_train_search.json")
        assert set(meta["chosen"]) == {str(e) for e in EPSILONS}
    output = {"scope": "Post-freeze Houston development scoring; all epsilon points reported",
              "seeds": {}}
    for seed in (202601, 202602, 202603):
        run = ROOT / f"investigations/distribution_correction_validation/runs/student_{seed}"
        cfg = json.loads((run / "config.json").read_text())
        with np.load(HERE / str(seed) / "predictions_before_gt.npz") as data:
            centers = data["centers"].copy()
            q = {name: data[name].copy() for name in data.files if name != "centers"}
        assert centers.shape == (53200, 2)
        assert set(q) == {"identity"} | {f"epsilon_{e}" for e in EPSILONS}
        assert all(v.shape == (53184, 7) for v in q.values())
        gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
        gt = hdf5storage.loadmat(str(gt_path))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        assert len(y) == 53184 and y.min() == 0 and y.max() == 6
        output["seeds"][str(seed)] = {name: metric(y, p) for name, p in q.items()}
    out_path.write_text(json.dumps(output, indent=2))
    for name in ("identity",) + tuple(f"epsilon_{e}" for e in EPSILONS):
        rows = [output["seeds"][str(seed)][name] for seed in (202601, 202602, 202603)]
        oa = np.array([100 * row["oa_official"] for row in rows])
        aa = np.array([100 * row["aa"] for row in rows])
        recall = np.array([row["recall"] for row in rows]) * 100
        print(name, "OA", np.round(oa, 2).tolist(),
              "mean±sd", round(oa.mean(), 2), round(oa.std(ddof=1), 2),
              "AA", round(aa.mean(), 2), round(aa.std(ddof=1), 2),
              "class recall", np.round(recall.mean(0), 1).tolist())


if __name__ == "__main__":
    main()
