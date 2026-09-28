"""Post-hoc candidate quality; this does not measure OT geometric coupling."""
from __future__ import annotations

import json
from pathlib import Path

import hdf5storage
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def details(q, y):
    pred = q.argmax(1)
    selected = q.max(1) >= .8
    out = {"coverage": float(selected.mean()),
           "candidate_precision": float((pred[selected] == y[selected]).mean()),
           "classes": {}}
    for c in range(7):
        candidate = selected & (pred == c)
        out["classes"][str(c + 1)] = {
            "candidate_n": int(candidate.sum()),
            "candidate_precision": float((y[candidate] == c).mean()) if candidate.any() else None,
            "correct_candidate_recall": float((candidate & (y == c)).sum() / (y == c).sum()),
        }
    return out


def main():
    out = {"warning": "post-hoc target GT; candidate purity is not Sinkhorn geometry", "seeds": {}}
    for seed in (202601, 202602, 202603):
        base = ROOT / "investigations/distribution_correction_validation/runs" / f"soft_{seed}"
        shift = HERE / f"formal_{seed}"
        with np.load(base / "probabilities_before_gt.npz") as data:
            centers = data["centers"].copy()
            q = {"A_raw": data["lambda_0"].copy(),
                 "A_corrected": data["lambda_1"].copy()}
        with np.load(shift / "predictions_before_gt.npz") as data:
            assert np.array_equal(centers, data["centers"])
            q.update(SHIFT_raw=data["raw"].copy(),
                     SHIFT_corrected=data["corrected"].copy())
        cfg = json.loads((base.parent / f"student_{seed}/config.json").read_text())
        gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        out["seeds"][str(seed)] = {name: details(v, y) for name, v in q.items()}
    (HERE / "CORRESPONDENCE_AUDIT.json").write_text(json.dumps(out, indent=2))
    for name in ("A_raw", "SHIFT_raw", "A_corrected", "SHIFT_corrected"):
        rows = [out["seeds"][str(s)][name] for s in (202601, 202602, 202603)]
        print(name, "coverage", round(np.mean([r["coverage"] for r in rows]) * 100, 2),
              "purity", round(np.mean([r["candidate_precision"] for r in rows]) * 100, 2),
              "class7_recall", round(np.mean([r["classes"]["7"]["correct_candidate_recall"]
                                             for r in rows]) * 100, 2),
              "class7_precision", [None if r["classes"]["7"]["candidate_precision"] is None
                                   else round(r["classes"]["7"]["candidate_precision"] * 100, 2)
                                   for r in rows])


if __name__ == "__main__":
    main()
