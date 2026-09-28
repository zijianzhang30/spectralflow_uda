"""Post-hoc paired prediction transitions for the frozen feature-refinement runs."""
from __future__ import annotations

import json
from pathlib import Path

import hdf5storage
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (202601, 202602, 202603)


def load(seed: int):
    base = ROOT / "investigations/distribution_correction_validation/runs"
    with np.load(base / f"soft_{seed}/probabilities_before_gt.npz") as data:
        centers = data["centers"].copy()
        probs = {"A": {"raw": data["lambda_0"].copy(),
                        "corrected": data["lambda_1"].copy()}}
    config = json.loads((base / f"student_{seed}/config.json").read_text())
    gt = hdf5storage.loadmat(str(Path(config["data"]) / "Houston18_7gt.mat"))["map"]
    y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
    for method in ("F1", "F2"):
        with np.load(HERE / f"runs/formal_{method}_{seed}/predictions_before_gt.npz") as data:
            assert np.array_equal(centers, data["centers"])
            probs[method] = {v: data[v].copy() for v in ("raw", "corrected")}
    return y, probs


def audit(y, qa, qb):
    pa, pb = qa.argmax(1), qb.argmax(1)
    result = {
        "overall_A_correct_to_F_wrong": int(((pa == y) & (pb != y)).sum()),
        "overall_A_wrong_to_F_correct": int(((pa != y) & (pb == y)).sum()),
        "classes": {},
    }
    for c in (5, 6):
        mask = y == c
        lost = mask & (pa == c) & (pb != c)
        gained = mask & (pa != c) & (pb == c)
        kept = mask & (pa == c) & (pb == c)
        a_top = qa.max(1)
        a_margin = np.sort(qa, axis=1)[:, -1] - np.sort(qa, axis=1)[:, -2]
        result["classes"][str(c + 1)] = {
            "count": int(mask.sum()),
            "A_correct_to_F_wrong": int(lost.sum()),
            "A_wrong_to_F_correct": int(gained.sum()),
            "lost_destination_counts": np.bincount(pb[lost], minlength=7).tolist(),
            "gained_A_prediction_counts": np.bincount(pa[gained], minlength=7).tolist(),
            "A_confidence_lost_mean": float(a_top[lost].mean()) if lost.any() else None,
            "A_confidence_kept_mean": float(a_top[kept].mean()) if kept.any() else None,
            "A_confidence_lost_ge_0.8": float((a_top[lost] >= .8).mean()) if lost.any() else None,
            "A_confidence_lost_ge_0.95": float((a_top[lost] >= .95).mean()) if lost.any() else None,
            "A_margin_lost_median": float(np.median(a_margin[lost])) if lost.any() else None,
        }
    return result


def main():
    out = {"scope": "Post-hoc GT diagnosis; not a model selection criterion", "seeds": {}}
    for seed in SEEDS:
        y, probs = load(seed)
        out["seeds"][str(seed)] = {
            variant: {method: audit(y, probs["A"][variant], probs[method][variant])
                      for method in ("F1", "F2")}
            for variant in ("raw", "corrected")
        }
    path = HERE / "TRANSITION_AUDIT.json"
    path.write_text(json.dumps(out, indent=2))
    for seed in SEEDS:
        for method in ("F1", "F2"):
            item = out["seeds"][str(seed)]["corrected"][method]
            print(seed, method, "net overall", item["overall_A_wrong_to_F_correct"] - item["overall_A_correct_to_F_wrong"])
            for c in ("6", "7"):
                row = item["classes"][c]
                print(" class", c, "lost", row["A_correct_to_F_wrong"], "gained", row["A_wrong_to_F_correct"],
                      "lost destinations", row["lost_destination_counts"],
                      "lost A conf >=0.8", round(row["A_confidence_lost_ge_0.8"], 3))


if __name__ == "__main__":
    main()
