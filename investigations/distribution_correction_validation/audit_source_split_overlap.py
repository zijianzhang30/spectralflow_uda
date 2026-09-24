"""Measure source train/validation patch overlap without target GT."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent


def main():
    result = {"seeds": [202601, 202602, 202603],
              "distance": "minimum Chebyshev pixel distance to a source-train center",
              "patch_overlap_rule": "two 7x7 patches share at least one pixel iff center distance <= 6 in both axes",
              "results": {}}
    for seed in result["seeds"]:
        with np.load(HERE / "runs" / f"student_{seed}" / "source_split.npz") as data:
            tr, tl = data["train_centers"].copy(), data["train_labels"].copy()
            va, vl = data["val_centers"].copy(), data["val_labels"].copy()
        any_distance = cKDTree(tr).query(va, p=np.inf)[0]
        same_distance = np.empty(len(va), dtype=float)
        for c in range(7):
            same_distance[vl == c] = cKDTree(tr[tl == c]).query(va[vl == c], p=np.inf)[0]
        result["results"][str(seed)] = {
            "source_val_n": len(va),
            "within_3_any_class_fraction": float(np.mean(any_distance <= 3)),
            "patch_overlap_any_class_fraction": float(np.mean(any_distance <= 6)),
            "within_3_same_class_fraction": float(np.mean(same_distance <= 3)),
            "patch_overlap_same_class_fraction": float(np.mean(same_distance <= 6)),
            "no_same_class_patch_overlap_n": int(np.sum(same_distance > 6)),
        }
    path = HERE / "source_split_overlap_audit.json"
    assert not path.exists(), f"Refusing to overwrite {path}"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
