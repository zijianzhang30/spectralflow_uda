"""Exploratory no-threshold fallback after inspecting the frozen audit.

If a source class has no hard target candidate in a batch, allocate that
class's target marginal proportional to its full soft probability vector.
This is an exploratory post-hoc comparison, not a prelocked method claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import hdf5storage
import numpy as np

from audit_correspondence_structure import (BATCH, GATE, GRID, HERE, SEEDS,
                                            source_batch_classes)


def compare(q: np.ndarray, y: np.ndarray, source_labels: np.ndarray) -> dict:
    acc = {rule: {"correct": np.zeros(7), "count": np.zeros(7, dtype=np.int64),
                  "effective_n": np.zeros(7), "fallback_count": np.zeros(7, dtype=np.int64)}
           for rule in ("hard", "soft", "empty_class_fallback")}
    for b, classes in enumerate(source_batch_classes(source_labels)):
        qb = q[b * BATCH:(b + 1) * BATCH].astype(np.float64)
        yb = y[b * BATCH:(b + 1) * BATCH]
        pred = qb.argmax(axis=1)
        confident = qb.max(axis=1) >= GATE
        for c in classes:
            c = int(c)
            hard = qb[:, c] * (confident & (pred == c))
            soft = qb[:, c]
            for rule, weights in (("hard", hard), ("soft", soft),
                                  ("empty_class_fallback", hard if hard.sum() else soft)):
                total = weights.sum()
                if not total:
                    continue
                rec = acc[rule]
                rec["count"][c] += 1
                rec["correct"][c] += weights[yb == c].sum() / total
                rec["effective_n"][c] += total**2 / np.square(weights).sum()
                if rule == "empty_class_fallback" and not hard.sum():
                    rec["fallback_count"][c] += 1
    return {rule: {
        "enabled_batches_by_class": rec["count"].tolist(),
        "fallback_batches_by_class": rec["fallback_count"].tolist(),
        "conditional_purity_by_class": [float(rec["correct"][c] / rec["count"][c])
                                        if rec["count"][c] else None for c in range(7)],
        "conditional_purity_overall": float(rec["correct"].sum() / rec["count"].sum()),
        "mean_effective_target_n_by_class": [float(rec["effective_n"][c] / rec["count"][c])
                                             if rec["count"][c] else None for c in range(7)],
    } for rule, rec in acc.items()}


def main() -> None:
    out = {"scope": "post-hoc exploratory fallback motivated by prior target-GT audit; not final method",
           "seeds": list(SEEDS), "grid": list(GRID), "first_200_batches": {}}
    previous = json.loads((HERE / "correspondence_structure_audit.json").read_text())
    for seed in SEEDS:
        student = HERE / "runs" / f"student_{seed}"
        with np.load(HERE / "runs" / f"soft_{seed}" / "probabilities_before_gt.npz") as data:
            centers = data["centers"].copy()
            probabilities = {name: data[name].copy() for name in GRID}
        with np.load(student / "source_split.npz") as split:
            assert np.array_equal(centers, split["target_centers"])
            source_labels = split["train_labels"].copy()
        cfg = json.loads((student / "config.json").read_text())
        gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        out["first_200_batches"][str(seed)] = {}
        for name, q in probabilities.items():
            result = compare(q, y, source_labels)
            older = previous["results"][str(seed)][name]["first_200_batches"]
            for rule in ("hard", "soft"):
                assert result[rule]["enabled_batches_by_class"] == older[rule]["enabled_batches_by_class"]
                assert np.isclose(result[rule]["conditional_purity_overall"],
                                  older[rule]["conditional_purity_overall"], atol=1e-12)
            assert np.all(
                np.asarray(result["empty_class_fallback"]["enabled_batches_by_class"])
                >= np.asarray(result["hard"]["enabled_batches_by_class"])
            )
            out["first_200_batches"][str(seed)][name] = result
    path = HERE / "empty_class_fallback_audit.json"
    assert not path.exists(), f"Refusing to overwrite {path}"
    path.write_text(json.dumps(out, indent=2))
    for seed in SEEDS:
        for name in ("lambda_0", "lambda_05", "lambda_1", "hard"):
            r = out["first_200_batches"][str(seed)][name]
            print(seed, name, *[(rule, r[rule]["enabled_batches_by_class"][6],
                                    round(r[rule]["conditional_purity_by_class"][6] or 0, 4),
                                    round(r[rule]["conditional_purity_overall"], 4))
                                   for rule in ("hard", "soft", "empty_class_fallback")])


if __name__ == "__main__":
    main()
