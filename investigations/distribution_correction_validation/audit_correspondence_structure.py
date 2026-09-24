"""Read-only post-hoc audit of class decisions and soft OT mass allocation."""
from __future__ import annotations

import json
from pathlib import Path

import hdf5storage
import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (202601, 202602, 202603)
GRID = ("lambda_0", "lambda_025", "lambda_05", "lambda_1", "lambda_2",
        "lambda_4", "lambda_8", "lambda_16", "hard")
GATE = 0.8
PAIR_BATCHES = 200
BATCH = 32


def full_scene(q: np.ndarray, y: np.ndarray) -> dict:
    pred = q.argmax(axis=1)
    conf = q.max(axis=1)
    support = np.bincount(y, minlength=7)
    rows = []
    for c in range(7):
        truth = y == c
        right = truth & (pred == c)
        wrong = truth & (pred != c)
        low = right & (conf < GATE)
        kept = right & (conf >= GATE)
        n = int(support[c])
        assert int(wrong.sum() + low.sum() + kept.sum()) == n
        mass = q[:, c].astype(np.float64)
        rows.append({
            "support": n,
            "wrong_argmax_fraction": float(wrong.sum() / n),
            "correct_below_gate_fraction": float(low.sum() / n),
            "correct_kept_fraction": float(kept.sum() / n),
            "correct_given_ungated_fraction": float(kept.sum() / right.sum()) if right.any() else None,
            "mean_true_class_probability": float(mass[truth].mean()),
            "soft_class_mass_fraction": float(mass.mean()),
            "soft_assignment_precision": float(mass[truth].sum() / mass.sum()),
            "hard_kept_count": int(kept.sum()),
        })
    return {"classwise": rows,
            "soft_expected_correct_fraction": float(q[np.arange(len(y)), y].mean()),
            "ungated_correct_fraction": float(np.mean(pred == y)),
            "hard_candidate_fraction": float(np.mean(conf >= GATE))}


def source_batch_classes(labels: np.ndarray):
    batches = [labels[start:start + BATCH] for start in range(0, len(labels) - BATCH + 1, BATCH)]
    assert len(batches) > 0 and all(len(x) == BATCH for x in batches)
    for index in range(PAIR_BATCHES):
        yield np.unique(batches[index % len(batches)])


def paired_scene(q: np.ndarray, y: np.ndarray, source_labels: np.ndarray) -> dict:
    records = {kind: {"correct": np.zeros(7), "enabled": np.zeros(7, dtype=np.int64),
                      "exposure": np.zeros(7), "effective_n": np.zeros(7)}
               for kind in ("hard", "soft")}
    for b, source_classes in enumerate(source_batch_classes(source_labels)):
        qb = q[b * BATCH:(b + 1) * BATCH].astype(np.float64)
        yb = y[b * BATCH:(b + 1) * BATCH]
        pred = qb.argmax(axis=1)
        keep = qb.max(axis=1) >= GATE
        for c in source_classes:
            c = int(c)
            for kind in ("hard", "soft"):
                weights = qb[:, c].copy()
                if kind == "hard":
                    weights *= keep & (pred == c)
                total = weights.sum()
                if total <= 0:
                    continue
                rec = records[kind]
                rec["enabled"][c] += 1
                rec["correct"][c] += weights[yb == c].sum() / total
                rec["exposure"][c] += total
                rec["effective_n"][c] += total**2 / np.square(weights).sum()
    output = {}
    for kind, rec in records.items():
        enabled = rec["enabled"]
        valid = enabled > 0
        output[kind] = {
            "enabled_batches_by_class": enabled.tolist(),
            "conditional_purity_by_class": [float(rec["correct"][c] / enabled[c]) if valid[c] else None
                                           for c in range(7)],
            "conditional_purity_overall": float(rec["correct"].sum() / enabled.sum()),
            "mean_effective_target_n_by_class": [float(rec["effective_n"][c] / enabled[c]) if valid[c] else None
                                                 for c in range(7)],
            "mean_raw_target_mass_by_class": [float(rec["exposure"][c] / enabled[c]) if valid[c] else None
                                              for c in range(7)],
        }
    return output


def main() -> None:
    output = {"seeds": list(SEEDS), "grid": list(GRID),
              "scope": "post-hoc read-only diagnostic; no lambda selection or Flow training",
              "results": {}}
    for seed in SEEDS:
        run = HERE / "runs" / f"soft_{seed}"
        student = HERE / "runs" / f"student_{seed}"
        existing = json.loads((run / "audit.json").read_text())
        with np.load(run / "probabilities_before_gt.npz") as data:
            centers = data["centers"].copy()
            probabilities = {name: data[name].copy() for name in GRID}
        with np.load(student / "source_split.npz") as split:
            assert np.array_equal(centers, split["target_centers"])
            source_labels = split["train_labels"].copy()
        cfg = json.loads((student / "config.json").read_text())
        gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        assert len(y) == 53184 and np.all((0 <= y) & (y < 7))
        variants = {}
        for name, q in probabilities.items():
            assert q.shape == (len(y), 7)
            full = full_scene(q, y)
            paired = paired_scene(q, y, source_labels)
            original = existing["variants"][name]["correspondence"]
            assert np.isclose(full["classwise"][6]["correct_kept_fraction"],
                              original["true_class_candidate_recall"][6], atol=1e-12)
            assert paired["hard"]["enabled_batches_by_class"] == original["pair_enabled_batches_by_class"]
            assert np.isclose(paired["hard"]["conditional_purity_overall"],
                              original["pair_expected_purity_overall"], atol=2e-6), (seed, name)
            variants[name] = {"full_scene": full, "first_200_batches": paired}
        output["results"][str(seed)] = variants
    path = HERE / "correspondence_structure_audit.json"
    assert not path.exists(), f"Refusing to overwrite {path}"
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps({str(seed): {
        name: {"class7_wrong": output["results"][str(seed)][name]["full_scene"]["classwise"][6]["wrong_argmax_fraction"],
               "class7_below_gate": output["results"][str(seed)][name]["full_scene"]["classwise"][6]["correct_below_gate_fraction"],
               "class7_kept": output["results"][str(seed)][name]["full_scene"]["classwise"][6]["correct_kept_fraction"],
               "hard_purity": output["results"][str(seed)][name]["first_200_batches"]["hard"]["conditional_purity_overall"],
               "soft_purity": output["results"][str(seed)][name]["first_200_batches"]["soft"]["conditional_purity_overall"]}
        for name in ("lambda_0", "lambda_05", "lambda_1", "hard")}
        for seed in SEEDS}, indent=2))


if __name__ == "__main__":
    main()
