"""Post-hoc target-GT audit; no fitting or selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import hdf5storage
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "distribution_correction_validation"
SEEDS = (202601, 202602, 202603)
METHODS = ("student", "teacher", "jcpot_0p1", "jcpot_0p2",
           "correction_hypersigma", "correction_jcpot_0p1", "correction_jcpot_0p2")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classification(q, y):
    p = q.argmax(axis=1)
    cm = confusion_matrix(y, p, labels=np.arange(7))
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    conf = q.max(axis=1)
    kept = conf >= 0.8
    true7 = y == 6
    pred7 = p == 6
    candidate7 = kept & pred7
    return {"accuracy_pilot": float(np.trace(cm) / len(y)),
            "aa": float(recall.mean()),
            "kappa": float(cohen_kappa_score(y, p, labels=np.arange(7))),
            "recall_by_class": recall.tolist(),
            "class7_correct_candidate_recall": float(np.mean(candidate7[true7])),
            "class7_candidate_precision": float(np.mean(true7[candidate7])) if candidate7.any() else None,
            "candidate_coverage": float(kept.mean()),
            "soft_expected_correct_fraction": float(q[np.arange(len(y)), y].mean()),
            "soft_class7_precision": float(q[true7, 6].sum() / q[:, 6].sum()),
            "soft_mass": q.mean(axis=0).tolist(),
            "confusion_true_by_pred": cm.tolist()}


def main():
    output = {"seeds": list(SEEDS), "target_n": 8192,
              "scope": "first 8192 official-order target positions; pilot, not official full-scene OA",
              "methods": list(METHODS), "results": {}}
    for seed in SEEDS:
        run = HERE / "runs" / str(seed)
        fit = json.loads((run / "fit.json").read_text())
        correction_fit = json.loads((run / "correction_fit.json").read_text())
        assert not fit["target_gt_opened"] and not correction_fit["target_gt_opened"]
        assert fit["before_gt_sha256"] == digest(run / "before_gt.npz")
        assert correction_fit["correction_sha256"] == digest(run / "corrections_before_gt.npz")
        with np.load(run / "before_gt.npz") as data:
            centers = data["centers"].copy()
            student = data["student"].copy()
            teacher = data["teacher"].copy()
            jcpot = {tag: data[f"posterior_{tag}"].copy() for tag in ("0p1", "0p2")}
            priors = {"student": student.mean(axis=0),
                      "hypersigma": teacher.mean(axis=0),
                      "jcpot_0p1": data["prior_0p1"].copy(),
                      "jcpot_0p2": data["prior_0p2"].copy()}
            for tag in ("0p1", "0p2"):
                assert np.allclose(jcpot[tag].mean(axis=0), priors[f"jcpot_{tag}"], atol=1e-5)
        with np.load(run / "corrections_before_gt.npz") as data:
            assert np.array_equal(student, data["student"])
            corrected = {name: data[name].copy() for name in
                         ("hypersigma", "jcpot_0p1", "jcpot_0p2")}
        student_run = PARENT / "runs" / f"student_{seed}"
        cfg = json.loads((student_run / "config.json").read_text())
        gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
        gt = hdf5storage.loadmat(str(gt_path))["map"]
        y = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
        assert len(y) == 8192 and np.all((0 <= y) & (y < 7))
        truth = np.bincount(y, minlength=7) / len(y)
        q = {"student": student, "teacher": teacher,
             "jcpot_0p1": jcpot["0p1"], "jcpot_0p2": jcpot["0p2"],
             **{f"correction_{key}": value for key, value in corrected.items()}}
        output["results"][str(seed)] = {
            "true_prior": truth.tolist(),
            "priors": {name: {"values": np.asarray(prior).tolist(),
                              "l1_error": float(np.abs(prior - truth).sum()),
                              "class7_abs_error": float(abs(prior[6] - truth[6]))}
                       for name, prior in priors.items()},
            "fit": fit["fit"],
            "classification": {name: classification(q[name], y) for name in METHODS},
            "target_gt_sha256": digest(gt_path)}
    path = HERE / "PILOT_AUDIT.json"
    assert not path.exists(), f"Refusing to overwrite {path}"
    path.write_text(json.dumps(output, indent=2))
    for seed in SEEDS:
        row = output["results"][str(seed)]
        print("seed", seed, "true prior", np.round(row["true_prior"], 4).tolist())
        for name, prior in row["priors"].items():
            print("prior", name, "class7", round(prior["values"][6], 4),
                  "L1", round(prior["l1_error"], 4))
        for name in METHODS:
            m = row["classification"][name]
            print("classification", name, "acc", round(m["accuracy_pilot"], 4),
                  "aa", round(m["aa"], 4), "class7", round(m["recall_by_class"][6], 4))


if __name__ == "__main__":
    main()
