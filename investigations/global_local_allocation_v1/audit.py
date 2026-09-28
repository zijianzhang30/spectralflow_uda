"""Post-freeze GT audit; never used in prediction construction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import hdf5storage
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RUNS = PROJECT / "investigations/distribution_correction_validation/runs"
SEEDS = (202601, 202602, 202603)
VARIANTS = ("raw", "hard", "soft1", "v1", "class_only", "sample_only")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(y, p):
    pred = p.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(7))
    recall = np.diag(cm) / cm.sum(axis=1)
    return {"oa": float(np.trace(cm) / 53200), "aa": float(recall.mean()),
            "kappa": float(cohen_kappa_score(y, pred, labels=np.arange(7))),
            "recall": recall.tolist(), "confusion_matrix": cm.tolist(),
            "predicted_count": cm.sum(axis=0).tolist()}


def main():
    out = HERE / "AUDIT.json"
    assert not out.exists(), "Refusing to replace an audit"
    manifest = json.loads((HERE / "FREEZE_MANIFEST.json").read_text())
    assert manifest["target_gt_opened"] is False
    assert manifest["lock_sha256"] == sha(HERE / "EXPERIMENT_LOCK.md")
    rows = {}
    for seed in SEEDS:
        record = manifest["seeds"][str(seed)]
        pred_path = HERE / f"seed_{seed}_before_gt.npz"
        assert record["prediction_sha256"] == sha(pred_path)
        with np.load(pred_path) as f:
            centers = f["centers"].copy()
            q = {name: f[name].copy() for name in VARIANTS}
        cfg = json.loads((RUNS / f"student_{seed}" / "config.json").read_text())
        gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
        gt = hdf5storage.loadmat(str(gt_path))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        assert np.array_equal(np.unique(y), np.arange(7))
        metrics = {name: metric(y, q[name]) for name in VARIANTS}
        old = json.loads((RUNS / f"correction_{seed}" / "audit.json").read_text())
        soft_old = json.loads((RUNS / f"soft_{seed}" / "audit.json").read_text())
        for new_name, original in (("raw", old["variants"]["A"]["classification"]),
                                   ("hard", old["variants"]["B"]["classification"]),
                                   ("soft1", soft_old["variants"]["lambda_1"]["classification"])):
            assert metrics[new_name]["confusion_matrix"] == original["confusion_matrix"]
        flips = {}
        base_correct = q["raw"].argmax(axis=1) == y
        for name in VARIANTS[1:]:
            changed_correct = q[name].argmax(axis=1) == y
            flips[name] = {"wrong_to_correct": int((~base_correct & changed_correct).sum()),
                           "correct_to_wrong": int((base_correct & ~changed_correct).sum()),
                           "changed_predictions": int((q[name].argmax(axis=1) != q["raw"].argmax(axis=1)).sum())}
        rows[str(seed)] = {"target_gt_sha256": sha(gt_path), "metrics": metrics, "flips_from_raw": flips}
        print(json.dumps({"seed": seed, "metrics": {name: {"oa": metrics[name]["oa"],
              "aa": metrics[name]["aa"], "class7": metrics[name]["recall"][6]}
              for name in VARIANTS}}), flush=True)
    summary = {}
    for name in VARIANTS:
        summary[name] = {field: {"mean": float(np.mean([rows[str(seed)]["metrics"][name][field] for seed in SEEDS])),
                                 "sd": float(np.std([rows[str(seed)]["metrics"][name][field] for seed in SEEDS], ddof=1))}
                         for field in ("oa", "aa", "kappa")}
        summary[name]["class7_recall_mean"] = float(np.mean([rows[str(seed)]["metrics"][name]["recall"][6] for seed in SEEDS]))
    positive = (summary["v1"]["oa"]["mean"] > summary["soft1"]["oa"]["mean"]
                and summary["v1"]["aa"]["mean"] >= summary["soft1"]["aa"]["mean"]
                and all(rows[str(seed)]["metrics"]["v1"]["recall"][6]
                        >= rows[str(seed)]["metrics"]["raw"]["recall"][6] - 0.05 for seed in SEEDS))
    out.write_text(json.dumps({"GT_role": "post-freeze audit only", "manifest_sha256": sha(HERE / "FREEZE_MANIFEST.json"),
                               "seeds": rows, "summary": summary, "predeclared_positive_gate": positive}, indent=2))
    print(json.dumps({"summary": summary, "predeclared_positive_gate": positive}), flush=True)


if __name__ == "__main__":
    main()
