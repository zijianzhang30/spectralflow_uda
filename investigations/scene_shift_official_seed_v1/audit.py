"""Post-freeze matched checkpoint and prior comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import hdf5storage
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (1341, 1174, 1370)
SELECTIONS = ("fixed100", "source_val_best")
VARIANTS = ("mluda_raw", "mluda_corrected", "shift_raw", "shift_corrected",
            "fusion_raw", "fusion_corrected")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metric(y, q):
    pred = q.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(7))
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return {"oa": float(np.trace(cm) / 53200), "aa": float(recall.mean()),
            "kappa": float(cohen_kappa_score(y, pred, labels=np.arange(7))),
            "recall": recall.tolist(), "confusion_matrix": cm.tolist(),
            "predicted_count": cm.sum(axis=0).tolist()}


def main():
    out = HERE / "AUDIT.json"
    assert not out.exists()
    frozen = {}
    for seed in SEEDS:
        for selection in SELECTIONS:
            path = HERE / f"seed_{seed}_{selection}_before_gt.npz"
            meta = json.loads((HERE / f"seed_{seed}_{selection}_freeze.json").read_text())
            assert meta["target_gt_opened"] is False and meta["prediction_sha256"] == sha(path)
            with np.load(path) as f:
                centers = f["centers"].copy()
                q = {name: f[name].copy() for name in VARIANTS}
            assert centers.shape == (53200, 2)
            assert all(x.shape == (53184, 7) for x in q.values())
            frozen[(seed, selection)] = (centers, q, meta)
    records = {}
    for seed in SEEDS:
        records[str(seed)] = {}
        cfg_path = ROOT / "official_aligned/runs/round1/formal" / str(seed) / "MLUDA_full/config.json"
        cfg = json.loads(cfg_path.read_text())
        gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
        gt = hdf5storage.loadmat(str(gt_path))["map"]
        for selection in SELECTIONS:
            centers, q, meta = frozen[(seed, selection)]
            y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
            assert np.array_equal(np.unique(y), np.arange(7))
            metrics = {name: metric(y, q[name]) for name in VARIANTS}
            if selection == "fixed100":
                old = json.loads((ROOT / "investigations/mluda_fixed100_v1/AUDIT.json").read_text())
                assert metrics["mluda_raw"]["confusion_matrix"] == old["seeds"][str(seed)]["confusion_matrix"]
            else:
                old = json.loads((ROOT / "official_aligned/runs/round1/formal" / str(seed) /
                                  "MLUDA_full/final_target.json").read_text())
                assert metrics["mluda_raw"]["confusion_matrix"] == old["confusion_matrix"]
            records[str(seed)][selection] = {"metrics": metrics, "checkpoint_epochs":
                {"mluda": meta["mluda_epoch"], "shift": meta["shift_epoch"]},
                "target_gt_sha256": sha(gt_path), "prediction_sha256": meta["prediction_sha256"]}
            print(json.dumps({"seed": seed, "selection": selection,
                              "scores": {name: {"oa": metrics[name]["oa"], "aa": metrics[name]["aa"]}
                                         for name in VARIANTS}}), flush=True)
    summary = {}
    for selection in SELECTIONS:
        summary[selection] = {}
        for name in VARIANTS:
            summary[selection][name] = {}
            for field in ("oa", "aa", "kappa"):
                vals = np.array([records[str(seed)][selection]["metrics"][name][field] for seed in SEEDS])
                summary[selection][name][field] = {"mean": float(vals.mean()),
                                                      "sd": float(vals.std(ddof=1))}
    out.write_text(json.dumps({"scope": "post-freeze matched-seed audit; Houston development scene",
                               "seeds": records, "summary": summary}, indent=2))
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
