"""Score all three frozen epoch-100 predictions after freezing is complete."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import hdf5storage
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OFFICIAL = PROJECT / "official_aligned"
SEEDS = (1341, 1174, 1370)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out = HERE / "AUDIT.json"
    assert not out.exists()
    frozen = {}
    for seed in SEEDS:
        path = HERE / f"seed_{seed}_before_gt.npz"
        meta = json.loads((HERE / f"seed_{seed}_freeze.json").read_text())
        assert meta["target_gt_opened"] is False and meta["prediction_sha256"] == sha(path)
        with np.load(path) as f:
            centers, pred = f["centers"].copy(), f["predictions"].copy()
        assert centers.shape == (53200, 2) and pred.shape == (53184,)
        frozen[seed] = (centers, pred, meta)
    results = {}
    for seed in SEEDS:
        centers, pred, meta = frozen[seed]
        run = OFFICIAL / "runs/round1/formal" / str(seed) / "MLUDA_full"
        cfg = json.loads((run / "config.json").read_text())
        gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
        gt = hdf5storage.loadmat(str(gt_path))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        assert np.array_equal(np.unique(y), np.arange(7))
        cm = confusion_matrix(y, pred, labels=np.arange(7))
        recall = np.diag(cm) / cm.sum(axis=1)
        best = json.loads((run / "final_target.json").read_text())
        results[str(seed)] = {"oa": float(np.trace(cm) / 53200),
                              "aa": float(recall.mean()),
                              "kappa": float(cohen_kappa_score(y, pred, labels=np.arange(7))),
                              "recall": recall.tolist(), "confusion_matrix": cm.tolist(),
                              "source_val_best_epoch": best["selected_epoch"],
                              "source_val_best_oa": best["oa"],
                              "source_val_best_aa": best["aa"],
                              "target_gt_sha256": sha(gt_path),
                              "prediction_sha256": meta["prediction_sha256"]}
        print(json.dumps({"seed": seed, "fixed100_oa": results[str(seed)]["oa"],
                          "fixed100_aa": results[str(seed)]["aa"],
                          "source_val_best_oa": best["oa"]}), flush=True)
    summary = {}
    for field in ("oa", "aa", "kappa"):
        x = np.array([results[str(seed)][field] for seed in SEEDS])
        summary[field] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1))}
    out.write_text(json.dumps({"scope": "post-freeze target-GT audit; existing full MLUDA checkpoints",
                               "selection": "fixed_epoch_100", "seeds": results, "summary": summary}, indent=2))
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
