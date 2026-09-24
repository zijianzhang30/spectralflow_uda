"""Verify and aggregate the post-locked, same-seed full-MLUDA comparator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs"
SEEDS = (202601, 202602, 202603)
SPLIT_KEYS = ("train_centers", "train_labels", "val_centers", "val_labels", "target_centers")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(x):
    return {"per_seed": x.tolist(), "mean": float(x.mean()),
            "std_ddof1": float(x.std(ddof=1))}


def main():
    rows = []
    for seed in SEEDS:
        run = ROOT / f"mluda_{seed}"
        cfg = json.loads((run / "config.json").read_text())
        history = json.loads((run / "history.json").read_text())
        best = json.loads((run / "best_source_val.json").read_text())
        complete = json.loads((run / "training_complete.json").read_text())
        final = json.loads((run / "final_target.json").read_text())
        correction = json.loads((ROOT / f"correction_{seed}" / "audit.json").read_text())
        assert cfg["method"] == "MLUDA_full" and cfg["seed"] == seed
        assert cfg["selection"] == final["selection"] == "source_val_best"
        assert cfg["steps_per_epoch"] == 38 and len(history) == complete["epochs"] == 100
        assert best == max(history, key=lambda x: x["source_val_accuracy"])
        assert best["epoch"] == final["selected_epoch"]
        assert final["evaluated_n"] == 53184 and final["dataset_n"] == 53200
        assert final["dropped_n"] == 16
        assert final["checkpoint_sha256"] == sha(run / "best_source_val.pth")
        with np.load(run / "source_split.npz") as m, np.load(ROOT / f"student_{seed}" / "source_split.npz") as s:
            assert all(np.array_equal(m[key], s[key]) for key in SPLIT_KEYS)
        cm = np.asarray(final["confusion_matrix"])
        n = cm.sum()
        assert n == 53184 and abs(np.trace(cm) / 53200 - final["oa"]) < 1e-12
        assert abs(np.mean(np.diag(cm) / cm.sum(1)) - final["aa"]) < 1e-12
        observed = np.trace(cm) / n
        expected = np.dot(cm.sum(0), cm.sum(1)) / n ** 2
        assert abs((observed - expected) / (1 - expected) - final["kappa"]) < 1e-12
        for variant in ("A", "B"):
            c = correction["variants"][variant]["classification"]
            assert np.asarray(c["confusion_matrix"]).sum() == 53184
        rows.append({"seed": seed, "MLUDA_full": final,
                     "A": correction["variants"]["A"]["classification"],
                     "B": correction["variants"]["B"]["classification"]})
    out = {"seeds": list(SEEDS), "selection": "source_val_best", "methods": {},
           "scope": "same-seed MLUDA comparator; not upstream ten-seed final-epoch average"}
    for method in ("A", "B", "MLUDA_full"):
        def val(row, key):
            d = row[method]
            return d["recall"][6] if key == "class7_recall" and method != "MLUDA_full" else (
                d["per_class_accuracy"][6] if key == "class7_recall" else
                d["oa_official"] if key == "oa" and method != "MLUDA_full" else d[key])
        out["methods"][method] = {key: stats(np.asarray([val(row, key) for row in rows]))
                                  for key in ("oa", "aa", "kappa", "class7_recall")}
    out["B_minus_MLUDA"] = {key: stats(np.asarray(out["methods"]["B"][key]["per_seed"]) -
                                             np.asarray(out["methods"]["MLUDA_full"][key]["per_seed"]))
                            for key in ("oa", "aa", "kappa", "class7_recall")}
    out["selected_epochs"] = [row["MLUDA_full"]["selected_epoch"] for row in rows]
    (HERE / "mluda_paired_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"MLUDA_full": out["methods"]["MLUDA_full"],
                      "B_minus_MLUDA": out["B_minus_MLUDA"],
                      "selected_epochs": out["selected_epochs"]}, indent=2))


if __name__ == "__main__":
    main()
