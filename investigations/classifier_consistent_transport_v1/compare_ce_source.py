"""Source-only class support of paired CE checkpoints; no target GT."""
from __future__ import annotations

import json
import sys
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/round9"))
from data import Patches, load_images
from model import Backbone


@torch.inference_mode()
def predict(model, cube, centers):
    output = []
    for x in DataLoader(Patches(cube, centers), batch_size=32,
                        shuffle=False, drop_last=False, num_workers=0):
        _, logits = model(x)
        output.append(logits.argmax(1).numpy())
    return np.concatenate(output)


def main():
    torch.set_num_threads(2)
    out = {"scope": "paired CE source-only training/validation evaluation; no target GT",
           "seeds": {}}
    for seed in (202601, 202602, 202603):
        run = ROOT / f"investigations/distribution_correction_validation/runs/student_{seed}"
        cfg = json.loads((run / "config.json").read_text())
        assert cfg["method"] == "A"
        with np.load(run / "source_split.npz") as split:
            train_centers, train_y = split["train_centers"].copy(), split["train_labels"].copy()
            val_centers, val_y = split["val_centers"].copy(), split["val_labels"].copy()
        model = Backbone().eval().requires_grad_(False)
        cp = torch.load(run / "best_source_val.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(cp["model"])
        source, _ = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
        pred_train = predict(model, source, train_centers)
        pred_val = predict(model, source, val_centers)
        recalibrated = copy.deepcopy(model).train()
        with torch.inference_mode():
            for x in DataLoader(Patches(source, train_centers), batch_size=32,
                                shuffle=False, drop_last=False, num_workers=0):
                recalibrated(x)
        recalibrated.eval()
        pred_val_recal = predict(recalibrated, source, val_centers)
        row = {
            "selected_epoch": int(cp["epoch"]),
            "train_accuracy": float((pred_train == train_y).mean()),
            "val_accuracy": float((pred_val == val_y).mean()),
            "train_class_correct_count": [int(((train_y == c) & (pred_train == c)).sum())
                                          for c in range(7)],
            "val_class_recall": [float((pred_val[val_y == c] == c).mean())
                                 for c in range(7)],
            "source_only_bn_recalibration": {
                "rule": "one source-train pass, batch 32, shuffle false, frozen weights",
                "val_accuracy": float((pred_val_recal == val_y).mean()),
                "val_class_recall": [float((pred_val_recal[val_y == c] == c).mean())
                                     for c in range(7)],
            },
        }
        recorded = json.loads((run / "best_source_val.json").read_text())
        row["val_accuracy_recorded"] = recorded["source_val_accuracy"]
        row["val_accuracy_diff_count"] = int(round(
            abs(row["val_accuracy"] - recorded["source_val_accuracy"]) * len(val_y)))
        out["seeds"][str(seed)] = row
    (HERE / "CE_SOURCE_SUPPORT.json").write_text(json.dumps(out, indent=2))
    for seed, row in out["seeds"].items():
        print(seed, "train_correct", row["train_class_correct_count"],
              "val_recall", [round(x, 3) for x in row["val_class_recall"]],
              "recal_val_recall", [round(x, 3) for x in
                                    row["source_only_bn_recalibration"]["val_class_recall"]],
              "val_diff", row["val_accuracy_diff_count"])


if __name__ == "__main__":
    main()
