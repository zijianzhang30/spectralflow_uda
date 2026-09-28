"""Label-free target diagnostic: source endpoint support inside classifier regions."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/round9"))
from model import Backbone
from data import Patches, load_images


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    out = {"scope": "frozen SceneShift source training features only; no target GT or new training",
           "rule": "source endpoint eligible iff frozen classifier argmax equals its source label",
           "seeds": {}}
    for seed in (202601, 202602, 202603):
        run = ROOT / f"investigations/scene_shift_flow_v1/formal_{seed}"
        shift = ROOT / f"investigations/scene_shift_v1/formal_{seed}"
        feature_path = run / "frozen_features_before_gt.npz"
        cp_path = shift / "best_source_val.pth"
        with np.load(feature_path) as data:
            z = data["source_z"].copy()
            y = data["source_labels"].copy()
        with np.load(shift / "source_split.npz") as split:
            val_centers = split["val_centers"].copy()
            val_y = split["val_labels"].copy()
        assert z.shape == (1260, 288) and y.shape == (1260,)
        assert np.bincount(y, minlength=7).tolist() == [180]*7
        model = Backbone().eval().requires_grad_(False)
        model.load_state_dict(torch.load(cp_path, map_location="cpu", weights_only=False)["model"])
        with torch.inference_mode():
            q = model.classifier(torch.from_numpy(z)).softmax(1).numpy()
        cfg = json.loads((shift / "config.json").read_text())
        source, _ = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
        val_probs = []
        with torch.inference_mode():
            for x in DataLoader(Patches(source, val_centers), batch_size=32, shuffle=False,
                                drop_last=False, num_workers=0):
                _, logits = model(x)
                val_probs.append(logits.softmax(1).numpy())
        val_pred = np.concatenate(val_probs).argmax(1)
        recorded = json.loads((shift / "best_source_val.json").read_text())
        val_accuracy = float((val_pred == val_y).mean())
        p = q.argmax(1)
        eligible = p == y
        rows = []
        for c in range(7):
            idx = y == c
            correct = idx & eligible
            margins = np.sort(q[idx], axis=1)[:, -1] - np.sort(q[idx], axis=1)[:, -2]
            rows.append({
                "class": c+1,
                "source_count": int(idx.sum()),
                "eligible_count": int(correct.sum()),
                "eligible_fraction": float(eligible[idx].mean()),
                "median_true_class_probability": float(np.median(q[idx, c])),
                "median_top_two_margin": float(np.median(margins)),
                "predicted_class_counts": np.bincount(p[idx], minlength=7).tolist(),
            })
        out["seeds"][str(seed)] = {
            "checkpoint_sha256": digest(cp_path),
            "frozen_features_sha256": digest(feature_path),
            "overall_eligible_fraction": float(eligible.mean()),
            "per_class": rows,
            "source_val_accuracy_recomputed": val_accuracy,
            "source_val_accuracy_recorded": recorded["source_val_accuracy"],
            "source_val_accuracy_matches_record": bool(np.isclose(
                val_accuracy, recorded["source_val_accuracy"])),
            "source_val_recall": [float((val_pred[val_y == c] == c).mean())
                                  for c in range(7)],
        }
    (HERE / "SOURCE_SUPPORT.json").write_text(json.dumps(out, indent=2))
    for seed, row in out["seeds"].items():
        print(seed, "overall", round(row["overall_eligible_fraction"], 4),
              "eligible_count_by_class", [x["eligible_count"] for x in row["per_class"]],
              "val_recall", [round(x, 3) for x in row["source_val_recall"]])


if __name__ == "__main__":
    main()
