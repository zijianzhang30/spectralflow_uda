"""Evaluate already frozen target-BN checkpoints on labeled source validation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/round9"))
from data import Patches
from model import Backbone


@torch.inference_mode()
def score(model, x, labels):
    model.eval()
    pred = []
    for batch in x.split(32):
        pred.append(model(batch)[1].argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    return {
        "accuracy": float((pred == labels).mean()),
        "class_recall": [float((pred[labels == c] == c).mean()) for c in range(7)],
        "predicted_class_counts": np.bincount(pred, minlength=7).tolist(),
    }


def main():
    torch.set_num_threads(2)
    out = {"scope": "Existing target-only BN states evaluated on source validation; no target GT", "results": {}}
    for method in ("A", "SHIFT"):
        out["results"][method] = {}
        for seed in (202601, 202602, 202603):
            run = (ROOT / f"investigations/distribution_correction_validation/runs/student_{seed}"
                   if method == "A" else ROOT / f"investigations/scene_shift_v1/formal_{seed}")
            cfg = json.loads((run / "config.json").read_text())
            with np.load(run / "source_split.npz") as split:
                centers = split["val_centers"].copy()
                labels = split["val_labels"].copy()
            with np.load(cfg["ilda_cache"]) as cache:
                source = cache["s"].astype(np.float32)
            ds = Patches(source, centers)
            x = torch.stack([ds[i] for i in range(len(ds))])
            original_state = torch.load(run / "best_source_val.pth", map_location="cpu", weights_only=False)
            target_state = torch.load(
                ROOT / f"investigations/target_bn_recal_v1/{method}_{seed}/calibrated_state.pth",
                map_location="cpu", weights_only=False)
            original = Backbone().eval().requires_grad_(False)
            target_bn = Backbone().eval().requires_grad_(False)
            original.load_state_dict(original_state["model"])
            target_bn.load_state_dict(target_state["model"])
            for (a_name, a), (b_name, b) in zip(original.named_parameters(), target_bn.named_parameters()):
                assert a_name == b_name and torch.equal(a, b), a_name
            out["results"][method][str(seed)] = {
                "original": score(original, x, labels),
                "target_bn": score(target_bn, x, labels),
                "parameter_tensors_identical": True,
            }
            print(method, seed, round(out["results"][method][str(seed)]["original"]["accuracy"], 4),
                  round(out["results"][method][str(seed)]["target_bn"]["accuracy"], 4), flush=True)
    (HERE / "CROSS_AUDIT.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
