"""Source-validation sensitivity to each frozen target-BN layer."""
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
from cross_audit import score


def main():
    torch.set_num_threads(2)
    output = {"scope": "Source validation only; copy one target-calibrated BN layer at a time into original frozen model", "results": {}}
    for method in ("A", "SHIFT"):
        output["results"][method] = {}
        for seed in (202601, 202602, 202603):
            run = (ROOT / f"investigations/distribution_correction_validation/runs/student_{seed}"
                   if method == "A" else ROOT / f"investigations/scene_shift_v1/formal_{seed}")
            cfg = json.loads((run / "config.json").read_text())
            with np.load(run / "source_split.npz") as split:
                centers, labels = split["val_centers"].copy(), split["val_labels"].copy()
            with np.load(cfg["ilda_cache"]) as cache:
                source = cache["s"].astype(np.float32)
            ds = Patches(source, centers)
            x = torch.stack([ds[i] for i in range(len(ds))])
            original_state = torch.load(run / "best_source_val.pth", map_location="cpu", weights_only=False)["model"]
            target_state = torch.load(
                ROOT / f"investigations/target_bn_recal_v1/{method}_{seed}/calibrated_state.pth",
                map_location="cpu", weights_only=False)["model"]
            assert set(original_state) == set(target_state)
            assert all(torch.equal(original_state[k], target_state[k])
                       for k in original_state if not k.endswith(("running_mean", "running_var", "num_batches_tracked")))
            row = {}
            for layer in range(1, 8):
                hybrid_state = dict(original_state)
                for suffix in ("running_mean", "running_var", "num_batches_tracked"):
                    key = f"bn{layer}.{suffix}"
                    hybrid_state[key] = target_state[key]
                hybrid = Backbone().eval().requires_grad_(False)
                hybrid.load_state_dict(hybrid_state)
                row[f"bn{layer}"] = score(hybrid, x, labels)
            output["results"][method][str(seed)] = row
            print(method, seed, [round(row[f"bn{layer}"]["accuracy"], 4)
                                 for layer in range(1, 8)], flush=True)
    (HERE / "LAYER_AUDIT.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
