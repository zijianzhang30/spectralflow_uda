"""Verify paired MLUDA+SHIFT runs without reading target labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (202601, 202602, 202603)
OFFICIAL_KEYS = (
    "seed", "epochs", "lr_horizon", "data", "ilda_cache", "selection",
    "tie_rule", "batch_size", "patch_size", "train_n", "val_n", "target_n",
    "steps_per_epoch", "lr", "momentum", "weight_decay",
    "optimizer_reset_each_epoch", "target_metrics_during_training",
    "source_validation_forward", "target_test_forward", "test_drop_last", "pooling",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out = {"target_gt_opened": False, "seeds": {}}
    for seed in SEEDS:
        old = ROOT / f"investigations/distribution_correction_validation/runs/mluda_{seed}"
        new = HERE / f"formal_{seed}"
        a = json.loads((old / "config.json").read_text())
        b = json.loads((new / "config.json").read_text())
        assert all(a[k] == b[k] for k in OFFICIAL_KEYS)
        assert b["method"] == "MLUDA_SHIFT"
        assert (old / "source_split.npz").read_bytes() == (new / "source_split.npz").read_bytes()
        pa = json.loads((old / "provenance.json").read_text())
        pb = json.loads((new / "provenance.json").read_text())
        assert pa["inputs"] == pb["inputs"]
        assert all(digest(ROOT / "official_aligned" / name) == hash_value
                   for name, hash_value in pb["code"].items()
                   if name != "train_mluda_shift.py")
        assert digest(ROOT / "experiments/round9/train_mluda_shift.py") == pb["code"]["train_mluda_shift.py"]
        history = json.loads((new / "history.json").read_text())
        complete = json.loads((new / "training_complete.json").read_text())
        best = json.loads((new / "best_source_val.json").read_text())
        cp = torch.load(new / "best_source_val.pth", map_location="cpu", weights_only=False)
        assert len(history) == complete["epochs"] == 100 and complete["formal"]
        assert b["steps_per_epoch"] == 38 and b["target_n"] == 53200
        assert cp["metrics"] == best == max(history, key=lambda row: row["source_val_accuracy"])
        assert cp["epoch"] == best["epoch"]
        out["seeds"][str(seed)] = {
            "official_config_equal": True, "source_split_equal": True,
            "input_hashes_equal": True, "snapshotted_code_hashes_equal": True,
            "completed_epochs": 100, "updates_per_epoch": 38,
            "target_training_positions": 53200,
            "source_val_best_epoch": best["epoch"],
            "checkpoint_sha256": digest(new / "best_source_val.pth"),
        }
    (HERE / "PROTOCOL_AUDIT.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
