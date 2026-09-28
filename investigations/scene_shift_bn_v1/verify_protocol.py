"""Audit full SHIFT versus SHIFT_BN traces without reading target labels."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIFT = HERE.parent / "scene_shift_v1"
SEEDS = (202601, 202602, 202603)
EQUAL_STEP_KEYS = (
    "epoch", "step", "ce", "shifted_ce", "backbone_hash",
    "classifier_hash", "gradient_hash", "source_logits_hash",
    "source_input_hash", "target_input_hash", "cpu_rng_hash", "cuda_rng_hash",
)
OFFICIAL_KEYS = (
    "batch_size", "patch_size", "steps_per_epoch", "lr", "momentum",
    "weight_decay", "optimizer_reset_each_epoch", "epochs", "lr_horizon",
    "parent_protocol_sha256",
)


def records(path):
    return [json.loads(line) for line in path.open()]


def main():
    report = {"description": "SHIFT and SHIFT_BN differ only in auxiliary BN buffers", "seeds": {}}
    for seed in SEEDS:
        old = SHIFT / f"formal_{seed}"
        new = HERE / f"formal_{seed}"
        a, b = records(old / "steps.jsonl"), records(new / "steps.jsonl")
        assert len(a) == len(b) == 3800
        mismatches = {key: sum(x[key] != y[key] for x, y in zip(a, b))
                      for key in EQUAL_STEP_KEYS}
        assert all(value == 0 for value in mismatches.values()), (seed, mismatches)
        cfg_a = json.loads((old / "config.json").read_text())
        cfg_b = json.loads((new / "config.json").read_text())
        assert all(cfg_a[key] == cfg_b[key] for key in OFFICIAL_KEYS)
        prov_a = json.loads((old / "provenance.json").read_text())
        prov_b = json.loads((new / "provenance.json").read_text())
        assert prov_a["initial_model_hash"] == prov_b["initial_model_hash"]
        assert prov_a["inputs"] == prov_b["inputs"]
        assert (old / "source_split.npz").read_bytes() == (new / "source_split.npz").read_bytes()
        row = {
            "steps": len(a), "equal_trace_mismatches": mismatches,
            "buffer_hash_different_steps": sum(x["buffers_hash"] != y["buffers_hash"]
                                               for x, y in zip(a, b)),
            "model_hash_different_steps": sum(x["model_hash"] != y["model_hash"]
                                              for x, y in zip(a, b)),
            "initial_model_equal": True, "source_split_equal": True,
            "dataset_hashes_equal": True, "official_settings_equal": True,
            "shift_best_epoch": json.loads((old / "best_source_val.json").read_text())["epoch"],
            "bn_best_epoch": json.loads((new / "best_source_val.json").read_text())["epoch"],
        }
        assert row["buffer_hash_different_steps"] == 3800
        report["seeds"][str(seed)] = row
    (HERE / "PROTOCOL_AUDIT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
