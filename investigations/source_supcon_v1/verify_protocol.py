"""Check paired F1/F1S provenance and all 3,800 input/RNG steps per seed."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
KEYS = ("source_input_hash", "target_input_hash", "cpu_rng_hash", "cuda_rng_hash")


def main():
    out = {"seeds": {}}
    for seed in (202601, 202602, 202603):
        a = ROOT / "investigations/feature_refinement_v1/runs" / f"formal_F1_{seed}"
        b = HERE / f"formal_{seed}"
        ca = json.loads((a / "config.json").read_text())
        cb = json.loads((b / "config.json").read_text())
        pa = json.loads((a / "provenance.json").read_text())
        pb = json.loads((b / "provenance.json").read_text())
        xa = [json.loads(line) for line in (a / "steps.jsonl").open()]
        xb = [json.loads(line) for line in (b / "steps.jsonl").open()]
        mismatch = {key: sum(ra[key] != rb[key] for ra, rb in zip(xa, xb)) for key in KEYS}
        row = {
            "step_count_F1": len(xa), "step_count_F1S": len(xb),
            "step_hash_mismatches": mismatch,
            "initial_model_equal": pa["initial_model_hash"] == pb["initial_model_hash"],
            "source_split_equal": (a / "source_split.npz").read_bytes() ==
                                  (b / "source_split.npz").read_bytes(),
            "parent_protocol_equal": ca["parent_protocol_sha256"] == cb["parent_protocol_sha256"],
            "dataset_hashes_equal": pa["inputs"] == pb["inputs"],
            "official_settings_equal": all(ca[k] == cb[k] for k in
                ("batch_size", "patch_size", "steps_per_epoch", "lr", "momentum",
                 "weight_decay", "optimizer_reset_each_epoch", "epochs", "lr_horizon")),
            "selected_F1_epoch": json.loads((a / "best_source_val.json").read_text())["epoch"],
            "selected_F1S_epoch": json.loads((b / "best_source_val.json").read_text())["epoch"],
        }
        assert len(xa) == len(xb) == 3800
        assert all(v == 0 for v in mismatch.values())
        assert all(row[k] for k in ("initial_model_equal", "source_split_equal",
                                     "parent_protocol_equal", "dataset_hashes_equal",
                                     "official_settings_equal"))
        out["seeds"][str(seed)] = row
    (HERE / "PROTOCOL_AUDIT.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
