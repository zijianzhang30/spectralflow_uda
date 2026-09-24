"""Verify paired F1/F2 and CE-only A sampling/protocol alignment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "distribution_correction_validation/runs"
SEEDS = (202601, 202602, 202603)


def rows(path):
    result = [json.loads(line) for line in path.open()]
    assert len(result) == 3800
    assert [(r["epoch"], r["step"]) for r in result] == [
        (epoch, step) for epoch in range(1, 101) for step in range(1, 39)]
    return result


def main():
    report = {"seeds": list(SEEDS), "results": {}}
    for seed in SEEDS:
        a = PARENT / f"student_{seed}"
        runs = {method: HERE / "runs" / f"formal_{method}_{seed}"
                for method in ("F1", "F2")}
        base_steps = rows(a / "steps.jsonl")
        with np.load(a / "source_split.npz") as data:
            split = {k: data[k].copy() for k in data.files}
        initial_hash = json.loads((a / "provenance.json").read_text())["initial_model_hash"]
        result = {}
        for method, run in runs.items():
            cfg = json.loads((run / "config.json").read_text())
            complete = json.loads((run / "training_complete.json").read_text())
            assert cfg["seed"] == seed and cfg["method"] == method
            assert cfg["batch_size"] == 32 and cfg["patch_size"] == 7
            assert cfg["steps_per_epoch"] == 38 and complete["formal"]
            assert complete["epochs"] == 100
            assert json.loads((run / "provenance.json").read_text())["initial_model_hash"] == initial_hash
            with np.load(run / "source_split.npz") as data:
                assert all(np.array_equal(split[k], data[k]) for k in split)
            current = rows(run / "steps.jsonl")
            assert all(x["source_input_hash"] == y["source_input_hash"] and
                       x["target_input_hash"] == y["target_input_hash"]
                       for x, y in zip(base_steps, current))
            history = json.loads((run / "history.json").read_text())
            selected = json.loads((run / "best_source_val.json").read_text())
            assert len(history) == 100
            assert selected == max(history, key=lambda entry: entry["source_val_accuracy"])
            result[method] = {"updates": len(current), "same_inputs_as_A": True,
                              "same_source_split_as_A": True,
                              "same_initial_model_as_A": True,
                              "selected_epoch": selected["epoch"]}
        report["results"][str(seed)] = result
    path = HERE / "PROTOCOL_AUDIT.json"
    assert not path.exists(), f"Refusing to overwrite {path}"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
