"""Post-hoc official classification and class-7 audit of frozen readouts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "investigations/distribution_correction_v1"))
from run import hash_file, metric
from audit_correspondence_structure import full_scene

SEEDS = (202601, 202602, 202603)
METHODS = ("original", "classifier_calibrated", "prototype", "ridge", "source_val_choice")


def main():
    report = {"seeds": list(SEEDS), "methods": list(METHODS),
              "scope": "post-hoc Houston18 diagnostic; all readouts fitted/selected on source labels only",
              "results": {}}
    for seed in SEEDS:
        student = HERE / "runs" / f"student_{seed}"
        run = HERE / "runs" / f"readout_{seed}"
        selection = json.loads((run / "selection_before_gt.json").read_text())
        assert selection["target_gt_opened"] is False
        assert selection["probability_sha256"] == hash_file(run / "probabilities_before_gt.npz")
        assert selection["checkpoint_sha256"] == hash_file(student / "best_source_val.pth")
        with np.load(run / "probabilities_before_gt.npz") as data:
            centers = data["centers"].copy()
            candidates = {name: data[name].copy() for name in METHODS}
        with np.load(student / "source_split.npz") as split:
            assert np.array_equal(centers, split["target_centers"])
        with np.load(HERE / "runs" / f"correction_{seed}" / "predictions_before_gt.npz") as old:
            assert np.array_equal(candidates["original"], old["A"])
        assert np.array_equal(candidates["source_val_choice"],
                              candidates[selection["source_val_choice"]])
        cfg = json.loads((student / "config.json").read_text())
        gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
        gt = hdf5storage.loadmat(str(gt_path))["map"]
        y = gt[centers[:53184, 0], centers[:53184, 1]].astype(np.int64) - 1
        assert len(y) == 53184 and np.all((0 <= y) & (y < 7))
        rows = {name: {"classification": metric(y, q), "decisions": full_scene(q, y)}
                for name, q in candidates.items()}
        previous = json.loads((HERE / "runs" / f"soft_{seed}" / "audit.json").read_text())
        assert rows["original"]["classification"] == previous["variants"]["lambda_0"]["classification"]
        report["results"][str(seed)] = {
            "source_val_choice": selection["source_val_choice"],
            "source_val_macro_nll": {name: selection["candidate_validation"][name]["macro_nll"]
                                     for name in ("classifier_calibrated", "prototype", "ridge")},
            "checkpoint_sha256": selection["checkpoint_sha256"],
            "target_gt_sha256": hash_file(gt_path), "variants": rows}
    path = HERE / "source_readout_audit.json"
    assert not path.exists(), f"Refusing to overwrite {path}"
    path.write_text(json.dumps(report, indent=2))
    for seed in SEEDS:
        row = report["results"][str(seed)]
        print("seed", seed, "source choice", row["source_val_choice"])
        for name in METHODS[:-1]:
            v = row["variants"][name]
            c = v["classification"]
            d = v["decisions"]["classwise"][6]
            print(name, "OA", round(c["oa_official"] * 100, 2),
                  "AA", round(c["aa"] * 100, 2),
                  "class7", round(c["recall"][6] * 100, 2),
                  "class7 retained", round(d["correct_kept_fraction"] * 100, 2))


if __name__ == "__main__":
    main()
