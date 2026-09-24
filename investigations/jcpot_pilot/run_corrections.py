"""Freeze same-prefix, equal-strength soft-KL prior comparisons before GT."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "distribution_correction_validation"))
from soft_projection import soft_project


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(202601, 202602, 202603), required=True)
    args = p.parse_args()
    run = HERE / "runs" / str(args.seed)
    output = run / "corrections_before_gt.npz"
    assert not output.exists()
    fit = json.loads((run / "fit.json").read_text())
    assert fit["target_gt_opened"] is False
    assert fit["before_gt_sha256"] == digest(run / "before_gt.npz")
    with np.load(run / "before_gt.npz") as data:
        q = data["student"].copy()
        teacher = data["teacher"].copy()
        priors = {"hypersigma": teacher.astype(np.float64).mean(axis=0),
                  "jcpot_0p1": data["prior_0p1"].copy(),
                  "jcpot_0p2": data["prior_0p2"].copy()}
    arrays = {"student": q}
    records = {}
    for name, prior in priors.items():
        arrays[name], records[name] = soft_project(q, prior, strength=1.0)
    np.savez_compressed(output, **arrays)
    (run / "correction_fit.json").write_text(json.dumps({
        "seed": args.seed, "target_gt_opened": False, "strength": 1.0,
        "jcpot_fit_sha256": digest(run / "before_gt.npz"),
        "lock_sha256": digest(HERE / "PILOT_LOCK.md"),
        "solver": records,
        "correction_sha256": digest(output)}, indent=2))
    print(json.dumps({"seed": args.seed,
                      "residuals": {name: value["stationarity_max_abs_residual"]
                                    for name, value in records.items()}}), flush=True)


if __name__ == "__main__":
    main()
