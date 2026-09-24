"""Freeze the full predeclared soft-KL curve before any GT audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from soft_projection import soft_project

HERE = Path(__file__).resolve().parent
STRENGTHS = ((0.25, "lambda_025"), (0.5, "lambda_05"), (1.0, "lambda_1"),
             (2.0, "lambda_2"), (4.0, "lambda_4"), (8.0, "lambda_8"),
             (16.0, "lambda_16"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(202601, 202602, 202603), required=True)
    args = p.parse_args()
    source_file = HERE / "runs" / f"correction_{args.seed}" / "predictions_before_gt.npz"
    out = HERE / "runs" / f"soft_{args.seed}"
    assert not out.exists(), f"Refusing to overwrite {out}"
    with np.load(source_file) as source:
        centers = source["centers"].copy()
        q = source["A"].copy()
        hard = source["B"].copy()
        prior = source["prior"].copy()
    assert centers.shape == (53200, 2) and q.shape == hard.shape == (53184, 7)
    arrays = {"centers": centers, "prior": prior,
              "lambda_0": q, "hard": hard}
    records = {}
    for strength, name in STRENGTHS:
        arrays[name], records[name] = soft_project(q, prior, strength)
        print(json.dumps({"seed": args.seed, "lambda": strength,
                          "residual": records[name]["stationarity_max_abs_residual"],
                          "iterations": records[name]["optimizer_iterations"]}), flush=True)
    out.mkdir(parents=True)
    np.savez_compressed(out / "probabilities_before_gt.npz", **arrays)
    metadata = {"seed": args.seed, "source_probabilities_sha256": sha(source_file),
                "soft_lock_sha256": sha(HERE / "SOFT_KL_LOCK.md"),
                "solver_code_sha256": sha(HERE / "soft_projection.py"),
                "grid": ["lambda_0"] + [name for _, name in STRENGTHS] + ["hard"],
                "solver": records, "target_gt_opened_by_soft_script": False}
    (out / "solver.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"seed": args.seed, "saved": str(out / "probabilities_before_gt.npz")}), flush=True)


if __name__ == "__main__":
    main()
