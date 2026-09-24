"""Freeze single-source JCPOT proportions/couplings before target-GT audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import ot

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "distribution_correction_validation"
REGS = (0.1, 0.2)
TARGET_N = 8192


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def solve(zs, ys, zt, reg):
    proportion, log = ot.bregman.jcpot_barycenter(
        [zs, zs.copy()], [ys.copy(), ys.copy()], zt, reg=reg, metric="sqeuclidean",
        numItermax=200, stopThr=1e-6, log=True)
    gamma = np.asarray(log["gamma"][0], dtype=np.float64)
    assert np.allclose(gamma, log["gamma"][1], rtol=1e-5, atol=1e-10)
    proportion = np.asarray(proportion, dtype=np.float64)
    assert gamma.shape == (1260, TARGET_N)
    assert proportion.shape == (7,) and np.isfinite(proportion).all()
    assert np.isfinite(gamma).all() and (gamma >= 0).all()
    assert np.isclose(proportion.sum(), 1, atol=1e-5)
    class_mass = np.stack([gamma[ys == c].sum(axis=0) for c in range(7)], axis=1)
    columns = class_mass.sum(axis=1)
    assert np.all(columns > 0)
    posterior = class_mass / columns[:, None]
    row_mass = gamma.sum(axis=1)
    expected_row_mass = proportion[ys] / np.bincount(ys, minlength=7)[ys]
    record = {"reg": reg, "iterations": int(log["niter"]) + 1,
              "last_change": float(log["err"][-1]),
              "target_column_marginal_max_abs_error": float(np.max(np.abs(columns - 1 / TARGET_N))),
              "source_row_marginal_max_abs_error": float(np.max(np.abs(row_mass - expected_row_mass))),
              "posterior_mean": posterior.mean(axis=0).tolist(),
              "estimated_prior": proportion.tolist()}
    return proportion.astype(np.float32), gamma.astype(np.float32), posterior.astype(np.float32), record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(202601, 202602, 202603), required=True)
    args = p.parse_args()
    source = PARENT / "runs" / f"readout_{args.seed}"
    original = PARENT / "runs" / f"correction_{args.seed}" / "predictions_before_gt.npz"
    out = HERE / "runs" / str(args.seed)
    assert not out.exists(), f"Refusing to overwrite {out}"
    with np.load(source / "features_before_gt.npz") as cache:
        zs = normalize(cache["train_z"])
        zt = normalize(cache["target_z"][:TARGET_N])
        ys = cache["train_labels"].copy()
        centers = cache["centers"][:TARGET_N].copy()
    with np.load(original) as data:
        assert np.array_equal(centers, data["centers"][:TARGET_N])
        student = data["A"][:TARGET_N].copy()
        teacher = data["teacher"][:TARGET_N].copy()
        teacher_full_prior = data["prior"].copy()
    assert zs.shape == (1260, 288) and zt.shape == (TARGET_N, 288)
    assert np.array_equal(np.bincount(ys, minlength=7), np.full(7, 180))
    arrays = {"centers": centers, "source_labels": ys, "student": student,
              "teacher": teacher, "teacher_full_prior": teacher_full_prior}
    records = {}
    for reg in REGS:
        tag = str(reg).replace(".", "p")
        prior, gamma, posterior, record = solve(zs, ys, zt, reg)
        arrays[f"prior_{tag}"] = prior
        arrays[f"gamma_{tag}"] = gamma
        arrays[f"posterior_{tag}"] = posterior
        records[tag] = record
        print(json.dumps({"seed": args.seed, **record}), flush=True)
    out.mkdir(parents=True)
    np.savez_compressed(out / "before_gt.npz", **arrays)
    (out / "fit.json").write_text(json.dumps({
        "seed": args.seed, "target_gt_opened": False,
        "feature_sha256": digest(source / "features_before_gt.npz"),
        "student_teacher_probability_sha256": digest(original),
        "lock_sha256": digest(HERE / "PILOT_LOCK.md"),
        "pot_version": ot.__version__, "fit": records,
        "before_gt_sha256": digest(out / "before_gt.npz")}, indent=2))


if __name__ == "__main__":
    main()
