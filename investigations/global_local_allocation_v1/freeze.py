"""Freeze label-free global-to-local predictions before the GT audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RUNS = PROJECT / "investigations/distribution_correction_validation/runs"
SEEDS = (202601, 202602, 202603)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def supported_prior(q, prior):
    floor = np.mean(np.square(q.astype(np.float64)), axis=0)
    assert floor.sum() < 1
    fn = lambda t: np.maximum(floor, t * prior).sum() - 1
    scale = brentq(fn, 0, 1 / prior.min(), xtol=1e-14)
    effective = np.maximum(floor, scale * prior)
    assert abs(effective.sum() - 1) < 1e-10
    return effective, floor


def allocate(q, prior, weight):
    logq = np.log(q.astype(np.float64))
    logp = np.log(prior.astype(np.float64))
    weight = np.asarray(weight, dtype=np.float64)
    assert weight.shape == (len(q),) and (weight >= 1).all()

    def dual(v):
        bias = np.r_[0., v]
        logits = logq + bias / weight[:, None]
        logz = logsumexp(logits, axis=1)
        marginal = np.exp(logits - logz[:, None]).mean(axis=0)
        target_logits = logp - bias
        target = np.exp(target_logits - logsumexp(target_logits))
        value = np.mean(weight * logz) + logsumexp(target_logits)
        return value, (marginal - target)[1:]

    result = minimize(dual, np.zeros(q.shape[1] - 1), jac=True, method="BFGS",
                      options={"gtol": 1e-10, "maxiter": 1000})
    bias = np.r_[0., result.x]
    logits = logq + bias / weight[:, None]
    logr = logits - logsumexp(logits, axis=1, keepdims=True)
    r = np.exp(logr)
    marginal = r.mean(axis=0)
    target_logits = logp - bias
    target = np.exp(target_logits - logsumexp(target_logits))
    residual = float(np.max(np.abs(marginal - target)))
    objective = float(np.mean(weight * np.sum(r * (logr - logq), axis=1))
                      + np.dot(marginal, np.log(marginal) - logp))
    original = float(np.dot(q.mean(axis=0), np.log(q.mean(axis=0)) - logp))
    assert residual < 1e-5, (result.message, residual)
    assert objective <= original + 1e-7, (objective, original)
    assert np.allclose(r.sum(axis=1), 1, atol=1e-10)
    return r.astype(np.float32), {"class_bias": bias.tolist(), "residual": residual,
                                  "objective": objective, "original_objective": original,
                                  "iterations": int(result.nit),
                                  "marginal": marginal.tolist()}


def main():
    manifest_path = HERE / "FREEZE_MANIFEST.json"
    assert not manifest_path.exists(), "Refusing to replace a frozen prediction set"
    lock = HERE / "EXPERIMENT_LOCK.md"
    records = {}
    for seed in SEEDS:
        source = RUNS / f"correction_{seed}" / "predictions_before_gt.npz"
        soft = RUNS / f"soft_{seed}" / "probabilities_before_gt.npz"
        with np.load(source) as f:
            q, prior, centers = f["A"].copy(), f["prior"].copy(), f["centers"].copy()
            hard = f["B"].copy()
        with np.load(soft) as f:
            assert np.array_equal(f["lambda_0"], q)
            assert np.array_equal(f["hard"], hard)
            soft1 = f["lambda_1"].copy()
        assert q.shape == soft1.shape == (53184, 7)
        assert centers.shape == (53200, 2)
        assert np.isfinite(q).all() and (q > 0).all()
        effective, floor = supported_prior(q, prior)
        sorted_q = np.sort(q, axis=1)
        weight = 1 + sorted_q[:, -1] - sorted_q[:, -2]
        v1, v1_info = allocate(q, effective, weight)
        class_only, class_info = allocate(q, effective, np.ones(len(q)))
        sample_only, sample_info = allocate(q, prior, weight)
        out = HERE / f"seed_{seed}_before_gt.npz"
        assert not out.exists()
        np.savez_compressed(out, centers=centers, raw=q, hard=hard, soft1=soft1,
                            v1=v1, class_only=class_only, sample_only=sample_only,
                            prior=prior, effective_prior=effective, support_floor=floor)
        records[str(seed)] = {"source_sha256": sha(source), "soft_sha256": sha(soft),
                              "prediction_sha256": sha(out), "prior": prior.tolist(),
                              "effective_prior": effective.tolist(), "support_floor": floor.tolist(),
                              "weight_min_mean_max": [float(weight.min()), float(weight.mean()), float(weight.max())],
                              "solver": {"v1": v1_info, "class_only": class_info,
                                         "sample_only": sample_info}}
        print(json.dumps({"seed": seed, "prediction_sha256": sha(out),
                          "floor": floor.tolist()}), flush=True)
    manifest_path.write_text(json.dumps({"lock_sha256": sha(lock),
                                          "freeze_code_sha256": sha(__file__),
                                          "target_gt_opened": False, "seeds": records}, indent=2))


if __name__ == "__main__":
    main()
