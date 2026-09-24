"""Convex soft prior correction with a single, explicit lambda."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def soft_project(q, prior, strength):
    """Minimize mean KL(r||q) + strength * KL(mean(r)||prior)."""
    q = np.asarray(q, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    strength = float(strength)
    assert q.ndim == 2 and q.shape[1] == prior.shape[0] == 7
    assert strength > 0 and np.isfinite(strength)
    assert np.isfinite(q).all() and np.all(q > 0)
    assert np.isfinite(prior).all() and np.all(prior > 0)
    assert abs(prior.sum() - 1) < 1e-5
    logq, logp = np.log(q), np.log(prior)

    def dual(v):
        bias = np.r_[0.0, v]
        logits = logq + bias
        log_partition = logsumexp(logits, axis=1)
        marg = np.exp(logits - log_partition[:, None]).mean(axis=0)
        target_logits = logp - bias / strength
        target = np.exp(target_logits - logsumexp(target_logits))
        value = log_partition.mean() + strength * logsumexp(target_logits)
        return value, (marg - target)[1:]

    solution = minimize(dual, np.zeros(6), jac=True, method="BFGS",
                        options={"gtol": 1e-10, "maxiter": 1000})
    bias = np.r_[0.0, solution.x]
    logits = logq + bias
    log_r = logits - logsumexp(logits, axis=1, keepdims=True)
    r = np.exp(log_r)
    marginal = r.mean(axis=0)
    target_logits = logp - bias / strength
    dual_target = np.exp(target_logits - logsumexp(target_logits))
    residual = float(np.max(np.abs(marginal - dual_target)))
    student_marginal = q.mean(axis=0)
    sample_kl = float(np.mean(np.sum(r * (log_r - logq), axis=1)))
    marginal_kl = float(np.sum(marginal * (np.log(marginal) - logp)))
    objective = sample_kl + strength * marginal_kl
    original_objective = float(strength * np.sum(
        student_marginal * (np.log(student_marginal) - logp)))
    assert residual < 1e-5, (solution.message, residual)
    assert np.isfinite(r).all() and np.allclose(r.sum(axis=1), 1, atol=1e-10)
    assert objective <= original_objective + 1e-7, (objective, original_objective)
    return r.astype(np.float32), {
        "lambda": strength, "class_bias": bias.tolist(),
        "optimizer_iterations": int(solution.nit),
        "optimizer_message": str(solution.message),
        "stationarity_max_abs_residual": residual,
        "sample_mean_kl_to_student": sample_kl,
        "marginal_kl_to_teacher": marginal_kl,
        "objective": objective,
        "original_student_objective": original_objective,
        "result_marginal": marginal.tolist(),
    }
