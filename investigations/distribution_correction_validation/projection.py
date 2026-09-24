"""Numerically identical KL projection to the exploratory screen, isolated from model imports."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def project_kl(q, prior):
    """argmin sum_i KL(r_i||q_i), subject to mean_i r_i == prior."""
    q = np.asarray(q, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    assert q.ndim == 2 and q.shape[1] == 7 and np.all(q > 0)
    assert np.all(prior > 0) and abs(prior.sum() - 1) < 1e-5
    logq = np.log(q)

    def objective(v):
        bias = np.r_[0.0, v]
        z = logq + bias
        lse = logsumexp(z, axis=1)
        r = np.exp(z - lse[:, None])
        value = lse.mean() - np.dot(prior[1:], v)
        gradient = (r.mean(axis=0) - prior)[1:]
        return value, gradient

    solution = minimize(objective, np.zeros(6), jac=True, method="BFGS",
                        options={"gtol": 1e-10, "maxiter": 1000})
    bias = np.r_[0.0, solution.x]
    z = logq + bias
    r = np.exp(z - logsumexp(z, axis=1, keepdims=True))
    error = float(np.max(np.abs(r.mean(axis=0) - prior)))
    assert np.isfinite(r).all() and error < 1e-5, (solution.message, error)
    return r.astype(np.float32), {"class_bias": bias.tolist(),
                                  "optimizer_iterations": int(solution.nit),
                                  "prior_max_abs_error": error}
