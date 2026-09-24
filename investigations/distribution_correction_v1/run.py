"""Frozen A/B/C distribution-correction and OT-candidate diagnostic.

B is the minimum-KL projection of DCRN target probabilities onto the
independently source-trained HyperSIGMA teacher's unlabeled-target soft
class marginal. C leaves student-low-confidence rows unchanged. Target GT
is opened only after both corrections are saved and only for diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "investigations/correspondence_v1"))
sys.path.insert(0, str(PROJECT / "experiments/round9"))
from screen import empty_stats, score_candidates, score_pairs, finish  # noqa: E402
from data import Patches, load_images  # noqa: E402

spec = importlib.util.spec_from_file_location("round9_dcrn_model", PROJECT / "experiments/round9/model.py")
student_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(student_module)
Backbone = student_module.Backbone

SEEDS = (1341, 1174, 1370)
GATE = 0.8
PAIR_BATCHES = 200


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    # BFGS can report precision loss near machine precision even when the
    # achieved marginal constraint is tighter than the required tolerance.
    assert np.isfinite(r).all() and error < 1e-5, (solution.message, error)
    return r.astype(np.float32), {"class_bias": bias.tolist(),
                                  "optimizer_iterations": int(solution.nit),
                                  "prior_max_abs_error": error}


def metric(y, q):
    p = q.argmax(1)
    cm = confusion_matrix(y, p, labels=np.arange(7))
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    precision = np.diag(cm) / np.maximum(cm.sum(0), 1)
    return {"oa_official": float(np.trace(cm) / 53200),
            "aa": float(recall.mean()),
            "kappa": float(cohen_kappa_score(y, p, labels=np.arange(7))),
            "recall": recall.tolist(), "precision": precision.tolist(),
            "confusion_matrix": cm.tolist(),
            "soft_mass_fraction": q.mean(axis=0).tolist(),
            "hard_count": np.bincount(p, minlength=7).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = PROJECT / "experiments/round9/runs" / f"formal_A_{args.seed}"
    prior_run = PROJECT / "investigations/hypersigma_teacher_gate/runs" / f"seed_{args.seed}"
    cache_path = prior_run / "target_probabilities.npz"
    with np.load(cache_path) as f:
        q = f["student"].copy()
        teacher = f["teacher"].copy()
        centers = f["centers"].copy()
    assert q.shape == teacher.shape == (53184, 7) and centers.shape == (53184, 2)
    prior = teacher.astype(np.float64).mean(axis=0)
    b, projection = project_kl(q, prior)
    keep_original = q.max(axis=1) < GATE
    c = b.copy()
    c[keep_original] = q[keep_original]
    variants = {"A": q, "B": b, "C": c}
    out = HERE / "runs" / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "corrected_probabilities.npz", centers=centers,
                        A=q, B=b, C=c, prior=prior)

    # The correction is frozen before opening target GT. GT enters only the
    # candidate and expected-pair-purity audit below.
    cfg = json.loads((run / "config.json").read_text())
    gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
    gt = hdf5storage.loadmat(str(gt_path))["map"]
    y = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    assert y.min() == 0 and y.max() == 6
    support = np.bincount(y, minlength=7)
    stats = {name: empty_stats() for name in variants}
    for name, x in variants.items():
        pred = x.argmax(axis=1)
        mask = x.max(axis=1) >= GATE
        score_candidates(stats[name], torch.from_numpy(mask), torch.from_numpy(pred),
                         torch.from_numpy(y))

    checkpoint = run / "best_source_val.pth"
    model = Backbone().to(args.device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    source, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    with np.load(run / "source_split.npz") as f:
        train_centers, train_labels = f["train_centers"].copy(), f["train_labels"].copy()
        assert np.array_equal(f["target_centers"][:53184], centers)
    target_loader = DataLoader(Patches(target, centers[:PAIR_BATCHES * 32]),
                               batch_size=32, shuffle=False, drop_last=True)
    source_loader = DataLoader(Patches(source, train_centers, train_labels),
                               batch_size=32, shuffle=False, drop_last=True)
    source_iter = iter(source_loader)
    with torch.inference_mode():
        for batch, tx in enumerate(target_loader):
            try:
                sx, sy = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                sx, sy = next(source_iter)
            zs, _ = model(sx.to(args.device))
            zt, _ = model(tx.to(args.device))
            ty = torch.from_numpy(y[batch * 32:(batch + 1) * 32])
            for name, x in variants.items():
                xb = x[batch * 32:(batch + 1) * 32]
                score_pairs(stats[name], zs, zt, sy, ty,
                            torch.from_numpy(xb.max(axis=1) >= GATE),
                            torch.from_numpy(xb.argmax(axis=1)),
                            torch.from_numpy(xb))
    result = {"seed": args.seed, "target_n": len(y), "official_oa_denominator": 53200,
              "source_val_best_checkpoint_sha256": hash_file(checkpoint),
              "input_probability_cache_sha256": hash_file(cache_path),
              "prior_estimator": "mean HyperSIGMA probabilities over same unlabeled target centers; no target GT",
              "prior": prior.tolist(), "projection": projection,
              "C_rule": "retain original student q where original max q < 0.8; elsewhere use B",
              "candidate_gate": GATE,
              "pair_audit": "first 200 official-order target batches; same frozen source/target features and source batches for A/B/C; cosine cost / 0.05, 100 Sinkhorn iterations",
              "target_gt_use": "post-hoc metrics and expected purity only; no selection or fitting",
              "variants": {}}
    for name, x in variants.items():
        entry = {"classification": metric(y, x),
                 "changed_argmax_fraction_vs_A": float(np.mean(x.argmax(1) != q.argmax(1))),
                 "correspondence": finish(stats[name], support, len(y) // 32)}
        result["variants"][name] = entry
    official = json.loads((run / "final_target_source_val_best.json").read_text())
    assert result["variants"]["A"]["classification"]["confusion_matrix"] == official["metrics"]["confusion_matrix"]
    (out / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"seed": args.seed,
                      "outcomes": {name: {"oa": result["variants"][name]["classification"]["oa_official"],
                                          "class7_recall": result["variants"][name]["classification"]["recall"][6],
                                          "candidate_coverage": result["variants"][name]["correspondence"]["candidate_coverage_fraction"],
                                          "pair_expected_purity": result["variants"][name]["correspondence"]["pair_expected_purity_overall"]}
                                   for name in variants}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
