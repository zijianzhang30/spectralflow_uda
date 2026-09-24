"""Frozen-feature, source-calibrated correspondence screen; no training."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "investigations/round6_diagnostics"))
from diagnose import Backbone, Patches, file_hash, load_images, seed_everything, source_prototypes, atomic_json
from compare_matching import choose_temperature

METHODS = ("q95", "q80", "proto95", "proto80", "fused95", "fused80",
           "fused50_95", "fused50_80", "agree95")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def decisions(q, p, alpha):
    fused = alpha*q + (1-alpha)*p
    fused50 = .5*q + .5*p
    qc, qy = q.max(1)
    pc, py = p.max(1)
    fc, fy = fused.max(1)
    hc, hy = fused50.max(1)
    return {
        "q95": ((qc >= .95), qy, q),
        "q80": ((qc >= .80), qy, q),
        "proto95": ((pc >= .95), py, p),
        "proto80": ((pc >= .80), py, p),
        "fused95": ((fc >= .95), fy, fused),
        "fused80": ((fc >= .80), fy, fused),
        "fused50_95": ((hc >= .95), hy, fused50),
        "fused50_80": ((hc >= .80), hy, fused50),
        "agree95": ((qy == py) & (qc >= .95) & (pc >= .95), qy,
                    torch.sqrt(q*p)),
    }


@torch.no_grad()
def source_val_alpha(model, source, centers, labels, prototypes, temperature, device):
    loader = DataLoader(Patches(source, centers, labels), batch_size=32,
                        shuffle=False, drop_last=False)
    qs, ps, ys = [], [], []
    for x, y in loader:
        z, logits = model(x.to(device))
        q = logits.softmax(1)
        cosine = F.normalize(z, dim=1) @ F.normalize(prototypes, dim=1).T
        qs.append(q.cpu())
        ps.append((cosine/temperature).softmax(1).cpu())
        ys.append(y)
    q, p, y = map(torch.cat, (qs, ps, ys))
    nll = {str(a): float(F.nll_loss((a*q+(1-a)*p).clamp_min(1e-12).log(), y))
           for a in ALPHAS}
    selected = min(ALPHAS, key=lambda a: nll[str(a)])
    return selected, {"fusion_alpha": selected, "source_val_n": len(y),
                      "source_val_nll_by_alpha": nll,
                      "source_val_accuracy_q": float((q.argmax(1)==y).float().mean()),
                      "source_val_accuracy_p": float((p.argmax(1)==y).float().mean()),
                      "source_val_accuracy_fused": float(((selected*q+(1-selected)*p).argmax(1)==y).float().mean())}


def empty_stats():
    return {"candidate_by_pred_class": np.zeros(7, np.int64),
            "candidate_correct_by_pred_class": np.zeros(7, np.int64),
            "candidate_true_by_class": np.zeros(7, np.int64),
            "candidate_confusion": np.zeros((7, 7), np.int64),
            "candidate_any_batches": np.zeros(7, np.int64),
            "pair_expected_correct": np.zeros(7, np.float64),
            "pair_n": np.zeros(7, np.int64),
            "pair_enabled_batches": np.zeros(7, np.int64)}


@torch.no_grad()
def score_candidates(stats, mask, pred, true):
    m = mask.cpu().numpy()
    py = pred.cpu().numpy()[m]
    ty = true.cpu().numpy()[m]
    stats["candidate_by_pred_class"] += np.bincount(py, minlength=7)
    stats["candidate_correct_by_pred_class"] += np.bincount(py[py==ty], minlength=7)
    # Recall requires a correct class assignment, not merely that a sample
    # passed some class gate.
    stats["candidate_true_by_class"] += np.bincount(ty[py==ty], minlength=7)
    np.add.at(stats["candidate_confusion"], (py, ty), 1)
    stats["candidate_any_batches"] += np.bincount(np.unique(py), minlength=7).astype(bool)


@torch.no_grad()
def score_pairs(stats, source, target, sy, ty, mask, pred, weights):
    # Matches round6 cosine/Sinkhorn settings. Expected purity is computed
    # from the normalized finite-iteration coupling, without sampling noise.
    costs = (1-F.normalize(source, dim=1) @ F.normalize(target, dim=1).T).cpu()
    sy, ty, mask, pred, weights = sy.cpu(), ty.cpu(), mask.cpu(), pred.cpu(), weights.cpu()
    for c in range(7):
        src = (sy==c).nonzero().flatten()
        dst = (mask & (pred==c)).nonzero().flatten()
        if not len(src) or not len(dst):
            continue
        cost = costs[src][:, dst].double()
        a = torch.full((len(src),), 1/len(src), dtype=torch.float64)
        b = weights[dst, c].double().clamp_min(1e-12)
        b /= b.sum()
        kernel = (-cost/.05).exp().clamp_min(1e-30)
        u, v = torch.ones_like(a), torch.ones_like(b)
        for _ in range(100):
            u = a/(kernel@v).clamp_min(1e-30)
            v = b/(kernel.T@u).clamp_min(1e-30)
        coupling = u[:, None]*kernel*v[None, :]
        marginal = coupling.sum(0)/coupling.sum()
        stats["pair_expected_correct"][c] += 32*float(marginal[ty[dst]==c].sum())
        stats["pair_n"][c] += 32
        stats["pair_enabled_batches"][c] += 1


def finish(s, support, full_batches):
    cand = s["candidate_by_pred_class"]
    pairs = s["pair_n"]
    return {
        "candidate_n_by_pred_class": cand.tolist(),
        "candidate_precision_by_pred_class": [None if cand[c]==0 else float(s["candidate_correct_by_pred_class"][c]/cand[c]) for c in range(7)],
        "true_class_candidate_recall": [None if support[c]==0 else float(s["candidate_true_by_class"][c]/support[c]) for c in range(7)],
        "candidate_confusion_pred_by_true": s["candidate_confusion"].tolist(),
        "candidate_coverage_fraction": float(cand.sum()/support.sum()),
        "candidate_macro_recall": float(np.mean(s["candidate_true_by_class"]/np.maximum(support, 1))),
        "candidate_micro_precision": None if cand.sum()==0 else float(s["candidate_correct_by_pred_class"].sum()/cand.sum()),
        "classes_with_candidates": int((cand>0).sum()),
        "candidate_batch_presence_by_class": (s["candidate_any_batches"]/full_batches).tolist(),
        "pair_n_by_class": pairs.tolist(),
        "pair_expected_purity_by_class": [None if pairs[c]==0 else float(s["pair_expected_correct"][c]/pairs[c]) for c in range(7)],
        "pair_expected_purity_overall": None if pairs.sum()==0 else float(s["pair_expected_correct"].sum()/pairs.sum()),
        "pair_enabled_batches_by_class": s["pair_enabled_batches"].tolist(),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pair-batches", type=int, default=200)
    parser.add_argument("--run", type=Path, default=None,
                        help="Optional alternative frozen run with the same Backbone and source split")
    parser.add_argument("--tag", default=None,
                        help="Output tag; required when --run is supplied")
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(2)
    if args.run is not None and args.tag is None:
        parser.error("--tag is required with --run to avoid overwriting baseline diagnostics")
    run = (args.run.resolve() if args.run is not None else
           REPO/"experiments/round6/runs"/f"formal_{args.seed}")
    cfg = json.loads((run/"config.json").read_text())
    assert cfg["seed"] == args.seed
    cp_path = run/"last.pth"
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    assert cp["epoch"] == 100
    model = Backbone().to(args.device).eval()
    model.load_state_dict(cp["model"])
    source, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    with np.load(run/"source_split.npz") as split:
        train_centers = split["train_centers"].copy()
        train_labels = split["train_labels"].copy()
        val_centers = split["val_centers"].copy()
        val_labels = split["val_labels"].copy()
        target_centers = split["target_centers"].copy()
    gt = hdf5storage.loadmat(str(Path(cfg["data"])/"Houston18_7gt.mat"))["map"]
    target_labels = gt[target_centers[:, 0], target_centers[:, 1]].astype(np.int64)-1
    assert len(target_labels)==53200 and np.all((0<=target_labels)&(target_labels<7))
    prototypes = source_prototypes(model, source, train_centers, train_labels, args.device)
    temperature, temp_calibration = choose_temperature(model, source, val_centers,
                                                        val_labels, prototypes, args.device)
    alpha, fusion_calibration = source_val_alpha(model, source, val_centers,
                                                 val_labels, prototypes, temperature,
                                                 args.device)
    target_loader = DataLoader(Patches(target, target_centers, target_labels),
                               batch_size=32, shuffle=False, drop_last=False)
    source_loader = DataLoader(Patches(source, train_centers, train_labels),
                               batch_size=32, shuffle=False, drop_last=True)
    source_iter = iter(source_loader)
    support = np.zeros(7, np.int64)
    stats = {m: empty_stats() for m in METHODS}
    soft_mass_q = np.zeros(7, np.float64)
    soft_mass_p = np.zeros(7, np.float64)
    raw_confusion = np.zeros((7, 7), np.int64)
    prototype_confusion = np.zeros((7, 7), np.int64)
    entropy_sum = 0.0
    total_batches = 0
    for batch, (tx, ty) in enumerate(target_loader):
        zt, logits = model(tx.to(args.device))
        q = logits.softmax(1)
        cosine = F.normalize(zt, dim=1) @ F.normalize(prototypes, dim=1).T
        p = (cosine/temperature).softmax(1)
        rules = decisions(q, p, alpha)
        support += np.bincount(ty.numpy(), minlength=7)
        soft_mass_q += q.double().sum(0).cpu().numpy()
        soft_mass_p += p.double().sum(0).cpu().numpy()
        entropy_sum += float((-(q*q.clamp_min(1e-12).log()).sum(1)).sum())
        np.add.at(raw_confusion, (ty.numpy(), q.argmax(1).cpu().numpy()), 1)
        np.add.at(prototype_confusion, (ty.numpy(), p.argmax(1).cpu().numpy()), 1)
        for method, (mask, pred, _) in rules.items():
            score_candidates(stats[method], mask, pred, ty)
        if batch < args.pair_batches and len(ty)==32:
            try:
                sx, sy = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                sx, sy = next(source_iter)
            zs, _ = model(sx.to(args.device))
            for method, (mask, pred, weights) in rules.items():
                score_pairs(stats[method], zs, zt, sy, ty, mask, pred, weights)
        total_batches += 1
    assert support.sum()==53200 and total_batches==1663
    result = {"seed": args.seed, "selection": "fixed_epoch_100",
              "run": str(run),
              "checkpoint_sha256": file_hash(cp_path), "target_n": int(support.sum()),
              "target_true_support_by_class": support.tolist(),
              "raw_confusion_true_by_pred": raw_confusion.tolist(),
              "raw_recall_by_class": (np.diag(raw_confusion)/support).tolist(),
              "prototype_confusion_true_by_pred": prototype_confusion.tolist(),
              "prototype_recall_by_class": (np.diag(prototype_confusion)/support).tolist(),
              "target_soft_mass_q_by_class": soft_mass_q.tolist(),
              "target_soft_mass_p_by_class": soft_mass_p.tolist(),
              "target_mean_entropy_q": entropy_sum/support.sum(),
              "pair_audit_batches": args.pair_batches,
              "pair_audit_order": "first official-order target batches; sequential cycling source train batches",
              "target_gt_use": "post-hoc diagnostic metrics only; no target-label hyperparameter selection",
              "prototype_temperature": temperature, "source_val_temperature_calibration": temp_calibration,
              "fusion_alpha": alpha, "source_val_fusion_calibration": fusion_calibration,
              "methods": {m: finish(stats[m], support, total_batches) for m in METHODS}}
    out = Path(__file__).resolve().parent/f"screen_{args.tag or args.seed}.json"
    atomic_json(out, result)
    print(json.dumps({"seed": args.seed, "temperature": temperature,
                      "alpha": alpha,
                      "methods": {m: {k: result["methods"][m][k] for k in
                                      ("candidate_coverage_fraction", "candidate_micro_precision",
                                       "candidate_macro_recall", "classes_with_candidates",
                                       "pair_expected_purity_overall")}
                                  for m in METHODS}}, indent=2))


if __name__ == "__main__":
    main()
