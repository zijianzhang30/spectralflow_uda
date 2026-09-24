"""Compare target class gates on frozen round6 features and identical OT."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diagnose import (REPO, Backbone, Patches, file_hash, load_images,
                      seed_everything, source_prototypes, atomic_json)

GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
METHODS = ("classifier_q", "classifier_q_prototype_mass",
           "classifier_q_batch_prototype_mass", "prototype_q", "agreement")


@torch.no_grad()
def choose_temperature(model, source, centers, labels, prototypes, device):
    loader = DataLoader(Patches(source, centers, labels), batch_size=32,
                        shuffle=False, drop_last=False)
    features, truths = [], []
    for x, y in loader:
        z, _ = model(x.to(device))
        features.append(F.normalize(z, dim=1))
        truths.append(y.to(device))
    cosine = torch.cat(features) @ F.normalize(prototypes, dim=1).T
    y = torch.cat(truths)
    losses = {str(t): float(F.cross_entropy(cosine/t, y)) for t in GRID}
    selected = min(GRID, key=lambda t: losses[str(t)])
    return selected, {"grid_nll": losses,
                      "selected_temperature": selected,
                      "source_val_prototype_accuracy": float((cosine.argmax(1) == y).float().mean()),
                      "source_val_n": len(y)}


def candidate_mask(q, p, c, method):
    qconf, qclass = q.max(1)
    pconf, pclass = p.max(1)
    if method == "classifier_q":
        return (qclass == c) & (qconf >= .95), q[:, c]
    if method == "classifier_q_prototype_mass":
        return (qclass == c) & (qconf >= .95), q[:, c] * p[:, c]
    if method == "classifier_q_batch_prototype_mass":
        return (qclass == c) & (qconf >= .95), q[:, c] * p[:, c]
    if method == "prototype_q":
        return (pclass == c) & (pconf >= .95), p[:, c]
    if method == "agreement":
        return ((qclass == c) & (pclass == c) & (qconf >= .95)
                & (pconf >= .95)), (q[:, c] * p[:, c]).sqrt()
    raise ValueError(method)


@torch.no_grad()
def score_batch(method, source, target, sy, ty, q, p, rng, stats):
    # Same cosine cost, balanced Sinkhorn, regularization, iteration count, and
    # 32 pairs per present class as round6. Only the class gate and mass change.
    costs = (1-F.normalize(source, dim=1) @ F.normalize(target, dim=1).T).cpu()
    q, p = q.cpu(), p.cpu()
    sy, ty = sy.cpu(), ty.cpu()
    for c in range(7):
        src = (sy == c).nonzero().flatten()
        mask, weights = candidate_mask(q, p, c, method)
        dst = mask.nonzero().flatten()
        stats["candidate_n"][c] += len(dst)
        stats["candidate_correct"][c] += int((ty[dst] == c).sum())
        stats["candidate_absent_batches"][c] += int(len(dst) == 0)
        if not len(src) or not len(dst):
            continue
        cost = costs[src][:, dst].double()
        a = torch.full((len(src),), 1/len(src), dtype=torch.float64)
        b = weights[dst].double().clamp_min(1e-12)
        b /= b.sum()
        kernel = (-cost/.05).exp().clamp_min(1e-30)
        u, v = torch.ones_like(a), torch.ones_like(b)
        for _ in range(100):
            u = a/(kernel @ v).clamp_min(1e-30)
            v = b/(kernel.T @ u).clamp_min(1e-30)
        coupling = u[:, None]*kernel*v[None, :]
        assert torch.isfinite(coupling).all() and coupling.sum() > 0
        sampled = torch.multinomial(coupling.flatten(), 32, replacement=True,
                                    generator=rng)
        pair_target = dst[sampled % len(dst)]
        stats["pair_n"][c] += len(pair_target)
        stats["pair_correct"][c] += int((ty[pair_target] == c).sum())


def finish(stats):
    cand = np.array(stats["candidate_n"])
    pairs = np.array(stats["pair_n"])
    return {**stats,
            "candidate_precision": [None if cand[c] == 0 else
                                    float(stats["candidate_correct"][c]/cand[c]) for c in range(7)],
            "pair_purity_by_class": [None if pairs[c] == 0 else
                                      float(stats["pair_correct"][c]/pairs[c]) for c in range(7)],
            "pair_purity_overall": sum(stats["pair_correct"])/max(sum(pairs), 1),
            "class_coverage": int(sum(x > 0 for x in pairs))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selection", choices=("fixed_epoch_100", "source_val_best"),
                        default="fixed_epoch_100")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = REPO / "experiments/round6/runs" / f"formal_{args.seed}"
    cfg = json.loads((run/"config.json").read_text())
    provenance = json.loads((run/"provenance.json").read_text())
    for path, digest in provenance["code"].items():
        assert file_hash(REPO/"experiments/round6"/Path(path).name) == digest
    cp_name = "last.pth" if args.selection == "fixed_epoch_100" else "best_source_val.pth"
    cp_path = run/cp_name
    cp = torch.load(cp_path, map_location=args.device, weights_only=False)
    seed_everything(args.seed)
    torch.set_num_threads(2)
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
    with torch.no_grad():
        prototypes = source_prototypes(model, source, train_centers, train_labels, args.device)
        temperature, calibration = choose_temperature(model, source, val_centers,
                                                       val_labels, prototypes, args.device)
        source_loader = DataLoader(Patches(source, train_centers, train_labels),
                                   batch_size=32, shuffle=False, drop_last=True)
        target_loader = DataLoader(Patches(target, target_centers, target_labels),
                                   batch_size=32, shuffle=False, drop_last=True)
        source_iter = iter(source_loader)
        rngs = {m: torch.Generator().manual_seed(args.seed+51001) for m in METHODS}
        stats = {m: {k: [0]*7 for k in ("candidate_n", "candidate_correct",
                                         "candidate_absent_batches", "pair_n", "pair_correct")}
                 for m in METHODS}
        true_support = np.zeros(7, dtype=np.int64)
        agreement_n = 0
        for batch, (tx, ty) in enumerate(target_loader):
            if batch == args.batches:
                break
            try:
                sx, sy = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                sx, sy = next(source_iter)
            zs, _ = model(sx.to(args.device))
            zt, logits = model(tx.to(args.device))
            q = logits.softmax(1)
            cosine = F.normalize(zt, dim=1) @ F.normalize(prototypes, dim=1).T
            p = (cosine/temperature).softmax(1)
            batch_sum = torch.zeros_like(prototypes)
            batch_count = torch.bincount(sy.to(args.device), minlength=7)
            batch_sum.index_add_(0, sy.to(args.device), zs)
            batch_prototypes = batch_sum / batch_count.clamp_min(1)[:, None]
            batch_cosine = F.normalize(zt, dim=1) @ F.normalize(batch_prototypes, dim=1).T
            batch_cosine[:, batch_count == 0] = -torch.inf
            p_batch = (batch_cosine/0.05).softmax(1)
            agreement_n += int((q.argmax(1) == p.argmax(1)).sum())
            true_support += np.bincount(ty.numpy(), minlength=7)
            for method in METHODS:
                score_batch(method, zs, zt, sy, ty, q,
                            p_batch if method == "classifier_q_batch_prototype_mass" else p,
                            rngs[method], stats[method])
        assert batch == args.batches and true_support.sum() == 32*args.batches
    result = {"seed": args.seed, "selection": args.selection,
              "checkpoint_epoch": cp["epoch"], "checkpoint_sha256": file_hash(cp_path),
              "batches": args.batches,
              "audit_sample_order": "first official-order target batches; sequential cycling source train batches",
              "target_gt_use": "post-hoc purity scoring only; temperature selected on source validation",
              "prototype_calibration": calibration,
              "q_proto_argmax_agreement": agreement_n/(32*args.batches),
              "true_support_by_class": true_support.tolist(),
              "methods": {m: finish(stats[m]) for m in METHODS}}
    out = Path(__file__).resolve().parent/f"matching_{args.seed}_{args.selection}.json"
    atomic_json(out, result)
    print(json.dumps({"seed": args.seed, "selection": args.selection,
                      "temperature": temperature,
                      "source_val_proto_acc": calibration["source_val_prototype_accuracy"],
                      "q_proto_agreement": result["q_proto_argmax_agreement"],
                      "methods": {m: {"coverage": result["methods"][m]["class_coverage"],
                                      "pair_purity": result["methods"][m]["pair_purity_overall"],
                                      "candidate_n": result["methods"][m]["candidate_n"]}
                                  for m in METHODS}}, indent=2))


if __name__ == "__main__":
    main()
