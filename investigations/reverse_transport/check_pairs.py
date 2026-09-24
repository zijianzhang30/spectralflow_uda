"""Post-hoc OT pair-label purity diagnostic; target GT never enters training."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/round5"))
from model import Backbone
from flow import ot_pairs
from data import Patches, load_images, file_hash
from runtime import atomic_json, seed_everything


@torch.no_grad()
def filtered_pairs(source, target, labels, q, generator):
    """Pair only target predictions above the predeclared 0.95 confidence."""
    costs = (1 - F.normalize(source, dim=1) @ F.normalize(target, dim=1).T).cpu()
    labels, q = labels.cpu(), q.cpu()
    confidence, pseudo = q.max(1)
    source_ids, target_ids = [], []
    for c in range(7):
        src = (labels == c).nonzero().flatten()
        dst = ((pseudo == c) & (confidence >= .95)).nonzero().flatten()
        if len(src) == 0 or len(dst) == 0:
            continue
        cost = costs[src][:, dst].double()
        a = torch.full((len(src),), 1/len(src), dtype=torch.float64)
        b = q[dst, c].double().clamp_min(1e-12)
        b /= b.sum()
        kernel = (-cost/.05).exp().clamp_min(1e-30)
        u, v = torch.ones_like(a), torch.ones_like(b)
        for _ in range(100):
            u = a/(kernel @ v).clamp_min(1e-30)
            v = b/(kernel.T @ u).clamp_min(1e-30)
        coupling = u[:, None]*kernel*v[None, :]
        sampled = torch.multinomial(coupling.flatten(), 32, replacement=True,
                                    generator=generator)
        source_ids.append(src[sampled//len(dst)])
        target_ids.append(dst[sampled % len(dst)])
    if not source_ids:
        return None
    return torch.cat(source_ids), torch.cat(target_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1341)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--filtered", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = REPO / f"experiments/round5/runs/formal_{args.seed}"
    cp = torch.load(run / "last.pth", map_location=args.device,
                    weights_only=False)
    assert cp["epoch"] == 100
    cfg = json.loads((run / "config.json").read_text())
    seed_everything(args.seed)
    torch.set_num_threads(2)
    model = Backbone().to(args.device).eval()
    model.load_state_dict(cp["model"])
    source, target = load_images(Path(cfg["data"]), cfg["protocol"],
                                 Path(cfg["ilda_cache"]))
    with np.load(run / "source_split.npz") as split:
        source_centers = split["train_centers"]
        source_labels = split["train_labels"]
        target_centers = split["target_centers"]
    gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
    target_labels = gt[target_centers[:, 0], target_centers[:, 1]].astype(np.int64)-1
    source_loader = DataLoader(Patches(source, source_centers, source_labels),
                               batch_size=32, shuffle=False, drop_last=True)
    target_loader = DataLoader(Patches(target, target_centers, target_labels),
                               batch_size=32, shuffle=False, drop_last=True)
    source_iter = iter(source_loader)
    generator = torch.Generator().manual_seed(args.seed + 51001)
    pair_cm = np.zeros((7, 7), dtype=np.int64)
    raw_cm = np.zeros((7, 7), dtype=np.int64)
    with torch.no_grad():
        for step, (target_x, ty) in enumerate(target_loader):
            if step >= args.batches:
                break
            try:
                source_x, sy = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_x, sy = next(source_iter)
            sz, _ = model(source_x.to(args.device))
            tz, logits = model(target_x.to(args.device))
            q = logits.softmax(dim=1)
            pairs = (filtered_pairs(sz, tz, sy.to(args.device), q, generator)
                     if args.filtered else ot_pairs(sz, tz, sy.to(args.device), q, generator))
            if pairs is None:
                continue
            si, ti = pairs
            np.add.at(pair_cm, (sy[si.cpu()].numpy(), ty[ti.cpu()].numpy()), 1)
            np.add.at(raw_cm, (ty.numpy(), logits.argmax(1).cpu().numpy()), 1)
    count = int(pair_cm.sum())
    result = dict(seed=args.seed, batches=args.batches,
                  pairing="confident_argmax_0.95" if args.filtered else "original_forced_per_class",
                  target_gt_used_only_for_posthoc_diagnostic=True,
                  checkpoint_sha256=file_hash(run / "last.pth"),
                  pair_n=count, pair_true_class_agreement=float(np.trace(pair_cm)/count),
                  raw_classifier_accuracy_on_same_target_batches=float(np.trace(raw_cm)/raw_cm.sum()),
                  pair_confusion_matrix=pair_cm.tolist(),
                  pair_agreement_by_source_class=(np.diag(pair_cm)/np.maximum(pair_cm.sum(1), 1)).tolist(),
                  raw_confusion_matrix=raw_cm.tolist())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, result)
    print(json.dumps({key: result[key] for key in
                      ("pair_n", "pair_true_class_agreement",
                       "raw_classifier_accuracy_on_same_target_batches",
                       "pair_agreement_by_source_class")}, indent=2))


if __name__ == "__main__":
    main()
