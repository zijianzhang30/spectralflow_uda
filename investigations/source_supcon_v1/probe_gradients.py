"""Technical first-step gradient probe; target GT is used only for official centers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hdf5storage
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/round9"))
from augmentation import flip_augmentation, radiation_noise
from data import DEFAULT_DATA, load_images, loaders
from model import Backbone
from runtime import seed_everything
from train_feature_supcon import source_supcon_loss, target_feature_losses


def grad_vector(loss, params):
    parts = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                      for p, g in zip(params, parts)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=202601)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    seed_everything(args.seed)
    device = torch.device(args.device)
    source, target = load_images(DEFAULT_DATA, "official_ilda",
                                 ROOT / "official_aligned/preprocessing/official_ilda.npz")
    source_gt = hdf5storage.loadmat(str(DEFAULT_DATA / "Houston13_7gt.mat"))["map"]
    target_gt = hdf5storage.loadmat(str(DEFAULT_DATA / "Houston18_7gt.mat"))["map"]
    source_loader, target_loader, _, _ = loaders(source, target, source_gt,
                                                  args.seed, target_gt)
    model = Backbone().to(device).train()
    x, labels = next(iter(source_loader))
    target_x = next(iter(target_loader))
    source_noise = radiation_noise(x).float().to(device)
    source_flip = flip_augmentation(x).to(device)
    target_noise = radiation_noise(target_x).float().to(device)
    target_flip = flip_augmentation(target_x).to(device)
    x, labels, target_x = x.to(device), labels.to(device), target_x.to(device)
    source_z, source_logits = model(x)
    ce = F.cross_entropy(source_logits, labels)
    supcon = source_supcon_loss(source_z, labels)
    with torch.no_grad():
        target_weak_z, _ = model(target_x)
        model(source_noise)
    target_strong_z, _ = model(target_noise)
    cons, _, _ = target_feature_losses(target_weak_z, target_strong_z)
    with torch.no_grad():
        model(source_flip)
        model(target_flip)
    params = [v for n, v in model.named_parameters() if not n.startswith("classifier.")]
    grads = {k: grad_vector(v, params) for k, v in
             (("ce", ce), ("supcon_weighted", .1 * supcon), ("consistency_ramped_epoch1", .1 * cons))}
    norms = {k: float(g.norm()) for k, g in grads.items()}
    cosine = {f"{a}_vs_{b}": float(F.cosine_similarity(grads[a], grads[b], dim=0))
              for a, b in (("ce", "supcon_weighted"), ("ce", "consistency_ramped_epoch1"),
                           ("supcon_weighted", "consistency_ramped_epoch1"))}
    out = {"seed": args.seed, "step": 1, "source_labels": labels.cpu().tolist(),
           "ce": float(ce), "supcon": float(supcon), "consistency": float(cons),
           "backbone_gradient_norms": norms, "backbone_gradient_cosines": cosine}
    path = ROOT / "investigations/source_supcon_v1" / f"gradient_probe_{args.seed}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("backbone_gradient_norms", "backbone_gradient_cosines")}))


if __name__ == "__main__":
    main()
