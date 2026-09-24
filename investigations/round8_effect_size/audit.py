"""Read-only paired audit of round6 versus round8 at fixed epoch 100."""
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
sys.path.insert(0, str(REPO / "experiments/round8"))
from data import Patches, load_images
from flow import ReverseFlow, ot_pairs, transport_target
from model import Backbone
from runtime import atomic_json, seed_everything


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batches", type=int, default=200)
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(2)
    old_run = REPO / "experiments/round6/runs" / f"formal_{args.seed}"
    new_run = REPO / "experiments/round8/runs" / f"formal_{args.seed}"
    old_cp = torch.load(old_run / "last.pth", map_location="cpu", weights_only=False)
    new_cp = torch.load(new_run / "last.pth", map_location="cpu", weights_only=False)
    assert old_cp["epoch"] == new_cp["epoch"] == 100
    assert old_cp["model"].keys() == new_cp["model"].keys()
    assert all(torch.equal(old_cp["model"][k], new_cp["model"][k]) for k in old_cp["model"])
    old_flow = ReverseFlow().to(args.device).eval()
    new_flow = ReverseFlow().to(args.device).eval()
    old_flow.load_state_dict(old_cp["flow"])
    new_flow.load_state_dict(new_cp["flow"])
    model = Backbone().to(args.device).eval()
    model.load_state_dict(new_cp["model"])
    cfg = json.loads((new_run / "config.json").read_text())
    source, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    with np.load(new_run / "source_split.npz") as split:
        train_centers = split["train_centers"].copy()
        train_labels = split["train_labels"].copy()
        target_centers = split["target_centers"].copy()
    gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
    true = gt[target_centers[:, 0], target_centers[:, 1]].astype(np.int64) - 1
    source_loader = DataLoader(Patches(source, train_centers, train_labels),
                               batch_size=32, shuffle=False, drop_last=True)
    target_loader = DataLoader(Patches(target, target_centers),
                               batch_size=32, shuffle=False, drop_last=True)

    mass_tv = []
    candidate_sizes = []
    pair_total = pair_diff = pair_target_diff = 0
    source_iter = iter(source_loader)
    for batch, tx in enumerate(target_loader):
        if batch >= args.batches:
            break
        try:
            sx, sy = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            sx, sy = next(source_iter)
        zs, _ = model(sx.to(args.device))
        zt, logits = model(tx.to(args.device))
        q = logits.softmax(1)
        sy = sy.to(args.device)
        sums = torch.zeros((7, zs.shape[1]), device=args.device)
        counts = torch.bincount(sy, minlength=7)
        sums.index_add_(0, sy, zs)
        protos = sums / counts.clamp_min(1)[:, None]
        sim = F.normalize(zt, dim=1) @ F.normalize(protos, dim=1).T
        sim[:, counts == 0] = -torch.inf
        p = (sim / .05).softmax(1)
        confidence, pseudo = q.max(1)
        for c in range(7):
            candidates = (pseudo == c) & (confidence >= .95)
            if counts[c] == 0 or candidates.sum() == 0:
                continue
            b0 = q[candidates, c].double()
            b1 = b0 * p[candidates, c].double()
            b0 /= b0.sum()
            b1 /= b1.sum()
            mass_tv.append(float(.5 * (b0-b1).abs().sum()))
            candidate_sizes.append(int(candidates.sum()))
        rng0 = torch.Generator().manual_seed(args.seed + 51001 + batch)
        rng1 = torch.Generator().manual_seed(args.seed + 51001 + batch)
        si0, ti0 = ot_pairs(zs, zt, sy, q, rng0)
        si1, ti1 = ot_pairs(zs, zt, sy, q, rng1, p)
        assert len(si0) == len(si1)
        pair_total += len(si0)
        pair_diff += int(((si0 != si1) | (ti0 != ti1)).sum())
        pair_target_diff += int((ti0 != ti1).sum())

    old_pred = []
    new_pred = []
    raw_pred = []
    moved0 = moved1 = moved_diff = feat_norm = 0.0
    for tx in target_loader:
        z, raw_logits = model(tx.to(args.device))
        z0 = transport_target(old_flow, z, steps=4)
        z1 = transport_target(new_flow, z, steps=4)
        raw_pred.extend(raw_logits.argmax(1).cpu().tolist())
        old_pred.extend(model.classifier(z0).argmax(1).cpu().tolist())
        new_pred.extend(model.classifier(z1).argmax(1).cpu().tolist())
        moved0 += float((z0-z).norm(dim=1).sum())
        moved1 += float((z1-z).norm(dim=1).sum())
        moved_diff += float((z1-z0).norm(dim=1).sum())
        feat_norm += float(z.norm(dim=1).sum())
    raw_pred, old_pred, new_pred = map(np.asarray, (raw_pred, old_pred, new_pred))
    y = true[:len(old_pred)]
    old_eval = json.loads((old_run / "final_target_fixed_epoch_100.json").read_text())
    new_eval = json.loads((new_run / "final_target_fixed_epoch_100.json").read_text())
    assert np.isclose((old_pred == y).sum()/len(true), old_eval["transported"]["oa"])
    assert np.isclose((new_pred == y).sum()/len(true), new_eval["transported"]["oa"])
    assert np.isclose((raw_pred == y).sum()/len(true), new_eval["raw"]["oa"])
    changed = old_pred != new_pred
    n = len(y)
    param_old = sum(float(v.double().square().sum()) for v in old_cp["flow"].values())
    param_delta = sum(float((v.double()-new_cp["flow"][k].double()).square().sum())
                      for k, v in old_cp["flow"].items())
    result = {
        "seed": args.seed, "selection": "fixed_epoch_100", "target_n": n,
        "model_states_exact_equal": True,
        "pair_audit_batches": args.batches, "eligible_class_batches": len(mass_tv),
        "candidate_size_mean": float(np.mean(candidate_sizes)),
        "candidate_size_median": float(np.median(candidate_sizes)),
        "candidate_singleton_fraction": float(np.mean(np.asarray(candidate_sizes)==1)),
        "target_mass_tv_mean": float(np.mean(mass_tv)),
        "target_mass_tv_median": float(np.median(mass_tv)),
        "pair_total": pair_total,
        "pair_index_changed_fraction": pair_diff/pair_total,
        "pair_target_index_changed_fraction": pair_target_diff/pair_total,
        "flow_parameter_relative_l2": (param_delta/param_old)**.5,
        "old_transport_norm_mean": moved0/n,
        "new_transport_norm_mean": moved1/n,
        "flow_output_difference_norm_mean": moved_diff/n,
        "feature_norm_mean": feat_norm/n,
        "old_new_prediction_disagreement_n": int(changed.sum()),
        "old_new_prediction_disagreement_fraction": float(changed.mean()),
        "old_correct_new_wrong": int(((old_pred==y)&(new_pred!=y)).sum()),
        "old_wrong_new_correct": int(((old_pred!=y)&(new_pred==y)).sum()),
        "old_correct": int((old_pred==y).sum()),
        "new_correct": int((new_pred==y).sum()),
        "raw_correct": int((raw_pred==y).sum()),
    }
    out = Path(__file__).resolve().parent / f"audit_{args.seed}.json"
    atomic_json(out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
