"""Train only the legacy reverse Flow on frozen SHIFT features; freeze predictions before GT."""
from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/round9"))
sys.path.insert(0, str(ROOT / "investigations/distribution_correction_validation"))
from data import Patches, load_images
from model import Backbone
from runtime import seed_everything
from soft_projection import soft_project
from flow import ReverseFlow, reverse_matching_loss, transport_target

SEEDS = (202601, 202602, 202603)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@torch.inference_mode()
def extract(model, cube, centers, device):
    loader = DataLoader(Patches(cube, centers), batch_size=32, shuffle=False,
                        drop_last=False, num_workers=0)
    features, probabilities = [], []
    for x in loader:
        z, logits = model(x.to(device))
        features.append(z.cpu().numpy())
        probabilities.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(features), np.concatenate(probabilities)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--audit", action="store_true")
    a = p.parse_args()
    assert 1 <= a.epochs <= 100 and (a.epochs == 100 or a.audit)
    torch.set_num_threads(2)
    seed_everything(a.seed)
    device = torch.device(a.device)
    shift = ROOT / "investigations/scene_shift_v1" / f"formal_{a.seed}"
    cfg = json.loads((shift / "config.json").read_text())
    complete = json.loads((shift / "training_complete.json").read_text())
    assert cfg["seed"] == a.seed and cfg["method"] == "SHIFT"
    assert complete["formal"] and complete["epochs"] == 100
    cp_path = shift / "best_source_val.pth"
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    assert cp["metrics"] == json.loads((shift / "best_source_val.json").read_text())
    with np.load(shift / "source_split.npz") as split:
        source_centers = split["train_centers"].copy()
        labels = split["train_labels"].copy()
        target_centers = split["target_centers"].copy()
    assert len(source_centers) == len(labels) == 1260
    assert target_centers.shape == (53200, 2)
    model = Backbone().to(device).eval().requires_grad_(False)
    model.load_state_dict(cp["model"])
    source, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    source_z, _ = extract(model, source, source_centers, device)
    target_z, target_raw_q = extract(model, target, target_centers, device)
    assert source_z.shape == (1260, 288) and target_z.shape == (53200, 288)
    saved_path = shift / "predictions_before_gt.npz"
    with np.load(saved_path) as saved:
        assert np.array_equal(target_centers, saved["centers"])
        base_raw, base_corrected = saved["raw"].copy(), saved["corrected"].copy()
    assert np.array_equal(target_raw_q[:53184].argmax(1), base_raw.argmax(1))
    assert np.max(np.abs(target_raw_q[:53184] - base_raw)) < 1e-5
    prior_path = (ROOT / "investigations/distribution_correction_validation/runs" /
                  f"correction_{a.seed}/predictions_before_gt.npz")
    with np.load(prior_path) as prior_data:
        assert np.array_equal(target_centers, prior_data["centers"])
        prior = prior_data["prior"].copy()
    target_q, solver = soft_project(target_raw_q, prior, strength=1.0)
    out = HERE / f"{'smoke' if a.audit else 'formal'}_{a.seed}"
    assert not out.exists(), f"Refusing to overwrite {out}"
    out.mkdir()
    np.savez_compressed(out / "frozen_features_before_gt.npz",
                        source_z=source_z, source_labels=labels,
                        target_z=target_z, target_q=target_q,
                        source_centers=source_centers, target_centers=target_centers)
    torch.manual_seed(a.seed + 71039)
    flow = ReverseFlow().to(device)
    sampler = torch.Generator(device="cpu").manual_seed(a.seed + 99173)
    source_loader = DataLoader(TensorDataset(torch.from_numpy(source_z),
                                             torch.from_numpy(labels)),
                               batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    target_loader = DataLoader(TensorDataset(torch.from_numpy(target_z),
                                             torch.from_numpy(target_q)),
                               batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    assert len(source_loader) - 1 == 38
    history = []
    start = time.time()
    for epoch in range(1, a.epochs + 1):
        lr = .01 / (1 + 10 * (epoch - 1) / 100) ** .75
        optimizer = torch.optim.SGD(flow.parameters(), lr=lr, momentum=.9,
                                    weight_decay=5e-4)
        flow.train()
        si, ti = iter(source_loader), iter(target_loader)
        fm_total = pairs_total = active = 0
        for step in range(1, len(source_loader)):
            sz, sy = next(si)
            tz, tq = next(ti)
            fm, n = reverse_matching_loss(flow, sz.to(device), tz.to(device),
                                          sy.to(device), tq.to(device), sampler)
            if n:
                assert torch.isfinite(fm)
                optimizer.zero_grad(set_to_none=True)
                fm.backward()
                assert any(p.grad is not None and torch.isfinite(p.grad).all() and
                           p.grad.abs().sum() > 0 for p in flow.parameters())
                optimizer.step()
                active += 1
            fm_total += float(fm.detach())
            pairs_total += n
        row = {"epoch": epoch, "fm": fm_total / 38,
               "pairs_per_step": pairs_total / 38,
               "active_steps": active, "lr": lr}
        history.append(row)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"flow": flow.state_dict(), "epoch": epoch,
                    "optimizer": optimizer.state_dict()}, out / "last.pth")
        print(row, flush=True)
    (out / "training_complete.json").write_text(json.dumps({
        "epochs": a.epochs, "formal": a.epochs == 100 and not a.audit,
        "seconds": time.time() - start,
        "selection": "fixed Flow epoch100; source-val-best frozen SHIFT backbone",
    }, indent=2))
    if a.audit:
        print("SMOKE COMPLETE: no target GT opened or scored", flush=True)
        return
    flow.eval()
    moved = []
    with torch.inference_mode():
        for offset in range(0, 53184, 32):
            z = torch.from_numpy(target_z[offset:offset + 32]).to(device)
            moved.append(model.classifier(transport_target(flow, z, steps=4))
                         .softmax(1).cpu().numpy())
    moved_q = np.concatenate(moved)
    moved_corrected, moved_solver = soft_project(moved_q, prior, strength=1.0)
    prediction = out / "predictions_before_gt.npz"
    np.savez_compressed(prediction, centers=target_centers,
                        base_raw=base_raw, base_corrected=base_corrected,
                        flow_raw=moved_q, flow_corrected=moved_corrected, prior=prior)
    (out / "metadata.json").write_text(json.dumps({
        "seed": a.seed, "selected_backbone_epoch": cp["epoch"],
        "backbone_checkpoint_sha256": digest(cp_path),
        "source_split_sha256": digest(shift / "source_split.npz"),
        "base_prediction_sha256": digest(saved_path),
        "prior_sha256": digest(prior_path),
        "flow_checkpoint_sha256": digest(out / "last.pth"),
        "feature_cache_sha256": digest(out / "frozen_features_before_gt.npz"),
        "prediction_sha256": digest(prediction),
        "lock_sha256": digest(HERE / "EXPERIMENT_LOCK.md"),
        "target_gt_opened": False,
        "training_projection": solver,
        "inference_projection": moved_solver,
    }, indent=2))
    print(json.dumps({"seed": a.seed,
                      "mean_pairs_per_step": sum(r["pairs_per_step"] for r in history) / len(history),
                      "changed_argmax": float((base_raw.argmax(1) != moved_q.argmax(1)).mean())}),
          flush=True)


if __name__ == "__main__":
    main()
