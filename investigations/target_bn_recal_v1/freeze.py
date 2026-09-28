"""One-pass unlabeled target BN recalibration; freeze predictions before GT."""
from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/round9"))
sys.path.insert(0, str(ROOT / "investigations/distribution_correction_validation"))
from data import Patches, load_images
from model import Backbone
from runtime import tensor_hash, seed_everything
from soft_projection import soft_project


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@torch.inference_mode()
def forward_all(model, target, centers, device, *, drop_last):
    loader = DataLoader(Patches(target, centers), batch_size=32, shuffle=False,
                        drop_last=drop_last, num_workers=0)
    result = []
    for x in loader:
        _, logits = model(x.to(device))
        if drop_last:
            result.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(result) if drop_last else len(loader)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("A", "SHIFT"), required=True)
    p.add_argument("--seed", type=int, choices=(202601, 202602, 202603), required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.set_num_threads(2)
    seed_everything(a.seed)
    run = (ROOT / f"investigations/distribution_correction_validation/runs/student_{a.seed}"
           if a.method == "A" else ROOT / f"investigations/scene_shift_v1/formal_{a.seed}")
    out = HERE / f"{a.method}_{a.seed}"
    assert not out.exists(), f"Refusing to overwrite {out}"
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["method"] == a.method and cfg["seed"] == a.seed
    cp_path = run / "best_source_val.pth"
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    assert cp["metrics"] == json.loads((run / "best_source_val.json").read_text())
    with np.load(run / "source_split.npz") as split:
        centers = split["target_centers"].copy()
    assert centers.shape == (53200, 2)
    prior_path = (ROOT / "investigations/distribution_correction_validation/runs" /
                  f"correction_{a.seed}/predictions_before_gt.npz")
    with np.load(prior_path) as data:
        assert np.array_equal(centers, data["centers"])
        prior = data["prior"].copy()
    _, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    model = Backbone().to(a.device)
    model.load_state_dict(cp["model"])
    parameter_hash_before = tensor_hash(model.parameters())
    buffer_hash_before = tensor_hash(model.buffers())
    model.train()
    calibration_batches = forward_all(model, target, centers, a.device, drop_last=False)
    assert calibration_batches == 1663
    assert parameter_hash_before == tensor_hash(model.parameters())
    buffer_hash_after = tensor_hash(model.buffers())
    assert buffer_hash_before != buffer_hash_after
    model.eval()
    q = forward_all(model, target, centers, a.device, drop_last=True)
    assert q.shape == (53184, 7) and np.isfinite(q).all()
    corrected, solver = soft_project(q, prior, strength=1.0)
    out.mkdir()
    state_path = out / "calibrated_state.pth"
    torch.save({"model": model.state_dict(), "source_checkpoint_sha256": digest(cp_path),
                "calibration_batches": calibration_batches}, state_path)
    pred_path = out / "predictions_before_gt.npz"
    np.savez_compressed(pred_path, centers=centers, raw=q, corrected=corrected,
                        prior=prior)
    metadata = {
        "method": a.method, "seed": a.seed,
        "target_gt_opened": False,
        "source_checkpoint_sha256": digest(cp_path),
        "source_selected_epoch": int(cp["epoch"]),
        "source_split_sha256": digest(run / "source_split.npz"),
        "teacher_prior_sha256": digest(prior_path),
        "lock_sha256": digest(HERE / "EXPERIMENT_LOCK.md"),
        "parameter_hash_unchanged": parameter_hash_before == tensor_hash(model.parameters()),
        "bn_buffer_hash_before": buffer_hash_before,
        "bn_buffer_hash_after": buffer_hash_after,
        "calibration_batches": calibration_batches,
        "calibrated_state_sha256": digest(state_path),
        "prediction_sha256": digest(pred_path),
        "soft_kl": solver,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: metadata[k] for k in
                      ("method", "seed", "source_selected_epoch", "calibration_batches",
                       "parameter_hash_unchanged", "prediction_sha256")}), flush=True)


if __name__ == "__main__":
    main()
