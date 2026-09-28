"""Freeze official full-MLUDA target probabilities and fixed soft-prior correction."""
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
sys.path.insert(0, str(ROOT / "official_aligned"))
sys.path.insert(0, str(ROOT / "official_aligned/reference"))
sys.path.insert(0, str(ROOT / "investigations/distribution_correction_validation"))
from baseline_pool import install_deterministic_pool
from data import Patches, load_images
from net2 import DSANSS
from runtime import evaluation, seed_everything
from soft_projection import soft_project


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(202601, 202602, 202603), required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.set_num_threads(2)
    seed_everything(a.seed)
    run = ROOT / "investigations/distribution_correction_validation/runs" / f"mluda_{a.seed}"
    out = HERE / f"seed_{a.seed}"
    assert not out.exists(), f"Refusing to overwrite {out}"
    cfg = json.loads((run / "config.json").read_text())
    complete = json.loads((run / "training_complete.json").read_text())
    assert cfg["seed"] == a.seed and cfg["method"] == "MLUDA_full"
    assert complete["epochs"] == 100 and cfg["selection"] == "source_val_best"
    cp_path = run / "best_source_val.pth"
    checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
    best = json.loads((run / "best_source_val.json").read_text())
    assert checkpoint["metrics"] == best
    model = install_deterministic_pool(DSANSS(48, 7, 7).to(a.device))
    model.load_state_dict(checkpoint["model"])
    _, target = load_images(Path(cfg["data"]), "official_ilda", Path(cfg["ilda_cache"]))
    with np.load(run / "source_split.npz") as split:
        centers = split["target_centers"].copy()
    prior_run = ROOT / "investigations/distribution_correction_validation/runs" / f"correction_{a.seed}"
    prior_path = prior_run / "predictions_before_gt.npz"
    with np.load(prior_path) as data:
        assert np.array_equal(centers, data["centers"])
        prior = data["prior"].copy()
    loader = DataLoader(Patches(target, centers), batch_size=32, shuffle=False,
                        drop_last=True, num_workers=0)
    reference = checkpoint["last_source_batch"].to(a.device)
    probs = []
    with evaluation(model), torch.inference_mode():
        for x in loader:
            probs.append(model(reference, x.to(a.device))[8].softmax(1).cpu().numpy())
    q = np.concatenate(probs)
    assert q.shape == (53184, 7) and np.isfinite(q).all()
    corrected, solver = soft_project(q, prior, strength=1.0)
    out.mkdir()
    output = out / "predictions_before_gt.npz"
    np.savez_compressed(output, centers=centers, raw=q,
                        corrected=corrected, prior=prior)
    (out / "metadata.json").write_text(json.dumps({
        "seed": a.seed, "selected_epoch": checkpoint["epoch"],
        "checkpoint_sha256": digest(cp_path),
        "teacher_prior_sha256": digest(prior_path),
        "source_split_sha256": digest(run / "source_split.npz"),
        "lock_sha256": digest(HERE / "EXPERIMENT_LOCK.md"),
        "prediction_sha256": digest(output),
        "target_gt_opened": False, "soft_kl": solver,
    }, indent=2))
    print(json.dumps({"seed": a.seed, "epoch": checkpoint["epoch"],
                      "residual": solver["stationarity_max_abs_residual"]}))


if __name__ == "__main__":
    main()
