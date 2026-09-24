"""Freeze F1/F2 target predictions and fixed HyperSIGMA correction before GT."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "experiments/round9"))
sys.path.insert(0, str(PROJECT / "investigations/distribution_correction_validation"))
from data import Patches, load_images
from model import Backbone
from soft_projection import soft_project


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@torch.inference_mode()
def probabilities(model, image, centers, device):
    loader = DataLoader(Patches(image, centers), batch_size=32, shuffle=False,
                        drop_last=True, num_workers=0)
    result = []
    for x in loader:
        _, logits = model(x.to(device))
        result.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(202601, 202602, 202603), required=True)
    p.add_argument("--method", choices=("F1", "F2"), required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    run = HERE / "runs" / f"formal_{args.method}_{args.seed}"
    output = run / "predictions_before_gt.npz"
    assert not output.exists(), f"Refusing to overwrite {output}"
    cfg = json.loads((run / "config.json").read_text())
    complete = json.loads((run / "training_complete.json").read_text())
    assert cfg["seed"] == args.seed and cfg["method"] == args.method
    assert complete["formal"] and complete["epochs"] == 100
    cp_path = run / "best_source_val.pth"
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    with np.load(run / "source_split.npz") as split:
        centers = split["target_centers"].copy()
    prior_run = PROJECT / "investigations/distribution_correction_validation/runs" / f"correction_{args.seed}"
    with np.load(prior_run / "predictions_before_gt.npz") as data:
        assert np.array_equal(centers, data["centers"])
        prior = data["prior"].copy()
    model = Backbone().to(args.device).eval()
    model.load_state_dict(cp["model"])
    _, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    q = probabilities(model, target, centers, args.device)
    assert q.shape == (53184, 7)
    corrected, solver = soft_project(q, prior, strength=1.0)
    np.savez_compressed(output, centers=centers, raw=q,
                        corrected=corrected, prior=prior)
    (run / "prediction_metadata.json").write_text(json.dumps({
        "seed": args.seed, "method": args.method,
        "target_gt_opened": False,
        "selected_epoch": int(cp["epoch"]),
        "checkpoint_sha256": digest(cp_path),
        "teacher_prior_sha256": digest(prior_run / "predictions_before_gt.npz"),
        "lock_sha256": digest(HERE / "EXPERIMENT_LOCK.md"),
        "soft_kl": solver,
        "prediction_sha256": digest(output)}, indent=2))
    print(json.dumps({"seed": args.seed, "method": args.method,
                      "epoch": cp["epoch"],
                      "soft_kl_residual": solver["stationarity_max_abs_residual"]}), flush=True)


if __name__ == "__main__":
    main()
