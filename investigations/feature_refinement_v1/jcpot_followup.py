"""Frozen-feature JCPOT coupling check for the predeclared F1/F2 runs."""
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
sys.path.insert(0, str(PROJECT / "investigations/jcpot_pilot"))
from data import Patches, load_images
from model import Backbone
from run import normalize, solve


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@torch.inference_mode()
def features(model, cube, centers, device):
    loader = DataLoader(Patches(cube, centers), batch_size=64, shuffle=False,
                        drop_last=False, num_workers=0)
    result = []
    for x in loader:
        z, _ = model(x.to(device))
        result.append(z.cpu().numpy())
    return normalize(np.concatenate(result))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(202601, 202602, 202603), required=True)
    p.add_argument("--method", choices=("F1", "F2"), required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    run = HERE / "runs" / f"formal_{args.method}_{args.seed}"
    output = run / "jcpot_before_gt.npz"
    assert not output.exists(), f"Refusing to overwrite {output}"
    cfg = json.loads((run / "config.json").read_text())
    cp_path = run / "best_source_val.pth"
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    with np.load(run / "source_split.npz") as split:
        source_centers = split["train_centers"].copy()
        source_labels = split["train_labels"].copy()
        target_centers = split["target_centers"][:8192].copy()
    source, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    model = Backbone().to(args.device).eval()
    model.load_state_dict(cp["model"])
    zs = features(model, source, source_centers, args.device)
    zt = features(model, target, target_centers, args.device)
    arrays = {"centers": target_centers, "source_labels": source_labels}
    records = {}
    for reg in (0.1, 0.2):
        tag = str(reg).replace(".", "p")
        prior, gamma, posterior, record = solve(zs, source_labels, zt, reg)
        arrays[f"prior_{tag}"] = prior
        arrays[f"gamma_{tag}"] = gamma
        arrays[f"posterior_{tag}"] = posterior
        records[tag] = record
    np.savez_compressed(output, **arrays)
    (run / "jcpot_fit.json").write_text(json.dumps({
        "method": args.method, "seed": args.seed, "target_gt_opened_by_fit": False,
        "checkpoint_sha256": digest(cp_path),
        "prediction_sha256": digest(run / "predictions_before_gt.npz"),
        "lock_sha256": digest(HERE / "EXPERIMENT_LOCK.md"),
        "fit": records, "output_sha256": digest(output)}, indent=2))
    print(json.dumps({"method": args.method, "seed": args.seed,
                      "priors": {tag: rec["estimated_prior"] for tag, rec in records.items()}}), flush=True)


if __name__ == "__main__":
    main()
