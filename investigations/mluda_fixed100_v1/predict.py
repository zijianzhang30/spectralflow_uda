"""Freeze epoch-100 full-MLUDA predictions before reading Houston18 labels."""
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
OFFICIAL = PROJECT / "official_aligned"
sys.path.insert(0, str(OFFICIAL))
sys.path.insert(0, str(OFFICIAL / "reference"))
from data import Patches, load_images
from runtime import evaluation
from baseline_pool import install_deterministic_pool
from net2 import DSANSS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(1341, 1174, 1370), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = OFFICIAL / "runs/round1/formal" / str(args.seed) / "MLUDA_full"
    out = HERE / f"seed_{args.seed}_before_gt.npz"
    meta_path = HERE / f"seed_{args.seed}_freeze.json"
    assert not out.exists() and not meta_path.exists()
    cfg = json.loads((run / "config.json").read_text())
    complete = json.loads((run / "training_complete.json").read_text())
    assert cfg["seed"] == args.seed and cfg["method"] == "MLUDA_full"
    assert complete["epochs"] == 100
    checkpoint = torch.load(run / "last.pth", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 100
    with np.load(run / "source_split.npz") as split:
        centers = split["target_centers"].copy()
    assert centers.shape == (53200, 2)
    model = install_deterministic_pool(DSANSS(48, 7, 7).to(args.device))
    model.load_state_dict(checkpoint["model"])
    _, target = load_images(Path(cfg["data"]), "official_ilda", Path(cfg["ilda_cache"]))
    loader = DataLoader(Patches(target, centers), batch_size=32, shuffle=False,
                        drop_last=True, num_workers=0)
    source_reference = checkpoint["last_source_batch"].to(args.device)
    predictions = []
    with evaluation(model):
        with torch.inference_mode():
            for x in loader:
                predictions.append(model(source_reference, x.to(args.device))[8].argmax(1).cpu().numpy())
    pred = np.concatenate(predictions)
    assert pred.shape == (53184,) and np.isin(pred, np.arange(7)).all()
    np.savez_compressed(out, centers=centers, predictions=pred)
    meta = {"seed": args.seed, "selection": "fixed_epoch_100", "epoch": 100,
            "checkpoint_sha256": sha(run / "last.pth"),
            "source_centers_sha256": sha(run / "source_split.npz"),
            "prediction_sha256": sha(out), "target_gt_opened": False,
            "test_forward": "model(epoch100_last_source_batch, target)[8]",
            "evaluated_n": 53184, "official_denominator": 53200}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps({"seed": args.seed, "saved": str(out), "sha256": sha(out)}), flush=True)


if __name__ == "__main__":
    main()
