"""Freeze matched MLUDA, SceneShift, prior-corrected and fused probabilities."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/round9"))
sys.path.insert(0, str(ROOT / "official_aligned"))
sys.path.insert(0, str(ROOT / "official_aligned/reference"))
sys.path.insert(0, str(ROOT / "investigations/distribution_correction_validation"))
from data import Patches, load_images
from model import Backbone
from net2 import DSANSS
from baseline_pool import install_deterministic_pool
from runtime import evaluation
from soft_projection import soft_project


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def infer(model, loader, device, reference=None):
    output = []
    with evaluation(model), torch.inference_mode():
        for x in loader:
            x = x.to(device)
            logits = model(x)[1] if reference is None else model(reference, x)[8]
            output.append(logits.softmax(1).cpu().numpy())
    q = np.concatenate(output)
    assert q.shape == (53184, 7) and np.allclose(q.sum(axis=1), 1, atol=1e-5)
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(1341, 1174, 1370), required=True)
    p.add_argument("--selection", choices=("fixed100", "source_val_best"), required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    out = HERE / f"seed_{args.seed}_{args.selection}_before_gt.npz"
    meta_path = HERE / f"seed_{args.seed}_{args.selection}_freeze.json"
    assert not out.exists() and not meta_path.exists()
    mrun = ROOT / "official_aligned/runs/round1/formal" / str(args.seed) / "MLUDA_full"
    srun = HERE / f"formal_{args.seed}"
    assert json.loads((srun / "training_complete.json").read_text())["epochs"] == 100
    mp = mrun / ("last.pth" if args.selection == "fixed100" else "best_source_val.pth")
    sp = srun / ("last.pth" if args.selection == "fixed100" else "best_source_val.pth")
    mcp = torch.load(mp, map_location="cpu", weights_only=False)
    scp = torch.load(sp, map_location="cpu", weights_only=False)
    if args.selection == "fixed100":
        assert mcp["epoch"] == scp["epoch"] == 100
    with np.load(mrun / "source_split.npz") as f:
        centers = f["target_centers"].copy()
    with np.load(srun / "source_split.npz") as f:
        assert np.array_equal(centers, f["target_centers"])
        with np.load(mrun / "source_split.npz") as m:
            for key in ("train_centers", "train_labels", "val_centers", "val_labels"):
                assert np.array_equal(f[key], m[key])
    prior_file = HERE / f"prior_{args.seed}_before_gt.npz"
    prior_meta = json.loads((HERE / f"prior_{args.seed}_freeze.json").read_text())
    assert prior_meta["target_gt_opened"] is False and prior_meta["prior_prediction_sha256"] == sha(prior_file)
    with np.load(prior_file) as f:
        assert np.array_equal(centers, f["centers"])
        prior = f["prior"].copy()
    cfg = json.loads((mrun / "config.json").read_text())
    _, target = load_images(Path(cfg["data"]), "official_ilda", Path(cfg["ilda_cache"]))
    loader = DataLoader(Patches(target, centers), batch_size=32, shuffle=False,
                        drop_last=True, num_workers=0)
    mluda = install_deterministic_pool(DSANSS(48, 7, 7).to(args.device))
    mluda.load_state_dict(mcp["model"])
    reference = mcp["last_source_batch"].to(args.device)
    mq = infer(mluda, loader, args.device, reference)
    del mluda
    torch.cuda.empty_cache()
    shift = Backbone().to(args.device)
    shift.load_state_dict(scp["model"])
    sq = infer(shift, loader, args.device)
    del shift
    torch.cuda.empty_cache()
    if args.selection == "fixed100":
        with np.load(ROOT / "investigations/mluda_fixed100_v1" / f"seed_{args.seed}_before_gt.npz") as f:
            assert np.array_equal(mq.argmax(axis=1), f["predictions"])
    mc, mi = soft_project(mq, prior, 1.0)
    sc, si = soft_project(sq, prior, 1.0)
    raw_fusion = ((mq.astype(np.float64) + sq.astype(np.float64)) / 2).astype(np.float32)
    corrected_fusion = ((mc.astype(np.float64) + sc.astype(np.float64)) / 2).astype(np.float32)
    np.savez_compressed(out, centers=centers, prior=prior, mluda_raw=mq,
                        mluda_corrected=mc, shift_raw=sq, shift_corrected=sc,
                        fusion_raw=raw_fusion, fusion_corrected=corrected_fusion)
    meta_path.write_text(json.dumps({"seed": args.seed, "selection": args.selection,
                                     "mluda_epoch": int(mcp["epoch"]), "shift_epoch": int(scp["epoch"]),
                                     "mluda_checkpoint_sha256": sha(mp),
                                     "shift_checkpoint_sha256": sha(sp),
                                     "prior_sha256": sha(prior_file),
                                     "prediction_sha256": sha(out),
                                     "soft_kl": {"mluda": mi, "shift": si},
                                     "target_gt_opened": False}, indent=2))
    print(json.dumps({"seed": args.seed, "selection": args.selection,
                      "mluda_epoch": mcp["epoch"], "shift_epoch": scp["epoch"],
                      "prediction_sha256": sha(out)}), flush=True)


if __name__ == "__main__":
    main()
