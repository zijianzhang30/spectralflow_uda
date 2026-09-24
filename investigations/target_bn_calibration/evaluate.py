"""One predeclared, post-hoc unlabeled-target BN calibration diagnostic."""
import argparse
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/round3"))
from model import Backbone
from data import Patches, file_hash, load_images
from runtime import atomic_json, scores, seed_everything, tensor_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run / "config.json").read_text())
    done = json.loads((run / "training_complete.json").read_text())
    assert done["formal"] and done["epochs"] == 100
    original = json.loads((run / "final_target_fixed_epoch_100.json").read_text())
    checkpoint = torch.load(run / "last.pth", map_location=args.device,
                            weights_only=False)
    assert checkpoint["epoch"] == 100
    checkpoint_hash = file_hash(run / "last.pth")
    assert checkpoint_hash == original["checkpoint_sha256"]
    seed_everything(cfg["seed"])
    torch.set_num_threads(2)
    _, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    with np.load(run / "source_split.npz") as split:
        centers = split["target_centers"]
    # Calibration sees only patches and their official order, no target labels.
    loader = DataLoader(Patches(target, centers), batch_size=32, shuffle=False,
                        drop_last=True)
    model = Backbone().to(args.device)
    model.load_state_dict(checkpoint["model"])
    before = tensor_hash(model.parameters())
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.reset_running_stats()
            module.momentum = None
    model.train()
    with torch.no_grad():
        for x in loader:
            model(x.to(args.device))
    assert tensor_hash(model.parameters()) == before
    assert all(module.num_batches_tracked.item() == len(loader)
               for module in model.modules() if isinstance(module, nn.BatchNorm3d))

    # Open GT only after calibration has finished; it was not available to the
    # BN updates or the choice of cumulative estimator.
    gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
    gt = hdf5storage.loadmat(str(gt_path))["map"]
    labels = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    predictions = []
    model.eval()
    with torch.no_grad():
        for x in loader:
            predictions.extend(model(x.to(args.device))[1].argmax(1).cpu().tolist())
    predictions = np.asarray(predictions)
    evaluated = labels[:len(predictions)]
    result = scores(evaluated, predictions)
    result.update(oa_evaluated=result["oa"],
                  oa=float((evaluated == predictions).sum()) / len(labels),
                  oa_definition="official: correct / full dataset, despite drop_last",
                  method=cfg["method"], seed=cfg["seed"], epoch=100,
                  diagnostic="posthoc_target_only_cumulative_BN",
                  calibration_n=len(loader)*32,
                  calibration_batches=len(loader), calibration_labels_used=False,
                  model_parameters_unchanged=True,
                  original_oa=original["oa"],
                  delta_oa_pp=100*(float((evaluated == predictions).sum())/len(labels)
                                   - original["oa"]),
                  checkpoint_sha256=checkpoint_hash,
                  target_gt_sha256=file_hash(gt_path), dataset_n=len(labels),
                  dropped_n=len(labels)-len(predictions))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, result)
    assert file_hash(run / "last.pth") == checkpoint_hash
    print(json.dumps({key: result[key] for key in
                      ("method", "seed", "oa", "original_oa", "delta_oa_pp")}, indent=2))


if __name__ == "__main__":
    main()
