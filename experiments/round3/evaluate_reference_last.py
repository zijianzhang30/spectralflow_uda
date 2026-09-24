"""Read-only fixed-epoch-100 evaluation of round1's full MLUDA checkpoints."""
import argparse
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.utils.data import DataLoader

OFFICIAL = Path(__file__).resolve().parents[2] / "official_aligned"
sys.path.insert(0, str(OFFICIAL))
from full_mluda import DSANSS, install_deterministic_pool
from data import Patches, load_images, file_hash
from runtime import atomic_json, evaluation, scores, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = args.run.resolve()
    complete = json.loads((run / "training_complete.json").read_text())
    assert complete["epochs"] == 100
    config = json.loads((run / "config.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    for name, digest in provenance["code"].items():
        assert file_hash(OFFICIAL / name) == digest, name
    for name, digest in provenance["inputs"].items():
        assert file_hash(name) == digest, name
    history = json.loads((run / "history.json").read_text())
    assert len(history) == 100
    checkpoint = torch.load(run / "last.pth", map_location=args.device,
                            weights_only=False)
    assert checkpoint["epoch"] == 100 and checkpoint["metrics"] == history[-1]
    seed_everything(config["seed"])
    torch.set_num_threads(2)
    model = install_deterministic_pool(DSANSS(48, 7, 7).to(args.device))
    model.load_state_dict(checkpoint["model"])
    data = Path(config["data"])
    _, target = load_images(data, "official_ilda", Path(config["ilda_cache"]))
    gt_path = data / "Houston18_7gt.mat"
    gt = hdf5storage.loadmat(str(gt_path))["map"]
    with np.load(run / "source_split.npz") as split:
        centers = split["target_centers"]
    labels = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    loader = DataLoader(Patches(target, centers), batch_size=32, shuffle=False,
                        drop_last=True)
    reference = checkpoint["last_source_batch"].to(args.device)
    predictions = []
    with evaluation(model):
        for x in loader:
            predictions.extend(model(reference, x.to(args.device))[8].argmax(1).cpu().tolist())
    predictions = np.asarray(predictions)
    evaluated = labels[:len(predictions)]
    result = scores(evaluated, predictions)
    result.update(oa_evaluated=result["oa"],
                  oa=float((evaluated == predictions).sum()) / len(labels),
                  oa_definition="official: correct / full dataset, despite drop_last",
                  method="MLUDA_full", seed=config["seed"],
                  protocol="official_ilda", selection="fixed_epoch_100",
                  selected_epoch=100,
                  source_val_accuracy=checkpoint["metrics"]["source_val_accuracy"],
                  dataset_n=len(labels), dropped_n=len(labels)-len(predictions),
                  checkpoint_sha256=file_hash(run / "last.pth"),
                  target_gt_sha256=file_hash(gt_path))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, result)
    print(json.dumps({key: result[key] for key in
                      ("method", "seed", "oa", "aa", "kappa", "selection")}, indent=2))


if __name__ == "__main__":
    main()
