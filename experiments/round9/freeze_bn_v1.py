"""Freeze source-BN-only predictions before target-GT evaluation."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import Patches, file_hash, load_images, array_hash
from model import Backbone
from runtime import atomic_json, evaluation, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selection", choices=("source_val_best", "fixed_epoch_100"),
                        default="source_val_best")
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run/"config.json").read_text())
    complete = json.loads((run/"training_complete.json").read_text())
    assert cfg["method"] in ("A", "B", "C") and complete["formal"] and complete["epochs"]==100
    provenance = json.loads((run/"provenance.json").read_text())
    root = Path(__file__).resolve().parent
    for path, digest in provenance["code"].items():
        assert file_hash(root/Path(path).name) == digest, path
    data = Path(cfg["data"])
    for name, digest in provenance["inputs"].items():
        assert file_hash(data/name) == digest, name
    history = json.loads((run/"history.json").read_text())
    assert len(history)==100
    cp_name = "last.pth" if args.selection=="fixed_epoch_100" else "best_source_val.pth"
    cp = torch.load(run/cp_name, map_location=args.device, weights_only=False)
    assert cp["epoch"] == (100 if args.selection=="fixed_epoch_100" else
                           max(history, key=lambda row: row["source_val_accuracy"])["epoch"])
    seed_everything(cfg["seed"])
    torch.set_num_threads(2)
    model = Backbone().to(args.device)
    model.load_state_dict(cp["model"])
    _, target = load_images(data, cfg["protocol"], None)
    with np.load(run/"source_split.npz") as split:
        centers = split["target_centers"].copy()
    loader = DataLoader(Patches(target, centers), batch_size=32, shuffle=False,
                        drop_last=True)
    predictions = []
    with evaluation(model), torch.no_grad():
        for x in loader:
            _, logits = model(x.to(args.device))
            predictions.extend(logits.argmax(1).cpu().tolist())
    predictions = np.asarray(predictions)
    assert len(predictions)==53184 and len(centers)==53200
    np.savez_compressed(run/f"frozen_target_predictions_{args.selection}.npz",
                        predictions=predictions.astype(np.int8), centers=centers)
    result = dict(method=cfg["method"], seed=cfg["seed"], selection=args.selection,
                  selected_epoch=cp["epoch"], checkpoint_sha256=file_hash(run/cp_name),
                  center_hash=array_hash(centers), predictions_n=len(predictions),
                  dataset_n=len(centers), dropped_n=len(centers)-len(predictions))
    atomic_json(run/f"frozen_target_predictions_{args.selection}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
