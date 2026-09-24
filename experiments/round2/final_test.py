"""Final target metrics; checkpoint selection uses source validation only."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
from pathlib import Path
import json
import hdf5storage
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import Backbone
from data import load_images, Patches, file_hash
from runtime import seed_everything, evaluation, scores, atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run = args.run.resolve()
    complete = json.loads((run/"training_complete.json").read_text())
    if complete["epochs"] != 100 or not complete["formal"]:
        raise ValueError("Final target testing requires completed 100-epoch training")
    config = json.loads((run/"config.json").read_text())
    provenance = json.loads((run/"provenance.json").read_text())
    root = Path(__file__).resolve().parent
    for path, digest in provenance["code"].items():
        assert file_hash(root/Path(path).name)==digest, "Training/evaluation code changed"
    data = Path(config["data"])
    cache = Path(config["ilda_cache"]) if config["ilda_cache"] else None
    for name,digest in provenance["inputs"].items():
        assert file_hash(cache if name=="ilda_cache" else data/name)==digest
    seed_everything(config["seed"])
    torch.set_num_threads(2)
    _, target = load_images(data, config["protocol"], cache)
    checkpoint = torch.load(run/"best_source_val.pth", map_location=args.device, weights_only=False)
    model = Backbone().to(args.device)
    model.load_state_dict(checkpoint["model"])
    selected = json.loads((run/"best_source_val.json").read_text())
    assert selected == checkpoint["metrics"]
    history = json.loads((run/"history.json").read_text())
    assert max(history, key=lambda x:x["source_val_accuracy"]) == selected
    gt_path = data/"Houston18_7gt.mat"
    gt = hdf5storage.loadmat(str(gt_path))["map"]
    with np.load(run/"source_split.npz") as split:
        centers = split["target_centers"]
    labels = gt[centers[:,0],centers[:,1]].astype(np.int64)-1
    loader = DataLoader(Patches(target, centers, labels), batch_size=32, shuffle=False,
                        drop_last=True, generator=torch.Generator().manual_seed(config["seed"]))
    predictions = []
    with evaluation(model):
        for x,_ in loader:
            predictions.extend(model(x.to(args.device))[1].argmax(1).cpu().tolist())
    evaluated_labels = labels[:len(predictions)]
    result = dict(**scores(evaluated_labels, np.asarray(predictions)), method=config["method"],
                  protocol=config["protocol"], seed=config["seed"], selection="source_val_best",
                  selected_epoch=checkpoint["epoch"], source_val_accuracy=selected["source_val_accuracy"],
                  checkpoint_sha256=file_hash(run/"best_source_val.pth"),target_gt_sha256=file_hash(gt_path))
    result["oa_evaluated"] = result["oa"]
    result["oa"] = float((evaluated_labels == np.asarray(predictions)).sum())/len(labels)
    result["oa_definition"] = "official: correct / full dataset, despite drop_last"
    result["dataset_n"] = len(labels)
    result["dropped_n"] = len(labels)-len(predictions)
    atomic_json(run/"final_target.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
