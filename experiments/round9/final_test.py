"""Official target evaluation of the selected student, without Flow."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import Patches, file_hash, load_images
from model import Backbone
from runtime import atomic_json, evaluation, scores, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selection", choices=("fixed_epoch_100", "source_val_best"),
                        default="fixed_epoch_100")
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run/"config.json").read_text())
    complete = json.loads((run/"training_complete.json").read_text())
    assert cfg["method"] in ("A", "B", "C") and complete["formal"] and complete["epochs"]==100
    provenance = json.loads((run/"provenance.json").read_text())
    root = Path(__file__).resolve().parent
    for path, digest in provenance["code"].items():
        assert file_hash(root/Path(path).name) == digest, path
    data, cache = Path(cfg["data"]), Path(cfg["ilda_cache"])
    for name, digest in provenance["inputs"].items():
        assert file_hash(cache if name=="ilda_cache" else data/name) == digest, name
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
    _, target = load_images(data, cfg["protocol"], cache)
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
    gt_path = data/"Houston18_7gt.mat"
    gt = hdf5storage.loadmat(str(gt_path))["map"]
    labels = gt[centers[:, 0], centers[:, 1]].astype(np.int64)-1
    assert len(predictions)==53184 and len(labels)==53200
    evaluated = labels[:len(predictions)]
    metrics = scores(evaluated, predictions)
    metrics["oa_evaluated"] = metrics["oa"]
    metrics["oa"] = float((evaluated==predictions).sum())/len(labels)
    result = dict(method=cfg["method"], seed=cfg["seed"], selection=args.selection,
                  selected_epoch=cp["epoch"], metrics=metrics,
                  checkpoint_sha256=file_hash(run/cp_name),
                  target_gt_sha256=file_hash(gt_path), dataset_n=len(labels),
                  dropped_n=len(labels)-len(predictions))
    if cfg["method"]=="A":
        baseline = root.parent/"round3/runs"/f"formal_A_{cfg['seed']}"/f"final_target_{args.selection}.json"
        if baseline.exists():
            expected = json.loads(baseline.read_text())
            assert result["metrics"]["confusion_matrix"] == expected["confusion_matrix"]
            assert result["metrics"]["oa"] == expected["oa"]
            result["matches_round3_A"] = True
    atomic_json(run/f"final_target_{args.selection}.json", result)
    print(json.dumps({"method": cfg["method"], "seed": cfg["seed"],
                      "selection": args.selection, "epoch": cp["epoch"],
                      "OA": metrics["oa"], "AA": metrics["aa"],
                      "class7_accuracy": metrics["per_class_accuracy"][6],
                      "matches_round3_A": result.get("matches_round3_A")}, indent=2))


if __name__ == "__main__":
    main()
