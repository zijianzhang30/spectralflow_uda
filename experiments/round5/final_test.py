"""Final target test of class-agnostic reverse Flow versus exact raw A logits."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import Backbone
from flow import ReverseFlow, transport_target
from data import Patches, file_hash, load_images
from runtime import atomic_json, evaluation, scores, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selection", choices=["fixed_epoch_100", "source_val_best"],
                        default="fixed_epoch_100")
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run / "config.json").read_text())
    complete = json.loads((run / "training_complete.json").read_text())
    assert cfg["method"] == "reverse_flow"
    assert complete["formal"] and complete["epochs"] == 100
    provenance = json.loads((run / "provenance.json").read_text())
    root = Path(__file__).resolve().parent
    for path, digest in provenance["code"].items():
        assert file_hash(root / Path(path).name) == digest, path
    data = Path(cfg["data"])
    cache = Path(cfg["ilda_cache"])
    for name, digest in provenance["inputs"].items():
        assert file_hash(cache if name == "ilda_cache" else data/name) == digest
    history = json.loads((run / "history.json").read_text())
    assert len(history) == 100
    cp_name = "last.pth" if args.selection == "fixed_epoch_100" else "best_source_val.pth"
    cp = torch.load(run / cp_name, map_location=args.device, weights_only=False)
    assert cp["epoch"] == (100 if args.selection == "fixed_epoch_100" else
                           max(history, key=lambda r: r["source_val_accuracy"])["epoch"])
    seed_everything(cfg["seed"])
    torch.set_num_threads(2)
    model = Backbone().to(args.device)
    flow = ReverseFlow().to(args.device)
    model.load_state_dict(cp["model"])
    flow.load_state_dict(cp["flow"])
    _, target = load_images(data, cfg["protocol"], cache)
    with np.load(run / "source_split.npz") as split:
        centers = split["target_centers"]
    loader = DataLoader(Patches(target, centers), batch_size=32, shuffle=False,
                        drop_last=True)
    raw, moved = [], []
    with evaluation(model), torch.no_grad():
        flow.eval()
        for x in loader:
            z, logits = model(x.to(args.device))
            raw.extend(logits.argmax(1).cpu().tolist())
            moved.extend(model.classifier(transport_target(flow, z, steps=4))
                         .argmax(1).cpu().tolist())
    raw, moved = np.asarray(raw), np.asarray(moved)
    gt_path = data / "Houston18_7gt.mat"
    gt = hdf5storage.loadmat(str(gt_path))["map"]
    labels = gt[centers[:, 0], centers[:, 1]].astype(np.int64)-1
    evaluated = labels[:len(raw)]
    assert len(raw) == len(moved) == 53184 and len(labels) == 53200

    def metric(pred):
        row = scores(evaluated, pred)
        row["oa_evaluated"] = row["oa"]
        row["oa"] = float((evaluated == pred).sum())/len(labels)
        return row

    result = dict(method="reverse_flow", seed=cfg["seed"],
                  selection=args.selection, selected_epoch=cp["epoch"],
                  integration="class-agnostic target-to-source, 4 Euler steps",
                  raw=metric(raw), transported=metric(moved),
                  predictions_changed=int((raw != moved).sum()),
                  checkpoint_sha256=file_hash(run / cp_name),
                  target_gt_sha256=file_hash(gt_path), dataset_n=len(labels),
                  dropped_n=len(labels)-len(raw))
    result["delta_oa_pp"] = 100*(result["transported"]["oa"]-result["raw"]["oa"])
    baseline = root.parent / "round3/runs" / f"formal_A_{cfg['seed']}" / \
        f"final_target_{args.selection}.json"
    if baseline.exists():
        expected = json.loads(baseline.read_text())
        assert result["raw"]["confusion_matrix"] == expected["confusion_matrix"]
        assert result["raw"]["oa"] == expected["oa"]
        result["raw_matches_round3_A"] = True
    atomic_json(run / f"final_target_{args.selection}.json", result)
    print(json.dumps({key: result[key] for key in
                      ("seed", "selection", "selected_epoch", "delta_oa_pp",
                       "predictions_changed", "raw_matches_round3_A")}, indent=2))


if __name__ == "__main__":
    main()
