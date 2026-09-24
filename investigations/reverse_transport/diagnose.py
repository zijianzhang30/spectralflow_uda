"""Read-only reverse integration of an existing source-to-target Flow field."""
import argparse
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/round3"))
from model import Backbone
from flow import Flow
from data import Patches, load_images, file_hash
from runtime import atomic_json, scores, seed_everything


@torch.no_grad()
def reverse_predict(model, flow, x, steps=4):
    features, raw_logits = model(x)
    class_condition = raw_logits.argmax(dim=1)
    state = features
    for step in range(steps, 0, -1):
        midpoint = (step - .5) / steps
        t = torch.full((len(x), 1), midpoint, device=x.device,
                       dtype=features.dtype)
        state = state - flow(state, t, class_condition) / steps
    return raw_logits.argmax(dim=1), model.classifier(state).argmax(dim=1), \
        (state-features).norm(dim=1).mean(), features.norm(dim=1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[1341, 1174, 1370], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = REPO / f"experiments/round3/runs/formal_flow_cpu_{args.seed}"
    complete = json.loads((run / "training_complete.json").read_text())
    assert complete["formal"] and complete["epochs"] == 100
    config = json.loads((run / "config.json").read_text())
    original = json.loads((run / "final_target_fixed_epoch_100.json").read_text())
    assert original["selection"] == "fixed_epoch_100"
    checkpoint = torch.load(run / "last.pth", map_location=args.device,
                            weights_only=False)
    assert checkpoint["epoch"] == 100 and checkpoint["flow"] is not None
    assert file_hash(run / "last.pth") == original["checkpoint_sha256"]
    seed_everything(args.seed)
    torch.set_num_threads(2)
    model = Backbone().to(args.device).eval()
    flow = Flow().to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    flow.load_state_dict(checkpoint["flow"])
    _, target = load_images(Path(config["data"]), config["protocol"],
                            Path(config["ilda_cache"]))
    with np.load(run / "source_split.npz") as split:
        centers = split["target_centers"]
    loader = DataLoader(Patches(target, centers), batch_size=32, shuffle=False,
                        drop_last=True)
    raw, inverse, shifts, norms = [], [], [], []
    with torch.no_grad():
        for x in loader:
            a, b, shift, norm = reverse_predict(model, flow, x.to(args.device))
            raw.extend(a.cpu().tolist())
            inverse.extend(b.cpu().tolist())
            shifts.append(float(shift))
            norms.append(float(norm))
    raw, inverse = np.asarray(raw), np.asarray(inverse)
    gt_path = Path(config["data"]) / "Houston18_7gt.mat"
    gt = hdf5storage.loadmat(str(gt_path))["map"]
    labels = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    y = labels[:len(raw)]
    raw_scores, inverse_scores = scores(y, raw), scores(y, inverse)
    raw_scores["oa"] = float((raw == y).sum()) / len(labels)
    inverse_scores["oa"] = float((inverse == y).sum()) / len(labels)
    assert abs(raw_scores["oa"] - original["oa"]) < 1e-12
    assert raw_scores["confusion_matrix"] == original["confusion_matrix"]
    result = dict(seed=args.seed, selection="fixed_epoch_100", steps=4,
                  class_condition="raw target argmax, held fixed across steps",
                  target_labels_used_for_condition=False,
                  checkpoint_sha256=original["checkpoint_sha256"],
                  raw=raw_scores, reverse=inverse_scores,
                  delta_oa_pp=100*(inverse_scores["oa"]-raw_scores["oa"]),
                  predictions_changed=int((raw != inverse).sum()),
                  mean_shift_l2=float(np.mean(shifts)),
                  mean_feature_l2=float(np.mean(norms)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, result)
    print(json.dumps({key: result[key] for key in
                      ("seed", "delta_oa_pp", "predictions_changed",
                       "mean_shift_l2", "mean_feature_l2")}, indent=2))


if __name__ == "__main__":
    main()
