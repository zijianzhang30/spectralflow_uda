"""Post-hoc FM endpoint and classifier audit on fixed sampled OT pairs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/round9"))
from model import Backbone
from flow import ReverseFlow, ot_pairs, transport_target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.set_num_threads(2)
    out = {"scope": "Post-hoc 50 fixed random batches per seed; not training replay",
           "seeds": {}}
    for seed in (202601, 202602, 202603):
        run = HERE / f"formal_{seed}"
        shift = ROOT / "investigations/scene_shift_v1" / f"formal_{seed}"
        with np.load(run / "frozen_features_before_gt.npz") as data:
            sz = data["source_z"].copy()
            sy = data["source_labels"].copy()
            tz = data["target_z"].copy()
            tq = data["target_q"].copy()
            centers = data["target_centers"].copy()
        cfg = json.loads((shift / "config.json").read_text())
        gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
        ty = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
        model = Backbone().to(a.device).eval().requires_grad_(False)
        model.load_state_dict(torch.load(shift / "best_source_val.pth", map_location="cpu",
                                         weights_only=False)["model"])
        flow = ReverseFlow().to(a.device).eval().requires_grad_(False)
        flow.load_state_dict(torch.load(run / "last.pth", map_location="cpu",
                                        weights_only=False)["flow"])
        rng = np.random.RandomState(seed + 87031)
        sampler = torch.Generator(device="cpu").manual_seed(seed + 99173)
        sums = {k: 0.0 for k in ("pair_true_match", "source_endpoint_label_correct",
                                   "start_source_label_correct", "moved_source_label_correct",
                                   "start_target_gt_correct", "moved_target_gt_correct",
                                   "start_to_endpoint_l2", "moved_to_endpoint_l2",
                                   "velocity_residual_l2", "velocity_target_l2")}
        count = 0
        class6 = {"n": 0, "source_endpoint_correct": 0, "moved_source_label_correct": 0}
        with torch.inference_mode():
            for _ in range(50):
                source_batch = rng.choice(len(sz), 32, replace=False)
                target_batch = rng.choice(len(tz), 32, replace=False)
                si, ti = ot_pairs(torch.from_numpy(sz[source_batch]),
                                  torch.from_numpy(tz[target_batch]),
                                  torch.from_numpy(sy[source_batch]),
                                  torch.from_numpy(tq[target_batch]), sampler)
                if not len(si):
                    continue
                source_ids = source_batch[si.numpy()]
                target_ids = target_batch[ti.numpy()]
                start = torch.from_numpy(tz[target_ids]).to(a.device)
                endpoint = torch.from_numpy(sz[source_ids]).to(a.device)
                moved = transport_target(flow, start, steps=4)
                midpoint = .5 * (start + endpoint)
                t = torch.full((len(si), 1), .5, device=a.device)
                velocity = flow(midpoint, t)
                labels = torch.from_numpy(sy[source_ids]).to(a.device)
                truth = torch.from_numpy(ty[target_ids]).to(a.device)
                pred_source = model.classifier(endpoint).argmax(1)
                pred_start = model.classifier(start).argmax(1)
                pred_moved = model.classifier(moved).argmax(1)
                n = len(si)
                sums["pair_true_match"] += float((labels == truth).sum())
                sums["source_endpoint_label_correct"] += float((pred_source == labels).sum())
                sums["start_source_label_correct"] += float((pred_start == labels).sum())
                sums["moved_source_label_correct"] += float((pred_moved == labels).sum())
                sums["start_target_gt_correct"] += float((pred_start == truth).sum())
                sums["moved_target_gt_correct"] += float((pred_moved == truth).sum())
                sums["start_to_endpoint_l2"] += float((start - endpoint).norm(dim=1).sum())
                sums["moved_to_endpoint_l2"] += float((moved - endpoint).norm(dim=1).sum())
                sums["velocity_residual_l2"] += float((velocity - (endpoint-start)).norm(dim=1).sum())
                sums["velocity_target_l2"] += float((endpoint-start).norm(dim=1).sum())
                mask6 = labels == 5
                class6["n"] += int(mask6.sum())
                class6["source_endpoint_correct"] += int(((pred_source == labels) & mask6).sum())
                class6["moved_source_label_correct"] += int(((pred_moved == labels) & mask6).sum())
                count += n
        row = {k: v / count for k, v in sums.items()}
        row["pairs"] = count
        row["class6"] = {"pairs": class6["n"],
                          "source_endpoint_correct":
                          class6["source_endpoint_correct"] / class6["n"],
                          "moved_source_label_correct":
                          class6["moved_source_label_correct"] / class6["n"]}
        out["seeds"][str(seed)] = row
    (HERE / "ROLLOUT_AUDIT.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
