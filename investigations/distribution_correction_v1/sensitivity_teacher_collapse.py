"""Apply the frozen correction rule to Round-9 Mean Teacher B at epoch 100.

This is a sensitivity check, not part of the primary CE-control A/B/C screen.
The prior estimator, KL projection, confidence cutoff, and target order are
unchanged. No training or target-label fitting occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.utils.data import DataLoader

from run import HERE, PROJECT, Backbone, Patches, load_images, metric, project_kl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(1341, 1174, 1370), required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    run = PROJECT / "experiments/round9/runs" / f"formal_B_{args.seed}"
    cfg = json.loads((run / "config.json").read_text())
    cp = torch.load(run / "last.pth", map_location="cpu", weights_only=False)
    assert cp["epoch"] == 100
    model = Backbone().to(args.device).eval()
    model.load_state_dict(cp["model"])
    with np.load(run / "source_split.npz") as f:
        centers = f["target_centers"][:53184].copy()
    foundation_cache = (PROJECT / "investigations/hypersigma_teacher_gate/runs" /
                        f"seed_{args.seed}/target_probabilities.npz")
    with np.load(foundation_cache) as f:
        assert np.array_equal(centers, f["centers"])
        prior = f["teacher"].astype(np.float64).mean(axis=0)
    _, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    loader = DataLoader(Patches(target, centers), batch_size=32, shuffle=False, drop_last=True)
    q = []
    with torch.inference_mode():
        for x in loader:
            _, logits = model(x.to(args.device))
            q.append(torch.softmax(logits, 1).cpu().numpy())
    q = np.concatenate(q).astype(np.float32)
    assert q.shape == (53184, 7)
    b, projection = project_kl(q, prior)
    c = b.copy()
    low = q.max(1) < 0.8
    c[low] = q[low]
    out = HERE / "runs" / f"seed_{args.seed}"
    np.savez_compressed(out / "mean_teacher_fixed100_sensitivity_probabilities.npz",
                        centers=centers, A=q, B=b, C=c, prior=prior)

    gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
    y = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    variants = {name: metric(y, x) for name, x in (("A", q), ("B", b), ("C", c))}
    official = json.loads((run / "final_target_fixed_epoch_100.json").read_text())
    assert variants["A"]["confusion_matrix"] == official["metrics"]["confusion_matrix"]
    result = {"seed": args.seed, "selection": "round9 Mean Teacher B fixed epoch 100",
              "prior": prior.tolist(), "projection": projection,
              "confidence_cutoff": 0.8, "low_confidence_fraction": float(low.mean()),
              "target_gt_use": "post-hoc metrics only; corrections saved before GT read",
              "variants": variants}
    (out / "mean_teacher_fixed100_sensitivity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"seed": args.seed,
                      "results": {name: {"oa": m["oa_official"], "aa": m["aa"],
                                         "class7_recall": m["recall"][6],
                                         "class7_soft_mass": m["soft_mass_fraction"][6]}
                                  for name, m in variants.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
