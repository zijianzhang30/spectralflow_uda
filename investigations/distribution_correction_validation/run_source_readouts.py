"""Freeze source-trained readouts and target predictions before GT audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "experiments/round9"))
from data import Patches, load_images
from model import Backbone

TEMPERATURES = (0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
PROTO_TEMPERATURES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
RIDGE_PENALTIES = (0.01, 0.1, 1.0, 10.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


@torch.inference_mode()
def extract(model, cube, centers, device):
    loader = DataLoader(Patches(cube, centers), batch_size=64, shuffle=False,
                        drop_last=False, num_workers=0)
    zs, logits = [], []
    for x in loader:
        z, score = model(x.to(device))
        zs.append(z.cpu().numpy())
        logits.append(score.cpu().numpy())
    return np.concatenate(zs), np.concatenate(logits)


def macro_nll(prob, y):
    selected = prob[np.arange(len(y)), y].clip(min=1e-12)
    return float(np.mean([-np.log(selected[y == c]).mean() for c in range(7)]))


def choose_temperature(score_val, y, grid):
    rows = [(macro_nll(softmax(score_val / t, axis=1), y), t) for t in grid]
    return min(rows, key=lambda pair: pair[0]), rows


def ridge_scores(train, labels, val, target, alpha):
    x = normalize(train.astype(np.float64))
    v = normalize(val.astype(np.float64))
    t = normalize(target.astype(np.float64))
    hot = np.eye(7)[labels]
    xm, ym = x.mean(axis=0), hot.mean(axis=0)
    xc = x - xm
    weight = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]),
                             xc.T @ (hot - ym))
    bias = ym - xm @ weight
    return v @ weight + bias, t @ weight + bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(202601, 202602, 202603), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    student = HERE / "runs" / f"student_{args.seed}"
    out = HERE / "runs" / f"readout_{args.seed}"
    assert not out.exists(), f"Refusing to overwrite {out}"
    checkpoint = student / "best_source_val.pth"
    cfg = json.loads((student / "config.json").read_text())
    with np.load(student / "source_split.npz") as data:
        train_centers = data["train_centers"].copy()
        train_labels = data["train_labels"].copy()
        val_centers = data["val_centers"].copy()
        val_labels = data["val_labels"].copy()
        target_centers = data["target_centers"].copy()
    assert len(train_labels) == 1260 and all(np.bincount(train_labels, minlength=7) == 180)
    assert len(target_centers) == 53200
    model = Backbone().to(args.device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    source, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    train_z, _ = extract(model, source, train_centers, args.device)
    val_z, val_logits = extract(model, source, val_centers, args.device)
    target_z, target_logits = extract(model, target, target_centers[:53184], args.device)
    del model
    with np.load(HERE / "runs" / f"correction_{args.seed}" / "predictions_before_gt.npz") as data:
        assert np.array_equal(target_centers, data["centers"])
        original = data["A"].copy()
    recomputed = softmax(target_logits, axis=1)
    assert np.max(np.abs(original - recomputed)) < 1e-5
    candidates = {"original": original}
    choices = {}
    (score, temp), records = choose_temperature(val_logits, val_labels, TEMPERATURES)
    candidates["classifier_calibrated"] = softmax(target_logits / temp, axis=1).astype(np.float32)
    choices["classifier_calibrated"] = {"macro_nll": score, "temperature": temp,
                                       "source_val_grid": records}
    prototypes = np.stack([train_z[train_labels == c].mean(axis=0) for c in range(7)])
    val_cos = normalize(val_z) @ normalize(prototypes).T
    target_cos = normalize(target_z) @ normalize(prototypes).T
    (score, temp), records = choose_temperature(val_cos, val_labels, PROTO_TEMPERATURES)
    candidates["prototype"] = softmax(target_cos / temp, axis=1).astype(np.float32)
    choices["prototype"] = {"macro_nll": score, "temperature": temp,
                             "source_val_grid": records}
    ridge_rows = []
    ridge_models = []
    for alpha in RIDGE_PENALTIES:
        val_score, target_score = ridge_scores(train_z, train_labels, val_z, target_z, alpha)
        (score, temp), records = choose_temperature(val_score, val_labels, TEMPERATURES)
        ridge_rows.append({"alpha": alpha, "macro_nll": score,
                           "temperature": temp, "source_val_grid": records})
        ridge_models.append(softmax(target_score / temp, axis=1).astype(np.float32))
    best_ridge = int(np.argmin([row["macro_nll"] for row in ridge_rows]))
    candidates["ridge"] = ridge_models[best_ridge]
    choices["ridge"] = {**ridge_rows[best_ridge], "selected_grid_index": best_ridge,
                        "all_penalties": ridge_rows}
    selected = min(("classifier_calibrated", "prototype", "ridge"),
                   key=lambda name: choices[name]["macro_nll"])
    candidates["source_val_choice"] = candidates[selected].copy()
    assert all(q.shape == (53184, 7) and np.allclose(q.sum(axis=1), 1, atol=1e-5)
               for q in candidates.values())
    out.mkdir(parents=True)
    np.savez_compressed(out / "features_before_gt.npz", train_z=train_z, val_z=val_z,
                        target_z=target_z, train_labels=train_labels, val_labels=val_labels,
                        centers=target_centers)
    np.savez_compressed(out / "probabilities_before_gt.npz", centers=target_centers,
                        **candidates)
    metadata = {"seed": args.seed, "checkpoint_sha256": digest(checkpoint),
                "source_split_sha256": digest(student / "source_split.npz"),
                "lock_sha256": digest(HERE / "SOURCE_READOUT_LOCK.md"),
                "candidate_validation": choices, "source_val_choice": selected,
                "target_gt_opened": False,
                "feature_sha256": digest(out / "features_before_gt.npz"),
                "probability_sha256": digest(out / "probabilities_before_gt.npz")}
    (out / "selection_before_gt.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"seed": args.seed, "selected": selected,
                      "source_val_macro_nll": {name: row["macro_nll"] for name, row in choices.items()},
                      "saved": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
