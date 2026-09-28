"""Frozen-weight, source-only class-mixture BN diagnostic."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/round9"))
from data import Patches
from model import Backbone
from runtime import tensor_hash


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def patches(cube, centers, device):
    ds = Patches(cube, centers)
    return torch.stack([ds[i] for i in range(len(ds))]).to(device)


@torch.inference_mode()
def predict(model, x):
    model.eval()
    return torch.cat([model(batch)[1] for batch in x.split(32)]).cpu().numpy()


def metrics(logits, labels):
    pred = logits.argmax(1)
    other = logits.copy()
    other[np.arange(len(labels)), labels] = -np.inf
    margin = logits[np.arange(len(labels)), labels] - other.max(1)
    confusion = np.zeros((7, 7), dtype=np.int64)
    np.add.at(confusion, (labels, pred), 1)
    return {
        "accuracy": float((pred == labels).mean()),
        "class_recall": (confusion.diagonal() / confusion.sum(1)).tolist(),
        "class_mean_true_margin": [float(margin[labels == c].mean()) for c in range(7)],
        "predicted_class_counts": np.bincount(pred, minlength=7).tolist(),
        "confusion": confusion.tolist(),
    }


def draw_indices(labels, condition, draw_seed):
    rng = np.random.RandomState(draw_seed)
    majority = {"class1_majority": 0, "class6_majority": 5,
                "class7_majority": 6}.get(condition)
    indices = []
    for c in range(7):
        pool = np.flatnonzero(labels == c)
        assert len(pool) == 180
        count = 180 if majority is None else (630 if c == majority else 105)
        indices.append(rng.choice(pool, size=count, replace=count > len(pool)))
    result = np.concatenate(indices)
    rng.shuffle(result)
    assert len(result) == 1260
    counts = np.bincount(labels[result], minlength=7)
    assert np.array_equal(counts, [180] * 7 if majority is None else
                          [630 if c == majority else 105 for c in range(7)])
    return result, counts.tolist()


@torch.inference_mode()
def calibrate(model, x, indices):
    model.train()
    selected = x[torch.as_tensor(indices, device=x.device)]
    for batch in selected.split(32):
        model(batch)
    model.eval()
    return (len(selected) + 31) // 32


def bn_displacement(original, calibrated):
    result = {}
    before = dict(original.named_buffers())
    after = dict(calibrated.named_buffers())
    for name in before:
        if not (name.endswith("running_mean") or name.endswith("running_var")):
            continue
        a, b = before[name].float(), after[name].float()
        result[name] = {
            "l2": float(torch.linalg.vector_norm(b - a).item()),
            "relative_l2": float((torch.linalg.vector_norm(b - a) /
                                  torch.linalg.vector_norm(a).clamp_min(1e-12)).item()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--out", type=Path, default=HERE / "SOURCE_MIXTURE.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    results = {
        "scope": "Source-only frozen-weight mechanistic diagnostic; no target samples or labels",
        "lock_sha256": digest(HERE / "EXPERIMENT_LOCK.md"),
        "code_sha256": digest(__file__),
        "seeds": {},
    }
    for seed in (202601, 202602, 202603):
        run = ROOT / f"investigations/distribution_correction_validation/runs/student_{seed}"
        cfg = json.loads((run / "config.json").read_text())
        assert cfg["method"] == "A" and cfg["protocol"] == "official_ilda"
        with np.load(run / "source_split.npz") as split:
            train_centers = split["train_centers"].copy()
            train_labels = split["train_labels"].copy()
            val_centers = split["val_centers"].copy()
            val_labels = split["val_labels"].copy()
        assert len(train_centers) == 1260 and len(val_centers) == 1270
        with np.load(Path(cfg["ilda_cache"])) as cache:
            source = cache["s"].astype(np.float32)
        x_train = patches(source, train_centers, args.device)
        x_val = patches(source, val_centers, args.device)
        checkpoint_path = run / "best_source_val.pth"
        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        original = Backbone().to(args.device).eval().requires_grad_(False)
        original.load_state_dict(cp["model"])
        parameter_hash = tensor_hash(original.parameters())
        baseline = metrics(predict(original, x_val), val_labels)
        seed_result = {
            "checkpoint_sha256": digest(checkpoint_path),
            "split_sha256": digest(run / "source_split.npz"),
            "selected_epoch": int(cp["epoch"]),
            "original": baseline,
            "conditions": {},
        }
        for condition in ("balanced", "class1_majority", "class6_majority", "class7_majority"):
            rows = []
            for repetition in range(3):
                draw_seed = seed * 100 + repetition
                indices, counts = draw_indices(train_labels, condition, draw_seed)
                model = copy.deepcopy(original)
                batches = calibrate(model, x_train, indices)
                assert batches == 40 and tensor_hash(model.parameters()) == parameter_hash
                row = {
                    "draw_seed": draw_seed,
                    "sample_counts_by_class": counts,
                    "sample_index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                    "parameter_hash_unchanged": True,
                    "calibration_batches": batches,
                    "metrics": metrics(predict(model, x_val), val_labels),
                    "bn_displacement": bn_displacement(original, model),
                }
                rows.append(row)
            seed_result["conditions"][condition] = rows
            print(seed, condition,
                  [round(r["metrics"]["accuracy"], 4) for r in rows], flush=True)
        results["seeds"][str(seed)] = seed_result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print("saved", args.out, digest(args.out), flush=True)


if __name__ == "__main__":
    main()
