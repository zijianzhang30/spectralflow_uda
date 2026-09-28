"""Freeze five-way spectral-shift diagnostic predictions before target GT."""
from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/round9"))
from data import Patches
from model import Backbone
from runtime import seed_everything, tensor_hash


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def shifts(source, target, train_centers, train_labels, target_centers):
    xs = source[train_centers[:, 0], train_centers[:, 1]].astype(np.float64)
    xt = target[target_centers[:, 0], target_centers[:, 1]].astype(np.float64)
    assert xs.shape == (1260, 48) and xt.shape == (53200, 48)
    assert np.array_equal(np.bincount(train_labels, minlength=7), [180] * 7)
    mu_s, mu_t = xs.mean(0), xt.mean(0)
    sigma = np.maximum(xs.std(0), 1e-6)
    class_means = np.stack([xs[train_labels == c].mean(0) for c in range(7)])
    basis = ((class_means[:6] - class_means[6]) / sigma).T
    left, singular, _ = np.linalg.svd(basis, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-8))
    assert rank == 6, singular
    delta = mu_t - mu_s
    standardized = delta / sigma
    parallel = sigma * (left[:, :rank] @ (left[:, :rank].T @ standardized))
    orthogonal = delta - parallel
    assert np.allclose(delta, parallel + orthogonal, atol=1e-10)
    assert np.max(np.abs(left[:, :rank].T @ (orthogonal / sigma))) < 1e-10
    summary = {
        "source_train_centers": len(xs), "target_unlabeled_centers": len(xt),
        "class_subspace_rank": rank, "class_subspace_singular_values": singular.tolist(),
        "source_sigma_min": float(sigma.min()),
        "standardized_l2": {
            "global": float(np.linalg.norm(standardized)),
            "orthogonal": float(np.linalg.norm(orthogonal / sigma)),
            "parallel": float(np.linalg.norm(parallel / sigma)),
        },
        "standardized_orthogonal_fraction_of_global": float(
            np.linalg.norm(orthogonal / sigma) / np.linalg.norm(standardized)),
        "shift_vectors": {"global": delta.tolist(),
                          "orthogonal": orthogonal.tolist(),
                          "parallel": parallel.tolist()},
    }
    return {"original": np.zeros(48), "global": delta,
            "orthogonal": orthogonal, "parallel": parallel}, summary


def cache_patches(target, centers, device):
    loader = DataLoader(Patches(target, centers), batch_size=256,
                        shuffle=False, drop_last=False, num_workers=0)
    x = torch.empty((len(centers), 48, 7, 7), dtype=torch.float32, device=device)
    cursor = 0
    for batch in loader:
        x[cursor:cursor + len(batch)] = batch.to(device)
        cursor += len(batch)
    assert cursor == 53200
    return x


@torch.inference_mode()
def predict(model, x, shift, device):
    adjustment = torch.as_tensor(shift, dtype=x.dtype, device=device).view(1, 48, 1, 1)
    output = []
    for start in range(0, 53184, 32):
        _, logits = model(x[start:start + 32] - adjustment)
        output.append(logits.softmax(1).cpu())
    return torch.cat(output).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:5")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cache_path = ROOT / "official_aligned/preprocessing/official_ilda.npz"
    with np.load(cache_path) as cache:
        source, target = cache["s"].astype(np.float32), cache["t"].astype(np.float32)
    assert source.shape == target.shape == (210, 954, 48)
    for seed in (202601, 202602, 202603):
        out = HERE / str(seed)
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite {out}")
        seed_everything(seed)
        run = ROOT / f"investigations/distribution_correction_validation/runs/student_{seed}"
        cfg = json.loads((run / "config.json").read_text())
        assert cfg["method"] == "A" and cfg["protocol"] == "official_ilda"
        assert Path(cfg["ilda_cache"]).resolve() == cache_path.resolve()
        split_path = run / "source_split.npz"
        with np.load(split_path) as split:
            train_centers = split["train_centers"].copy()
            train_labels = split["train_labels"].copy()
            target_centers = split["target_centers"].copy()
        shift, shift_summary = shifts(source, target, train_centers,
                                      train_labels, target_centers)
        cp_path = run / "best_source_val.pth"
        cp = torch.load(cp_path, map_location="cpu", weights_only=False)
        model = Backbone().to(args.device).eval().requires_grad_(False)
        model.load_state_dict(cp["model"])
        params_before, buffers_before = tensor_hash(model.parameters()), tensor_hash(model.buffers())
        x = cache_patches(target, target_centers, args.device)
        q = {}
        for name in ("original", "global", "orthogonal", "parallel"):
            q[name] = predict(model, x, shift[name], args.device)
            assert q[name].shape == (53184, 7) and np.isfinite(q[name]).all()
            assert tensor_hash(model.parameters()) == params_before
            assert tensor_hash(model.buffers()) == buffers_before
            print(seed, name, "predicted", len(q[name]), flush=True)
        del x
        ref_path = ROOT / f"investigations/distribution_correction_validation/runs/soft_{seed}/probabilities_before_gt.npz"
        with np.load(ref_path) as reference:
            assert np.array_equal(target_centers, reference["centers"])
            old_original = reference["lambda_0"].copy()
        max_abs_diff = float(np.max(np.abs(q["original"] - old_original)))
        argmax_disagree = int(np.sum(q["original"].argmax(1) != old_original.argmax(1)))
        assert max_abs_diff < 1e-4 and argmax_disagree == 0, (max_abs_diff, argmax_disagree)
        target_bn_path = ROOT / f"investigations/target_bn_recal_v1/A_{seed}/predictions_before_gt.npz"
        with np.load(target_bn_path) as target_bn:
            assert np.array_equal(target_centers, target_bn["centers"])
            assert target_bn["raw"].shape == (53184, 7)
        out.mkdir(parents=True)
        pred_path = out / "predictions_before_gt.npz"
        np.savez_compressed(pred_path, centers=target_centers, **q)
        metadata = {
            "seed": seed, "target_gt_opened": False,
            "selected_epoch": int(cp["epoch"]),
            "source_checkpoint_sha256": digest(cp_path),
            "source_split_sha256": digest(split_path),
            "ilda_sha256": digest(cache_path),
            "lock_sha256": digest(HERE / "EXPERIMENT_LOCK.md"),
            "code_sha256": digest(__file__),
            "existing_original_sha256": digest(ref_path),
            "existing_target_bn_sha256": digest(target_bn_path),
            "original_reference_max_abs_diff": max_abs_diff,
            "original_reference_argmax_disagreement": argmax_disagree,
            "parameter_hash_unchanged": tensor_hash(model.parameters()) == params_before,
            "buffer_hash_unchanged": tensor_hash(model.buffers()) == buffers_before,
            "shift_summary": shift_summary,
            "prediction_sha256": digest(pred_path),
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(seed, "frozen", metadata["prediction_sha256"], flush=True)


if __name__ == "__main__":
    main()
