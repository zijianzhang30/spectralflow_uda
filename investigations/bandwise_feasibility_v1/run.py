"""Prelocked source-train-constrained band-wise feasibility search."""
from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import copy
import hashlib
import json
import math
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

EPSILONS = (0.0, 0.01, 0.02, 0.05, 0.10)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dct_basis(device):
    band = torch.arange(48, device=device, dtype=torch.float32).view(-1, 1)
    frequency = torch.arange(6, device=device, dtype=torch.float32).view(1, -1)
    basis = (2 / 48) ** 0.5 * torch.cos(math.pi * (band + .5) * frequency / 48)
    basis[:, 0] /= 2 ** .5
    return basis * 48 ** .5


def transform(u, v, basis, sigma):
    a = torch.exp(math.log(2) * torch.tanh(basis @ u))
    b = 3 * sigma * torch.tanh(basis @ v)
    return a, b


def simplex_projection(vector):
    sorted_values, _ = torch.sort(vector, descending=True)
    # CUDA cumsum lacks a deterministic kernel in the project Torch version.
    csum = torch.stack([sorted_values[:i].sum() - 1
                        for i in range(1, len(vector) + 1)])
    index = torch.arange(1, len(vector) + 1, device=vector.device)
    active = sorted_values - csum / index > 0
    rho = int(active.sum().item())
    threshold = csum[rho - 1] / rho
    return torch.clamp(vector - threshold, min=0)


@torch.no_grad()
def optimal_pi(target_mean, class_means, iterations):
    gram = class_means @ class_means.T
    linear = class_means @ target_mean
    step = 1 / (2 * torch.linalg.eigvalsh(gram)[-1].clamp_min(1e-8))
    pi = torch.full((7,), 1 / 7, device=class_means.device)
    for _ in range(iterations):
        pi = simplex_projection(pi - step * 2 * (gram @ pi - linear))
    return pi


def build_rff(xs, seed, device):
    rng = np.random.RandomState(seed + 2718)
    idx1 = rng.randint(0, len(xs), size=2048)
    idx2 = rng.randint(0, len(xs), size=2048)
    median = float(np.median(np.linalg.norm(xs[idx1] - xs[idx2], axis=1)))
    assert median > 0
    blocks = [rng.normal(size=(48, 256)).astype(np.float32) / (median * factor)
              for factor in (.5, 1., 2.)]
    w = torch.as_tensor(np.concatenate(blocks, axis=1), device=device)
    phase = torch.as_tensor(rng.uniform(0, 2 * math.pi, size=768).astype(np.float32), device=device)
    return w, phase, median


def phi(x, w, phase):
    return torch.cos(x @ w + phase) * (2 / 768) ** .5


@torch.inference_mode()
def logits_all(model, x):
    model.eval()
    return torch.cat([model(batch)[1].detach() for batch in x.split(32)])


def margin(logits, labels):
    true = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.scatter(1, labels[:, None], -torch.inf).max(1).values
    return true - other


def source_score(models, x, labels, a, b):
    with torch.inference_mode():
        changed = x * a.view(1, 48, 1, 1) + b.view(1, 48, 1, 1)
        output = {}
        for name, model in models.items():
            logits = logits_all(model, changed)
            pred = logits.argmax(1)
            m = margin(logits, labels)
            output[name] = {
                "recall": [float((pred[labels == c] == c).float().mean()) for c in range(7)],
                "mean_true_margin": [float(m[labels == c].mean()) for c in range(7)],
            }
        return output


def cached_patches(cube, centers, device):
    ds = Patches(cube, centers)
    loader = DataLoader(ds, batch_size=256, shuffle=False, drop_last=False, num_workers=0)
    output = torch.empty((len(centers), 48, 7, 7), device=device)
    cursor = 0
    for batch in loader:
        output[cursor:cursor + len(batch)] = batch.to(device)
        cursor += len(batch)
    assert cursor == len(centers)
    return output


def fit_seed(seed, device, source, target):
    out = HERE / str(seed)
    if out.exists():
        raise FileExistsError(out)
    seed_everything(seed)
    run = ROOT / f"investigations/distribution_correction_validation/runs/student_{seed}"
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["method"] == "A" and cfg["protocol"] == "official_ilda"
    with np.load(run / "source_split.npz") as split:
        train_centers, train_y = split["train_centers"].copy(), split["train_labels"].copy()
        target_centers = split["target_centers"].copy()
    assert np.array_equal(np.bincount(train_y, minlength=7), np.full(7, 180))
    xs_raw = source[train_centers[:, 0], train_centers[:, 1]].astype(np.float32)
    xt_raw = target[target_centers[:, 0], target_centers[:, 1]].astype(np.float32)
    mu_np = xs_raw.mean(0)
    sigma_np = np.maximum(xs_raw.std(0), 1e-6)
    mu = torch.as_tensor(mu_np, device=device)
    sigma = torch.as_tensor(sigma_np, device=device)
    xs = torch.as_tensor((xs_raw - mu_np) / sigma_np, device=device)
    target_rng = np.random.RandomState(seed + 31415)
    target_idx = target_rng.choice(len(xt_raw), size=8192, replace=False)
    xt_fit = torch.as_tensor(xt_raw[target_idx], device=device)
    w, phase, median = build_rff(xs.cpu().numpy(), seed, device)
    source_embedding = phi(xs, w, phase)
    y = torch.as_tensor(train_y, device=device, dtype=torch.long)
    class_embeddings = torch.stack([source_embedding[y == c].mean(0) for c in range(7)])
    basis = dct_basis(device)
    cp_path = run / "best_source_val.pth"
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    original = Backbone().to(device).eval().requires_grad_(False)
    original.load_state_dict(cp["model"])
    recal = copy.deepcopy(original).train()
    source_patches = cached_patches(source, train_centers, device)
    with torch.inference_mode():
        for batch in source_patches.split(32):
            recal(batch)
    recal.eval().requires_grad_(False)
    models = {"original_bn": original, "source_recal_bn": recal}
    original_param_hash = tensor_hash(original.parameters())
    original_buffer_hash = tensor_hash(original.buffers())
    recal_param_hash = tensor_hash(recal.parameters())
    recal_buffer_hash = tensor_hash(recal.buffers())
    assert original_param_hash == recal_param_hash and original_buffer_hash != recal_buffer_hash
    guard_rng = np.random.RandomState(seed + 1618)
    guard_indices = np.concatenate([guard_rng.choice(np.flatnonzero(train_y == c), 32, replace=False)
                                    for c in range(7)])
    guard_x = source_patches[torch.as_tensor(guard_indices, device=device)]
    guard_y = y[torch.as_tensor(guard_indices, device=device)]
    base_logits = {name: model(guard_x)[1].detach() for name, model in models.items()}
    temperatures = {name: margin(logits, guard_y).abs().median().clamp_min(1.0).detach()
                    for name, logits in base_logits.items()}
    base_soft_risk = torch.stack([
        torch.stack([torch.sigmoid(-margin(logits, guard_y)[guard_y == c] /
                                    temperatures[name]).mean() for c in range(7)])
        for name, logits in base_logits.items()
    ]).detach()
    identity_a = torch.ones(48, device=device)
    identity_b = torch.zeros(48, device=device)
    identity_train = source_score(models, source_patches, y, identity_a, identity_b)

    def discrepancy(a, b, pi_iterations=100):
        transformed = xt_fit * a + b
        z = (transformed - mu) / sigma
        target_embedding = phi(z, w, phase).mean(0)
        pi = optimal_pi(target_embedding.detach(), class_embeddings, pi_iterations)
        distance = torch.sum((target_embedding - pi @ class_embeddings) ** 2)
        return distance, pi

    with torch.no_grad():
        identity_distance, identity_pi = discrepancy(identity_a, identity_b, 1000)
        balanced_distance = torch.sum((phi((xt_fit - mu) / sigma, w, phase).mean(0)
                                       - class_embeddings.mean(0)) ** 2)
    assert float(identity_distance) > 1e-8
    candidates = []
    for search_epsilon in EPSILONS:
        u = torch.zeros(6, device=device, requires_grad=True)
        v = torch.zeros(6, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([u, v], lr=.05)
        dual = torch.zeros((2, 7), device=device)
        for step in range(81):
            if step % 10 == 0:
                with torch.no_grad():
                    a, b = transform(u, v, basis, sigma)
                    dist, pi = discrepancy(a, b, 1000)
                    train_score = source_score(models, source_patches, y, a, b)
                    candidate = {
                        "search_epsilon": search_epsilon, "step": step,
                        "u": u.detach().cpu().tolist(), "v": v.detach().cpu().tolist(),
                        "discrepancy": float(dist),
                        "relative_discrepancy": float(dist / identity_distance),
                        "pi": pi.cpu().tolist(), "source_train": train_score,
                    }
                    candidates.append(candidate)
            if step == 80:
                break
            optimizer.zero_grad(set_to_none=True)
            a, b = transform(u, v, basis, sigma)
            dist, _ = discrepancy(a, b)
            changed = guard_x * a.view(1, 48, 1, 1) + b.view(1, 48, 1, 1)
            soft_risk = []
            for name, model in models.items():
                current = model(changed)[1]
                m = margin(current, guard_y)
                soft_risk.append(torch.stack([
                    torch.sigmoid(-m[guard_y == c] / temperatures[name]).mean()
                    for c in range(7)]))
            violation = torch.stack(soft_risk) - base_soft_risk - search_epsilon
            positive = torch.relu(violation)
            loss = (dist / identity_distance +
                    (dual * positive + 50 * positive.square()).sum())
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                dual = torch.clamp(dual + 10 * positive.detach(), max=100)
        print(seed, "search eps", search_epsilon,
              "best relative", round(min(c["relative_discrepancy"] for c in candidates
                                         if c["search_epsilon"] == search_epsilon), 4), flush=True)

    chosen = {}
    for epsilon in EPSILONS:
        feasible = []
        for candidate in candidates:
            if all(candidate["source_train"][name]["recall"][c] >=
                   identity_train[name]["recall"][c] - epsilon - 1e-12
                   for name in models for c in range(7)):
                feasible.append(candidate)
        assert feasible, epsilon  # identity is always feasible
        winner = min(feasible, key=lambda c: (c["discrepancy"],
                                               c["search_epsilon"], c["step"]))
        chosen[str(epsilon)] = {
            "search_epsilon": winner["search_epsilon"], "step": winner["step"],
            "u": winner["u"], "v": winner["v"],
            "discrepancy": winner["discrepancy"],
            "relative_discrepancy": winner["relative_discrepancy"],
            "pi": winner["pi"], "source_train": winner["source_train"],
        }
    assert tensor_hash(original.parameters()) == original_param_hash
    assert tensor_hash(original.buffers()) == original_buffer_hash
    assert tensor_hash(recal.parameters()) == recal_param_hash
    assert tensor_hash(recal.buffers()) == recal_buffer_hash

    # Candidate selection is now frozen. Source-val is read for audit only.
    with np.load(run / "source_split.npz") as split:
        val_centers, val_labels = split["val_centers"].copy(), split["val_labels"].copy()
    val_patches = cached_patches(source, val_centers, device)
    val_y = torch.as_tensor(val_labels, device=device, dtype=torch.long)
    identity_val = source_score(models, val_patches, val_y, identity_a, identity_b)
    for row in chosen.values():
        with torch.no_grad():
            u = torch.as_tensor(row["u"], device=device)
            v = torch.as_tensor(row["v"], device=device)
            a, b = transform(u, v, basis, sigma)
            row["source_val_audit"] = source_score(models, val_patches, val_y, a, b)
            row["a"] = a.cpu().tolist()
            row["b"] = b.cpu().tolist()
            row["transform_negative_target_center_fraction"] = float(((
                torch.as_tensor(xt_raw, device=device) * a + b) < 0).float().mean())

    # Inference is always through the original-BN checkpoint, with official
    # target center order, patch size, batch size and drop_last.
    target_patches = cached_patches(target, target_centers, device)
    probs = {}
    for key, row in chosen.items():
        a = torch.as_tensor(row["a"], device=device)
        b = torch.as_tensor(row["b"], device=device)
        with torch.inference_mode():
            q = torch.cat([original(batch * a.view(1, 48, 1, 1) +
                                           b.view(1, 48, 1, 1))[1].softmax(1).cpu()
                           for batch in target_patches[:53184].split(32)])
        probs[f"epsilon_{key}"] = q.numpy()
        print(seed, "selected epsilon", key,
              "relative discrepancy", round(row["relative_discrepancy"], 4),
              "target predictions frozen", len(q), flush=True)
    with torch.inference_mode():
        identity_q = torch.cat([original(batch)[1].softmax(1).cpu()
                                for batch in target_patches[:53184].split(32)]).numpy()
    reference_path = ROOT / f"investigations/distribution_correction_validation/runs/soft_{seed}/probabilities_before_gt.npz"
    with np.load(reference_path) as reference:
        assert np.array_equal(target_centers, reference["centers"])
        ref = reference["lambda_0"]
        assert np.array_equal(identity_q, ref)
    probs["identity"] = identity_q
    assert tensor_hash(original.parameters()) == original_param_hash
    assert tensor_hash(original.buffers()) == original_buffer_hash
    out.mkdir(parents=True)
    pred_path = out / "predictions_before_gt.npz"
    np.savez_compressed(pred_path, centers=target_centers, **probs)
    search_path = out / "source_train_search.json"
    search_path.write_text(json.dumps(candidates, indent=2))
    metadata = {
        "seed": seed, "target_gt_opened": False, "source_val_used_for_selection": False,
        "checkpoint_sha256": digest(cp_path),
        "source_split_sha256": digest(run / "source_split.npz"),
        "ilda_sha256": digest(ROOT / "official_aligned/preprocessing/official_ilda.npz"),
        "lock_sha256": digest(HERE / "EXPERIMENT_LOCK.md"),
        "code_sha256": digest(__file__),
        "selected_epoch": int(cp["epoch"]),
        "rff_source_median_bandwidth": median,
        "identity_pi_only": identity_pi.cpu().tolist(),
        "identity_discrepancy": float(identity_distance),
        "identity_balanced_source_discrepancy": float(balanced_distance),
        "identity_source_train": identity_train,
        "identity_source_val_audit": identity_val,
        "chosen": chosen,
        "source_train_search_sha256": digest(search_path),
        "prediction_sha256": digest(pred_path),
        "original_params_unchanged": True,
        "original_buffers_unchanged": True,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(seed, "complete", metadata["prediction_sha256"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:5")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with np.load(ROOT / "official_aligned/preprocessing/official_ilda.npz") as cache:
        source, target = cache["s"].astype(np.float32), cache["t"].astype(np.float32)
    for seed in (202601, 202602, 202603):
        fit_seed(seed, args.device, source, target)


if __name__ == "__main__":
    main()
