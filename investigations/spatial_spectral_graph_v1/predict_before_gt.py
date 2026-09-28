"""One fixed unlabeled spatial-spectral graph step on frozen target probabilities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def edges(centers, spectra):
    lookup = np.full((210, 954), -1, dtype=np.int32)
    lookup[centers[:, 0], centers[:, 1]] = np.arange(len(centers))
    assert np.unique(centers, axis=0).shape[0] == len(centers)
    normalized = spectra / np.maximum(np.linalg.norm(spectra, axis=1, keepdims=True), 1e-12)
    source, neighbor, distance = [], [], []
    for dr, dc in ((1, 0), (0, 1)):
        rr, cc = centers[:, 0] + dr, centers[:, 1] + dc
        inbounds = (rr < 210) & (cc < 954)
        src = np.flatnonzero(inbounds)
        dst = lookup[rr[inbounds], cc[inbounds]]
        good = dst >= 0
        src, dst = src[good], dst[good]
        dist = np.maximum(1 - np.einsum("ij,ij->i", normalized[src], normalized[dst]), 0)
        source.append(src)
        neighbor.append(dst)
        distance.append(dist)
    src = np.concatenate(source)
    dst = np.concatenate(neighbor)
    dist = np.concatenate(distance)
    scale = max(float(np.median(dist)), float(np.finfo(np.float32).eps))
    weight = np.exp(-dist / scale).astype(np.float64)
    return src, dst, weight, {"undirected_edges": int(len(src)),
                              "mean_neighbors": float(2 * len(src) / len(centers)),
                              "median_spectral_distance": float(np.median(dist)),
                              "weight_mean": float(weight.mean())}


def diffuse(q, src, dst, weight):
    q = np.asarray(q, dtype=np.float64)
    numerator = np.zeros_like(q)
    denominator = np.zeros(len(q), dtype=np.float64)
    np.add.at(numerator, src, weight[:, None] * q[dst])
    np.add.at(numerator, dst, weight[:, None] * q[src])
    np.add.at(denominator, src, weight)
    np.add.at(denominator, dst, weight)
    result = q.copy()
    active = denominator > 0
    result[active] = .5 * q[active] + .5 * numerator[active] / denominator[active, None]
    assert np.isfinite(result).all() and np.allclose(result.sum(1), 1, atol=1e-6)
    return result.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(202601, 202602, 202603), required=True)
    a = p.parse_args()
    out = HERE / f"seed_{a.seed}"
    assert not out.exists(), f"Refusing to overwrite {out}"
    base = ROOT / "investigations/distribution_correction_validation/runs" / f"soft_{a.seed}"
    mluda = ROOT / "investigations/mluda_prior_probe" / f"seed_{a.seed}"
    with np.load(base / "probabilities_before_gt.npz") as data:
        centers = data["centers"].copy()
        q = {"A_raw": data["lambda_0"].copy(), "A_prior": data["lambda_1"].copy()}
    with np.load(mluda / "predictions_before_gt.npz") as data:
        assert np.array_equal(centers, data["centers"])
        q.update(MLUDA_raw=data["raw"].copy(), MLUDA_prior=data["corrected"].copy())
    centers = centers[:53184]
    with np.load(ROOT / "official_aligned/preprocessing/official_ilda.npz") as data:
        cube = data["t"]
        spectra = cube[centers[:, 0], centers[:, 1]].astype(np.float64)
    src, dst, weight, graph = edges(centers, spectra)
    result = {name: diffuse(probs, src, dst, weight) for name, probs in q.items()}
    out.mkdir()
    path = out / "predictions_before_gt.npz"
    np.savez_compressed(path, centers=centers, **result)
    (out / "metadata.json").write_text(json.dumps({
        "seed": a.seed, "graph": graph, "target_gt_opened": False,
        "input_DCRN_sha256": digest(base / "probabilities_before_gt.npz"),
        "input_MLUDA_sha256": digest(mluda / "predictions_before_gt.npz"),
        "ilda_cache_sha256": digest(ROOT / "official_aligned/preprocessing/official_ilda.npz"),
        "lock_sha256": digest(HERE / "EXPERIMENT_LOCK.md"),
        "prediction_sha256": digest(path),
    }, indent=2))
    print(json.dumps({"seed": a.seed, "graph": graph,
                      "changed_argmax": {name: float((p.argmax(1) != q[name].argmax(1)).mean())
                                         for name, p in result.items()}}))


if __name__ == "__main__":
    main()
