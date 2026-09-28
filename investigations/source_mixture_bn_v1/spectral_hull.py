"""Can unlabeled target mean spectra be explained by source class mixture alone?"""
from __future__ import annotations

import hashlib
import json
import itertools
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


def project(mean, class_means, scale):
    a = (class_means / scale).T.astype(np.float64)
    b = (mean / scale).astype(np.float64)
    # Seven classes permit exact active-set enumeration. For each possible
    # positive support, solve the equality-constrained least-squares problem.
    best = None
    for count in range(1, 8):
        for support in itertools.combinations(range(7), count):
            sub = a[:, support]
            gram = sub.T @ sub
            kkt = np.block([[gram, np.ones((count, 1))],
                            [np.ones((1, count)), np.zeros((1, 1))]])
            rhs = np.concatenate((sub.T @ b, [1.0]))
            solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:count]
            if solution.min() < -1e-8:
                continue
            weight = np.zeros(7)
            weight[list(support)] = solution
            residual = a @ weight - b
            distance = float(np.linalg.norm(residual))
            if best is None or distance < best[0]:
                best = distance, weight
    if best is None:
        raise RuntimeError("No feasible source class mixture found")
    return {"distance": best[0], "weights_nearest_source_mixture": best[1].tolist()}


def center_spectra(cube, centers):
    return cube[centers[:, 0], centers[:, 1]].astype(np.float64)


def main():
    out_path = HERE / "SPECTRAL_HULL_EXACT.json"
    if out_path.exists():
        raise FileExistsError(out_path)
    cache_path = ROOT / "official_aligned/preprocessing/official_ilda.npz"
    with np.load(cache_path) as cache:
        source, target = cache["s"].copy(), cache["t"].copy()
    result = {"scope": "Official-ILDA raw center spectra, source class labels and unlabeled target centers only",
              "ilda_sha256": digest(cache_path), "seeds": {}}
    for seed in (202601, 202602, 202603):
        split_path = ROOT / f"investigations/distribution_correction_validation/runs/student_{seed}/source_split.npz"
        with np.load(split_path) as split:
            train_centers, train_labels = split["train_centers"].copy(), split["train_labels"].copy()
            val_centers, val_labels = split["val_centers"].copy(), split["val_labels"].copy()
            target_centers = split["target_centers"].copy()
        assert len(target_centers) == 53200 and len(train_centers) == 1260
        train = center_spectra(source, train_centers)
        val = center_spectra(source, val_centers)
        unlabeled_target = center_spectra(target, target_centers)
        class_means = np.stack([train[train_labels == c].mean(0) for c in range(7)])
        val_class_means = np.stack([val[val_labels == c].mean(0) for c in range(7)])
        scale = np.maximum(train.std(0), 1e-6)
        val_mean = val.mean(0)
        target_mean = unlabeled_target.mean(0)
        row = {
            "split_sha256": digest(split_path),
            "source_val_mean_projection": project(val_mean, class_means, scale),
            "unlabeled_target_mean_projection": project(target_mean, class_means, scale),
            "unlabeled_target_mean_projection_to_val_class_means": project(target_mean, val_class_means, scale),
            "source_train_mean_projection_to_val_class_means": project(train.mean(0), val_class_means, scale),
            "source_val_mean_distance_to_train_mean": float(np.linalg.norm((val_mean - train.mean(0)) / scale)),
            "target_mean_distance_to_train_mean": float(np.linalg.norm((target_mean - train.mean(0)) / scale)),
            "source_class_means_diameter": float(max(
                np.linalg.norm((class_means[i] - class_means[j]) / scale)
                for i in range(7) for j in range(i + 1, 7))),
        }
        result["seeds"][str(seed)] = row
        print(seed, "source_val_hull", row["source_val_mean_projection"]["distance"],
              "target_hull", row["unlabeled_target_mean_projection"]["distance"], flush=True)
    out_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
