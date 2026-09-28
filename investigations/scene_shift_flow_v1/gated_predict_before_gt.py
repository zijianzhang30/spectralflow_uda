"""Apply only the Flow's existing 0.95 training-support gate at inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "investigations/distribution_correction_validation"))
from soft_projection import soft_project


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(202601, 202602, 202603), required=True)
    a = p.parse_args()
    run = HERE / f"formal_{a.seed}"
    out = HERE / f"gated_{a.seed}"
    assert not out.exists(), f"Refusing to overwrite {out}"
    feature_path = run / "frozen_features_before_gt.npz"
    prediction_path = run / "predictions_before_gt.npz"
    with np.load(feature_path) as data:
        q_train = data["target_q"][:53184].copy()
        centers = data["target_centers"].copy()
    with np.load(prediction_path) as data:
        assert np.array_equal(centers, data["centers"])
        base_raw = data["base_raw"].copy()
        flow_raw = data["flow_raw"].copy()
        prior = data["prior"].copy()
    mask = q_train.max(axis=1) >= .95
    hybrid = base_raw.copy()
    hybrid[mask] = flow_raw[mask]
    corrected, solver = soft_project(hybrid, prior, strength=1.0)
    out.mkdir()
    output = out / "predictions_before_gt.npz"
    np.savez_compressed(output, centers=centers, gated_raw=hybrid,
                        gated_corrected=corrected, moved_mask=mask, prior=prior)
    (out / "metadata.json").write_text(json.dumps({
        "seed": a.seed, "target_gt_opened": False,
        "training_support_threshold": .95, "moved_n": int(mask.sum()),
        "moved_fraction": float(mask.mean()),
        "feature_cache_sha256": digest(feature_path),
        "ungated_prediction_sha256": digest(prediction_path),
        "lock_sha256": digest(HERE / "GATED_INFERENCE_LOCK.md"),
        "prediction_sha256": digest(output), "soft_kl": solver,
    }, indent=2))
    print(json.dumps({"seed": a.seed, "moved_fraction": float(mask.mean()),
                      "changed_argmax_vs_base":
                      float((hybrid.argmax(1) != base_raw.argmax(1)).mean())}))


if __name__ == "__main__":
    main()
