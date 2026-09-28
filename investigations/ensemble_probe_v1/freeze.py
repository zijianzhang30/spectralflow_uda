"""Freeze equal-weight probability ensembles without loading target labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (202601, 202602, 202603)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = {"target_gt_opened": False, "fusion": "equal arithmetic probability mean",
                "lock_sha256": digest(HERE / "EXPERIMENT_LOCK.md"), "seeds": {}}
    for seed in SEEDS:
        m_path = ROOT / f"investigations/mluda_prior_probe/seed_{seed}/predictions_before_gt.npz"
        s_path = ROOT / f"investigations/scene_shift_v1/formal_{seed}/predictions_before_gt.npz"
        out = HERE / f"seed_{seed}_fused_before_gt.npz"
        assert not out.exists(), f"Refusing to overwrite {out}"
        with np.load(m_path) as m, np.load(s_path) as s:
            assert np.array_equal(m["centers"], s["centers"])
            assert np.array_equal(m["prior"], s["prior"])
            assert m["raw"].shape == s["raw"].shape == (53184, 7)
            assert m["corrected"].shape == s["corrected"].shape == (53184, 7)
            raw = (m["raw"].astype(np.float64) + s["raw"].astype(np.float64)) / 2
            corrected = (m["corrected"].astype(np.float64) +
                         s["corrected"].astype(np.float64)) / 2
            assert np.isfinite(raw).all() and np.isfinite(corrected).all()
            assert np.allclose(raw.sum(1), 1, atol=1e-5)
            assert np.allclose(corrected.sum(1), 1, atol=1e-5)
            np.savez_compressed(out, centers=m["centers"], raw=raw,
                                corrected=corrected)
        manifest["seeds"][str(seed)] = {
            "mluda_prediction_sha256": digest(m_path),
            "scene_shift_prediction_sha256": digest(s_path),
            "fused_prediction_sha256": digest(out),
        }
    (HERE / "FREEZE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
