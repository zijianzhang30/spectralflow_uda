"""Post-hoc, read-only audit of frozen MLUDA and HyperSIGMA evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import hdf5storage
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (202601, 202602, 202603)
BINS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.000001))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prevalence(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels, minlength=7).astype(np.float64) / len(labels)


def comparison(estimate: np.ndarray, truth: np.ndarray) -> dict:
    error = estimate - truth
    return {"estimate": estimate.tolist(), "truth": truth.tolist(),
            "signed_error": error.tolist(),
            "total_variation": float(np.abs(error).sum() / 2)}


def classification(labels: np.ndarray, pred: np.ndarray) -> dict:
    recalls = []
    for c in range(7):
        mask = labels == c
        recalls.append(float((pred[mask] == c).mean()))
    return {"oa_official": float((pred == labels).sum() / 53200),
            "aa": float(np.mean(recalls)), "recall": recalls}


def flips(labels: np.ndarray, raw: np.ndarray, corrected: np.ndarray) -> dict:
    before = raw.argmax(1)
    after = corrected.argmax(1)
    confidence = raw.max(1)
    changed = before != after
    corrected_error = (before != labels) & (after == labels)
    introduced_error = (before == labels) & (after != labels)
    wrong_to_wrong = changed & (before != labels) & (after != labels)

    def row(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        return {"n": n, "raw_accuracy": float((before[mask] == labels[mask]).mean()) if n else None,
                "changed": int((changed & mask).sum()),
                "wrong_to_correct": int((corrected_error & mask).sum()),
                "correct_to_wrong": int((introduced_error & mask).sum()),
                "wrong_to_different_wrong": int((wrong_to_wrong & mask).sum())}

    return {"overall": row(np.ones(len(labels), dtype=bool)),
            "by_true_class": {str(c + 1): row(labels == c) for c in range(7)},
            "by_original_confidence": {
                f"[{lo:.1f},{hi:.1f}{')' if hi < 1 else ']'}":
                    row((confidence >= lo) & (confidence < hi))
                for lo, hi in BINS}}


def main() -> None:
    output = {"scope": "Post-hoc Houston development diagnostic; no parameter selection",
              "lock_sha256": sha256(HERE / "EXPERIMENT_LOCK.md"), "seeds": {}}
    for seed in SEEDS:
        mluda_dir = ROOT / f"investigations/mluda_prior_probe/seed_{seed}"
        mluda_path = mluda_dir / "predictions_before_gt.npz"
        teacher_dir = ROOT / f"investigations/distribution_correction_validation/runs/correction_{seed}"
        teacher_path = teacher_dir / "predictions_before_gt.npz"
        mluda_meta = json.loads((mluda_dir / "metadata.json").read_text())
        assert mluda_meta["prediction_sha256"] == sha256(mluda_path)
        assert mluda_meta["teacher_prior_sha256"] == sha256(teacher_path)
        assert mluda_meta["target_gt_opened"] is False
        with np.load(mluda_path) as saved:
            centers = saved["centers"].copy()
            raw = saved["raw"].astype(np.float64)
            corrected = saved["corrected"].astype(np.float64)
            prior = saved["prior"].astype(np.float64)
        with np.load(teacher_path) as saved:
            assert np.array_equal(centers, saved["centers"])
            teacher = saved["teacher"].astype(np.float64)
            assert np.allclose(prior, saved["prior"], atol=1e-12)
        assert centers.shape == (53200, 2)
        assert raw.shape == corrected.shape == (53184, 7)
        assert teacher.shape == (53200, 7)
        assert np.allclose(teacher.mean(0), prior, atol=1e-7)
        assert np.allclose(raw.sum(1), 1, atol=1e-5)
        assert np.allclose(corrected.sum(1), 1, atol=1e-5)
        cfg = json.loads((ROOT / f"investigations/distribution_correction_validation/runs/mluda_{seed}/config.json").read_text())
        gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
        gt = hdf5storage.loadmat(str(gt_path))["map"]
        y_all = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
        assert y_all.shape == (53200,) and y_all.min() == 0 and y_all.max() == 6
        y = y_all[:53184]
        true_all, true_eval = prevalence(y_all), prevalence(y)
        raw_marginal = raw.mean(0)
        uniform = np.full(7, 1 / 7)
        result = {
            "frozen_mluda_sha256": sha256(mluda_path),
            "frozen_teacher_sha256": sha256(teacher_path),
            "target_gt_sha256": sha256(gt_path),
            "target_n_prior": 53200, "target_n_evaluation": 53184,
            "teacher_prior_vs_true_all": comparison(prior, true_all),
            "mluda_marginal_vs_true_evaluated": comparison(raw_marginal, true_eval),
            "balanced_source_prior_vs_true_all": comparison(uniform, true_all),
            "balanced_source_prior_vs_true_evaluated": comparison(uniform, true_eval),
            "raw": classification(y, raw.argmax(1)),
            "corrected": classification(y, corrected.argmax(1)),
            "flips": flips(y, raw, corrected),
        }
        prior_audit = json.loads((ROOT / "investigations/mluda_prior_probe/AUDIT.json").read_text())
        for variant in ("raw", "corrected"):
            expected = prior_audit["seeds"][str(seed)][variant]["metrics"]
            assert np.isclose(result[variant]["oa_official"], expected["oa_official"])
            assert np.isclose(result[variant]["aa"], expected["aa"])
            assert np.allclose(result[variant]["recall"], expected["recall"])
        output["seeds"][str(seed)] = result
        print(json.dumps({"seed": seed,
                          "TV_teacher": result["teacher_prior_vs_true_all"]["total_variation"],
                          "TV_mluda": result["mluda_marginal_vs_true_evaluated"]["total_variation"],
                          "class7_teacher_error": result["teacher_prior_vs_true_all"]["signed_error"][6],
                          "class7_mluda_error": result["mluda_marginal_vs_true_evaluated"]["signed_error"][6],
                          "flips": result["flips"]["overall"]}), flush=True)
    path = HERE / "AUDIT.json"
    assert not path.exists(), f"Refusing to overwrite {path}"
    path.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
