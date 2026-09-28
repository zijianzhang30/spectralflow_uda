"""Post-freeze, paired official-metric audit for the raw-vs-ILDA ablation."""
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "round9"))
from data import array_hash, file_hash  # noqa: E402
from runtime import scores, atomic_json  # noqa: E402


def main():
    base = Path(__file__).resolve().parent
    rows = []
    for seed in (202601, 202602, 202603):
        raw = base / f"raw_{seed}"
        ilda = ROOT / "investigations" / "distribution_correction_validation" / "runs" / f"student_{seed}"
        meta = json.loads((raw / "frozen_target_predictions.json").read_text())
        cfg = json.loads((raw / "config.json").read_text())
        gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
        assert file_hash(gt_path) == json.loads((raw / "provenance.json").read_text())["inputs"]["Houston18_7gt.mat"]
        with np.load(raw / "frozen_target_predictions.npz") as frozen, \
             np.load(ilda / "source_split.npz") as il_split:
            pred = frozen["predictions"].astype(np.int64)
            centers = frozen["centers"]
            assert np.array_equal(centers, il_split["target_centers"])
        assert array_hash(centers) == meta["center_hash"]
        gt = hdf5storage.loadmat(str(gt_path))["map"]
        labels = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
        assert len(labels) == 53200 and len(pred) == 53184
        evaluated = labels[:len(pred)]
        assert np.all((evaluated >= 0) & (evaluated < 7))
        metrics = scores(evaluated, pred)
        metrics["oa_evaluated"] = metrics["oa"]
        metrics["oa"] = float((evaluated == pred).sum()) / len(labels)
        il_result = json.loads((ilda / "final_target_source_val_best.json").read_text())
        assert il_result["target_gt_sha256"] == file_hash(gt_path)
        row = dict(seed=seed, raw_epoch=meta["selected_epoch"],
                   ilda_epoch=il_result["selected_epoch"], raw=metrics,
                   ilda=il_result["metrics"],
                   delta_oa=metrics["oa"] - il_result["metrics"]["oa"],
                   delta_aa=metrics["aa"] - il_result["metrics"]["aa"],
                   delta_kappa=metrics["kappa"] - il_result["metrics"]["kappa"],
                   raw_checkpoint_sha256=meta["checkpoint_sha256"],
                   ilda_checkpoint_sha256=il_result["checkpoint_sha256"])
        rows.append(row)
    summary = {}
    for arm in ("raw", "ilda"):
        summary[arm] = {}
        for key in ("oa", "aa", "kappa"):
            vals = np.array([row[arm][key] for row in rows])
            summary[arm][key] = dict(mean=float(vals.mean()), std=float(vals.std(ddof=1)))
    for key in ("delta_oa", "delta_aa", "delta_kappa"):
        vals = np.array([row[key] for row in rows])
        summary[key] = dict(mean=float(vals.mean()), std=float(vals.std(ddof=1)))
    atomic_json(base / "paired_metrics.json", dict(rows=rows, summary=summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
