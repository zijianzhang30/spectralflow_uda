"""Post-freeze paired evaluation and step-level BN/parameter audit."""
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "round9"))
from data import array_hash, file_hash  # noqa: E402
from runtime import atomic_json, scores  # noqa: E402


def step_audit(baseline, intervention):
    with np.load(baseline / "source_split.npz") as a, np.load(intervention / "source_split.npz") as b:
        split_equal = all(np.array_equal(a[k], b[k]) for k in a.files)
    x = [json.loads(line) for line in (baseline / "steps.jsonl").read_text().splitlines()]
    y = [json.loads(line) for line in (intervention / "steps.jsonl").read_text().splitlines()]
    assert len(x) == len(y) == 3800 and split_equal
    fields = ("source_input_hash", "target_input_hash", "source_logits_hash",
              "gradient_hash", "backbone_hash", "classifier_hash", "model_hash",
              "buffers_hash", "cpu_rng_hash", "cuda_rng_hash")
    counts = {f: sum(a[f] == b[f] for a, b in zip(x, y)) for f in fields}
    for key in fields[:6]:
        assert counts[key] == 3800, (key, counts[key])
    assert counts["buffers_hash"] == 0
    return dict(split_equal=split_equal, steps=3800, matches=counts)


def score_frozen(run, selection, baseline=False):
    suffix = "" if baseline and selection == "source_val_best" else f"_{selection}"
    meta = json.loads((run / f"frozen_target_predictions{suffix}.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    with np.load(run / f"frozen_target_predictions{suffix}.npz") as frozen:
        pred = frozen["predictions"].astype(np.int64)
        centers = frozen["centers"].copy()
    assert len(pred) == 53184 and len(centers) == 53200
    assert array_hash(centers) == meta["center_hash"]
    data = Path(cfg["data"])
    gt_path = data / "Houston18_7gt.mat"
    assert file_hash(gt_path) == json.loads((run / "provenance.json").read_text())["inputs"]["Houston18_7gt.mat"]
    gt = hdf5storage.loadmat(str(gt_path))["map"]
    labels = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    evaluated = labels[:len(pred)]
    assert np.all((evaluated >= 0) & (evaluated < 7))
    metrics = scores(evaluated, pred)
    metrics["oa_evaluated"] = metrics["oa"]
    metrics["oa"] = float((evaluated == pred).sum()) / len(labels)
    history = json.loads((run / "history.json").read_text())
    epoch = meta["selected_epoch"]
    assert history[epoch - 1]["epoch"] == epoch
    return dict(epoch=epoch, source_val_accuracy=history[epoch - 1]["source_val_accuracy"],
                metrics=metrics, checkpoint_sha256=meta["checkpoint_sha256"],
                centers_hash=meta["center_hash"])


def main():
    base = Path(__file__).resolve().parent
    rows = []
    for seed in (202601, 202602, 202603):
        baseline = ROOT / "investigations" / "ilda_input_ablation_v1" / f"raw_{seed}"
        intervention = base / f"formal_{seed}"
        audit = step_audit(baseline, intervention)
        selections = {}
        for selection in ("source_val_best", "fixed_epoch_100"):
            a = score_frozen(baseline, selection, baseline=True)
            b = score_frozen(intervention, selection)
            assert a["centers_hash"] == b["centers_hash"]
            selections[selection] = dict(baseline=a, source_bn_only=b,
                                         delta_oa=b["metrics"]["oa"] - a["metrics"]["oa"],
                                         delta_aa=b["metrics"]["aa"] - a["metrics"]["aa"],
                                         delta_kappa=b["metrics"]["kappa"] - a["metrics"]["kappa"])
        rows.append(dict(seed=seed, audit=audit, selections=selections))
    summary = {}
    for selection in ("source_val_best", "fixed_epoch_100"):
        summary[selection] = {}
        for arm in ("baseline", "source_bn_only"):
            summary[selection][arm] = {}
            for key in ("oa", "aa", "kappa"):
                vals = np.array([r["selections"][selection][arm]["metrics"][key] for r in rows])
                summary[selection][arm][key] = dict(mean=float(vals.mean()), std=float(vals.std(ddof=1)))
        for key in ("delta_oa", "delta_aa", "delta_kappa"):
            vals = np.array([r["selections"][selection][key] for r in rows])
            summary[selection][key] = dict(mean=float(vals.mean()), std=float(vals.std(ddof=1)))
    atomic_json(base / "AUDIT.json", dict(rows=rows, summary=summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
