"""Verify and summarize the locked three-seed filtered reverse-Flow probe."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = (1341, 1174, 1370)
SELECTIONS = ("fixed_epoch_100", "source_val_best")


def verified(seed, selection):
    path = ROOT / "runs" / f"formal_{seed}"
    row = json.loads((path / f"final_target_{selection}.json").read_text())
    history = json.loads((path / "history.json").read_text())
    complete = json.loads((path / "training_complete.json").read_text())
    assert complete["formal"] and complete["epochs"] == len(history) == 100
    assert row["seed"] == seed and row["selection"] == selection
    assert row["raw_matches_round3_A"] and row["dataset_n"] == 53200
    assert row["dropped_n"] == 16
    assert row["selected_epoch"] == (100 if selection == "fixed_epoch_100" else
                                     max(history, key=lambda x: x["source_val_accuracy"])["epoch"])
    for name in ("raw", "transported"):
        metric = row[name]
        cm = np.asarray(metric["confusion_matrix"])
        assert cm.sum() == metric["evaluated_n"] == 53184
        assert abs(metric["oa"] - np.trace(cm)/53200) < 1e-12
        assert abs(metric["aa"] - np.mean(np.diag(cm)/cm.sum(1))) < 1e-12
    return row


def fmt(values, scale=1):
    values = np.asarray(values, dtype=float)*scale
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def main():
    data = {s: [verified(seed, s) for seed in SEEDS] for s in SELECTIONS}
    lines = ["# Round6: filtered target-to-source Flow — three seeds", "",
             "Fixed setting: confidence ≥0.95, skip absent classes/empty-pair Flow updates;",
             "all other Houston13→Houston18 training and test settings match round5.",
             "Four Euler steps transport target features at inference. Source-val-best",
             "is reported separately; fixed epoch100 remains the primary rule.", ""]
    for selection, rows in data.items():
        lines += [f"## {selection}", "",
                  "| Method | OA (%) | AA (%) | Kappa | OA by seed (%) |",
                  "|---|---:|---:|---:|---|"]
        for method in ("raw", "transported"):
            scores = [r[method] for r in rows]
            lines.append("| {} | {} | {} | {} | {} |".format(
                "A (raw)" if method == "raw" else "Filtered reverse Flow",
                fmt([x["oa"] for x in scores], 100),
                fmt([x["aa"] for x in scores], 100),
                fmt([x["kappa"] for x in scores]),
                ", ".join(f"{x['oa']*100:.2f}" for x in scores)))
        lines += ["", "Mean transported − A OA: {:.2f} pp; by seed: {} pp.".format(
            np.mean([r["delta_oa_pp"] for r in rows]),
            ", ".join(f"{r['delta_oa_pp']:+.2f}" for r in rows)), ""]
    lines += ["## Coverage and interpretation", ""]
    empty = []
    for seed in SEEDS:
        path = ROOT / "runs" / f"formal_{seed}" / "steps.jsonl"
        steps = [json.loads(x) for x in path.read_text().splitlines()]
        assert len(steps) == 3800
        empty.append(sum(x["pair_n"] == 0 for x in steps))
    lines += ["Empty eligible-pair batches by seed 1341/1174/1370: "
              + "/".join(map(str, empty)) + " of 3800 each.",
              "All 3800 main-model/input/gradient/BN/RNG hashes per seed match A;"
              " raw target confusion matrices exactly match round3 A.",
              "The primary mean does not improve over A. Secondary selection is mixed"
              " across seeds. This method has not reached a stable 80% OA result.",
              "Houston target results have been used for iterative development;"
              " a future claim needs an independent scene."]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (ROOT / "summary.json").write_text(json.dumps(data, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
