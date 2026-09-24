"""Summarize only completed, predeclared three-seed round3 evaluations."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = (1341, 1174, 1370)
METHODS = {"A": "A", "linear_transport": "linear_cpu",
           "flow_transport": "flow_cpu"}
SELECTIONS = ("fixed_epoch_100", "source_val_best")


def load(method, tag, seed, selection):
    run = ROOT / "runs" / f"formal_{tag}_{seed}"
    complete = json.loads((run / "training_complete.json").read_text())
    history = json.loads((run / "history.json").read_text())
    result = json.loads((run / f"final_target_{selection}.json").read_text())
    assert complete["formal"] and complete["epochs"] == len(history) == 100
    assert result["method"] == method and result["seed"] == seed
    assert result["selection"] == selection and result["evaluated_n"] == 53184
    assert result["dataset_n"] == 53200 and result["dropped_n"] == 16
    cm = np.asarray(result["confusion_matrix"])
    assert int(cm.sum()) == 53184
    assert abs(result["oa"] - np.trace(cm)/53200) < 1e-12
    assert abs(result["aa"] - np.mean(np.diag(cm)/cm.sum(1))) < 1e-12
    assert len(result["per_class_accuracy"]) == 7
    assert result["selected_epoch"] == (100 if selection == "fixed_epoch_100" else
                                         max(history, key=lambda r: r["source_val_accuracy"])["epoch"])
    return result


def stats(rows, key):
    values = np.asarray([r[key] for r in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def main():
    data = {selection: {method: [load(method, tag, seed, selection)
                                 for seed in SEEDS] for method, tag in METHODS.items()}
            for selection in SELECTIONS}
    reference = {}
    for seed in SEEDS:
        row = json.loads((ROOT / f"reference_fixed100_{seed}.json").read_text())
        assert row["seed"] == seed and row["selection"] == "fixed_epoch_100"
        reference[seed] = row
    data["fixed_epoch_100"]["MLUDA_full"] = [reference[s] for s in SEEDS]
    mluda_best = []
    for seed in SEEDS:
        path = ROOT.parents[1] / f"official_aligned/runs/round1/formal/{seed}/MLUDA_full/final_target.json"
        row = json.loads(path.read_text())
        assert row["seed"] == seed and row["selection"] == "source_val_best"
        mluda_best.append(row)
    data["source_val_best"]["MLUDA_full"] = mluda_best
    rows = ["# Round3: source-labelled transport views", "",
            "Protocol: frozen Houston13→Houston18 data/training/test setting; 100 epochs,",
            "38 updates/epoch, three seeds. Primary checkpoint is fixed epoch100;",
            "source-val-best is a separately predeclared secondary table.", "",
            "Seed1341 target outcomes were seen before the other two seeds finished;",
            "this Houston set is a development benchmark, not an untouched test.", ""]
    for selection in SELECTIONS:
        rows += [f"## {selection}", "",
                 "| Method | OA (%) | AA (%) | Kappa | Seed OAs (%) | Epochs |",
                 "|---|---:|---:|---:|---|---|"]
        for method, items in data[selection].items():
            oa, oa_std = stats(items, "oa")
            aa, aa_std = stats(items, "aa")
            kappa, k_std = stats(items, "kappa")
            rows.append(f"| {method} | {oa*100:.2f} ± {oa_std*100:.2f} | "
                        f"{aa*100:.2f} ± {aa_std*100:.2f} | {kappa:.4f} ± {k_std:.4f} | "
                        + ", ".join(f"{r['oa']*100:.2f}" for r in items) + " | "
                        + ", ".join(str(r["selected_epoch"]) for r in items) + " |")
        rows += [""]
        if selection == "fixed_epoch_100":
            a = data[selection]["A"]
            linear = data[selection]["linear_transport"]
            flow = data[selection]["flow_transport"]
            rows += [f"Flow − A mean OA: {np.mean([100*(x['oa']-y['oa']) for x,y in zip(flow,a)]):+.2f} pp.",
                     f"Flow − linear mean OA: {np.mean([100*(x['oa']-y['oa']) for x,y in zip(flow,linear)]):+.2f} pp.", ""]
    rows += ["## Per-class accuracy, fixed epoch100 (%)", "",
             "| Method | Class1 | Class2 | Class3 | Class4 | Class5 | Class6 | Class7 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, items in data["fixed_epoch_100"].items():
        mean = np.mean([r["per_class_accuracy"] for r in items], axis=0)
        rows.append("| " + method + " | " + " | ".join(f"{x*100:.2f}" for x in mean) + " |")
    rows += ["", "Both auxiliary methods use the same detached FM, OT pairs,"
             " Flow network and auxiliary CE weight. Only the view displacement differs.",
             "Target labels were used to reproduce official center order and for final scoring,"
             " not for training or checkpoint selection."]
    (ROOT / "RESULTS.md").write_text("\n".join(rows) + "\n")
    (ROOT / "summary.json").write_text(json.dumps(data, indent=2))
    print("\n".join(rows[:25]))


if __name__ == "__main__":
    main()
