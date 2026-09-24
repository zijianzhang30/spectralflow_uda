"""Aggregate round7 with the frozen round3 paired controls."""
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
SEEDS = (1341, 1174, 1370)
METHODS = {
    "A": ROOT.parent / "round3" / "runs" / "formal_A_{}",
    "linear_transport": ROOT.parent / "round3" / "runs" / "formal_linear_cpu_{}",
    "flow_transport_detached": ROOT.parent / "round3" / "runs" / "formal_flow_cpu_{}",
    "flow_transport_coupled": ROOT / "runs" / "formal_coupled_{}",
}


def run():
    result = {"seeds": SEEDS, "selections": {}}
    for selection in ("fixed_epoch_100", "source_val_best"):
        section = {}
        for method, template in METHODS.items():
            rows = []
            for seed in SEEDS:
                path = Path(str(template).format(seed)) / f"final_target_{selection}.json"
                row = json.loads(path.read_text())
                assert row["seed"] == seed and row["selection"] == selection
                assert row["dataset_n"] == 53200 and row["evaluated_n"] == 53184
                rows.append(row)
            section[method] = {
                "oa_mean": mean(row["oa"] for row in rows),
                "oa_std_ddof1": stdev(row["oa"] for row in rows),
                "aa_mean": mean(row["aa"] for row in rows),
                "aa_std_ddof1": stdev(row["aa"] for row in rows),
                "kappa_mean": mean(row["kappa"] for row in rows),
                "kappa_std_ddof1": stdev(row["kappa"] for row in rows),
                "oa_by_seed": {str(seed): row["oa"] for seed, row in zip(SEEDS, rows)},
                "aa_by_seed": {str(seed): row["aa"] for seed, row in zip(SEEDS, rows)},
                "selected_epochs": {str(seed): row["selected_epoch"] for seed, row in zip(SEEDS, rows)},
                "per_class_mean": [mean(row["per_class_accuracy"][c] for row in rows)
                                   for c in range(7)],
            }
        coupled = section["flow_transport_coupled"]
        section["paired_oa_delta_pp"] = {
            control: {str(seed): 100 * (coupled["oa_by_seed"][str(seed)]
                                        - section[control]["oa_by_seed"][str(seed)])
                      for seed in SEEDS}
            for control in ("A", "linear_transport", "flow_transport_detached")
        }
        result["selections"][selection] = section
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for selection, section in result["selections"].items():
        print(selection)
        for method in METHODS:
            row = section[method]
            print(method, "OA", round(100 * row["oa_mean"], 2), "+/-",
                  round(100 * row["oa_std_ddof1"], 2), "AA",
                  round(100 * row["aa_mean"], 2), "Kappa", round(row["kappa_mean"], 4))
        print("paired delta pp", section["paired_oa_delta_pp"])


if __name__ == "__main__":
    run()
