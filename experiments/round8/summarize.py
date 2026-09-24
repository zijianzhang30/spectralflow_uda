"""Verify complete backbone isolation and summarize matched round6/8 results."""
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
SEEDS = (1341, 1174, 1370)
HASH_KEYS = ("source_input_hash", "target_input_hash", "model_hash",
             "backbone_hash", "classifier_hash", "buffers_hash", "gradient_hash",
             "source_logits_hash", "cpu_rng_hash", "cuda_rng_hash")


def rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def metric_summary(values):
    return {"mean": mean(values), "std_ddof1": stdev(values),
            "by_seed": {str(s): v for s, v in zip(SEEDS, values)}}


def main():
    output = {"seeds": SEEDS, "full_step_audit": {}, "selections": {}}
    for seed in SEEDS:
        old = ROOT.parent/"round6"/"runs"/f"formal_{seed}"
        new = ROOT/"runs"/f"formal_{seed}"
        a, b = rows(old/"steps.jsonl"), rows(new/"steps.jsonl")
        assert len(a) == len(b) == 3800
        different = {key: sum(x[key] != y[key] for x, y in zip(a, b))
                     for key in HASH_KEYS}
        assert all(n == 0 for n in different.values())
        assert all(x["pair_n"] == y["pair_n"] for x, y in zip(a, b))
        output["full_step_audit"][str(seed)] = {
            "steps": 3800, "hash_differences": different,
            "pair_count_differences": 0,
            "fm_value_changed_steps": sum(x["fm"] != y["fm"] for x, y in zip(a, b))}

    for selection in ("fixed_epoch_100", "source_val_best"):
        old_results, new_results = [], []
        for seed in SEEDS:
            old_path = ROOT.parent/"round6"/"runs"/f"formal_{seed}"/f"final_target_{selection}.json"
            new_path = ROOT/"runs"/f"formal_{seed}"/f"final_target_{selection}.json"
            old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
            assert old["seed"] == new["seed"] == seed
            assert old["raw"]["confusion_matrix"] == new["raw"]["confusion_matrix"]
            assert old["selected_epoch"] == new["selected_epoch"]
            assert old["raw"]["oa"] == new["raw"]["oa"]
            old_results.append(old); new_results.append(new)
        output["selections"][selection] = {
            "selected_epochs": {str(s): r["selected_epoch"] for s, r in zip(SEEDS, new_results)},
            "raw_oa": metric_summary([r["raw"]["oa"] for r in new_results]),
            "round6_transported_oa": metric_summary([r["transported"]["oa"] for r in old_results]),
            "round8_transported_oa": metric_summary([r["transported"]["oa"] for r in new_results]),
            "round6_transported_aa": metric_summary([r["transported"]["aa"] for r in old_results]),
            "round8_transported_aa": metric_summary([r["transported"]["aa"] for r in new_results]),
            "round6_transported_kappa": metric_summary([r["transported"]["kappa"] for r in old_results]),
            "round8_transported_kappa": metric_summary([r["transported"]["kappa"] for r in new_results]),
            "paired_oa_delta_pp": {str(s): 100*(new["transported"]["oa"]-
                                              old["transported"]["oa"])
                                   for s, old, new in zip(SEEDS, old_results, new_results)},
        }
    (ROOT/"summary.json").write_text(json.dumps(output, indent=2)+"\n")
    for selection, data in output["selections"].items():
        print(selection)
        for key in ("raw_oa", "round6_transported_oa", "round8_transported_oa",
                    "round6_transported_aa", "round8_transported_aa"):
            x = data[key]
            print(key, round(100*x["mean"], 3), "+/-", round(100*x["std_ddof1"], 3))
        print("paired delta pp", data["paired_oa_delta_pp"])


if __name__ == "__main__":
    main()
