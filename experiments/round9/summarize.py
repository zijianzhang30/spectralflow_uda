"""Validate formal runs, paired audits, target tests and correspondence screens."""
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SEEDS = (1341, 1174, 1370)
METHODS = ("A", "B", "C")
SELECTIONS = ("fixed_epoch_100", "source_val_best")
AUDIT_KEYS = ("model_hash", "backbone_hash", "classifier_hash", "buffers_hash",
              "gradient_hash", "source_logits_hash", "source_input_hash",
              "target_input_hash", "cpu_rng_hash")
SHARED_KEYS = ("source_input_hash", "target_input_hash", "cpu_rng_hash")


def read(path):
    return json.loads(path.read_text())


def steps(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    output = {"seeds": list(SEEDS), "audit": {}, "selections": {}, "correspondence_fixed100": {}}
    runs = {}
    for seed in SEEDS:
        rows = {}
        for method in METHODS:
            run = ROOT/"runs"/f"formal_{method}_{seed}"
            cfg, complete, history = (read(run/name) for name in
                                      ("config.json", "training_complete.json", "history.json"))
            assert cfg["method"]==method and cfg["seed"]==seed
            assert complete["formal"] and complete["epochs"]==100
            assert len(history)==100
            log = steps(run/"steps.jsonl")
            assert len(log)==3800
            assert all(sum(row["epoch"]==e for row in log)==38 for e in range(1,101))
            rows[method] = log
            runs[(method,seed)] = run
        old = steps(REPO/"experiments/round8/runs"/f"formal_{seed}"/"steps.jsonl")
        assert len(old)==3800
        a_old = {key: sum(a[key]!=b[key] for a,b in zip(rows["A"],old)) for key in AUDIT_KEYS}
        shared = {method: {key: sum(a[key]!=b[key] for a,b in zip(rows["A"],rows[method]))
                           for key in SHARED_KEYS} for method in ("B","C")}
        assert all(v==0 for v in a_old.values())
        assert all(v==0 for row in shared.values() for v in row.values())
        assert all(abs(row["weight_mean"]-1)<1e-6 for row in rows["C"])
        output["audit"][str(seed)] = {"steps":3800,"A_vs_round8_CE_hash_differences":a_old,
                                       "shared_input_rng_hash_differences":shared,
                                       "B_C_consistency_changed_steps":sum(b["consistency"]!=c["consistency"] for b,c in zip(rows["B"],rows["C"])),
                                       "B_C_gradient_changed_steps":sum(b["gradient_hash"]!=c["gradient_hash"] for b,c in zip(rows["B"],rows["C"]))}
    for selection in SELECTIONS:
        methods = {}
        for method in METHODS:
            by_seed = {}
            for seed in SEEDS:
                row = read(runs[(method,seed)]/f"final_target_{selection}.json")
                assert row["dataset_n"]==53200 and row["dropped_n"]==16
                assert row["metrics"]["evaluated_n"]==53184
                assert row["selection"]==selection
                if method=="A":
                    assert row["matches_round3_A"]
                by_seed[str(seed)] = {"epoch":row["selected_epoch"],
                                      "oa":row["metrics"]["oa"],
                                      "aa":row["metrics"]["aa"],
                                      "kappa":row["metrics"]["kappa"],
                                      "per_class_accuracy":row["metrics"]["per_class_accuracy"]}
            methods[method] = {"by_seed":by_seed,
                               "mean": {k:mean(by_seed[str(s)][k] for s in SEEDS) for k in ("oa","aa","kappa")},
                               "std_ddof1": {k:stdev(by_seed[str(s)][k] for s in SEEDS) for k in ("oa","aa","kappa")},
                               "per_class_mean": [mean(by_seed[str(s)]["per_class_accuracy"][c] for s in SEEDS) for c in range(7)]}
        output["selections"][selection] = {"methods":methods,
            "paired_oa_delta_pp": {method:{str(s):100*(methods[method]["by_seed"][str(s)]["oa"]-methods["A"]["by_seed"][str(s)]["oa"])
                                           for s in SEEDS} for method in ("B","C")}}
    for method in METHODS:
        by_seed = {}
        for seed in SEEDS:
            row = read(REPO/"investigations/correspondence_v1"/f"screen_round9_{method}_{seed}.json")
            assert row["target_n"]==53200 and row["pair_audit_batches"]==200
            assert abs(sum(row["target_soft_mass_q_by_class"])-53200)<.1
            assert abs(sum(row["target_soft_mass_p_by_class"])-53200)<.1
            q95 = row["methods"]["q95"]
            by_seed[str(seed)] = {
                "q95_pair_expected_purity":q95["pair_expected_purity_overall"],
                "q95_candidate_coverage":q95["candidate_coverage_fraction"],
                "q95_candidate_macro_recall":q95["candidate_macro_recall"],
                "q95_candidate_class7_recall":q95["true_class_candidate_recall"][6],
                "q95_pair_class5_purity":q95["pair_expected_purity_by_class"][4],
                "q95_classes_with_candidates":q95["classes_with_candidates"],
                "soft_mass_fraction_by_class":[x/53200 for x in row["target_soft_mass_q_by_class"]],
                "mean_entropy":row["target_mean_entropy_q"],
                "raw_class7_recall":row["raw_recall_by_class"][6]}
        keys = ("q95_pair_expected_purity","q95_candidate_coverage",
                "q95_candidate_macro_recall","q95_candidate_class7_recall",
                "soft_mass_fraction_by_class","mean_entropy","raw_class7_recall")
        output["correspondence_fixed100"][method] = {
            "by_seed":by_seed,
            "mean": {k:([mean(by_seed[str(s)][k][c] for s in SEEDS) for c in range(7)]
                         if k=="soft_mass_fraction_by_class" else mean(by_seed[str(s)][k] for s in SEEDS))
                     for k in keys}}
    (ROOT/"summary.json").write_text(json.dumps(output,indent=2)+"\n")
    for selection, detail in output["selections"].items():
        print(selection)
        for method in METHODS:
            m = detail["methods"][method]
            print(method,"OA",round(100*m["mean"]["oa"],2),"±",round(100*m["std_ddof1"]["oa"],2),
                  "AA",round(100*m["mean"]["aa"],2),
                  "class7",round(100*m["per_class_mean"][6],2))
    print("correspondence fixed100")
    for method in METHODS:
        m = output["correspondence_fixed100"][method]["mean"]
        print(method,"pair purity",round(100*m["q95_pair_expected_purity"],2),
              "class7 candidate recall",round(100*m["q95_candidate_class7_recall"],2),
              "soft class6/7",round(100*m["soft_mass_fraction_by_class"][5],2),
              round(100*m["soft_mass_fraction_by_class"][6],2))
    print("PASS: 9 complete runs, 3800-step audit, 18 tests, 9 correspondence screens")


if __name__=="__main__":
    main()
