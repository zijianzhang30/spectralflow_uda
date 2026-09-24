"""Summarize the three frozen-feature correspondence screens."""
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
SEEDS = (1341, 1174, 1370)
METHODS = ("q95", "q80", "proto95", "proto80", "fused95", "fused80",
           "fused50_95", "fused50_80", "agree95")


def main():
    rows = [json.loads((HERE/f"screen_{s}.json").read_text()) for s in SEEDS]
    assert all(r["target_n"] == 53200 and r["pair_audit_batches"] == 200 for r in rows)
    for r in rows:
        support = r["target_true_support_by_class"]
        assert sum(support) == r["target_n"]
        assert r["fusion_alpha"] == 0
        for method in METHODS:
            m = r["methods"][method]
            cm = m["candidate_confusion_pred_by_true"]
            assert sum(sum(row) for row in cm) == sum(m["candidate_n_by_pred_class"])
            assert [sum(row) for row in cm] == m["candidate_n_by_pred_class"]
            for c in range(7):
                assert abs(m["true_class_candidate_recall"][c] - cm[c][c]/support[c]) < 1e-12
                assert m["pair_n_by_class"][c] % 32 == 0
        assert r["methods"]["fused95"] == r["methods"]["proto95"]
        assert r["methods"]["fused80"] == r["methods"]["proto80"]
    output = {"seeds": list(SEEDS),
              "source_val_alpha": {str(r["seed"]): r["fusion_alpha"] for r in rows},
              "methods": {}}
    keys = ("candidate_coverage_fraction", "candidate_micro_precision",
            "candidate_macro_recall", "pair_expected_purity_overall")
    for method in METHODS:
        output["methods"][method] = {
            "mean": {k: mean(r["methods"][method][k] for r in rows) for k in keys},
            "by_seed": {str(r["seed"]): dict(
                {k: r["methods"][method][k] for k in keys},
                classes_with_candidates=r["methods"][method]["classes_with_candidates"],
                true_class_candidate_recall=r["methods"][method]["true_class_candidate_recall"],
                pair_expected_purity_by_class=r["methods"][method]["pair_expected_purity_by_class"])
                        for r in rows}}
    (HERE/"summary.json").write_text(json.dumps(output, indent=2)+"\n")
    print("method coverage precision macro_recall pair_purity class_counts")
    for method in METHODS:
        row = output["methods"][method]
        print(method, *(f"{100*row['mean'][k]:.2f}" for k in keys),
              [row["by_seed"][str(s)]["classes_with_candidates"] for s in SEEDS])
    print("per-class correct-candidate recall (%)")
    for method in ("q95", "q80", "fused50_80", "agree95"):
        print(method, [[round(100*x, 1) for x in output["methods"][method]["by_seed"][str(s)]["true_class_candidate_recall"]] for s in SEEDS])
    print("validated: candidate confusion, correct-class recall, pair counts, calibrated fusion")


if __name__ == "__main__":
    main()
