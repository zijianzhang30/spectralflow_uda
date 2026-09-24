"""Aggregate the three locked validation seeds and apply the predeclared gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (202601, 202602, 202603)


def values(rows, variant, field):
    return np.array([row["variants"][variant]["classification"][field] for row in rows])


def diagnostic(rows, variant, field):
    return np.array([row["variants"][variant]["correspondence"][field] for row in rows])


def summary(a):
    return {"per_seed": a.tolist(), "mean": float(a.mean()), "std_ddof1": float(a.std(ddof=1))}


def main():
    rows = [json.loads((HERE / "runs" / f"correction_{seed}" / "audit.json").read_text())
            for seed in SEEDS]
    assert [row["seed"] for row in rows] == list(SEEDS)
    fields = {}
    for name in ("A", "B"):
        fields[name] = {
            "oa": summary(values(rows, name, "oa_official")),
            "aa": summary(values(rows, name, "aa")),
            "kappa": summary(values(rows, name, "kappa")),
            "class7_recall": summary(np.array([row["variants"][name]["classification"]["recall"][6]
                                               for row in rows])),
            "candidate_coverage": summary(diagnostic(rows, name, "candidate_coverage_fraction")),
            "class7_correct_candidate_recall": summary(np.array([
                row["variants"][name]["correspondence"]["true_class_candidate_recall"][6]
                for row in rows])),
            "ot_expected_purity": summary(diagnostic(rows, name, "pair_expected_purity_overall")),
        }
    differences = {key: summary(np.array(fields["B"][key]["per_seed"]) -
                                np.array(fields["A"][key]["per_seed"]))
                   for key in fields["A"]}
    passed = {
        "oa_mean_positive": differences["oa"]["mean"] > 0,
        "aa_mean_drop_at_most_1pp": differences["aa"]["mean"] >= -0.01,
        "class7_mean_drop_at_most_5pp": differences["class7_recall"]["mean"] >= -0.05,
        "class7_each_seed_drop_at_most_10pp": min(differences["class7_recall"]["per_seed"]) >= -0.10,
        "coverage_mean_drop_at_most_5pp": differences["candidate_coverage"]["mean"] >= -0.05,
        "class7_candidate_recall_mean_drop_at_most_5pp":
            differences["class7_correct_candidate_recall"]["mean"] >= -0.05,
        "ot_purity_mean_positive": differences["ot_expected_purity"]["mean"] > 0,
    }
    report = {"seeds": list(SEEDS), "variants": fields, "B_minus_A": differences,
              "flow_progression_gate": {"checks": passed, "pass": all(passed.values())},
              "scope": "new-seed stability on already-inspected Houston18; not untouched target-domain test"}
    (HERE / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
