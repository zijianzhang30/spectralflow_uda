"""Aggregate every predeclared soft-KL strength without selecting a winner."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (202601, 202602, 202603)
GRID = ("lambda_0", "lambda_025", "lambda_05", "lambda_1", "lambda_2",
        "lambda_4", "lambda_8", "lambda_16", "hard")


def aggregate(values):
    values = np.asarray(values, dtype=np.float64)
    return {"per_seed": values.tolist(), "mean": float(values.mean()),
            "std_ddof1": float(values.std(ddof=1))}


def main():
    rows = [json.loads((HERE / "runs" / f"soft_{seed}" / "audit.json").read_text())
            for seed in SEEDS]
    assert [row["seed"] for row in rows] == list(SEEDS)
    assert all(tuple(row["grid"]) == GRID for row in rows)
    fields = ("oa", "aa", "kappa", "class7_recall", "candidate_coverage",
              "class7_correct_candidate_recall", "ot_expected_purity")
    report = {"seeds": list(SEEDS), "grid": list(GRID), "variants": {},
              "scope": "full predeclared curve on previously inspected Houston18; no target-selected lambda"}
    for name in GRID:
        entries = [row["variants"][name] for row in rows]
        report["variants"][name] = {}
        for field in fields:
            vals = []
            for entry in entries:
                classification = entry["classification"]
                corr = entry["correspondence"]
                value = {
                    "oa": classification["oa_official"],
                    "aa": classification["aa"],
                    "kappa": classification["kappa"],
                    "class7_recall": classification["recall"][6],
                    "candidate_coverage": corr["candidate_coverage_fraction"],
                    "class7_correct_candidate_recall": corr["true_class_candidate_recall"][6],
                    "ot_expected_purity": corr["pair_expected_purity_overall"],
                }[field]
                vals.append(value)
            report["variants"][name][field] = aggregate(vals)
        report["variants"][name]["soft_mass_fraction"] = {
            "per_seed": [entry["classification"]["soft_mass_fraction"] for entry in entries],
            "mean": np.mean([entry["classification"]["soft_mass_fraction"]
                             for entry in entries], axis=0).tolist()}
    base = report["variants"]["lambda_0"]
    for name in GRID:
        current = report["variants"][name]
        current["delta_vs_lambda_0"] = {
            field: aggregate(np.asarray(current[field]["per_seed"]) -
                             np.asarray(base[field]["per_seed"]))
            for field in fields}
    (HERE / "soft_kl_summary.json").write_text(json.dumps(report, indent=2))
    for name in GRID:
        row = report["variants"][name]
        print(name, *(f"{field}={row[field]['mean']:.5f}"
                      for field in ("oa", "aa", "class7_recall", "candidate_coverage",
                                    "class7_correct_candidate_recall", "ot_expected_purity")))


if __name__ == "__main__":
    main()
