"""Aggregate the complete predeclared feature-refinement grid."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIELDS = ("oa_official", "aa", "kappa", "class7_recall",
          "class7_correct_candidate_recall", "candidate_coverage")


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"per_seed": a.tolist(), "mean": float(a.mean()),
            "std_ddof1": float(a.std(ddof=1))}


def main():
    audit = json.loads((HERE / "AUDIT.json").read_text())
    seeds = audit["seeds"]
    out = {"seeds": seeds, "methods": audit["methods"],
           "variants": ("raw", "corrected"),
           "scope": "all predeclared 3-seed outcomes, no target-GT-selected winner",
           "results": {}}
    for method in audit["methods"]:
        out["results"][method] = {}
        for variant in out["variants"]:
            vals = {field: [] for field in FIELDS}
            epochs = []
            for seed in seeds:
                row = audit["results"][str(seed)]["methods"][method]
                epochs.append(row["selected_epoch"])
                v = row["variants"][variant]
                for field in FIELDS:
                    vals[field].append({
                        "oa_official": v["metrics"]["oa_official"],
                        "aa": v["metrics"]["aa"],
                        "kappa": v["metrics"]["kappa"],
                        "class7_recall": v["metrics"]["recall"][6],
                        "class7_correct_candidate_recall": v["class7_correct_candidate_recall"],
                        "candidate_coverage": v["candidate_coverage"],
                    }[field])
            out["results"][method][variant] = {field: stats(values)
                                                  for field, values in vals.items()}
            out["results"][method][variant]["selected_epochs"] = epochs
    for method in ("F1", "F2"):
        for variant in out["variants"]:
            row = out["results"][method][variant]
            base = out["results"]["A"][variant]
            row["delta_vs_A_same_variant"] = {
                field: stats(np.asarray(row[field]["per_seed"]) -
                             np.asarray(base[field]["per_seed"]))
                for field in FIELDS}
    path = HERE / "SUMMARY.json"
    assert not path.exists(), f"Refusing to overwrite {path}"
    path.write_text(json.dumps(out, indent=2))
    for method in audit["methods"]:
        for variant in out["variants"]:
            row = out["results"][method][variant]
            print(method, variant, "epoch", row["selected_epochs"],
                  *(f"{field}={row[field]['mean']:.5f}±{row[field]['std_ddof1']:.5f}"
                    for field in FIELDS))


if __name__ == "__main__":
    main()
