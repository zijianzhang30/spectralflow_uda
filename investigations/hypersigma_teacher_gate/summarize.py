"""Recompute the three-seed teacher gate from fixed probability caches."""
from __future__ import annotations

import json
from pathlib import Path

import hdf5storage
import numpy as np

HERE = Path(__file__).resolve().parent
GT = Path("/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/Houston18_7gt.mat")
SEEDS = (1341, 1174, 1370)


def main():
    gt = hdf5storage.loadmat(str(GT))["map"]
    rows = []
    for seed in SEEDS:
        run = HERE / "runs" / f"seed_{seed}"
        audit = json.loads((run / "target_audit.json").read_text())
        assert audit["student_matches_official_audit"]
        with np.load(run / "target_probabilities.npz") as f:
            centers, tq, sq = f["centers"], f["teacher"], f["student"]
        assert centers.shape == (53184, 2) and tq.shape == sq.shape == (53184, 7)
        y = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
        t, s = tq.argmax(1), sq.argmax(1)
        agree = t == s
        agree7 = agree & (t == 6)
        proposed7 = (t == 6) & (s != 6)
        row = {
            "seed": seed,
            "teacher_stage": audit["teacher_selected_stage"],
            "teacher_epoch": audit["stage_source_val"][1]["epoch"],
            "teacher_source_val": audit["stage_source_val"][1]["val_acc"],
            "teacher_oa": audit["teacher"]["oa_official"],
            "student_oa": audit["student"]["oa_official"],
            "teacher_aa": audit["teacher"]["aa"],
            "student_aa": audit["student"]["aa"],
            "teacher_kappa": audit["teacher"]["kappa"],
            "student_kappa": audit["student"]["kappa"],
            "teacher_class7_recall": audit["teacher"]["recall"][6],
            "student_class7_recall": audit["student"]["recall"][6],
            "teacher_class7_precision": audit["teacher"]["precision"][6],
            "student_class7_precision": audit["student"]["precision"][6],
            "teacher_class7_soft_mass_fraction": audit["teacher"]["soft_mass"][6] / len(y),
            "student_class7_soft_mass_fraction": audit["student"]["soft_mass"][6] / len(y),
            "teacher_ece_15": audit["teacher"]["ece_15"],
            "student_ece_15": audit["student"]["ece_15"],
            "teacher_only_correct": audit["complementarity"]["teacher_only_correct"],
            "student_only_correct": audit["complementarity"]["student_only_correct"],
            "teacher_only_class7_correct": audit["complementarity"]["per_class"][6]["teacher_only_correct"],
            "agreement_coverage": float(np.mean(agree)),
            "agreement_purity": float(np.mean(s[agree] == y[agree])),
            "agreement_correct_class7_recall": float(np.sum(agree7 & (y == 6)) / np.sum(y == 6)),
            "teacher7_student_not7_count": int(np.sum(proposed7)),
            "teacher7_student_not7_precision": float(np.mean(y[proposed7] == 6)),
        }
        rows.append(row)
    stats = {}
    for key in rows[0]:
        if isinstance(rows[0][key], (int, float)) and key not in ("seed", "teacher_epoch"):
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            stats[key] = {"mean": float(values.mean()), "std_ddof1": float(values.std(ddof=1))}
    result = {"seeds": SEEDS, "n_evaluated_per_seed": 53184,
              "oa_denominator": 53200,
              "protocol": "source-val selected teacher, source-val selected DCRN; target GT post-hoc only",
              "rows": rows, "stats": stats}
    (HERE / "summary.json").write_text(json.dumps(result, indent=2))
    for row in rows:
        print(row["seed"], "teacher OA", round(100 * row["teacher_oa"], 2),
              "student OA", round(100 * row["student_oa"], 2),
              "teacher class7 recall", round(100 * row["teacher_class7_recall"], 2))


if __name__ == "__main__":
    main()
