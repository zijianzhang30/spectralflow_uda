"""Audit and aggregate the preregistered frozen A/B/C comparison."""
from __future__ import annotations

import json
from pathlib import Path

import hdf5storage
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SEEDS = (1341, 1174, 1370)


def main():
    rows = []
    sensitivity_rows = []
    gt = hdf5storage.loadmat(
        "/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/Houston18_7gt.mat")["map"]
    for seed in SEEDS:
        run = HERE / "runs" / f"seed_{seed}"
        audit = json.loads((run / "audit.json").read_text())
        with np.load(run / "corrected_probabilities.npz") as f:
            centers, q, b, c, prior = (f[k] for k in ("centers", "A", "B", "C", "prior"))
        y = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
        assert len(y) == 53184 and q.shape == b.shape == c.shape == (53184, 7)
        for arr in (q, b, c):
            assert np.allclose(arr.sum(1), 1, atol=1e-5)
        assert np.max(np.abs(b.astype(np.float64).mean(0) - prior)) < 1e-5
        low = q.max(1) < 0.8
        assert np.array_equal(c[low], q[low])
        assert np.array_equal(c[~low], b[~low])
        assert np.array_equal(centers, np.load(
            PROJECT / "investigations/hypersigma_teacher_gate/runs" /
            f"seed_{seed}/target_probabilities.npz")["centers"])
        true_prior = np.bincount(y, minlength=7) / len(y)
        row = {"seed": seed,
               "student_prior_tv_to_true_posthoc": float(np.abs(q.astype(np.float64).mean(0) - true_prior).sum() / 2),
               "foundation_prior_tv_to_true_posthoc": float(np.abs(prior - true_prior).sum() / 2),
               "low_confidence_fraction_original_q": float(low.mean()),
               "variants": {}}
        for name, arr in (("A", q), ("B", b), ("C", c)):
            m = audit["variants"][name]
            cm = np.bincount(7 * y + arr.argmax(1), minlength=49).reshape(7, 7)
            assert cm.tolist() == m["classification"]["confusion_matrix"]
            corr = m["correspondence"]
            assert sum(corr["candidate_n_by_pred_class"]) <= len(y)
            assert all(n % 32 == 0 for n in corr["pair_n_by_class"])
            row["variants"][name] = {
                "oa": m["classification"]["oa_official"],
                "aa": m["classification"]["aa"],
                "kappa": m["classification"]["kappa"],
                "class7_recall": m["classification"]["recall"][6],
                "class7_precision": m["classification"]["precision"][6],
                "class7_soft_mass": m["classification"]["soft_mass_fraction"][6],
                "changed_argmax_fraction": m["changed_argmax_fraction_vs_A"],
                "candidate_coverage": corr["candidate_coverage_fraction"],
                "candidate_micro_precision": corr["candidate_micro_precision"],
                "candidate_correct_class7_recall": corr["true_class_candidate_recall"][6],
                "pair_expected_purity": corr["pair_expected_purity_overall"],
                "pair_expected_class7_purity": corr["pair_expected_purity_by_class"][6],
                "class7_pair_enabled_batches": corr["pair_enabled_batches_by_class"][6],
            }
        official = json.loads((PROJECT / "experiments/round9/runs" /
                               f"formal_A_{seed}/final_target_source_val_best.json").read_text())
        assert row["variants"]["A"]["oa"] == official["metrics"]["oa"]
        rows.append(row)
        sensitivity = json.loads((run / "mean_teacher_fixed100_sensitivity.json").read_text())
        pair_sensitivity = json.loads((run / "mean_teacher_fixed100_pair_audit.json").read_text())
        official_mt = json.loads((PROJECT / "experiments/round9/runs" /
                                  f"formal_B_{seed}/final_target_fixed_epoch_100.json").read_text())
        assert sensitivity["variants"]["A"]["confusion_matrix"] == official_mt["metrics"]["confusion_matrix"]
        sr = {"seed": seed, "variants": {}}
        for name in ("A", "B", "C"):
            m, corr = sensitivity["variants"][name], pair_sensitivity["methods"][name]
            sr["variants"][name] = {
                "oa": m["oa_official"], "aa": m["aa"], "kappa": m["kappa"],
                "class7_recall": m["recall"][6],
                "class7_precision": m["precision"][6],
                "class7_soft_mass": m["soft_mass_fraction"][6],
                "candidate_coverage": corr["candidate_coverage_fraction"],
                "candidate_micro_precision": corr["candidate_micro_precision"],
                "candidate_correct_class7_recall": corr["true_class_candidate_recall"][6],
                "pair_expected_purity": corr["pair_expected_purity_overall"],
                "class7_pair_enabled_batches": corr["pair_enabled_batches_by_class"][6],
            }
        sensitivity_rows.append(sr)
    def aggregate_rows(group):
        aggregate = {}
        for name in ("A", "B", "C"):
            aggregate[name] = {}
            for key in group[0]["variants"][name]:
                if group[0]["variants"][name][key] is None:
                    continue
                values = np.asarray([row["variants"][name][key] for row in group], dtype=np.float64)
                if not np.isfinite(values).all():
                    continue
                aggregate[name][key] = {"mean": float(values.mean()),
                                        "std_ddof1": float(values.std(ddof=1))}
        return aggregate
    aggregate = aggregate_rows(rows)
    sensitivity_aggregate = aggregate_rows(sensitivity_rows)
    result = {"seeds": SEEDS, "candidate_gate": 0.8,
              "pair_audit_batches": 200, "oa_denominator": 53200,
              "target_gt_use": "post-hoc metrics and prior-error audit only",
              "rows": rows, "aggregate": aggregate,
              "sensitivity_rows": sensitivity_rows,
              "sensitivity_aggregate": sensitivity_aggregate}
    (HERE / "summary.json").write_text(json.dumps(result, indent=2))
    for name in ("A", "B", "C"):
        x = aggregate[name]
        print(name, "OA", round(100 * x["oa"]["mean"], 2),
              "AA", round(100 * x["aa"]["mean"], 2),
              "C7 recall", round(100 * x["class7_recall"]["mean"], 2),
              "candidate coverage", round(100 * x["candidate_coverage"]["mean"], 2),
              "pair purity", round(100 * x["pair_expected_purity"]["mean"], 2))
    for name in ("A", "B", "C"):
        x = sensitivity_aggregate[name]
        print("MeanTeacher", name, "OA", round(100 * x["oa"]["mean"], 2),
              "AA", round(100 * x["aa"]["mean"], 2),
              "C7 recall", round(100 * x["class7_recall"]["mean"], 2),
              "pair purity", round(100 * x["pair_expected_purity"]["mean"], 2))


if __name__ == "__main__":
    main()
