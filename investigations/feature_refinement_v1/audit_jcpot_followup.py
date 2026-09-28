"""Post-hoc semantic audit of the frozen A/F1/F2 JCPOT couplings."""
from __future__ import annotations

import json
from pathlib import Path

import hdf5storage
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SEEDS = (202601, 202602, 202603)
METHODS = ("A", "F1", "F2")
REGS = ("0p1", "0p2")


def measure(prior, gamma, posterior, source_labels, y):
    truth = np.bincount(y, minlength=7) / len(y)
    coupling_correct = float(sum(gamma[source_labels == c][:, y == c].sum()
                                 for c in range(7)) / gamma.sum())
    posterior_correct = float(posterior[np.arange(len(y)), y].mean())
    assert np.isclose(coupling_correct, posterior_correct, atol=2e-5)
    prediction = posterior.argmax(axis=1)
    return {"prior": prior.tolist(), "prior_l1_error": float(np.abs(prior - truth).sum()),
            "class7_prior_abs_error": float(abs(prior[6] - truth[6])),
            "coupling_expected_correct_class_mass": coupling_correct,
            "posterior_pilot_accuracy": float(np.mean(prediction == y)),
            "posterior_class7_recall": float(np.mean(prediction[y == 6] == 6)),
            "class7_correct_candidate_recall_at_0p8": float(
                np.mean(((prediction == 6) & (posterior.max(axis=1) >= 0.8))[y == 6]))}


def main():
    output = {"seeds": list(SEEDS), "methods": list(METHODS),
              "scope": "first 8192 official target positions; GT used post-hoc only",
              "results": {}}
    for seed in SEEDS:
        student = PROJECT / "investigations/distribution_correction_validation/runs" / f"student_{seed}"
        cfg = json.loads((student / "config.json").read_text())
        gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
        row = {}
        centers_reference = None
        for method in METHODS:
            if method == "A":
                path = PROJECT / "investigations/jcpot_pilot/runs" / str(seed) / "before_gt.npz"
                fit = json.loads((path.parent / "fit.json").read_text())
                assert fit["target_gt_opened"] is False
            else:
                path = HERE / "runs" / f"formal_{method}_{seed}" / "jcpot_before_gt.npz"
                fit = json.loads((path.parent / "jcpot_fit.json").read_text())
                assert fit["target_gt_opened_by_fit"] is False
            with np.load(path) as data:
                centers = data["centers"].copy()
                source_labels = data["source_labels"].copy()
                assert len(centers) == 8192
                if centers_reference is None:
                    centers_reference = centers
                else:
                    assert np.array_equal(centers, centers_reference)
                y = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
                reg_rows = {tag: measure(data[f"prior_{tag}"].copy(),
                                         data[f"gamma_{tag}"].copy(),
                                         data[f"posterior_{tag}"].copy(),
                                         source_labels, y)
                            for tag in REGS}
            reg_rows["prior_reg_sensitivity_l1"] = float(
                np.abs(np.asarray(reg_rows["0p1"]["prior"]) -
                       np.asarray(reg_rows["0p2"]["prior"])).sum())
            row[method] = reg_rows
        output["results"][str(seed)] = row
    path = HERE / "JCPOT_AUDIT.json"
    assert not path.exists(), f"Refusing to overwrite {path}"
    path.write_text(json.dumps(output, indent=2))
    for seed in SEEDS:
        for method in METHODS:
            r = output["results"][str(seed)][method]
            print(seed, method, "prior sensitivity", round(r["prior_reg_sensitivity_l1"], 3),
                  *[(tag, round(r[tag]["prior_l1_error"], 3),
                     round(r[tag]["coupling_expected_correct_class_mass"], 3),
                     round(r[tag]["posterior_class7_recall"], 3)) for tag in REGS])


if __name__ == "__main__":
    main()
