"""Independent active-set audit of frozen mixture-prior solutions; no GT."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run import build_rff, dct_basis, phi, transform


def exact_pi(target, classes):
    m = classes.T.astype(np.float64)
    v = target.astype(np.float64)
    best = None
    for count in range(1, 8):
        for support in itertools.combinations(range(7), count):
            sub = m[:, support]
            kkt = np.block([[sub.T @ sub, np.ones((count, 1))],
                            [np.ones((1, count)), np.zeros((1, 1))]])
            rhs = np.concatenate((sub.T @ v, [1.0]))
            sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:count]
            if sol.min() < -1e-8:
                continue
            pi = np.zeros(7)
            pi[list(support)] = sol
            distance = float(np.sum((m @ pi - v) ** 2))
            if best is None or distance < best[0]:
                best = distance, pi
    assert best is not None
    return best


def main():
    torch.set_num_threads(2)
    with np.load(ROOT / "official_aligned/preprocessing/official_ilda.npz") as cache:
        source, target = cache["s"].copy(), cache["t"].copy()
    output = {}
    for seed in (202601, 202602, 202603):
        meta = json.loads((HERE / str(seed) / "metadata.json").read_text())
        split_path = ROOT / f"investigations/distribution_correction_validation/runs/student_{seed}/source_split.npz"
        with np.load(split_path) as split:
            sc, sy, tc = split["train_centers"], split["train_labels"], split["target_centers"]
        xs = source[sc[:, 0], sc[:, 1]].astype(np.float32)
        xt = target[tc[:, 0], tc[:, 1]].astype(np.float32)
        mu, sigma = xs.mean(0), np.maximum(xs.std(0), 1e-6)
        rng = np.random.RandomState(seed + 31415)
        fit = xt[rng.choice(len(xt), size=8192, replace=False)]
        w, phase, _ = build_rff((xs - mu) / sigma, seed, "cpu")
        xse = phi(torch.as_tensor((xs - mu) / sigma), w, phase).numpy()
        classes = np.stack([xse[sy == c].mean(0) for c in range(7)])
        basis = dct_basis("cpu")
        rows = {}
        for name, row in [("identity", {"u": [0.] * 6, "v": [0.] * 6,
                                          "discrepancy": meta["identity_discrepancy"],
                                          "pi": meta["identity_pi_only"]})] + list(meta["chosen"].items()):
            u, v = torch.as_tensor(row["u"]), torch.as_tensor(row["v"])
            a, b = transform(u, v, basis, torch.as_tensor(sigma))
            z = (torch.as_tensor(fit) * a + b - torch.as_tensor(mu)) / torch.as_tensor(sigma)
            target_mean = phi(z, w, phase).mean(0).numpy()
            exact_d, exact_weight = exact_pi(target_mean, classes)
            rows[name] = {
                "stored_distance": row["discrepancy"],
                "exact_distance": exact_d,
                "distance_relative_error": float((row["discrepancy"] - exact_d) / exact_d),
                "pi_l1_difference": float(np.abs(np.array(row["pi"]) - exact_weight).sum()),
            }
        output[str(seed)] = rows
        print(seed, {name: round(r["distance_relative_error"], 7)
                     for name, r in rows.items()}, flush=True)
    (HERE / "PI_AUDIT.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
