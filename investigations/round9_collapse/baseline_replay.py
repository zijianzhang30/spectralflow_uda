"""Deterministic A replay with the same unlabeled target probe as B/C."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"experiments/round9"))
import train
from data import Patches, file_hash, load_images
from runtime import atomic_json, evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--probe-n", type=int, default=2048)
    args = parser.parse_args()
    formal = REPO/"experiments/round9/runs"/f"formal_A_{args.seed}"
    cfg = json.loads((formal/"config.json").read_text())
    provenance = json.loads((formal/"provenance.json").read_text())
    assert file_hash(REPO/"experiments/round9/train.py") == provenance["code"][str(REPO/"experiments/round9/train.py")]
    out = args.out.resolve()
    original_validation = train.source_validation
    epochs = 0
    probe_loader = None
    records = []

    @torch.no_grad()
    def probe(student, epoch):
        nonlocal probe_loader
        if probe_loader is None:
            with np.load(out/"source_split.npz") as split:
                centers = split["target_centers"][:args.probe_n].copy()
            _, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
            probe_loader = DataLoader(Patches(target, centers), batch_size=32,
                                      shuffle=False, drop_last=False)
        mass = torch.zeros(7, dtype=torch.float64)
        hard = torch.zeros(7, dtype=torch.int64)
        entropy = 0.0
        n = 0
        with evaluation(student):
            for x in probe_loader:
                _, logits = student(x.to(args.device))
                q = logits.softmax(1)
                mass += q.double().sum(0).cpu()
                hard += torch.bincount(q.argmax(1).cpu(), minlength=7)
                entropy += float((-(q*q.clamp_min(1e-12).log()).sum(1)).sum())
                n += len(x)
        assert n==args.probe_n
        records.append({"epoch":epoch,"n":n,
                        "student_soft_mass":(mass/n).tolist(),
                        "student_hard_fraction":(hard.double()/n).tolist(),
                        "student_mean_entropy":entropy/n})
        atomic_json(out/"student_trajectory.json",records)

    def hooked_validation(model, loader, device):
        nonlocal epochs
        metrics = original_validation(model, loader, device)
        epochs += 1
        probe(model, epochs)
        return metrics

    train.source_validation = hooked_validation
    sys.argv = [str(REPO/"experiments/round9/train.py"),
                "--method","A","--seed",str(args.seed),
                "--device",args.device,"--out",str(out)]
    train.main()
    assert epochs==100 and len(records)==100
    formal_steps = [json.loads(line) for line in (formal/"steps.jsonl").read_text().splitlines()]
    replay_steps = [json.loads(line) for line in (out/"steps.jsonl").read_text().splitlines()]
    keys = ("model_hash","backbone_hash","classifier_hash","buffers_hash",
            "gradient_hash","source_logits_hash","source_input_hash",
            "target_input_hash","cpu_rng_hash")
    assert len(formal_steps)==len(replay_steps)==3800
    diffs = {k:sum(a[k]!=b[k] for a,b in zip(formal_steps,replay_steps)) for k in keys}
    assert all(v==0 for v in diffs.values())
    atomic_json(out/"replay_audit.json",{"method":"A","seed":args.seed,
                "probe_n":args.probe_n,"epochs":100,"steps":3800,
                "formal_step_hash_differences":diffs})
    print(json.dumps({"seed":args.seed,"formal_step_hash_differences":diffs},indent=2))


if __name__=="__main__":
    main()
