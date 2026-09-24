"""Deterministic B/C replay with read-only per-epoch teacher/student probes."""
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
from runtime import atomic_json, evaluation, tensor_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("B", "C"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--probe-n", type=int, default=2048)
    args = parser.parse_args()
    formal = REPO/"experiments/round9/runs"/f"formal_{args.method}_{args.seed}"
    cfg = json.loads((formal/"config.json").read_text())
    provenance = json.loads((formal/"provenance.json").read_text())
    assert file_hash(REPO/"experiments/round9/train.py") == provenance["code"][str(REPO/"experiments/round9/train.py")]
    assert cfg["method"]==args.method and cfg["seed"]==args.seed
    out = args.out.resolve()
    original_update = train.update_teacher
    updates = 0
    probe_loader = None
    records = []

    @torch.no_grad()
    def probe(student, teacher, epoch):
        nonlocal probe_loader
        if probe_loader is None:
            with np.load(out/"source_split.npz") as split:
                centers = split["target_centers"][:args.probe_n].copy()
            _, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
            probe_loader = DataLoader(Patches(target, centers), batch_size=32,
                                      shuffle=False, drop_last=False)
        mass_teacher = torch.zeros(7, dtype=torch.float64)
        mass_student = torch.zeros(7, dtype=torch.float64)
        hard_teacher = torch.zeros(7, dtype=torch.int64)
        hard_student = torch.zeros(7, dtype=torch.int64)
        entropy_teacher = entropy_student = 0.0
        count = 0
        with evaluation(student):
            teacher.eval()
            for x in probe_loader:
                x = x.to(args.device)
                _, t_logits = teacher(x)
                _, s_logits = student(x)
                tq, sq = t_logits.softmax(1), s_logits.softmax(1)
                mass_teacher += tq.double().sum(0).cpu()
                mass_student += sq.double().sum(0).cpu()
                hard_teacher += torch.bincount(tq.argmax(1).cpu(), minlength=7)
                hard_student += torch.bincount(sq.argmax(1).cpu(), minlength=7)
                entropy_teacher += float((-(tq*tq.clamp_min(1e-12).log()).sum(1)).sum())
                entropy_student += float((-(sq*sq.clamp_min(1e-12).log()).sum(1)).sum())
                count += len(x)
        assert count==args.probe_n
        row = {"epoch": epoch, "n": count,
               "teacher_soft_mass": (mass_teacher/count).tolist(),
               "student_soft_mass": (mass_student/count).tolist(),
               "teacher_hard_fraction": (hard_teacher.double()/count).tolist(),
               "student_hard_fraction": (hard_student.double()/count).tolist(),
               "teacher_mean_entropy": entropy_teacher/count,
               "student_mean_entropy": entropy_student/count}
        records.append(row)
        atomic_json(out/"teacher_trajectory.json", records)

    def hooked_update(teacher, student):
        nonlocal updates
        original_update(teacher, student)
        updates += 1
        if updates % 38 == 0:
            probe(student, teacher, updates//38)

    train.update_teacher = hooked_update
    sys.argv = [str(REPO/"experiments/round9/train.py"),
                "--method", args.method, "--seed", str(args.seed),
                "--device", args.device, "--out", str(out)]
    train.main()
    assert updates==3800 and len(records)==100
    formal_steps = [json.loads(line) for line in (formal/"steps.jsonl").read_text().splitlines()]
    replay_steps = [json.loads(line) for line in (out/"steps.jsonl").read_text().splitlines()]
    keys = ("model_hash", "backbone_hash", "classifier_hash", "buffers_hash",
            "gradient_hash", "source_logits_hash", "source_input_hash",
            "target_input_hash", "cpu_rng_hash")
    assert len(formal_steps)==len(replay_steps)==3800
    differences = {key:sum(a[key]!=b[key] for a,b in zip(formal_steps,replay_steps))
                   for key in keys}
    assert all(v==0 for v in differences.values())
    # The teacher must also be identical, although it is absent from the step log.
    cp_formal = torch.load(formal/"last.pth", map_location="cpu", weights_only=False)
    cp_replay = torch.load(out/"last.pth", map_location="cpu", weights_only=False)
    assert tensor_hash(cp_formal["teacher"].values())==tensor_hash(cp_replay["teacher"].values())
    atomic_json(out/"replay_audit.json", {"method":args.method,"seed":args.seed,
                "probe_n":args.probe_n,"probe_order":"first official-order target centers, no labels",
                "epochs":100,"steps":3800,"formal_step_hash_differences":differences,
                "teacher_checkpoint_exact_match":True})
    print(json.dumps({"method":args.method,"seed":args.seed,
                      "formal_step_hash_differences":differences,
                      "teacher_checkpoint_exact_match":True},indent=2))


if __name__=="__main__":
    main()
