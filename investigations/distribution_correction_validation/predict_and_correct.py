"""Save locked new-seed target predictions and KL projection before GT audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "investigations/hypersigma_teacher_gate"))
from evaluate_matched import CenterPatches, SSFusionFramework, Backbone, probabilities, hash_file
from projection import project_kl

SEEDS = (202601, 202602, 202603)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(4)
    student_run = HERE / "runs" / f"student_{args.seed}"
    teacher_run = HERE / "runs" / f"seed_{args.seed}"
    out = HERE / "runs" / f"correction_{args.seed}"
    assert not out.exists(), f"Refusing to overwrite {out}"
    teacher_cfg = json.loads((teacher_run / "config.json").read_text())
    assert teacher_cfg["seed"] == args.seed
    assert teacher_cfg["source_split_sha256"] == hash_file(student_run / "source_split.npz")
    stage_files = [teacher_run / "stage1/best.pth", teacher_run / "full/best.pth"]
    stage_meta = [torch.load(f, map_location="cpu", weights_only=False)["best"] for f in stage_files]
    selected = int(stage_meta[1]["val_acc"] > stage_meta[0]["val_acc"])
    teacher_ckpt = stage_files[selected]
    teacher = SSFusionFramework(img_size=33, in_channels=48, patch_size=2,
                                classes=7, model_size="base")
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location="cpu", weights_only=False)["model"])
    teacher.to(args.device).eval()
    student_ckpt = student_run / "best_source_val.pth"
    student = Backbone().to(args.device).eval()
    student.load_state_dict(torch.load(student_ckpt, map_location="cpu", weights_only=False)["model"])
    cfg = json.loads((student_run / "config.json").read_text())
    assert cfg["seed"] == args.seed and cfg["method"] == "A"
    with np.load(student_run / "source_split.npz") as split:
        centers = split["target_centers"].copy()
    assert centers.shape == (53200, 2)
    with np.load(cfg["ilda_cache"]) as cache:
        ilda = cache["t"].copy()
    raw_path = Path(cfg["data"]) / "Houston18.mat"
    raw = hdf5storage.loadmat(str(raw_path))["ori_data"].astype(np.float32)
    assert raw.shape == ilda.shape == (210, 954, 48)
    teacher_q = probabilities(teacher, DataLoader(CenterPatches(raw, centers, 33),
                              batch_size=32, shuffle=False, num_workers=0), args.device)
    del teacher
    torch.cuda.empty_cache()
    student_q = probabilities(student, DataLoader(CenterPatches(ilda, centers[:53184], 7),
                              batch_size=32, shuffle=False, num_workers=0), args.device,
                              student=True)
    assert teacher_q.shape == (53200, 7) and student_q.shape == (53184, 7)
    assert np.isfinite(teacher_q).all() and np.isfinite(student_q).all()
    prior = teacher_q.astype(np.float64).mean(axis=0)
    corrected, projection = project_kl(student_q, prior)
    out.mkdir(parents=True)
    np.savez_compressed(out / "predictions_before_gt.npz", centers=centers,
                        teacher=teacher_q, A=student_q, B=corrected, prior=prior)
    metadata = {
        "seed": args.seed, "target_prior_n": 53200, "target_test_n": 53184,
        "prior": prior.tolist(), "projection": projection,
        "selected_teacher_stage": "full" if selected else "stage1",
        "teacher_checkpoint": str(teacher_ckpt),
        "teacher_checkpoint_sha256": hash_file(teacher_ckpt),
        "student_checkpoint": str(student_ckpt),
        "student_checkpoint_sha256": hash_file(student_ckpt),
        "student_source_val_epoch": int(torch.load(student_ckpt, map_location="cpu", weights_only=False)["epoch"]),
        "validation_lock_sha256": hash_file(HERE / "VALIDATION_LOCK.md"),
        "correction_code_sha256": hash_file(Path(__file__)),
        "projection_code_sha256": hash_file(HERE / "projection.py"),
        "teacher_train_code_sha256": hash_file(PROJECT / "investigations/hypersigma_teacher_gate/train_matched.py"),
        "target_gt_opened_by_correction_script": False,
    }
    (out / "selection_and_projection.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"seed": args.seed, "prior": prior.tolist(),
                      "residual": projection["prior_max_abs_error"],
                      "selected_stage": metadata["selected_teacher_stage"]}), flush=True)


if __name__ == "__main__":
    main()
