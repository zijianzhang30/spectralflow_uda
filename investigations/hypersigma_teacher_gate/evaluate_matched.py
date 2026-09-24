"""Post-hoc HyperSIGMA–DCRN probability and complementarity audit.

Both checkpoints are selected before this script opens target ground truth.
The saved official target-center order comes from the DCRN run artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
LEGACY = Path("/home/zhangzj26/TGRS_MLUDA-2024")
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(LEGACY / "third_party/HyperSIGMA/ImageClassification"))
from hypersigma_teacher_smoke_test import SSFusionFramework  # noqa: E402
spec = importlib.util.spec_from_file_location("round9_dcrn_model", PROJECT / "experiments/round9/model.py")
student_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(student_module)
Backbone = student_module.Backbone


class CenterPatches(Dataset):
    def __init__(self, cube, centers, size):
        self.centers = centers
        self.size = size
        half = size // 2
        self.padded = np.pad(cube.astype(np.float32), ((half, half), (half, half), (0, 0)),
                             mode="constant")

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, i):
        row, col = self.centers[i]
        patch = self.padded[row:row + self.size, col:col + self.size]
        return torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probabilities(model, loader, device, student=False):
    output = []
    model.eval()
    with torch.inference_mode():
        for x in loader:
            x = x.to(device)
            logits = model(x)[1] if student else model(x)
            output.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def metrics(y, q):
    pred = q.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(7))
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    precision = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
    return {"n": len(y), "correct": int(np.trace(cm)),
            "oa_official": float(np.trace(cm) / 53200),
            "oa_evaluated": float(np.mean(pred == y)),
            "aa": float(np.mean(recall)),
            "kappa": float(cohen_kappa_score(y, pred, labels=np.arange(7))),
            "recall": recall.tolist(), "precision": precision.tolist(),
            "confusion_matrix": cm.tolist(),
            "soft_mass": q.sum(axis=0).tolist(),
            "hard_count": np.bincount(pred, minlength=7).tolist(),
            "mean_confidence": float(q.max(axis=1).mean()),
            "ece_15": ece(y, q, 15),
            "brier": float(np.mean(np.sum((q - np.eye(7)[y]) ** 2, axis=1)))}


def ece(y, q, bins):
    conf, pred = q.max(axis=1), q.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < bins - 1 else (conf <= 1))
        if mask.any():
            total += mask.mean() * abs((pred[mask] == y[mask]).mean() - conf[mask].mean())
    return float(total)


def complementarity(y, teacher, student):
    t, s = teacher.argmax(axis=1), student.argmax(axis=1)
    tc, sc = t == y, s == y
    out = {"agreement": float(np.mean(t == s)),
           "both_correct": int(np.sum(tc & sc)),
           "teacher_only_correct": int(np.sum(tc & ~sc)),
           "student_only_correct": int(np.sum(~tc & sc)),
           "both_wrong": int(np.sum(~tc & ~sc))}
    per_class = []
    for cls in range(7):
        mask = y == cls
        per_class.append({"class": cls + 1, "n": int(mask.sum()),
                          "agreement": float(np.mean(t[mask] == s[mask])),
                          "teacher_only_correct": int(np.sum(mask & tc & ~sc)),
                          "student_only_correct": int(np.sum(mask & ~tc & sc)),
                          "both_correct": int(np.sum(mask & tc & sc)),
                          "both_wrong": int(np.sum(mask & ~tc & ~sc))})
    out["per_class"] = per_class
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=(1341, 1174, 1370), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    out = HERE / "runs" / f"seed_{args.seed}"
    student_run = PROJECT / "experiments/round9/runs" / f"formal_A_{args.seed}"
    stage_files = [out / "stage1/best.pth", out / "full/best.pth"]
    stage_meta = [torch.load(p, map_location="cpu", weights_only=False)["best"] for p in stage_files]
    selected = int(stage_meta[1]["val_acc"] > stage_meta[0]["val_acc"])
    teacher_ckpt = stage_files[selected]
    teacher = SSFusionFramework(img_size=33, in_channels=48, patch_size=2,
                                classes=7, model_size="base")
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location="cpu", weights_only=False)["model"])
    teacher.to(device).eval()
    student_ckpt = student_run / "best_source_val.pth"
    student = Backbone().to(device)
    student.load_state_dict(torch.load(student_ckpt, map_location="cpu", weights_only=False)["model"])
    student.eval()

    cfg = json.loads((student_run / "config.json").read_text())
    with np.load(student_run / "source_split.npz") as f:
        centers = f["target_centers"].copy()
    assert centers.shape == (53200, 2)
    raw_path = Path(cfg["data"]) / "Houston18.mat"
    with np.load(cfg["ilda_cache"]) as f:
        ilda = f["t"].copy()
    raw = hdf5storage.loadmat(str(raw_path))["ori_data"].astype(np.float32)
    assert raw.shape == ilda.shape == (210, 954, 48)
    n = 53184
    teacher_q = probabilities(teacher, DataLoader(CenterPatches(raw, centers[:n], 33),
                              batch_size=args.batch_size, shuffle=False, num_workers=0), device)
    print(f"seed={args.seed} teacher predictions={len(teacher_q)}", flush=True)
    del teacher
    torch.cuda.empty_cache()
    student_q = probabilities(student, DataLoader(CenterPatches(ilda, centers[:n], 7),
                              batch_size=32, shuffle=False, num_workers=0), device, student=True)
    assert teacher_q.shape == student_q.shape == (n, 7)
    # Freeze the predictions before target GT is opened; labels are audit-only.
    np.savez_compressed(out / "target_probabilities.npz", teacher=teacher_q, student=student_q,
                        centers=centers[:n])
    gt_path = Path(cfg["data"]) / "Houston18_7gt.mat"
    gt = hdf5storage.loadmat(str(gt_path))["map"]
    y = gt[centers[:n, 0], centers[:n, 1]].astype(np.int64) - 1
    assert y.min() == 0 and y.max() == 6
    report = {"seed": args.seed, "teacher_selected_stage": "full" if selected else "stage1",
              "stage_source_val": stage_meta, "teacher_checkpoint": str(teacher_ckpt),
              "teacher_checkpoint_sha256": hash_file(teacher_ckpt),
              "student_checkpoint": str(student_ckpt),
              "student_checkpoint_sha256": hash_file(student_ckpt),
              "target_gt_sha256": hash_file(gt_path),
              "protocol": "source-val checkpoint selection; target GT post-hoc only; first 53184 official centers",
              "teacher": metrics(y, teacher_q), "student": metrics(y, student_q),
              "complementarity": complementarity(y, teacher_q, student_q)}
    official = json.loads((student_run / "final_target_source_val_best.json").read_text())
    assert report["student"]["confusion_matrix"] == official["metrics"]["confusion_matrix"]
    assert report["student"]["oa_official"] == official["metrics"]["oa"]
    report["student_matches_official_audit"] = True
    (out / "target_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"seed": args.seed, "selected": report["teacher_selected_stage"],
                      "teacher_oa": report["teacher"]["oa_official"],
                      "student_oa": report["student"]["oa_official"],
                      "teacher_class7_recall": report["teacher"]["recall"][6],
                      "student_class7_recall": report["student"]["recall"][6],
                      "teacher_only_correct": report["complementarity"]["teacher_only_correct"]}), flush=True)


if __name__ == "__main__":
    main()
