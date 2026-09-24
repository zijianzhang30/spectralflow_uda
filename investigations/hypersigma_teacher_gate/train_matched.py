"""Source-only HyperSIGMA adaptation on the saved DCRN source splits.

Stage 1 and full fine-tuning match the prior HyperSIGMA implementation;
only the source split and output location change. No target image or label is
opened by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
LEGACY = Path("/home/zhangzj26/TGRS_MLUDA-2024")
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(LEGACY / "third_party/HyperSIGMA/ImageClassification"))
from hypersigma_stage1_protocol import load_pretrained, patches, seed_all, set_stage1  # noqa: E402
from hypersigma_ft_protocol import configure_trainable  # noqa: E402
from hypersigma_teacher_smoke_test import SSFusionFramework  # noqa: E402

SEEDS = (1341, 1174, 1370)
OUT = HERE / "runs"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum = correct = count = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += ce(logits, y).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
    return loss_sum / count, correct / count


def train_stage(model, mode, train_loader, val_loader, device, epochs, out):
    if mode == "stage1":
        set_stage1(model)
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": 6e-5}]
    else:
        groups, _ = configure_trainable(model, "full")
    model.to(device)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    ce = nn.CrossEntropyLoss()
    best_acc = -1.0
    history = []
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        if mode == "stage1":
            model.spat_encoder.blocks.eval()
            model.spec_encoder.blocks.eval()
        loss_sum = correct = count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        row = {"epoch": epoch, "stage": mode, "train_loss": loss_sum / count,
               "train_acc": correct / count, "val_loss": val_loss,
               "val_acc": val_acc, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "best": row, "stage": mode}, out / "best.pth")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stage1-epochs", type=int, default=20)
    p.add_argument("--full-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--stage", choices=("stage1", "full", "both"), default="both")
    args = p.parse_args()
    assert args.stage1_epochs == 20 and args.full_epochs == 20, "fixed diagnostic schedule"
    assert args.batch_size == 32, "fixed diagnostic batch size"
    seed_all(args.seed)
    torch.set_num_threads(4)
    run = PROJECT / "experiments/round9/runs" / f"formal_A_{args.seed}"
    split_path = run / "source_split.npz"
    with np.load(split_path) as split:
        train_centers, val_centers = split["train_centers"], split["val_centers"]
        assert len(train_centers) == 1260 and len(val_centers) == 1270
    cfg = json.loads((run / "config.json").read_text())
    source_path = Path(cfg["data"]) / "Houston13.mat"
    gt_path = Path(cfg["data"]) / "Houston13_7gt.mat"
    source = hdf5storage.loadmat(str(source_path))["ori_data"].astype(np.float32)
    gt = hdf5storage.loadmat(str(gt_path))["map"]
    train_y = gt[train_centers[:, 0], train_centers[:, 1]].astype(np.int64) - 1
    val_y = gt[val_centers[:, 0], val_centers[:, 1]].astype(np.int64) - 1
    assert np.array_equal(np.bincount(train_y, minlength=7), np.full(7, 180))
    out = OUT / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "seed": args.seed, "source_split_sha256": sha256(split_path),
        "source_image_sha256": sha256(source_path), "source_gt_sha256": sha256(gt_path),
        "source_train_n": 1260, "source_val_n": 1270,
        "input": "raw 48-band 33x33, constant padding, patch_size=2",
        "stage1_epochs": 20, "full_epochs": 20, "batch_size": 32,
        "stage1_rng_seed": args.seed, "full_rng_seed": args.seed + 10000,
        "optimizer": {"type": "AdamW", "betas": [0.9, 0.999], "weight_decay": 0.05,
                      "stage1_lr": 6e-5, "full_new_module_lr": 6e-5,
                      "full_last_block_lr": 3e-6, "full_block_decay": 0.92,
                      "schedule": "CosineAnnealingLR over each 20-epoch stage"},
        "selection": "source-val accuracy; first maximum within each stage",
        "target_used_for_training_or_selection": False,
    }, indent=2))
    train_loader = DataLoader(TensorDataset(torch.from_numpy(patches(source, train_centers, 48)),
                                            torch.from_numpy(train_y)),
                              batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(patches(source, val_centers, 48)),
                                          torch.from_numpy(val_y)),
                            batch_size=32, shuffle=False, num_workers=0)
    model = SSFusionFramework(img_size=33, in_channels=48, patch_size=2, classes=7, model_size="base")
    stage1_file = out / "stage1/best.pth"
    if args.stage in ("stage1", "both"):
        load_pretrained(model)
        train_stage(model, "stage1", train_loader, val_loader, torch.device(args.device),
                    20, out / "stage1")
    if args.stage in ("full", "both"):
        checkpoint = torch.load(stage1_file, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        seed_all(args.seed + 10000)
        train_stage(model, "full", train_loader, val_loader, torch.device(args.device),
                    20, out / "full")


if __name__ == "__main__":
    main()
