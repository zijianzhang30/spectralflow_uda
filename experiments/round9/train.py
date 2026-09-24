"""Locked A/B/C target consistency refinement, without Flow."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.nn import functional as F

from augmentation import radiation_noise, flip_augmentation
from data import DEFAULT_DATA, array_hash, file_hash, load_images, loaders
from model import Backbone
from runtime import atomic_json, evaluation, save_checkpoint, seed_everything, tensor_hash

ROOT = Path(__file__).resolve().parent
EMA_MOMENTUM = 0.99
CONS_MAX_WEIGHT = 1.0
RAMP_EPOCHS = 10
ENTROPY_FLOOR = 0.1


def source_validation(model, loader, device):
    correct = count = 0
    loss = 0.0
    with evaluation(model):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, logits = model(x)
            correct += int((logits.argmax(1) == y).sum())
            count += len(y)
            loss += float(F.cross_entropy(logits, y, reduction="sum"))
    return dict(source_val_accuracy=correct/count, source_val_loss=loss/count,
                source_val_n=count)


@torch.no_grad()
def update_teacher(teacher, student):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(EMA_MOMENTUM).add_(s.detach(), alpha=1-EMA_MOMENTUM)
    # EMA parameters and copied BN statistics are fixed across B and C.
    for t, s in zip(teacher.buffers(), student.buffers()):
        t.copy_(s)


def consistency_loss(student_logits, teacher_logits, weighted):
    q = teacher_logits.detach().softmax(1)
    kl = F.kl_div(F.log_softmax(student_logits, dim=1), q,
                  reduction="none").sum(1)
    if weighted:
        entropy = -(q*q.clamp_min(1e-12).log()).sum(1)
        weights = (1-entropy/np.log(q.shape[1])).clamp_min(ENTROPY_FLOOR)
        weights = (weights/weights.mean()).detach()
    else:
        weights = torch.ones_like(kl)
    return (weights*kl).mean(), weights, q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("A", "B", "C"), required=True)
    parser.add_argument("--protocol", choices=("official_ilda",), default="official_ilda")
    parser.add_argument("--ilda-cache", type=Path,
                        default=ROOT.parents[1]/"official_aligned/preprocessing/official_ilda.npz")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1341)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr-horizon", type=int, choices=(100,), default=100)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 1 <= args.epochs <= args.lr_horizon:
        parser.error("Require 1 <= epochs <= lr-horizon")
    if args.epochs != 100 and not args.audit:
        parser.error("Formal runs require 100 epochs; use --audit for short runs")
    parent_lock = ROOT.parents[1]/"official_aligned/PROTOCOL_LOCK.json"
    locked = json.loads(parent_lock.read_text())
    if file_hash(args.ilda_cache) != locked["ilda_sha256"]:
        raise ValueError("ILDA cache differs from frozen protocol")
    for path, expected in locked["inputs"].items():
        if Path(path).suffix == ".mat" and file_hash(args.data/Path(path).name) != expected:
            raise ValueError("Dataset differs from frozen protocol: " + Path(path).name)
    torch.set_num_threads(2)
    args.data = args.data.resolve()
    args.ilda_cache = args.ilda_cache.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device(args.device)
    source, target = load_images(args.data, args.protocol, args.ilda_cache)
    gt = hdf5storage.loadmat(str(args.data/"Houston13_7gt.mat"))["map"]
    target_gt = hdf5storage.loadmat(str(args.data/"Houston18_7gt.mat"))["map"]
    train_loader, target_loader, val_loader, split = loaders(source, target, gt,
                                                              args.seed, target_gt)
    del target_gt
    np.savez(out/"source_split.npz", **split)
    model = Backbone().to(device)
    teacher = copy.deepcopy(model).to(device).eval()
    teacher.requires_grad_(False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(batch_size=32, patch_size=7, train_n=len(train_loader.dataset),
                  val_n=len(val_loader.dataset), target_n=len(target_loader.dataset),
                  steps_per_epoch=len(train_loader)-1, lr=.01, momentum=.9,
                  weight_decay=5e-4, optimizer_reset_each_epoch=True,
                  target_sampling="official nonzero GT centers; target labels not used by loss",
                  target_labels_loaded_for_sampling=True,
                  selection_primary="fixed_epoch_100", selection_secondary="source_val_best",
                  augmentation="official radiation noise / flip; same six student passes A/B/C",
                  weak_view="unaltered target_x", strong_view="official target radiation_noise",
                  teacher_mode="eval; EMA parameters; BN buffers copied after each student step",
                  ema_momentum=EMA_MOMENTUM, consistency_max_weight=CONS_MAX_WEIGHT,
                  consistency_ramp_epochs=RAMP_EPOCHS,
                  consistency_ramp="min(epoch/10, 1) shared B/C",
                  consistency="per-sample KL(stopgrad teacher weak || student strong)",
                  C_weight="max(1-H(q_teacher)/log(7),0.1), detached, batch-mean normalized",
                  parent_protocol_sha256=file_hash(parent_lock),
                  source_train_labels_hash=array_hash(split["train_labels"]),
                  source_val_labels_hash=array_hash(split["val_labels"]),
                  feature_dim=288, flow=False)
    atomic_json(out/"config.json", config)
    snapshot = out/"code"
    snapshot.mkdir()
    paths = [ROOT/name for name in ("train.py", "model.py", "data.py", "runtime.py",
                                    "final_test.py", "augmentation.py", "official_preprocessing.py")]
    hashes = {}
    for path in paths:
        shutil.copy2(path, snapshot/path.name)
        hashes[str(path)] = file_hash(path)
    inputs = {name: file_hash(args.data/name) for name in
              ("Houston13.mat", "Houston18.mat", "Houston13_7gt.mat", "Houston18_7gt.mat")}
    inputs["ilda_cache"] = file_hash(args.ilda_cache)
    atomic_json(out/"provenance.json", dict(code=hashes, inputs=inputs,
                 torch=torch.__version__, initial_model_hash=tensor_hash(model.state_dict().values())))
    history, best = [], -1.0
    started = time.time()
    for epoch in range(1, args.epochs+1):
        lr = .01/(1+10*(epoch-1)/args.lr_horizon)**.75
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9,
                                    weight_decay=5e-4)
        model.train()
        teacher.eval()
        train_iterator = iter(train_loader)
        target_iterator = iter(target_loader)
        ce_sum = cons_sum = 0.0
        for step in range(1, len(train_loader)):
            x, labels = next(train_iterator)
            target_x = next(target_iterator)
            source_noise = radiation_noise(x).float().to(device)
            source_flip = flip_augmentation(x).to(device)
            target_noise = radiation_noise(target_x).float().to(device)
            target_flip = flip_augmentation(target_x).to(device)
            x, labels, target_x = x.to(device), labels.to(device), target_x.to(device)
            _, source_logits = model(x)
            ce = F.cross_entropy(source_logits, labels)
            with torch.no_grad():
                model(target_x)
                model(source_noise)
            if args.method == "A":
                with torch.no_grad():
                    model(target_noise)
                cons = ce.new_zeros(())
                weights = None
            else:
                _, student_strong_logits = model(target_noise)
                with torch.no_grad():
                    _, teacher_weak_logits = teacher(target_x)
                cons, weights, _ = consistency_loss(student_strong_logits,
                                                     teacher_weak_logits,
                                                     weighted=args.method=="C")
            with torch.no_grad():
                model(source_flip)
                model(target_flip)
            ramp = min(epoch/RAMP_EPOCHS, 1.0)
            loss = ce + (CONS_MAX_WEIGHT*ramp*cons if args.method != "A" else 0)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {epoch}/{step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_hash = tensor_hash(p.grad for p in model.parameters() if p.grad is not None)
            optimizer.step()
            if args.method != "A":
                update_teacher(teacher, model)
            row = dict(epoch=epoch, step=step, ce=float(ce.detach()),
                       consistency=float(cons.detach()), ramp=ramp,
                       weight_min=None if weights is None else float(weights.min()),
                       weight_mean=None if weights is None else float(weights.mean()),
                       weight_max=None if weights is None else float(weights.max()),
                       model_hash=tensor_hash(model.state_dict().values()),
                       backbone_hash=tensor_hash(p for n,p in model.named_parameters()
                                                 if not n.startswith("classifier.")),
                       classifier_hash=tensor_hash(model.classifier.parameters()),
                       buffers_hash=tensor_hash(model.buffers()), gradient_hash=gradient_hash,
                       source_logits_hash=tensor_hash([source_logits]),
                       source_input_hash=tensor_hash([x, labels]),
                       target_input_hash=tensor_hash([target_x]),
                       cpu_rng_hash=tensor_hash([torch.get_rng_state()]),
                       cuda_rng_hash=tensor_hash(torch.cuda.get_rng_state_all()) if device.type=="cuda" else None)
            with (out/"steps.jsonl").open("a") as handle:
                handle.write(json.dumps(row)+"\n")
            ce_sum += float(ce.detach())
            cons_sum += float(cons.detach())
        metrics = dict(epoch=epoch, lr=lr, ce=ce_sum/(len(train_loader)-1),
                       consistency=cons_sum/(len(train_loader)-1),
                       **source_validation(model, val_loader, device))
        history.append(metrics)
        checkpoint = dict(model=model.state_dict(), teacher=teacher.state_dict(),
                          epoch=epoch, metrics=metrics, config=config,
                          optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else [])
        if metrics["source_val_accuracy"] > best:
            best = metrics["source_val_accuracy"]
            save_checkpoint(out/"best_source_val.pth", checkpoint)
            atomic_json(out/"best_source_val.json", metrics)
        save_checkpoint(out/"last.pth", checkpoint)
        atomic_json(out/"history.json", history)
        print(metrics, flush=True)
    atomic_json(out/"training_complete.json", dict(epochs=args.epochs,
                lr_horizon=args.lr_horizon, seconds=time.time()-started,
                formal=args.epochs==100 and not args.audit,
                selection_primary="fixed_epoch_100", selection_secondary="source_val_best"))
    print("COMPLETE. Run final_test.py separately.", flush=True)


if __name__ == "__main__":
    main()
