"""CE plus historical composite source SceneShift, official protocol preserved."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
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
SHIFT_ALPHA = 0.7
SHIFT_CE_WEIGHT = 0.5
SHIFT_EPS = 1e-5


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


def scene_shift(x, source_mean, source_std, target_mean, target_std, generator):
    """Historical affine transfer, multiplicative noise, smooth noise and clamp."""
    shifted = (x - source_mean) / (source_std + SHIFT_EPS)
    shifted = shifted * ((1 - SHIFT_ALPHA) * source_std + SHIFT_ALPHA * target_std)
    shifted = shifted + (1 - SHIFT_ALPHA) * source_mean + SHIFT_ALPHA * target_mean
    scale = 1 + .04 * torch.randn((len(x), 1, 1, 1), generator=generator,
                                  device=x.device, dtype=x.dtype)
    noise = torch.randn(shifted.shape, generator=generator, device=x.device,
                        dtype=x.dtype)
    return (shifted * scale + .015 * F.avg_pool2d(noise, 5, 1, 2)).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("SHIFT",), default="SHIFT")
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
    def band_stats(cube):
        return (torch.from_numpy(cube.mean(axis=(0, 1), dtype=np.float64).astype(np.float32))
                .to(device)[None, :, None, None],
                torch.from_numpy(cube.std(axis=(0, 1), ddof=0, dtype=np.float64)
                                 .astype(np.float32)).to(device)[None, :, None, None])
    source_mean, source_std = band_stats(source)
    target_mean, target_std = band_stats(target)
    gt = hdf5storage.loadmat(str(args.data/"Houston13_7gt.mat"))["map"]
    target_gt = hdf5storage.loadmat(str(args.data/"Houston18_7gt.mat"))["map"]
    train_loader, target_loader, val_loader, split = loaders(source, target, gt,
                                                              args.seed, target_gt)
    del target_gt
    np.savez(out/"source_split.npz", **split)
    model = Backbone().to(device)
    shift_rng = torch.Generator(device=device).manual_seed(args.seed + 99173)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(batch_size=32, patch_size=7, train_n=len(train_loader.dataset),
                  val_n=len(val_loader.dataset), target_n=len(target_loader.dataset),
                  steps_per_epoch=len(train_loader)-1, lr=.01, momentum=.9,
                  weight_decay=5e-4, optimizer_reset_each_epoch=True,
                  target_sampling="official nonzero GT centers; target labels not used by loss",
                  target_labels_loaded_for_sampling=True,
                  selection_primary="source_val_best", selection_secondary="fixed_epoch_100",
                  augmentation="official six passes, then additional historical SceneShift supervised source pass",
                  scene_shift_alpha=SHIFT_ALPHA, scene_shift_epsilon=SHIFT_EPS,
                  scene_shift_ce_weight=SHIFT_CE_WEIGHT,
                  scene_shift_source="ILDA full-scene source/target channel mean/std, ddof=0",
                  scene_shift_randomness="separate GPU generator; official RNG trajectory unchanged",
                  scene_shift_bn="train mode, running stats updated on extra shifted pass",
                  loss="CE(original source) + 0.5 CE(shifted source)",
                  parent_protocol_sha256=file_hash(parent_lock),
                  source_train_labels_hash=array_hash(split["train_labels"]),
                  source_val_labels_hash=array_hash(split["val_labels"]),
                  feature_dim=288, flow=False)
    atomic_json(out/"config.json", config)
    snapshot = out/"code"
    snapshot.mkdir()
    paths = [ROOT/name for name in ("train_scene_shift.py", "model.py", "data.py", "runtime.py",
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
        train_iterator = iter(train_loader)
        target_iterator = iter(target_loader)
        ce_sum = shift_ce_sum = 0.0
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
                model(target_noise)
                model(source_flip)
                model(target_flip)
            shifted = scene_shift(x, source_mean, source_std,
                                  target_mean, target_std, shift_rng)
            _, shifted_logits = model(shifted)
            shift_ce = F.cross_entropy(shifted_logits, labels)
            loss = ce + SHIFT_CE_WEIGHT * shift_ce
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {epoch}/{step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_hash = tensor_hash(p.grad for p in model.parameters() if p.grad is not None)
            optimizer.step()
            row = dict(epoch=epoch, step=step, ce=float(ce.detach()),
                       shifted_ce=float(shift_ce.detach()),
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
            shift_ce_sum += float(shift_ce.detach())
        metrics = dict(epoch=epoch, lr=lr, ce=ce_sum/(len(train_loader)-1),
                       shifted_ce=shift_ce_sum/(len(train_loader)-1),
                       **source_validation(model, val_loader, device))
        history.append(metrics)
        checkpoint = dict(model=model.state_dict(),
                          epoch=epoch, metrics=metrics, config=config,
                          optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                          shift_rng=shift_rng.get_state(),
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
                selection_primary="source_val_best", selection_secondary="fixed_epoch_100"))
    print("COMPLETE. Run final_test.py separately.", flush=True)


if __name__ == "__main__":
    main()
