"""Round7 CE-to-Flow coupling ablation on the frozen Houston protocol."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
from pathlib import Path
import shutil
import time
import hdf5storage
import numpy as np
import torch
from torch.nn import functional as F
from model import Backbone
from augmentation import radiation_noise, flip_augmentation
from flow import Flow, transport_objective
from data import DEFAULT_DATA, load_images, loaders, file_hash, array_hash
from runtime import (seed_everything, evaluation, tensor_hash, atomic_json, save_checkpoint)

ROOT = Path(__file__).resolve().parent


def source_validation(model, loader, device):
    correct = count = 0
    loss = 0.
    with evaluation(model):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, logits = model(x)
            correct += int((logits.argmax(1) == y).sum())
            count += len(y)
            loss += float(F.cross_entropy(logits, y, reduction="sum"))
    return dict(source_val_accuracy=correct/count, source_val_loss=loss/count, source_val_n=count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["A", "flow_transport_detached",
                                             "flow_transport_coupled", "linear_transport"],
                        required=True)
    parser.add_argument("--protocol", choices=["official_ilda"], default="official_ilda")
    parser.add_argument("--ilda-cache", type=Path,
                        default=ROOT.parents[1]/"official_aligned/preprocessing/official_ilda.npz")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1341)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr-horizon", type=int, choices=[100], default=100)
    parser.add_argument("--audit", action="store_true", help="Explicit short-run audit; never target tested")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ot-membership", choices=["soft_q"], default="soft_q")
    args = parser.parse_args()
    if not 1 <= args.epochs <= args.lr_horizon:
        parser.error("Require 1 <= epochs <= lr-horizon")
    if args.epochs != 100 and not args.audit:
        parser.error("Formal runs require 100 epochs; use --audit for short diagnostic runs")
    # The shared preprocessing realization must match the frozen first-round study.
    import json
    parent_lock = ROOT.parents[1]/"official_aligned/PROTOCOL_LOCK.json"
    locked = json.loads(parent_lock.read_text())
    if file_hash(args.ilda_cache) != locked["ilda_sha256"]:
        raise ValueError("ILDA cache differs from the frozen round1 protocol")
    for path, expected in locked["inputs"].items():
        if Path(path).suffix == ".mat" and file_hash(args.data/Path(path).name) != expected:
            raise ValueError("Dataset differs from round1: " + Path(path).name)
    torch.set_num_threads(2)
    args.data = args.data.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.ilda_cache:
        args.ilda_cache = args.ilda_cache.resolve()
    seed_everything(args.seed)
    device = torch.device(args.device)
    source, target = load_images(args.data, args.protocol, args.ilda_cache)
    gt = hdf5storage.loadmat(str(args.data/"Houston13_7gt.mat"))["map"]
    target_gt = hdf5storage.loadmat(str(args.data/"Houston18_7gt.mat"))["map"]
    train_loader, target_loader, val_loader, split = loaders(source, target, gt, args.seed, target_gt)
    del target_gt
    np.savez(out/"source_split.npz", **split)
    model = Backbone().to(device)
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(args.seed + 71039)
        flow = Flow().to(device) if args.method != "A" else None
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 99173)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(batch_size=32, patch_size=7, train_n=len(train_loader.dataset),
                  val_n=len(val_loader.dataset), target_n=len(target_loader.dataset),
                  steps_per_epoch=len(train_loader)-1, lr=.01, momentum=.9, weight_decay=5e-4,
                  optimizer_reset_each_epoch=True, target_labels_loaded=True,
                  target_sampling="official nonzero GT centers, class-wise then global shuffle; no target labels in loss",
                  selection_primary="fixed_epoch_100", selection_secondary="source_val_best",
                  tie_rule="earliest",
                  target_feature_mode="train, no-grad; target updates shared BN buffers",
                  source_train_labels_hash=array_hash(split["train_labels"]),
                  source_val_labels_hash=array_hash(split["val_labels"]),
                  augmentation="official radiation/flip, BN-only views; no contrastive loss",
                  feature_dim=288, fm_weight=1., transport_ce_max_weight=.2,
                  transport_ce_warmup_epochs=20, transport_fraction=.5,
                  ot_reg=.05, ot_iterations=100,
                  pairs_per_present_class=32,
                  flow_fm_endpoint_gradient="detached",
                  transport_ce_to_flow=args.method == "flow_transport_coupled",
                  transport_ce_to_source=True,
                  experiment="round7_ce_to_flow_coupling", parent_protocol_sha256=file_hash(parent_lock),
                  pooling="global torch.mean; no custom backward")
    atomic_json(out/"config.json", config)
    snapshot = out/"code"; snapshot.mkdir()
    paths = [ROOT/name for name in ["train.py", "model.py", "data.py", "flow.py",
                                    "runtime.py", "final_test.py", "augmentation.py", "official_preprocessing.py"]]
    hashes = {}
    for path in paths:
        shutil.copy2(path, snapshot/path.name)
        hashes[str(path)] = file_hash(path)
    inputs = {name: file_hash(args.data/name) for name in [
        "Houston13.mat", "Houston18.mat", "Houston13_7gt.mat", "Houston18_7gt.mat"]}
    if args.ilda_cache:
        inputs["ilda_cache"] = file_hash(args.ilda_cache)
    atomic_json(out/"provenance.json", dict(code=hashes, inputs=inputs, torch=torch.__version__,
                initial_model_hash=tensor_hash(model.state_dict().values())))
    history, best = [], -1.
    started = time.time()
    for epoch in range(1, args.epochs+1):
        lr = .01/(1+10*(epoch-1)/args.lr_horizon)**.75
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=5e-4)
        flow_optimizer = (torch.optim.SGD(flow.parameters(), lr=lr, momentum=.9, weight_decay=5e-4)
                          if flow is not None else None)
        model.train()
        if flow is not None:
            flow.train()
        train_iterator = iter(train_loader)
        target_iterator = iter(target_loader)
        ce_sum = fm_sum = aux_sum = 0.
        aux_weight = .2*min(epoch/20, 1.)
        for step in range(1, len(train_loader)):
            x, labels = next(train_iterator)
            target_x = next(target_iterator)
            source_noise = radiation_noise(x).float().to(device)
            source_flip = flip_augmentation(x).to(device)
            target_noise = radiation_noise(target_x).float().to(device)
            target_flip = flip_augmentation(target_x).to(device)
            x, labels, target_x = x.to(device), labels.to(device), target_x.to(device)
            source_features, logits = model(x)
            ce = F.cross_entropy(logits, labels)
            # All variants share transductive BN updates, including the CE control.
            with torch.no_grad():
                target_features, target_logits = model(target_x)
                q = target_logits.softmax(1)
            # Preserve official six backbone passes and BN update order, without SCL losses.
            with torch.no_grad():
                model(source_noise)
                model(target_noise)
                model(source_flip)
                model(target_flip)
            if flow is None:
                fm = aux_ce = ce.new_zeros(())
            else:
                fm, aux_ce = transport_objective(
                    flow, model.classifier, source_features, target_features,
                    labels, q, generator, args.method)
            loss = ce + fm + aux_weight*aux_ce
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {epoch}/{step}")
            optimizer.zero_grad(set_to_none=True)
            if flow_optimizer:
                flow_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_hash = tensor_hash(p.grad for p in model.parameters() if p.grad is not None)
            optimizer.step()
            if flow_optimizer:
                flow_optimizer.step()
            row = dict(epoch=epoch, step=step, ce=float(ce.detach()),
                       fm=float(fm.detach()), transport_ce=float(aux_ce.detach()),
                       transport_ce_weight=aux_weight,
                       model_hash=tensor_hash(model.state_dict().values()),
                       backbone_hash=tensor_hash(p for n,p in model.named_parameters()
                                                 if not n.startswith("classifier.")),
                       classifier_hash=tensor_hash(model.classifier.parameters()),
                       buffers_hash=tensor_hash(model.buffers()), gradient_hash=gradient_hash,
                       source_logits_hash=tensor_hash([logits]),
                       source_input_hash=tensor_hash([x, labels]),
                       target_input_hash=tensor_hash([target_x]),
                       cpu_rng_hash=tensor_hash([torch.get_rng_state()]),
                       cuda_rng_hash=tensor_hash(torch.cuda.get_rng_state_all()) if devices else None)
            with (out/"steps.jsonl").open("a") as handle:
                import json
                handle.write(json.dumps(row)+"\n")
            ce_sum += float(ce.detach())
            fm_sum += float(fm.detach())
            aux_sum += float(aux_ce.detach())
        metrics = dict(epoch=epoch, lr=lr, ce=ce_sum/(len(train_loader)-1),
                       fm=fm_sum/(len(train_loader)-1),
                       transport_ce=aux_sum/(len(train_loader)-1),
                       transport_ce_weight=aux_weight,
                       **source_validation(model, val_loader, device))
        history.append(metrics)
        checkpoint = dict(model=model.state_dict(), flow=flow.state_dict() if flow else None,
                          epoch=epoch, metrics=metrics, config=config,
                          optimizer=optimizer.state_dict(),
                          flow_optimizer=flow_optimizer.state_dict() if flow_optimizer else None,
                          rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if devices else [],
                          flow_rng=generator.get_state(),
                          source_loader_rng=None,
                          target_loader_rng=None)
        if metrics["source_val_accuracy"] > best:
            best = metrics["source_val_accuracy"]
            save_checkpoint(out/"best_source_val.pth", checkpoint)
            atomic_json(out/"best_source_val.json", metrics)
        save_checkpoint(out/"last.pth", checkpoint)
        atomic_json(out/"history.json", history)
        print(metrics, flush=True)
    atomic_json(out/"training_complete.json", dict(epochs=args.epochs, lr_horizon=args.lr_horizon,
                seconds=time.time()-started, target_labels_loaded=True, formal=args.epochs==100 and not args.audit,
                selection_primary="fixed_epoch_100", selected_checkpoint="last.pth",
                selection_secondary="source_val_best"))
    print("COMPLETE. Run final_test.py separately for final target testing.", flush=True)


if __name__ == "__main__":
    main()
