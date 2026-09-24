"""Read-only post-hoc diagnostics for the frozen round6 reverse Flow."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/round6"))
from model import Backbone
from flow import ReverseFlow, ot_pairs, transport_target
from data import Patches, file_hash, load_images
from runtime import atomic_json, evaluation, seed_everything


def training_pair_log(path):
    epochs = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            epoch = row["epoch"]
            rec = epochs.setdefault(epoch, {"updates": 0, "pair_total": 0,
                                           "empty_updates": 0})
            rec["updates"] += 1
            rec["pair_total"] += row["pair_n"]
            rec["empty_updates"] += int(row["pair_n"] == 0)
    assert len(epochs) == 100 and all(x["updates"] == 38 for x in epochs.values())
    return {"updates": 3800,
            "total_pairs": sum(x["pair_total"] for x in epochs.values()),
            "empty_updates": sum(x["empty_updates"] for x in epochs.values()),
            "by_epoch": epochs,
            "limitation": "Per-class candidates and pair truth were not logged during training."}


@torch.no_grad()
def source_prototypes(model, cube, centers, labels, device):
    loader = DataLoader(Patches(cube, centers, labels), batch_size=32,
                        shuffle=False, drop_last=False)
    sums = torch.zeros(7, 288, device=device)
    counts = torch.zeros(7, device=device)
    for x, y in loader:
        z, _ = model(x.to(device))
        y = y.to(device)
        sums.index_add_(0, y, z)
        counts += torch.bincount(y, minlength=7)
    assert (counts == 180).all()
    return sums / counts[:, None]


@torch.no_grad()
def pair_audit(model, source, target, source_centers, source_labels,
               target_centers, target_labels, device, seed, batches):
    source_loader = DataLoader(Patches(source, source_centers, source_labels),
                               batch_size=32, shuffle=False, drop_last=True)
    target_loader = DataLoader(Patches(target, target_centers, target_labels),
                               batch_size=32, shuffle=False, drop_last=True)
    source_iter = iter(source_loader)
    rng = torch.Generator().manual_seed(seed + 51001)
    candidate_n = np.zeros(7, dtype=np.int64)
    candidate_correct = np.zeros(7, dtype=np.int64)
    candidate_absent = np.zeros(7, dtype=np.int64)
    source_absent = np.zeros(7, dtype=np.int64)
    pair_n = np.zeros(7, dtype=np.int64)
    pair_correct = np.zeros(7, dtype=np.int64)
    confusion = np.zeros((7, 7), dtype=np.int64)
    raw_correct = raw_n = 0
    audited = 0
    for target_x, ty in target_loader:
        if audited == batches:
            break
        try:
            source_x, sy = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_x, sy = next(source_iter)
        zs, _ = model(source_x.to(device))
        zt, logits = model(target_x.to(device))
        q = logits.softmax(1)
        confidence, pseudo = q.max(1)
        sy_np, ty_np = sy.numpy(), ty.numpy()
        raw_correct += int((logits.argmax(1).cpu() == ty).sum())
        raw_n += len(ty)
        for c in range(7):
            mask = ((pseudo == c) & (confidence >= .95)).cpu().numpy()
            count = int(mask.sum())
            candidate_n[c] += count
            candidate_correct[c] += int((ty_np[mask] == c).sum())
            candidate_absent[c] += int(count == 0)
            source_absent[c] += int((sy_np == c).sum() == 0)
        si, ti = ot_pairs(zs, zt, sy.to(device), q, rng)
        if len(si):
            source_classes = sy[si.cpu()].numpy()
            true_target_classes = ty[ti.cpu()].numpy()
            np.add.at(confusion, (source_classes, true_target_classes), 1)
            pair_n += np.bincount(source_classes, minlength=7)
            pair_correct += np.bincount(source_classes[source_classes == true_target_classes],
                                        minlength=7)
        audited += 1
    assert audited == batches
    return {"batches": audited, "sampling": "fixed sequential source-train and official-order target batches; not historical train-loader shuffles",
            "target_gt_use": "post-hoc diagnostic only",
            "raw_accuracy_on_audit_batches": raw_correct / raw_n,
            "candidate_n_by_pred_class": candidate_n.tolist(),
            "candidate_precision_by_pred_class":
                (candidate_correct / np.maximum(candidate_n, 1)).tolist(),
            "candidate_absent_batches": candidate_absent.tolist(),
            "source_absent_batches": source_absent.tolist(),
            "pair_n_by_source_class": pair_n.tolist(),
            "pair_purity_by_source_class":
                (pair_correct / np.maximum(pair_n, 1)).tolist(),
            "pair_purity_overall": float(pair_correct.sum() / max(pair_n.sum(), 1)),
            "pair_confusion": confusion.tolist()}


@torch.no_grad()
def transport_audit(model, flow, target, centers, labels, prototypes, device):
    loader = DataLoader(Patches(target, centers, labels), batch_size=32,
                        shuffle=False, drop_last=True)
    cm = np.zeros((2, 2), dtype=np.int64)  # raw correctness x moved correctness
    class_cm = np.zeros((7, 2, 2), dtype=np.int64)
    pred_changes = np.zeros((7, 7), dtype=np.int64)
    displacements = []
    true_proto_cosine_before = true_proto_cosine_after = 0.
    nearest_proto_correct_before = nearest_proto_correct_after = 0
    true_proto_improved = 0
    n = 0
    normalized_prototypes = F.normalize(prototypes, dim=1)
    for x, y in loader:
        z, logits = model(x.to(device))
        moved = transport_target(flow, z, steps=4)
        raw_pred = logits.argmax(1)
        moved_pred = model.classifier(moved).argmax(1)
        y = y.to(device)
        raw_ok = (raw_pred == y)
        moved_ok = (moved_pred == y)
        np.add.at(cm, (raw_ok.cpu().numpy().astype(int),
                       moved_ok.cpu().numpy().astype(int)), 1)
        np.add.at(class_cm, (y.cpu().numpy(), raw_ok.cpu().numpy().astype(int),
                             moved_ok.cpu().numpy().astype(int)), 1)
        np.add.at(pred_changes, (raw_pred.cpu().numpy(), moved_pred.cpu().numpy()), 1)
        before = F.normalize(z, dim=1) @ normalized_prototypes.T
        after = F.normalize(moved, dim=1) @ normalized_prototypes.T
        true_before = before.gather(1, y[:, None]).squeeze(1)
        true_after = after.gather(1, y[:, None]).squeeze(1)
        true_proto_cosine_before += float(true_before.sum())
        true_proto_cosine_after += float(true_after.sum())
        true_proto_improved += int((true_after > true_before).sum())
        nearest_proto_correct_before += int((before.argmax(1) == y).sum())
        nearest_proto_correct_after += int((after.argmax(1) == y).sum())
        displacements.extend((moved-z).norm(dim=1).cpu().tolist())
        n += len(y)
    assert n == 53184
    return {"evaluated_n": n, "target_gt_use": "post-hoc diagnostic only",
            "raw_wrong_to_moved_correct": int(cm[0, 1]),
            "raw_correct_to_moved_wrong": int(cm[1, 0]),
            "both_correct": int(cm[1, 1]),
            "both_wrong": int(cm[0, 0]),
            "class_correctness_transitions": class_cm.tolist(),
            "prediction_transition_matrix": pred_changes.tolist(),
            "mean_displacement_l2": float(np.mean(displacements)),
            "median_displacement_l2": float(np.median(displacements)),
            "mean_true_class_prototype_cosine_before": true_proto_cosine_before/n,
            "mean_true_class_prototype_cosine_after": true_proto_cosine_after/n,
            "fraction_true_class_prototype_cosine_improved": true_proto_improved/n,
            "nearest_source_prototype_accuracy_before": nearest_proto_correct_before/n,
            "nearest_source_prototype_accuracy_after": nearest_proto_correct_after/n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selection", choices=("fixed_epoch_100", "source_val_best"),
                        default="fixed_epoch_100")
    parser.add_argument("--pair-batches", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = REPO / "experiments/round6/runs" / f"formal_{args.seed}"
    cfg = json.loads((run / "config.json").read_text())
    complete = json.loads((run / "training_complete.json").read_text())
    assert complete["formal"] and complete["epochs"] == 100
    provenance = json.loads((run / "provenance.json").read_text())
    for path, digest in provenance["code"].items():
        assert file_hash(REPO / "experiments/round6" / Path(path).name) == digest
    for name, digest in provenance["inputs"].items():
        path = Path(cfg["ilda_cache"]) if name == "ilda_cache" else Path(cfg["data"]) / name
        assert file_hash(path) == digest
    cp_path = run / ("last.pth" if args.selection == "fixed_epoch_100" else "best_source_val.pth")
    cp = torch.load(cp_path, map_location=args.device, weights_only=False)
    assert cp["epoch"] == (100 if args.selection == "fixed_epoch_100" else
                           json.loads((run / "best_source_val.json").read_text())["epoch"])
    seed_everything(args.seed)
    torch.set_num_threads(2)
    model = Backbone().to(args.device)
    flow = ReverseFlow().to(args.device)
    model.load_state_dict(cp["model"])
    flow.load_state_dict(cp["flow"])
    model.eval(); flow.eval()
    source, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    with np.load(run / "source_split.npz") as split:
        source_centers = split["train_centers"].copy()
        source_labels = split["train_labels"].copy()
        target_centers = split["target_centers"].copy()
    target_gt = hdf5storage.loadmat(str(Path(cfg["data"]) / "Houston18_7gt.mat"))["map"]
    target_labels = target_gt[target_centers[:, 0], target_centers[:, 1]].astype(np.int64)-1
    del target_gt
    with evaluation(model):
        prototypes = source_prototypes(model, source, source_centers, source_labels, args.device)
        pairs = pair_audit(model, source, target, source_centers, source_labels,
                           target_centers, target_labels, args.device, args.seed,
                           args.pair_batches)
        moved = transport_audit(model, flow, target, target_centers, target_labels,
                                prototypes, args.device)
    result = {"seed": args.seed, "selection": args.selection, "checkpoint_epoch": cp["epoch"],
              "checkpoint_sha256": file_hash(cp_path),
              "training_pair_log": training_pair_log(run / "steps.jsonl"),
              "fixed_batch_pair_audit": pairs, "full_target_transport_audit": moved}
    out = Path(__file__).resolve().parent / f"{args.seed}_{args.selection}.json"
    atomic_json(out, result)
    print(json.dumps({"seed": args.seed, "selection": args.selection,
                      "pair_purity": pairs["pair_purity_overall"],
                      "candidate_n": pairs["candidate_n_by_pred_class"],
                      "pair_n": pairs["pair_n_by_source_class"],
                      "wrong_to_correct": moved["raw_wrong_to_moved_correct"],
                      "correct_to_wrong": moved["raw_correct_to_moved_wrong"],
                      "prototype_cosine_before": moved["mean_true_class_prototype_cosine_before"],
                      "prototype_cosine_after": moved["mean_true_class_prototype_cosine_after"]},
                     indent=2))


if __name__ == "__main__":
    main()
