"""Read-only CE/KL backbone gradient geometry on fixed train-style batches."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

import hdf5storage
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"experiments/round9"))
from augmentation import radiation_noise, flip_augmentation
from data import Patches, load_images
from model import Backbone
from runtime import atomic_json, seed_everything
from train import consistency_loss


def geometry(a, b):
    dot = sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b))
    na = sum(float(x.double().square().sum()) for x in a)**.5
    nb = sum(float(x.double().square().sum()) for x in b)**.5
    return {"norm_a":na,"norm_b":nb,"ratio_b_to_a":nb/max(na,1e-30),
            "cosine":dot/max(na*nb,1e-30)}


def grad(loss, parameters, retain_graph=True):
    out = torch.autograd.grad(loss, parameters, retain_graph=retain_graph)
    return tuple(x.detach() for x in out)


def summarize(rows):
    if not rows:
        return None
    return {"n_batches":len(rows),
            "cosine_mean":mean(x["cosine"] for x in rows),
            "cosine_median":median(x["cosine"] for x in rows),
            "cosine_negative_fraction":mean(x["cosine"]<0 for x in rows),
            "ratio_mean":mean(x["ratio_b_to_a"] for x in rows),
            "ratio_median":median(x["ratio_b_to_a"] for x in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("B","C"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--selection", choices=("fixed_epoch_100","source_val_best"),
                        default="fixed_epoch_100")
    args = parser.parse_args()
    seed_everything(args.seed+88111)
    torch.set_num_threads(2)
    run = REPO/"experiments/round9/runs"/f"formal_{args.method}_{args.seed}"
    cfg = json.loads((run/"config.json").read_text())
    cp_name = "last.pth" if args.selection=="fixed_epoch_100" else "best_source_val.pth"
    cp = torch.load(run/cp_name, map_location="cpu", weights_only=False)
    assert cp["epoch"]==(100 if args.selection=="fixed_epoch_100" else
                          json.loads((run/"best_source_val.json").read_text())["epoch"])
    student = Backbone().to(args.device).train()
    teacher = Backbone().to(args.device).eval()
    student.load_state_dict(cp["model"])
    teacher.load_state_dict(cp["teacher"])
    teacher.requires_grad_(False)
    params = tuple(p for n,p in student.named_parameters() if not n.startswith("classifier."))
    source, target = load_images(Path(cfg["data"]),cfg["protocol"],Path(cfg["ilda_cache"]))
    with np.load(run/"source_split.npz") as split:
        sc, sl = split["train_centers"].copy(), split["train_labels"].copy()
        tc = split["target_centers"].copy()
    gt = hdf5storage.loadmat(str(Path(cfg["data"])/"Houston18_7gt.mat"))["map"]
    tl = gt[tc[:,0],tc[:,1]].astype(np.int64)-1
    source_loader = DataLoader(Patches(source,sc,sl),batch_size=32,shuffle=False,drop_last=True)
    target_loader = DataLoader(Patches(target,tc,tl),batch_size=32,shuffle=False,drop_last=True)
    global_rows, class_rows, margin_rows = [], {5:[],6:[]}, []
    teacher_q6_true7 = teacher_q7_true7 = teacher_pred6_true7 = 0.0
    true7_n = 0
    weight_true7_sum = weight_all_sum = 0.0
    source_iter = iter(source_loader)
    for batch,(tx,ty) in enumerate(target_loader):
        if batch>=args.batches:
            break
        try:
            sx,sy = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            sx,sy = next(source_iter)
        source_noise = radiation_noise(sx).float().to(args.device)
        source_flip = flip_augmentation(sx).to(args.device)
        target_noise = radiation_noise(tx).float().to(args.device)
        target_flip = flip_augmentation(tx).to(args.device)
        sx,sy,tx,ty = sx.to(args.device),sy.to(args.device),tx.to(args.device),ty.to(args.device)
        _,slogits = student(sx)
        ce_per = F.cross_entropy(slogits,sy,reduction="none")
        ce = ce_per.mean()
        with torch.no_grad():
            student(tx)
            student(source_noise)
        _,strong_logits = student(target_noise)
        with torch.no_grad():
            _,weak_logits = teacher(tx)
        kl, weights, q = consistency_loss(strong_logits,weak_logits,args.method=="C")
        kl_per = F.kl_div(F.log_softmax(strong_logits,dim=1),q,reduction="none").sum(1)
        with torch.no_grad():
            student(source_flip)
            student(target_flip)
        gce, gkl = grad(ce,params), grad(kl,params)
        row = geometry(gce,gkl)
        row.update(batch=batch,ce=float(ce.detach()),kl=float(kl.detach()))
        global_rows.append(row)
        target7 = ty==6
        n7 = int(target7.sum())
        if n7:
            true7_n += n7
            teacher_q6_true7 += float(q[target7,5].sum())
            teacher_q7_true7 += float(q[target7,6].sum())
            teacher_pred6_true7 += int((q[target7].argmax(1)==5).sum())
            weight_true7_sum += float(weights[target7].sum())
            # Positive cosine means a KL descent step tends to reduce the
            # true-class-7 versus class-6 margin on these target samples.
            margin = (strong_logits[target7,6]-strong_logits[target7,5]).mean()
            gmargin = grad(margin,params)
            margin_rows.append(geometry(gmargin,gkl))
        weight_all_sum += float(weights.sum())
        for c in (5,6):
            src_mask = sy==c
            dst_mask = ty==c
            if not int(src_mask.sum()) or not int(dst_mask.sum()):
                continue
            gce_c = grad(ce_per[src_mask].sum()/len(sy),params)
            gkl_c = grad((weights[dst_mask]*kl_per[dst_mask]).sum()/len(ty),params)
            r = geometry(gce_c,gkl_c)
            r.update(source_n=int(src_mask.sum()),target_n=int(dst_mask.sum()),batch=batch)
            class_rows[c].append(r)
    assert len(global_rows)==args.batches
    output = {"method":args.method,"seed":args.seed,"selection":args.selection,
              "checkpoint_epoch":cp["epoch"],
              "batches":args.batches,"sampling":"sequential cycling source train and first official-order target batches",
              "target_gt_use":"gradient attribution and class-7 diagnostic only; no optimizer step or checkpoint selection",
              "global_CE_vs_KL_backbone":summarize(global_rows),
              "class6_CE_vs_KL_backbone":summarize(class_rows[5]),
              "class7_CE_vs_KL_backbone":summarize(class_rows[6]),
              "true7_margin_vs_total_KL_backbone":summarize(margin_rows),
              "true7_margin_harmful_KL_fraction":mean(x["cosine"]>0 for x in margin_rows),
              "teacher_q6_on_true7_mean":teacher_q6_true7/true7_n,
              "teacher_q7_on_true7_mean":teacher_q7_true7/true7_n,
              "teacher_pred6_on_true7_fraction":teacher_pred6_true7/true7_n,
              "true7_weight_mean":weight_true7_sum/true7_n,
              "overall_weight_mean":weight_all_sum/(32*args.batches),
              "true7_sample_n":true7_n,
              "batch_rows":global_rows}
    suffix = "" if args.selection=="fixed_epoch_100" else "_sourcebest"
    out = Path(__file__).resolve().parent/f"grad_{args.method}_{args.seed}{suffix}.json"
    atomic_json(out,output)
    print(json.dumps({k:v for k,v in output.items() if k!="batch_rows"},indent=2))


if __name__=="__main__":
    main()
