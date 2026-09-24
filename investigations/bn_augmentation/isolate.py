"""Isolate domain/view/EMA effects on BN without optimizer steps or target GT.

All source validation is in eval mode. BN calibration uses source TRAIN images
and, in explicit mixed conditions, unlabeled target images at saved centers.
No calibrated checkpoint is saved; published experiments remain unchanged.
"""
import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"official_aligned"))
from model import Backbone
from data import Patches, file_hash
from augmentation import radiation_noise

CONDITIONS = [
    ("source_clean_cumulative", False, "none", None),
    ("source_clean_ema", False, "none", .1),
    ("source_official_cumulative", False, "official", None),
    ("source_spatial_cumulative", False, "spatial", None),
    ("mixed_clean_cumulative", True, "none", None),
    ("mixed_official_cumulative", True, "official", None),
    ("mixed_spatial_cumulative", True, "spatial", None),
    ("mixed_official_ema", True, "official", .1),
]


def run_dir(seed, method):
    r = ROOT/"official_aligned/runs"
    return r/"official_seed1341_100ep"/method if seed==1341 else r/"round1/formal"/str(seed)/method


def flip_views(x):
    horizontal, vertical = np.random.random()>.5, np.random.random()>.5
    official, spatial = x, x
    if horizontal:
        official, spatial = official.flip(1), spatial.flip(3)
    if vertical:
        official, spatial = official.flip(0), spatial.flip(2)
    return official, spatial


def metrics(model, batches):
    model.eval()
    correct=n=0; loss=0.
    with torch.no_grad():
        for x,y in batches:
            logits=model(x)[1]
            correct+=int((logits.argmax(1)==y).sum()); n+=len(y)
            loss+=float(F.cross_entropy(logits,y,reduction="sum"))
    return dict(accuracy=correct/n,loss=loss/n,n=n)


def inputs(dataset, ids, labelled):
    examples=[dataset[int(i)] for i in ids]
    if labelled:
        return torch.stack([x for x,_ in examples]),torch.tensor([y for _,y in examples])
    return torch.stack(examples)


def geometry(train, val):
    distance=np.abs(val[:,None,:]-train[None,:,:]).max(axis=2).min(axis=1)
    return dict(shared_centers=int((distance==0).sum()),validation_n=len(val),
                val_center_in_any_train_patch=int((distance<=3).sum()),
                val_patch_overlaps_any_train_patch=int((distance<=6).sum()),
                nearest_center_chebyshev_median=float(np.median(distance)))


def inspect(seed, method, source, target):
    run=run_dir(seed,method); cp_path=run/"best_source_val.pth"
    before=file_hash(cp_path)
    cp=torch.load(cp_path,map_location="cpu",weights_only=False)
    with np.load(run/"source_split.npz") as split:
        sd=Patches(source,split["train_centers"],split["train_labels"])
        td=Patches(target,split["target_centers"])
        vd=Patches(source,split["val_centers"],split["val_labels"])
        overlap=geometry(split["train_centers"],split["val_centers"])
    rng=np.random.RandomState(seed+75000)
    si,ti=rng.permutation(len(sd)),rng.permutation(len(td))
    sb=[inputs(sd,si[k:k+32],True) for k in range(0,1248,32)]
    tb=[inputs(td,ti[k:k+32],False) for k in range(0,1248,32)]
    vb=[inputs(vd,range(k,min(k+32,len(vd))),True) for k in range(0,len(vd),32)]
    model=Backbone();model.load_state_dict(cp["model"])
    rows=[dict(condition="saved_statistics",**metrics(model,vb))]
    for name,mixed,augmentation,momentum in CONDITIONS:
        model.load_state_dict(cp["model"])
        for module in model.modules():
            if isinstance(module,nn.BatchNorm3d):
                module.reset_running_stats();module.momentum=momentum
        model.train()
        np.random.seed(seed+88000)
        with torch.no_grad():
            for (sx,_),tx in zip(sb,tb):
                # Identical RNG draws and base batches in every condition.
                sn=radiation_noise(sx).float(); so,ss=flip_views(sx)
                tn=radiation_noise(tx).float(); to,ts=flip_views(tx)
                model(sx)
                if mixed:model(tx)
                if augmentation!="none":
                    model(sn)
                    if mixed:model(tn)
                    model(so if augmentation=="official" else ss)
                    if mixed:model(to if augmentation=="official" else ts)
        row=dict(condition=name,**metrics(model,vb))
        row["bn_updates_per_layer"]=int(model.bn1.num_batches_tracked)
        rows.append(row)
        print(seed,method,json.dumps(row),flush=True)
    assert file_hash(cp_path)==before
    return dict(seed=seed,method=method,epoch=cp["epoch"],checkpoint_sha256=before,
                calibration_source_n=1248,calibration_target_n=1248,
                overlap=overlap,conditions=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(__file__).parent/"results.json")
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(0)
    cache=ROOT/"official_aligned/preprocessing/official_ilda.npz"
    with np.load(cache) as f:source,target=f["s"].astype(np.float32),f["t"].astype(np.float32)
    report=dict(scope="post-hoc diagnostic only, selected checkpoints; no training or target evaluation",
                script_sha256=file_hash(Path(__file__)),cache_sha256=file_hash(cache),
                target_gt_opened=False,calibration_uses_validation=False,rows=[])
    started=time.time()
    for seed,method in [(1341,"A"),(1174,"A"),(1370,"A"),(1174,"C")]:
        report["rows"].append(inspect(seed,method,source,target))
        report["seconds"]=time.time()-started
        tmp=args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(report,indent=2)+"\n");tmp.replace(args.out)
    report["complete"]=True
    args.out.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
