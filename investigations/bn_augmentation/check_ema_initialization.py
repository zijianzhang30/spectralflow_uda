"""Separate a short cold-reset EMA calibration artifact from domain mismatch."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from isolate import ROOT, run_dir, Patches, Backbone, inputs, metrics


def main():
    torch.set_num_threads(2)
    with np.load(ROOT/"official_aligned/preprocessing/official_ilda.npz") as f:
        source=f["s"].astype(np.float32)
    rows=[]
    for seed in [1341,1174,1370]:
        run=run_dir(seed,"A")
        cp=torch.load(run/"best_source_val.pth",map_location="cpu",weights_only=False)
        with np.load(run/"source_split.npz") as split:
            sd=Patches(source,split["train_centers"],split["train_labels"])
            vd=Patches(source,split["val_centers"],split["val_labels"])
        si=np.random.RandomState(seed+75000).permutation(len(sd))
        sb=[inputs(sd,si[k:k+32],True) for k in range(0,1248,32)]
        vb=[inputs(vd,range(k,min(k+32,len(vd))),True) for k in range(0,len(vd),32)]
        row=dict(seed=seed,updates=39,initial_variance_residual=.9**39)
        for cold in [True,False]:
            model=Backbone();model.load_state_dict(cp["model"])
            for mod in model.modules():
                if isinstance(mod,nn.BatchNorm3d):
                    if cold:mod.reset_running_stats()
                    mod.momentum=.1
            model.train()
            with torch.no_grad():
                for x,_ in sb:model(x)
            row["cold_reset" if cold else "warm_start"]=metrics(model,vb)
            if cold:
                residual=.9**39
                negative=0
                for mod in model.modules():
                    if isinstance(mod,nn.BatchNorm3d):
                        corrected=(mod.running_var-residual)/(1-residual)
                        negative+=int((corrected<0).sum())
                        mod.running_var.copy_(corrected.clamp_min(0))
                        mod.running_mean.div_(1-residual)
                row["negative_variances_before_clamp"]=negative
                row["cold_reset_debiased"]=metrics(model,vb)
        rows.append(row);print(json.dumps(row),flush=True)
    (Path(__file__).parent/"ema_initialization.json").write_text(json.dumps(dict(
        scope="source-only diagnostic; not a proposed new training protocol",rows=rows),indent=2)+"\n")


if __name__=="__main__":main()
