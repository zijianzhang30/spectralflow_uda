"""Independently reload selected checkpoints and recompute source validation."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import hdf5storage
from data import load_images,loaders
from model import Backbone
from train import source_validation
from runtime import seed_everything,atomic_json

torch.set_num_threads(2)
root=Path(__file__).resolve().parents[1]
reports=[]
for protocol in ["raw","official_ilda"]:
    for method in ["A","B"]:
        run=root/"runs/audit"/protocol/method
        config=json.loads((run/"config.json").read_text())
        seed_everything(config["seed"])
        cache=Path(config["ilda_cache"]) if config["ilda_cache"] else None
        source,target=load_images(Path(config["data"]),protocol,cache)
        gt=hdf5storage.loadmat(str(Path(config["data"])/"Houston13_7gt.mat"))["map"]
        _,_,loader,_=loaders(source,target,gt,config["seed"])
        ckpt=torch.load(run/"best_source_val.pth",map_location="cuda",weights_only=False)
        model=Backbone().cuda();model.load_state_dict(ckpt["model"])
        actual=source_validation(model,loader,"cuda")
        history=json.loads((run/"history.json").read_text())
        selected=max(history,key=lambda x:x["source_val_accuracy"])
        assert selected==ckpt["metrics"]
        assert all(actual[k]==selected[k] for k in actual),(protocol,method,actual,selected)
        reports.append(dict(protocol=protocol,method=method,selected_epoch=selected["epoch"],
                            source_val_accuracy=actual["source_val_accuracy"],passed=True))
atomic_json(root/"runs/audit/checkpoint_recheck.json",reports)
print(reports)
