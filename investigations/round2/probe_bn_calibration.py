"""Source-only BN recalibration diagnostic; never uses validation to calibrate."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from audit_implementation import ROOT, run_dir, assess, Backbone, Patches, file_hash


def main():
    torch.set_num_threads(2)
    with np.load(ROOT/"official_aligned/preprocessing/official_ilda.npz") as cache:
        source = cache["s"].astype(np.float32)
    rows = []
    for seed in [1341,1174,1370]:
        for method in ["A","C"]:
            run = run_dir(seed,method)
            path = run/"best_source_val.pth"; before = file_hash(path)
            cp = torch.load(path,map_location="cpu",weights_only=False)
            with np.load(run/"source_split.npz") as split:
                train = Patches(source,split["train_centers"],split["train_labels"])
                val = Patches(source,split["val_centers"],split["val_labels"])
            model = Backbone(); model.load_state_dict(cp["model"])
            for module in model.modules():
                if isinstance(module,nn.BatchNorm3d):
                    module.reset_running_stats()
                    module.momentum = None
            model.train()
            loader = DataLoader(train,batch_size=32,shuffle=True,drop_last=True,
                                generator=torch.Generator().manual_seed(seed+75000))
            with torch.no_grad():
                for x,_ in loader:
                    model(x)
            metrics = assess(model,val,False)
            assert file_hash(path)==before
            row = dict(seed=seed,method=method,epoch=cp["epoch"],
                       original_source_val=cp["metrics"]["source_val_accuracy"],
                       source_only_recalibrated_eval=metrics,
                       calibration_n=39*32,checkpoint_sha256=before)
            rows.append(row)
            print(json.dumps(row),flush=True)
    report=dict(scope="diagnostic only; no target evaluation or new checkpoint selection",
                checkpoint_files_unchanged=True, target_gt_opened=False,
                calibration="39 shuffled source-training batches, reset BN, cumulative statistics; no weight updates",
                rows=rows)
    (Path(__file__).parent/"bn_calibration.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
