"""Independent CPU diagnostics; no updates to experimental code or checkpoints."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "official_aligned"))
from augmentation import flip_augmentation
from model import Backbone
from data import Patches, file_hash
from flow import Flow


def run_dir(seed, method):
    root = ROOT / "official_aligned/runs"
    return (root / "official_seed1341_100ep" / method if seed == 1341 else
            root / "round1/formal" / str(seed) / method)


def assess(model, dataset, batch_statistics):
    model.train(batch_statistics)
    correct = count = 0
    loss = 0.
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    with torch.no_grad():
        for x,y in loader:
            logits = model(x)[1]
            correct += int((logits.argmax(1)==y).sum())
            count += len(y)
            loss += float(F.cross_entropy(logits, y, reduction="sum"))
    return dict(accuracy=correct/count, loss=loss/count, n=count)


def sinkhorn_check(source, target, labels, q):
    costs = 1-F.normalize(source, dim=1) @ F.normalize(target, dim=1).T
    rows = []
    for c in labels.unique():
        cost = costs[labels==c].double()
        a = torch.full((len(cost),), 1/len(cost), dtype=torch.double)
        b = q[:,c].double().clamp_min(1e-12); b /= b.sum()
        kernel = (-cost/.05).exp().clamp_min(1e-30)
        u,v = torch.ones_like(a), torch.ones_like(b)
        for _ in range(100):
            u = a/(kernel@v).clamp_min(1e-30)
            v = b/(kernel.T@u).clamp_min(1e-30)
        actual = u[:,None]*kernel*v[None,:]
        # Independent log-domain, much longer convergence reference.
        logk = -cost/.05
        lu,lv = torch.zeros_like(a), torch.zeros_like(b)
        for _ in range(2000):
            lu = a.log()-torch.logsumexp(logk+lv[None,:],dim=1)
            lv = b.log()-torch.logsumexp(logk+lu[:,None],dim=0)
        expected = (lu[:,None]+logk+lv[None,:]).exp()
        rows.append(dict(class_id=int(c)+1, row_l1=float((actual.sum(1)-a).abs().sum()),
                         column_l1=float((actual.sum(0)-b).abs().sum()),
                         log_reference_l1=float((actual-expected).abs().sum())))
    return rows


def main():
    torch.set_num_threads(2)
    torch.manual_seed(4)
    # Characterize actual official tensor axes with unique entries.
    x = torch.arange(4*48*7*7).reshape(4,48,7,7)
    np.random.seed(0)  # Both official flip decisions true.
    aug = flip_augmentation(x)
    axes = dict(batch_and_band_reversed=torch.equal(aug, x.flip((0,1))),
                spatially_flipped=torch.equal(aug, x.flip((2,3))))
    assert axes["batch_and_band_reversed"] and not axes["spatially_flipped"]
    # Verify the source-endpoint derivative with a fixed pair/time and linear field.
    source = torch.randn(2,5,dtype=torch.double,requires_grad=True)
    target = torch.randn_like(source)
    t = torch.tensor([[.2],[.7]],dtype=torch.double)
    w = torch.randn(5,5,dtype=torch.double)
    state = (1-t)*source+t*target
    error = state@w.T-(target-source)
    loss = error.square().mean()
    numerical = torch.autograd.grad(loss,source)[0]
    analytic = 2/error.numel()*((1-t)*(error@w)+error)
    torch.testing.assert_close(numerical,analytic,rtol=1e-12,atol=1e-12)
    with np.load(ROOT/"official_aligned/preprocessing/official_ilda.npz") as cache:
        s,tgt = cache["s"].astype(np.float32), cache["t"].astype(np.float32)
    bn_rows,ot_rows = [],[]
    for seed in [1341,1174,1370]:
        for method in ["A","C"]:
            run = run_dir(seed,method)
            cp = run/"best_source_val.pth"
            original_hash = file_hash(cp)
            checkpoint = torch.load(cp,map_location="cpu",weights_only=False)
            with np.load(run/"source_split.npz") as split:
                val = Patches(s,split["val_centers"],split["val_labels"])
                train = Patches(s,split["train_centers"],split["train_labels"])
                target_ds = Patches(tgt,split["target_centers"])
            model = Backbone(); model.load_state_dict(checkpoint["model"])
            frozen = assess(model,val,False)
            model.load_state_dict(checkpoint["model"])
            batch = assess(model,val,True)
            row = dict(seed=seed,method=method,epoch=checkpoint["epoch"],
                       stored_source_val=checkpoint["metrics"]["source_val_accuracy"],
                       eval_running_statistics=frozen, diagnostic_batch_statistics=batch)
            bn_rows.append(row)
            print("BN",json.dumps(row),flush=True)
            if method=="C":
                rng = np.random.RandomState(seed+50000)
                si,ti = rng.permutation(len(train))[:32],rng.permutation(len(target_ds))[:32]
                samples = [train[int(i)] for i in si]
                sx = torch.stack([p[0] for p in samples]); sy = torch.tensor([p[1] for p in samples])
                tx = torch.stack([target_ds[int(i)] for i in ti])
                model.load_state_dict(checkpoint["model"]); model.train()
                with torch.no_grad():
                    zs,_ = model(sx); zt,logits = model(tx)
                    checks = sinkhorn_check(zs,zt,sy,logits.softmax(1))
                ot_rows.append(dict(seed=seed,batches=1,classes=checks))
            assert file_hash(cp)==original_hash
    report=dict(target_gt_opened=False,checkpoint_files_unchanged=True,
                official_flip_axes=axes,source_fm_gradient_matches_analytic=True,
                bn_diagnostics=bn_rows,sinkhorn_diagnostics=ot_rows,
                limitations="Selected checkpoints only; BN batch-stat validation is diagnostic, not a new model selection/evaluation protocol. OT test uses one batch per seed, not all training steps.")
    (Path(__file__).parent/"implementation_audit.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
