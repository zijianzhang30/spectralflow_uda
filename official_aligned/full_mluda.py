"""Full official MLUDA comparator, same data and source-val-best selection."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import argparse,json,sys,time,shutil
from pathlib import Path
import numpy as np
import hdf5storage
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data import DEFAULT_DATA,load_images,loaders,Patches,file_hash
from runtime import seed_everything,evaluation,atomic_json,save_checkpoint,scores
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"reference"))
from net2 import DSANSS
from baseline_pool import install_deterministic_pool
from full_mluda_objective import objective

def validate(model,loader):
    correct=n=0; total=0.
    with evaluation(model):
        for x,y in loader:
            x,y=x.cuda(),y.cuda()
            logits=model(x,x)[3]
            correct+=int((logits.argmax(1)==y).sum()); n+=len(y)
            total+=float(F.cross_entropy(logits,y,reduction="sum"))
    return dict(source_val_accuracy=correct/n,source_val_loss=total/n,source_val_n=n)

def final_test(out):
    cfg=json.loads((out/"config.json").read_text())
    assert json.loads((out/"training_complete.json").read_text())["epochs"]==100
    prov=json.loads((out/"provenance.json").read_text())
    for n,h in prov["code"].items(): assert file_hash(ROOT/n)==h,n
    for n,h in prov["inputs"].items(): assert file_hash(n)==h,n
    seed_everything(cfg["seed"])
    checkpoint=torch.load(out/"best_source_val.pth",map_location="cuda",weights_only=False)
    selected=json.loads((out/"best_source_val.json").read_text())
    history=json.loads((out/"history.json").read_text())
    assert checkpoint["metrics"]==selected==max(history,key=lambda r:r["source_val_accuracy"])
    model=install_deterministic_pool(DSANSS(48,7,7).cuda())
    model.load_state_dict(checkpoint["model"])
    _,target=load_images(Path(cfg["data"]),"official_ilda",Path(cfg["ilda_cache"]))
    gt=hdf5storage.loadmat(str(Path(cfg["data"])/"Houston18_7gt.mat"))["map"]
    with np.load(out/"source_split.npz") as split: centers=split["target_centers"]
    labels=gt[centers[:,0],centers[:,1]].astype(np.int64)-1
    loader=DataLoader(Patches(target,centers),batch_size=32,shuffle=False,drop_last=True)
    reference=checkpoint["last_source_batch"].cuda()
    predictions=[]
    with evaluation(model):
        for x in loader:predictions.extend(model(reference,x.cuda())[8].argmax(1).cpu().tolist())
    pred=np.asarray(predictions); y=labels[:len(pred)]
    result=scores(y,pred)
    result.update(oa_evaluated=result["oa"],oa=float((y==pred).sum())/len(labels),
        oa_definition="official: correct / full dataset, despite drop_last",
        method="MLUDA_full",seed=cfg["seed"],protocol="official_ilda",
        selection="source_val_best",selected_epoch=selected["epoch"],
        source_val_accuracy=selected["source_val_accuracy"],dataset_n=len(labels),
        dropped_n=len(labels)-len(pred),checkpoint_sha256=file_hash(out/"best_source_val.pth"),
        target_gt_sha256=file_hash(Path(cfg["data"])/"Houston18_7gt.mat"))
    atomic_json(out/"final_target.json",result)
    print("FINAL",json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--data",type=Path,default=DEFAULT_DATA)
    p.add_argument("--ilda-cache",type=Path,default=ROOT/"preprocessing/official_ilda.npz")
    p.add_argument("--test-only",action="store_true")
    a=p.parse_args()
    assert 1<=a.epochs<=100
    torch.set_num_threads(2)
    out=a.out.resolve()
    if a.test_only:final_test(out);return
    out.mkdir(parents=True,exist_ok=False)
    seed_everything(a.seed)
    source,target=load_images(a.data,"official_ilda",a.ilda_cache)
    sg=hdf5storage.loadmat(str(a.data/"Houston13_7gt.mat"))["map"]
    tg=hdf5storage.loadmat(str(a.data/"Houston18_7gt.mat"))["map"]
    sl,tl,vl,split=loaders(source,target,sg,a.seed,tg)
    np.savez(out/"source_split.npz",**split)
    model=install_deterministic_pool(DSANSS(48,7,7).cuda())
    config=dict(method="MLUDA_full",seed=a.seed,epochs=a.epochs,lr_horizon=100,
        data=str(a.data.resolve()),ilda_cache=str(a.ilda_cache.resolve()),
        selection="source_val_best",tie_rule="earliest",batch_size=32,patch_size=7,
        train_n=len(sl.dataset),val_n=len(vl.dataset),target_n=len(tl.dataset),
        steps_per_epoch=len(sl)-1,lr=.01,momentum=.9,weight_decay=5e-4,
        optimizer_reset_each_epoch=True,target_metrics_during_training=False,
        source_validation_forward="model(x,x)[3]",
        target_test_forward="model(selected_epoch_last_source_batch,target)[8]",
        test_drop_last=True,objective="official full MLUDA objective, verbatim",
        pooling="official forward; deterministic equivalent global avg/max backward")
    atomic_json(out/"config.json",config)
    code=[Path(__file__).resolve(),ROOT/"full_mluda_objective.py",ROOT/"baseline_pool.py",
        ROOT/"data.py",ROOT/"runtime.py"]+list((ROOT/"reference").glob("*.py"))
    snapshot=out/"code";snapshot.mkdir()
    hashes={}
    for path in code:
        rel=path.relative_to(ROOT); dst=snapshot/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dst);hashes[str(rel)]=file_hash(path)
    inputs={str((a.data/n).resolve()):file_hash(a.data/n) for n in
        ["Houston13.mat","Houston18.mat","Houston13_7gt.mat","Houston18_7gt.mat"]}
    inputs[str(a.ilda_cache.resolve())]=file_hash(a.ilda_cache)
    atomic_json(out/"provenance.json",dict(code=hashes,inputs=inputs,torch=torch.__version__))
    history=[];best=-1.;started=time.time()
    for epoch in range(1,a.epochs+1):
        lr=.01/(1+10*(epoch-1)/100)**.75
        optimizer=torch.optim.SGD([
            {"params":model.feature_layers.parameters()},
            {"params":model.fc1.parameters(),"lr":lr},
            {"params":model.fc2.parameters(),"lr":lr},
            {"params":model.head1.parameters(),"lr":lr},
            {"params":model.head2.parameters(),"lr":lr}],
            lr=lr,momentum=.9,weight_decay=5e-4)
        model.train();si,ti=iter(sl),iter(tl); totals={}
        for step in range(1,len(sl)):
            x,y=next(si);tx=next(ti)
            loss,parts=objective(model,x,tx,y,epoch,100)
            if not torch.isfinite(loss):raise FloatingPointError(str((epoch,step,parts)))
            optimizer.zero_grad()
            loss.backward();optimizer.step()
            for k,v in parts.items():totals[k]=totals.get(k,0.)+v
        row=dict(epoch=epoch,lr=lr,**{k:v/38 for k,v in totals.items()},**validate(model,vl))
        history.append(row)
        checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            epoch=epoch,metrics=row,config=config,last_source_batch=x.clone(),
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        if row["source_val_accuracy"]>best:
            best=row["source_val_accuracy"]
            save_checkpoint(out/"best_source_val.pth",checkpoint)
            atomic_json(out/"best_source_val.json",row)
        save_checkpoint(out/"last.pth",checkpoint)
        atomic_json(out/"history.json",history)
        print(json.dumps(row),flush=True)
    atomic_json(out/"training_complete.json",dict(epochs=a.epochs,seconds=time.time()-started))
    # Target metrics remain a separate operation after formal training.
if __name__=="__main__":main()
