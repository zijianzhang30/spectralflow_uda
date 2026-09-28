"""Full MLUDA objective plus one fixed source SceneShift CE branch."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import argparse,json,sys,time,shutil
from pathlib import Path
import numpy as np
import hdf5storage
import torch
from torch.nn import functional as F
SCRIPT=Path(__file__).resolve()
ROOT=SCRIPT.parents[2]/"official_aligned"
sys.path.insert(0,str(ROOT))
from data import DEFAULT_DATA,load_images,loaders,Patches,file_hash
from runtime import seed_everything,evaluation,atomic_json,save_checkpoint
sys.path.insert(0,str(ROOT/"reference"))
from net2 import DSANSS
from baseline_pool import install_deterministic_pool
from full_mluda_objective import objective
SHIFT_ALPHA=.7
SHIFT_CE_WEIGHT=.5
SHIFT_EPS=1e-5

def band_stats(cube):
    mean=cube.mean(axis=(0,1),dtype=np.float64).astype(np.float32)
    std=cube.std(axis=(0,1),ddof=0,dtype=np.float64).astype(np.float32)
    return (torch.from_numpy(mean).cuda()[None,:,None,None],
            torch.from_numpy(std).cuda()[None,:,None,None])

def scene_shift(x,source_mean,source_std,target_mean,target_std,generator):
    shifted=(x-source_mean)/(source_std+SHIFT_EPS)
    shifted=shifted*((1-SHIFT_ALPHA)*source_std+SHIFT_ALPHA*target_std)
    shifted=shifted+(1-SHIFT_ALPHA)*source_mean+SHIFT_ALPHA*target_mean
    scale=1+.04*torch.randn((len(x),1,1,1),generator=generator,device=x.device,dtype=x.dtype)
    noise=torch.randn(shifted.shape,generator=generator,device=x.device,dtype=x.dtype)
    return (shifted*scale+.015*F.avg_pool2d(noise,5,1,2)).clamp(0,1)

def validate(model,loader):
    correct=n=0; total=0.
    with evaluation(model):
        for x,y in loader:
            x,y=x.cuda(),y.cuda()
            logits=model(x,x)[3]
            correct+=int((logits.argmax(1)==y).sum()); n+=len(y)
            total+=float(F.cross_entropy(logits,y,reduction="sum"))
    return dict(source_val_accuracy=correct/n,source_val_loss=total/n,source_val_n=n)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--data",type=Path,default=DEFAULT_DATA)
    p.add_argument("--ilda-cache",type=Path,default=ROOT/"preprocessing/official_ilda.npz")
    p.add_argument("--audit",action="store_true")
    a=p.parse_args()
    assert 1<=a.epochs<=100
    assert a.epochs==100 or a.audit,"Formal runs require 100 epochs"
    torch.set_num_threads(2)
    out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    seed_everything(a.seed)
    source,target=load_images(a.data,"official_ilda",a.ilda_cache)
    source_mean,source_std=band_stats(source)
    target_mean,target_std=band_stats(target)
    sg=hdf5storage.loadmat(str(a.data/"Houston13_7gt.mat"))["map"]
    tg=hdf5storage.loadmat(str(a.data/"Houston18_7gt.mat"))["map"]
    sl,tl,vl,split=loaders(source,target,sg,a.seed,tg)
    np.savez(out/"source_split.npz",**split)
    model=install_deterministic_pool(DSANSS(48,7,7).cuda())
    shift_rng=torch.Generator(device="cuda").manual_seed(a.seed+99173)
    config=dict(method="MLUDA_SHIFT",seed=a.seed,epochs=a.epochs,lr_horizon=100,
        data=str(a.data.resolve()),ilda_cache=str(a.ilda_cache.resolve()),
        selection="source_val_best",tie_rule="earliest",batch_size=32,patch_size=7,
        train_n=len(sl.dataset),val_n=len(vl.dataset),target_n=len(tl.dataset),
        steps_per_epoch=len(sl)-1,lr=.01,momentum=.9,weight_decay=5e-4,
        optimizer_reset_each_epoch=True,target_metrics_during_training=False,
        source_validation_forward="model(x,x)[3]",
        target_test_forward="model(selected_epoch_last_source_batch,target)[8]",
        test_drop_last=True,objective="verbatim full MLUDA objective + 0.5 CE on shifted source",
        pooling="official forward; deterministic equivalent global avg/max backward",
        scene_shift_alpha=SHIFT_ALPHA,scene_shift_ce_weight=SHIFT_CE_WEIGHT,
        scene_shift_stats="post-ILDA full-scene source/target mean/std, ddof=0",
        scene_shift_noise="multiplicative 0.04, smoothed 0.015, clamp [0,1]",
        scene_shift_forward="model(shifted_source, original_target)[3]",
        scene_shift_rng="independent CUDA generator seed+99173",
        target_gt_loaded_for_sampling_only=True,
        target_gt_read_for_metrics=False)
    atomic_json(out/"config.json",config)
    code=[SCRIPT,ROOT/"full_mluda_objective.py",ROOT/"baseline_pool.py",
        ROOT/"data.py",ROOT/"runtime.py"]+list((ROOT/"reference").glob("*.py"))
    snapshot=out/"code";snapshot.mkdir()
    hashes={}
    for path in code:
        rel=Path("train_mluda_shift.py") if path==SCRIPT else path.relative_to(ROOT)
        dst=snapshot/rel;dst.parent.mkdir(parents=True,exist_ok=True)
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
            shifted=scene_shift(x.cuda(),source_mean,source_std,
                                target_mean,target_std,shift_rng)
            shifted_logits=model(shifted,tx.cuda())[3]
            shifted_ce=F.cross_entropy(shifted_logits,y.cuda())
            loss=loss+SHIFT_CE_WEIGHT*shifted_ce
            parts["shifted_ce"]=float(shifted_ce.detach())
            if not torch.isfinite(loss):raise FloatingPointError(str((epoch,step,parts)))
            optimizer.zero_grad()
            loss.backward();optimizer.step()
            for k,v in parts.items():totals[k]=totals.get(k,0.)+v
        row=dict(epoch=epoch,lr=lr,**{k:v/38 for k,v in totals.items()},**validate(model,vl))
        history.append(row)
        checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            epoch=epoch,metrics=row,config=config,last_source_batch=x.clone(),
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            shift_rng=shift_rng.get_state())
        if row["source_val_accuracy"]>best:
            best=row["source_val_accuracy"]
            save_checkpoint(out/"best_source_val.pth",checkpoint)
            atomic_json(out/"best_source_val.json",row)
        save_checkpoint(out/"last.pth",checkpoint)
        atomic_json(out/"history.json",history)
        print(json.dumps(row),flush=True)
    atomic_json(out/"training_complete.json",dict(epochs=a.epochs,seconds=time.time()-started,
        formal=a.epochs==100 and not a.audit))
    # Target metrics remain a separate operation after formal training.
if __name__=="__main__":main()
