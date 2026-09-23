"""Contract tests against archived local MLUDA code, never used by training."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import ast
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.metrics import accuracy_score,cohen_kappa_score,confusion_matrix
from model import Backbone
from flow import Flow,matching_loss
from data import load_images,DEFAULT_DATA,loaders,source_split,Patches
from runtime import evaluation,seed_everything,scores,tensor_hash
import hdf5storage

root=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
seed_everything(1341)
device="cuda" if torch.cuda.is_available() else "cpu"
spec=importlib.util.spec_from_file_location("reference_network",root/"reference/net2.py")
refmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(refmodule)
ref=refmodule.DCRN_02(48,7,7).to(device).eval()
model=Backbone().to(device).eval()
state=model.state_dict()
for k in state:
    if not k.startswith("classifier."): state[k]=ref.state_dict()[k].clone()
model.load_state_dict(state)
captured={}
h=ref.ca.register_forward_pre_hook(lambda module,args:captured.update(x=args[0].detach().clone()) if "x" not in captured else None)
x=torch.rand(8,48,7,7,device=device);y=torch.rand_like(x)
with torch.no_grad():
    ref(x,y)
    expected=captured["x"].mean((2,3))
    actual=model.features(x)
assert torch.allclose(actual,expected,rtol=1e-6,atol=1e-7)
h.remove()
assert not any(k.startswith(("ca.","sa.","atten.","head","fc2")) for k in model.state_dict())
print("PASS: convolutional features match official pre-attention concatenation; no attention/adaptation modules")

# Compile only the archived data-sampling function for reference comparison.
tree=ast.parse((root/"reference/utils.py").read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="get_sample_data")
namespace={"np":np}
exec(compile(ast.Module(body=[fn],type_ignores=[]),"<reference sample function>","exec"),namespace)
source,target=load_images(DEFAULT_DATA,"raw")
gt=hdf5storage.loadmat(str(DEFAULT_DATA/"Houston13_7gt.mat"))["map"]
for seed in [1341,1174,1370]:
    np.random.seed(seed)
    rx,ry=namespace["get_sample_data"](source,gt,3,180)
    train,val=source_split(gt,seed)
    ty=gt[train[:,0],train[:,1]].astype(np.int64)-1
    ds=Patches(source,train,ty)
    assert np.array_equal(ry,ty)
    assert all(np.array_equal(rx[i],ds[i][0].numpy()) for i in range(len(ds)))
    ls,lt,lv,split=loaders(source,target,gt,seed)
    assert (len(ds),len(lv.dataset),len(ls)-1,len(lt.dataset))==(1260,1270,38,200340)
print("PASS: 3 source splits and all 1260 patches match official; held-out 1270, 38 steps")
truth=np.arange(7).repeat(3);pred=truth.copy();pred[::4]=(pred[::4]+1)%7
v=scores(truth,pred);cm=confusion_matrix(truth,pred,labels=np.arange(7))
assert v["oa"]==accuracy_score(truth,pred)
assert np.isclose(v["kappa"],cohen_kappa_score(truth,pred))
assert np.isclose(v["aa"],(np.diag(cm)/cm.sum(1)).mean())
print("PASS: OA/AA/Kappa match sklearn")

model.train()
before=tensor_hash(model.state_dict().values())
rng=torch.get_rng_state().clone()
cuda=torch.cuda.get_rng_state().clone() if device=="cuda" else None
with evaluation(model):
    model(x)
assert tensor_hash(model.state_dict().values())==before
assert torch.equal(rng,torch.get_rng_state())
assert cuda is None or torch.equal(cuda,torch.cuda.get_rng_state())
assert model.training
print("PASS: target feature extraction/evaluation preserve model buffers and RNG")
for backprop in [False,True]:
    zs=torch.randn(14,288,device=device,requires_grad=True)
    zt=torch.randn(14,288,device=device,requires_grad=True)
    logits=torch.randn(14,7,device=device,requires_grad=True)
    labels=torch.arange(14,device=device)%7
    flow=Flow().to(device)
    loss=matching_loss(flow,zs,zt,labels,logits.softmax(1),torch.Generator().manual_seed(10),backprop)
    loss.backward()
    assert zt.grad is None and logits.grad is None
    assert (zs.grad is not None and zs.grad.norm()>0) if backprop else zs.grad is None
    assert sum(float(p.grad.abs().sum()) for p in flow.parameters())>0
print("PASS: B updates Flow only; C has source gradient; target/q detached")

# Every production trainer/data file must exclude target-GT filenames.
assert "Houston18_7gt" not in (root/"train.py").read_text()
assert "Houston18_7gt" not in (root/"data.py").read_text()
assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and
               "scene_semantic_uda" in ast.dump(n) for n in ast.walk(ast.parse((root/"train.py").read_text())))
print("ALL CONTRACT TESTS PASSED")
