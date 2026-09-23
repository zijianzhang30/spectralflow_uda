"""Compare against actual official helpers, including order, patches, RNG and metrics."""
import ast
from pathlib import Path
import numpy as np
import torch
import hdf5storage
from sklearn.metrics import accuracy_score, cohen_kappa_score
from data import DEFAULT_DATA, load_images, loaders, Patches
from runtime import scores, atomic_json
from augmentation import radiation_noise, flip_augmentation

root=Path(__file__).resolve().parent
source=(root/"reference/utils.py").read_text()
tree=ast.parse(source)
names={"get_sample_data","get_all_data","radiation_noise","flip_augmentation"}
ns={"np":np,"torch":torch}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),"<official test reference>","exec"),ns)
s,t=load_images(DEFAULT_DATA,"official_ilda",root/"preprocessing/official_ilda.npz")
sg=hdf5storage.loadmat(str(DEFAULT_DATA/"Houston13_7gt.mat"))["map"]
tg=hdf5storage.loadmat(str(DEFAULT_DATA/"Houston18_7gt.mat"))["map"]
report=[]
for seed in [1341,1174,1370]:
    np.random.seed(seed)
    sx,sy=ns["get_sample_data"](s,sg,3,180)
    _,tx,ty,_,_,_,_=ns["get_all_data"](t,tg,3)
    official_state=np.random.get_state()
    sl,tl,vl,split=loaders(s,t,sg,seed,tg)
    actual_state=np.random.get_state()
    assert official_state[0]==actual_state[0]
    assert np.array_equal(official_state[1],actual_state[1])
    assert official_state[2:]==actual_state[2:]
    assert np.array_equal(sy,split["train_labels"])
    centers=split["target_centers"]
    assert np.array_equal(ty,tg[centers[:,0],centers[:,1]]-1)
    for i in range(len(sx)):
        assert np.array_equal(sl.dataset[i][0].numpy(),sx[i])
    for i in range(len(tx)):
        assert np.array_equal(tl.dataset[i].numpy(),tx[i])
    # Exact sampler behavior for the same global torch state and iterator order.
    torch.manual_seed(seed)
    official_s=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(sx),torch.tensor(sy)),batch_size=32,shuffle=True,drop_last=True)
    official_t=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(tx),torch.tensor(ty)),batch_size=32,shuffle=True,drop_last=True)
    si,ti=iter(official_s),iter(official_t)
    expected=[(next(si),next(ti)[0]) for _ in range(38)]
    torch.manual_seed(seed)
    si,ti=iter(sl),iter(tl)
    for (ex,ey),et in expected:
        ax,ay=next(si); at=next(ti)
        assert torch.equal(ex,ax) and torch.equal(ey,ay) and torch.equal(et,at)
    del official_s,official_t,expected,tx,sx
    x=next(iter(sl))[0]
    np.random.set_state(official_state)
    a=ns["radiation_noise"](x); b=ns["flip_augmentation"](x)
    np.random.set_state(official_state)
    assert torch.equal(a,radiation_noise(x))
    assert torch.equal(b,flip_augmentation(x))
    assert len(sl)==39 and len(tl)==1662 and len(vl.dataset)==1270
    assert len(split["dropped_test_centers"])==16
    report.append(dict(seed=seed,source_patches=1260,target_patches=53200,steps=38,passed=True))
y=np.tile(np.arange(7),7600)[:53184]
pred=y.copy(); pred[::11]=(pred[::11]+1)%7
metrics=scores(y,pred)
assert metrics["oa"]==accuracy_score(y,pred)
assert abs(metrics["kappa"]-cohen_kappa_score(y,pred))<1e-14
assert metrics["oa"]*len(y)/53200==float((y==pred).sum())/53200
atomic_json(root/"protocol_checks.json",dict(passed=True,seeds=report,metrics=True))
print("ALL PROTOCOL CHECKS PASSED",flush=True)
