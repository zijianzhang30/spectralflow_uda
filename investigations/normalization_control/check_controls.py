"""One real 38-update CPU epoch for A/C: shared vs routed BN control.

Same fixed images/augmentation/optimization within each pair. Not a replay of
the official global RNG trajectory, a new formal run, or target evaluation.
"""
import copy
import io
import json
from contextlib import nullcontext
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

from controls import DomainBNBuffers, spatial_flip

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"official_aligned"))
from model import Backbone
from flow import Flow, matching_loss
from augmentation import flip_augmentation, radiation_noise
from data import Patches, file_hash


def assert_equal(values1, values2):
    values1,values2=list(values1),list(values2)
    assert len(values1)==len(values2)
    for a,b in zip(values1,values2):
        if a is None or b is None: assert a is b
        else: assert torch.equal(a,b)


def one_step(model, flow, router, views, labels, generator, optimizer, flow_optimizer):
    sx,tx,sn,tn,sf,tf = views
    def forward(x,domain,update):
        with router.use(domain,update=update) if router else nullcontext():
            return model(x)
    zs,logits=forward(sx,"source",True)
    with torch.no_grad():
        zt,tl=forward(tx,"target",True)
        forward(sn,"source",False);forward(tn,"target",False)
        forward(sf,"source",False);forward(tf,"target",False)
    ce=F.cross_entropy(logits,labels)
    fm=matching_loss(flow,zs,zt,labels,tl.softmax(1),generator,True) if flow else ce.new_zeros(())
    optimizer.zero_grad(set_to_none=True)
    if flow_optimizer:flow_optimizer.zero_grad(set_to_none=True)
    (ce+fm).backward()
    optimizer.step()
    if flow_optimizer:flow_optimizer.step()
    return logits.detach(),tl.detach(),ce.detach(),fm.detach()


def main():
    torch.set_num_threads(2)
    with np.load(ROOT/"official_aligned/preprocessing/official_ilda.npz") as f:
        source,target=f["s"].astype(np.float32),f["t"].astype(np.float32)
    path=ROOT/"official_aligned/runs/official_seed1341_100ep/A/source_split.npz"
    with np.load(path) as split:
        sd=Patches(source,split["train_centers"],split["train_labels"])
        td=Patches(target,split["target_centers"])
    rng=np.random.RandomState(45001)
    si,ti=rng.permutation(len(sd)),rng.permutation(len(td))
    rows=[]
    for method in ["A","C"]:
        torch.manual_seed(1341);np.random.seed(1341)
        shared=Backbone().train();routed=copy.deepcopy(shared)
        router=DomainBNBuffers(routed)
        flows=[Flow(),None] if method=="C" else [None,None]
        if flows[0] is not None:flows[1]=copy.deepcopy(flows[0])
        optimizers=[torch.optim.SGD(m.parameters(),lr=.01,momentum=.9,weight_decay=.0005)
                    for m in [shared,routed]]
        fos=[torch.optim.SGD(f.parameters(),lr=.01,momentum=.9,weight_decay=.0005) if f else None for f in flows]
        generators=[torch.Generator().manual_seed(1341+99173) for _ in range(2)]
        for step in range(38):
            samples=[sd[int(i)] for i in si[32*step:32*(step+1)]]
            sx=torch.stack([x for x,_ in samples]);labels=torch.tensor([y for _,y in samples])
            tx=torch.stack([td[int(i)] for i in ti[32*step:32*(step+1)]])
            sn=radiation_noise(sx).float();sf=flip_augmentation(sx)
            tn=radiation_noise(tx).float();tf=flip_augmentation(tx)
            views=(sx,tx,sn,tn,sf,tf)
            results=[]
            for model,flow,route,gen,opt,fo in zip([shared,routed],flows,[None,router],generators,optimizers,fos):
                results.append(one_step(model,flow,route,views,labels,gen,opt,fo))
            assert_equal(*results)
            assert_equal(shared.parameters(),routed.parameters())
            assert_equal((p.grad for p in shared.parameters()),(p.grad for p in routed.parameters()))
            if flows[0]:
                assert_equal(flows[0].parameters(),flows[1].parameters())
                assert_equal((p.grad for p in flows[0].parameters()),(p.grad for p in flows[1].parameters()))
            assert torch.equal(generators[0].get_state(),generators[1].get_state())
        assert int(shared.bn1.num_batches_tracked)==228
        assert all(int(router.banks[d][n]["num_batches_tracked"])==38 for d in router.banks for n in router.modules)
        # Eval uses a named bank and never changes its statistics.
        banks=router.state_dict();routed.eval()
        with torch.no_grad(),router.use("source"):
            before=routed(sx)[1]
        routed.train()
        with torch.no_grad(),router.use("target",update=True):routed(tx)
        routed.eval()
        with torch.no_grad(),router.use("source"):
            after=routed(sx)[1]
        assert torch.equal(before,after)
        router.load_state_dict(banks)
        # Real serialization roundtrip includes both model parameters and BN banks.
        stream=io.BytesIO();torch.save(dict(model=routed.state_dict(),bn_banks=router.state_dict()),stream)
        stream.seek(0);checkpoint=torch.load(stream,map_location="cpu",weights_only=False)
        reloaded=Backbone().eval();reloaded.load_state_dict(checkpoint["model"])
        other=DomainBNBuffers(reloaded);other.load_state_dict(checkpoint["bn_banks"])
        for domain in ["source","target"]:
            with torch.no_grad(),router.use(domain):expected=routed(sx)[1]
            with torch.no_grad(),other.use(domain):actual=reloaded(sx)[1]
            assert torch.equal(expected,actual)
        # Failed forwards restore module references and do not commit a bank.
        refs={n:m.running_mean for n,m in router.modules.items()}
        try:
            with router.use("source"):
                raise RuntimeError("test exception")
        except RuntimeError as e:
            assert str(e)=="test exception"
        assert all(m.running_mean is refs[n] for n,m in router.modules.items())
        rows.append(dict(method=method,updates=38,logits_losses_parameters_gradients_exact=True,
                         flow_rng_exact=True,raw_bn_updates_per_domain=38,shared_bn_updates=228,
                         checkpoint_roundtrip_exact=True,target_updates_preserve_source_eval=True))
        print(json.dumps(rows[-1]),flush=True)
    # Distinct sample IDs and channel IDs survive all four spatial-flip choices.
    x=torch.arange(4*48*7*7).reshape(4,48,7,7)
    for seed in [0,1,2,6]:
        np.random.seed(seed);y=spatial_flip(x)
        assert torch.equal(x.flatten(2).sort(2).values,y.flatten(2).sort(2).values)
        assert torch.equal(x[:,:,3,3],y[:,:,3,3])
        assert not torch.equal(x,y) or seed==2
    report=dict(passed=True,scope="38-step CPU pairwise control, seed1341; no target metrics",
                source_target_parameters_shared=True,spatial_flip_preserves_sample_band_center=True,
                rows=rows,code={p.name:file_hash(p) for p in Path(__file__).parent.glob("*.py")})
    (Path(__file__).parent/"checks.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
