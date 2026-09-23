"""Compare actual A/B step traces, source validation and selected checkpoints."""
import argparse
import json
from pathlib import Path
import torch
from runtime import atomic_json, tensor_hash


def compare(a, b, epochs):
    configs = [json.loads((d/"config.json").read_text()) for d in [a,b]]
    assert [c["method"] for c in configs]==["A","B"]
    allowed={"method","out","flow_source_gradient"}
    assert {k:v for k,v in configs[0].items() if k not in allowed} == {
        k:v for k,v in configs[1].items() if k not in allowed}
    provenance=[json.loads((d/"provenance.json").read_text()) for d in [a,b]]
    assert provenance[0]==provenance[1], "Code, initialization or input files differ"
    traces=[[json.loads(x) for x in (d/"steps.jsonl").read_text().splitlines()] for d in [a,b]]
    expected=[(e,s) for e in range(1,epochs+1) for s in range(1,39)]
    assert all([(x["epoch"],x["step"]) for x in rows]==expected for rows in traces)
    differences=[dict(index=i,fields=[k for k in x if x[k]!=y.get(k)])
                 for i,(x,y) in enumerate(zip(*traces)) if x!=y]
    histories=[json.loads((d/"history.json").read_text()) for d in [a,b]]
    validation_equal=all(all(x[k]==y[k] for k in ["epoch","ce","source_val_accuracy","source_val_loss"])
                         for x,y in zip(*histories))
    assert all(len(h)==epochs for h in histories)
    best=[torch.load(d/"best_source_val.pth",map_location="cpu",weights_only=False) for d in [a,b]]
    selected_equal=(best[0]["epoch"]==best[1]["epoch"] and
                    tensor_hash(best[0]["model"].values())==tensor_hash(best[1]["model"].values()))
    passed=not differences and validation_equal and selected_equal
    return dict(passed=passed,epochs=epochs,steps=len(expected),mismatches=differences[:5],
                validation_equal=validation_equal,selected_checkpoint_equal=selected_equal)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--a",type=Path,required=True)
    p.add_argument("--b",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    report=compare(args.a,args.b,args.epochs)
    atomic_json(args.out,report)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report["passed"] else 1)
