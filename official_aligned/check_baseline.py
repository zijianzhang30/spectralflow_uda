"""Check the comparator against official code and pooling kernels before formal runs."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import ast,json,textwrap,sys
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"reference"))
from baseline_pool import _DeterministicGlobalPool,_DeterministicGlobalMaxPool
from runtime import atomic_json
official=(ROOT/"reference/MLUDA_hu.py").read_text()
block=textwrap.dedent(official[official.index("            # 0\n"):official.index("            # Update parameters")])
expected=ast.parse(block).body
actual=next(n for n in ast.parse((ROOT/"full_mluda_objective.py").read_text()).body if isinstance(n,ast.FunctionDef)).body[:-1]
assert ast.dump(ast.Module(body=actual,type_ignores=[]))==ast.dump(ast.Module(body=expected,type_ignores=[]))
torch.manual_seed(1341)
for shape in [(32,288,7,7),(32,24,1,7,7)]:
    x=torch.randn(shape,device="cuda",requires_grad=True)
    y=x.detach().clone().requires_grad_(True)
    if len(shape)==4:
        a=F.adaptive_max_pool2d(x,1);b=_DeterministicGlobalMaxPool.apply(y)
    else:
        a=F.avg_pool3d(x,(1,7,7));b=_DeterministicGlobalPool.apply(y)
    assert torch.equal(a,b)
    grad=torch.randn_like(a);a.backward(grad);b.backward(grad)
    assert torch.equal(x.grad,y.grad)
smoke=json.loads((ROOT/"runs/round1_baseline_smoke_v2/training_complete.json").read_text())
assert smoke["epochs"]==1
atomic_json(ROOT/"baseline_checks.json",dict(passed=True,official_objective_ast_equal=True,
    native_pool_forward_backward_equal=True,one_epoch_smoke=True))
print("FULL MLUDA CONTRACT CHECKS PASSED")
