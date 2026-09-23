"""Run A/B audit and C smoke; never launch formal jobs on a failed audit."""
import os,sys,json,subprocess,time
from pathlib import Path
from runtime import atomic_json
from audit import compare
root=Path(__file__).resolve().parent
out=root/"runs/audit"
out.mkdir(parents=True,exist_ok=False)
jobs=[]
for method,gpu,epochs in [("A","0",5),("B","1",5),("C","4",1)]:
    free=int(subprocess.check_output(["nvidia-smi","-i",gpu,"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    if free<12000: raise RuntimeError("Insufficient GPU memory: "+gpu)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    log=(out/(method+".log")).open("w")
    p=subprocess.Popen([sys.executable,"-u","train.py","--method",method,"--ilda-cache",str(root/"preprocessing/official_ilda.npz"),"--out",str(out/method),"--epochs",str(epochs)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    jobs.append((method,p,log))
atomic_json(out/"status.json",{m:dict(pid=p.pid,status="running") for m,p,_ in jobs})
codes={m:p.wait() for m,p,_ in jobs}
for _,_,f in jobs:f.close()
atomic_json(out/"status.json",codes)
if any(codes.values()): raise RuntimeError("Audit training failed: "+str(codes))
result=compare(out/"A",out/"B",5)
atomic_json(out/"audit.json",result)
assert result["passed"]
c=json.loads((out/"C/history.json").read_text())
assert len(c)==1
atomic_json(out/"complete.json",dict(passed=True,ab=result,c_smoke=True))
print("AUDIT PASSED. Formal 100-epoch runs may now start.",flush=True)
