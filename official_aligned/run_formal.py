"""First official-aligned A/B/C 100-epoch experiment; target tests only after training/audit."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from audit import compare
from data import file_hash
from runtime import atomic_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs/official_seed1341_100ep"
GPUS={"A":"0","B":"1","C":"4"}


def main():
    report=compare(ROOT/"runs/audit/A",ROOT/"runs/audit/B",5)
    assert report["passed"]
    provenance=json.loads((ROOT/"runs/audit/B/provenance.json").read_text())
    for path,digest in provenance["code"].items():
        assert file_hash(ROOT/Path(path).name)==digest,path
    assert json.loads((ROOT/"protocol_checks.json").read_text())["passed"]
    assert file_hash(ROOT/"preprocessing/official_ilda.npz")==provenance["inputs"]["ilda_cache"]
    OUT.mkdir(parents=True,exist_ok=False)
    atomic_json(OUT/"plan.json",dict(seed=1341,epochs=100,protocol="official_ilda",
        selection="source_val_best",gpus=GPUS,methods=["A","B","C"],
        target_labels="official sampling during training; metrics after training/audit",
        code=provenance["code"]))
    lock=threading.Lock()
    states={m:dict(status="queued",gpu=g) for m,g in GPUS.items()}
    def update(method,**values):
        with lock:
            states[method].update(values)
            atomic_json(OUT/"status.json",states)
    def run(method):
        gpu=GPUS[method]
        free=int(subprocess.check_output(["nvidia-smi",f"--id={gpu}",
            "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
        assert free>=12000,f"GPU {gpu} lacks free memory"
        cmd=[sys.executable,"-u",str(ROOT/"train.py"),"--method",method,
             "--protocol","official_ilda","--ilda-cache",str(ROOT/"preprocessing/official_ilda.npz"),"--seed","1341","--epochs","100","--lr-horizon","100",
             "--out",str(OUT/method)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
        try:
            with (OUT/(method+".log")).open("w") as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                update(method,status="training",pid=proc.pid)
                assert proc.wait()==0,f"{method} training failed"
            update(method,status="trained")
        except Exception as exc:
            update(method,status="failed",error=repr(exc));raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,m) for m in GPUS]
        for future in futures:future.result()
    report=compare(OUT/"A",OUT/"B",100)
    atomic_json(OUT/"ab_audit.json",report)
    assert report["passed"],"No target evaluation: A/B audit failed"
    results={}
    for method,gpu in GPUS.items():
        update(method,status="final_testing")
        with (OUT/(method+"_test.log")).open("w") as log:
            subprocess.run([sys.executable,str(ROOT/"final_test.py"),"--run",str(OUT/method)],
                           cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"),
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        results[method]=json.loads((OUT/method/"final_target.json").read_text())
        update(method,status="complete")
    atomic_json(OUT/"summary.json",results)
    assert all(results["A"][k]==results["B"][k] for k in [
        "oa","aa","kappa","per_class_accuracy","selected_epoch"])
    print("COMPLETE: official-aligned A/B/C, full A/B audit and final target testing.",flush=True)


if __name__=="__main__":main()
