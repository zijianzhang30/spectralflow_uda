"""Run fresh five-epoch A/B traces on two GPUs, then source-gradient smoke."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
from audit import compare
from runtime import atomic_json

ROOT=Path(__file__).resolve().parent


def main():
    root=ROOT/"runs"/"audit"
    root.mkdir(parents=True,exist_ok=False)
    cache=ROOT/"preprocessing"/"official_ilda.npz"
    def run(method,gpu,protocol,epochs):
        out=root/protocol/method
        out.parent.mkdir(exist_ok=True)
        cmd=[sys.executable,"-u",str(ROOT/"train.py"),"--method",method,
             "--protocol",protocol,"--seed","1341","--epochs",str(epochs),
             "--lr-horizon","100","--out",str(out)]
        if protocol=="official_ilda":cmd+=["--ilda-cache",str(cache)]
        with (out.parent/(method+".log")).open("w") as log:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    reports={}
    for protocol in ["raw","official_ilda"]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run,method,gpu,protocol,5) for method,gpu in [("A","0"),("B","1")]]
            for f in futures:f.result()
        report=compare(root/protocol/"A",root/protocol/"B",5)
        atomic_json(root/protocol/"audit.json",report)
        assert report["passed"],report
        reports[protocol]=report
        print(protocol,report,flush=True)
    run("C","0","raw",1)
    atomic_json(root/"complete.json",dict(audits=reports,C_smoke=True,target_test_run=False))
    print("ALL AUDITS COMPLETE; no target labels loaded and no formal run launched.",flush=True)


if __name__=="__main__":main()
