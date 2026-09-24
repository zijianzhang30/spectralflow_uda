"""Locked first formal study: three seeds, A/B/C and full MLUDA."""
import os,sys,json,time,queue,threading,subprocess,concurrent.futures,shutil
from pathlib import Path
import numpy as np
from data import file_hash,DEFAULT_DATA
from runtime import atomic_json
from audit import compare
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs/round1"
SEEDS=[1341,1174,1370]
GPUS=["0","1","4","5"]
LOCK=ROOT/"PROTOCOL_LOCK.json"
mutex=threading.Lock()
states={}
def verify_lock():
    lock=json.loads(LOCK.read_text())
    for n,h in lock["code"].items():assert file_hash(ROOT/n)==h,n
    for n,h in lock["inputs"].items():assert file_hash(n)==h,n
    return lock
def update(key,**values):
    with mutex:
        states.setdefault(key,{}).update(values)
        atomic_json(OUT/"status.json",states)
def command(args,gpu,log,key):
    verify_lock()
    free=int(subprocess.check_output(["nvidia-smi","-i",gpu,"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    while free<12000:
        update(key,status="waiting_memory",gpu=gpu,free_mib=free)
        time.sleep(20)
        free=int(subprocess.check_output(["nvidia-smi","-i",gpu,"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    with log.open("w") as f:
        p=subprocess.Popen([sys.executable,"-u"]+args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        update(key,status="running",gpu=gpu,pid=p.pid)
        code=p.wait()
    update(key,status="complete" if code==0 else "failed",exit_code=code)
    if code:raise RuntimeError(str((key,code,log)))
def stage(jobs):
    work=queue.Queue()
    for job in jobs:work.put(job)
    def worker(gpu):
        while True:
            try:key,args,log=work.get_nowait()
            except queue.Empty:return
            command(args,gpu,log,key)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        for f in [pool.submit(worker,g) for g in GPUS]:f.result()
def trainer(method,seed,out,epochs=100):
    return ["train.py","--method",method,"--seed",str(seed),"--epochs",str(epochs),
        "--lr-horizon","100","--protocol","official_ilda","--ilda-cache",
        str(ROOT/"preprocessing/official_ilda.npz"),"--out",str(out)]
def run_dir(seed,method):
    if seed==1341 and method in "ABC":return ROOT/"runs/official_seed1341_100ep"/method
    return OUT/"formal"/str(seed)/method
def aggregate():
    verify_lock()
    results={}
    for method in ["A","B","C","MLUDA_full"]:
        rows=[]
        for seed in SEEDS:
            run=run_dir(seed,method)
            cfg=json.loads((run/"config.json").read_text())
            assert cfg["seed"]==seed and cfg["epochs"]==100
            assert cfg["selection"]=="source_val_best"
            assert file_hash(Path(cfg["ilda_cache"]))==json.loads(LOCK.read_text())["ilda_sha256"]
            row=json.loads((run/"final_target.json").read_text())
            best=json.loads((run/"best_source_val.json").read_text())
            history=json.loads((run/"history.json").read_text())
            assert len(history)==100 and max(history,key=lambda x:x["source_val_accuracy"])==best
            assert row["selected_epoch"]==best["epoch"] and row["evaluated_n"]==53184
            assert row["dataset_n"]==53200 and row["selection"]=="source_val_best"
            with np.load(run/"source_split.npz") as split, np.load(run_dir(seed,"A")/"source_split.npz") as ref:
                for key in ["train_centers","train_labels","val_centers","val_labels","target_centers"]:
                    assert np.array_equal(split[key],ref[key]),(method,seed,key)
            assert row["seed"]==seed
            rows.append(row)
        results[method]=dict(seeds=rows,mean={},std={})
        for k in ["oa","oa_evaluated","aa","kappa","per_class_accuracy"]:
            values=np.asarray([r[k] for r in rows])
            results[method]["mean"][k]=values.mean(axis=0).tolist()
            results[method]["std"][k]=values.std(axis=0,ddof=1).tolist()
    report=dict(protocol_lock_sha256=file_hash(LOCK),std_definition="sample standard deviation, ddof=1",
        methods=results,selection="source_val_best",seeds=SEEDS)
    atomic_json(OUT/"summary.json",report)
    lines=["# First formal Houston13 -> Houston18 study","",
        "Official ILDA input; source-val-best checkpoints; standard deviation uses ddof=1.",
        "OA follows official correct/53200; AA/Kappa use 53184 predictions.","",
        "| Method | Seed | OA (%) | AA (%) | Kappa | Source-val (%) | Selected epoch |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for m,d in results.items():
        for r in d["seeds"]:
            lines.append("| {} | {} | {:.4f} | {:.4f} | {:.6f} | {:.4f} | {} |".format(m,r["seed"],r["oa"]*100,r["aa"]*100,r["kappa"],r["source_val_accuracy"]*100,r["selected_epoch"]))
    lines+=["","| Method | OA mean +/- std (%) | AA mean +/- std (%) | Kappa mean +/- std |","|---|---:|---:|---:|"]
    for m,d in results.items():
        vals=["{:.4f} +/- {:.4f}".format(d["mean"][k]*scale,d["std"][k]*scale) for k,scale in [("oa",100),("aa",100),("kappa",1)]]
        lines.append("| "+m+" | "+" | ".join(vals)+" |")
    lines+=["","## Per-class accuracy (%)","","| Method | Seed | Class1 | Class2 | Class3 | Class4 | Class5 | Class6 | Class7 |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m,d in results.items():
        for r in d["seeds"]:
            lines.append("| "+m+" | "+str(r["seed"])+" | "+" | ".join("{:.4f}".format(v*100) for v in r["per_class_accuracy"])+" |")
        lines.append("| "+m+" | mean +/- std | "+" | ".join("{:.4f} +/- {:.4f}".format(a*100,b*100) for a,b in zip(d["mean"]["per_class_accuracy"],d["std"]["per_class_accuracy"]))+" |")
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    published=ROOT.parent/"results/round1";published.mkdir(parents=True,exist_ok=False)
    for n in ["summary.json","RESULTS.md"]:shutil.copy2(OUT/n,published/n)
    shutil.copy2(LOCK,published/"PROTOCOL_LOCK.json")
    for seed in SEEDS:
        dst=published/str(seed);dst.mkdir()
        shutil.copy2(OUT/("audit_"+str(seed)+".json"),dst/"five_epoch_audit.json")
        shutil.copy2(OUT/("formal_audit_"+str(seed)+".json"),dst/"full_ab_audit.json")
        for m in results:
            folder=dst/m;folder.mkdir()
            for n in ["config.json","provenance.json","best_source_val.json","final_target.json"]:
                shutil.copy2(run_dir(seed,m)/n,folder/n)
    print("ROUND1 COMPLETE",flush=True)
def main():
    verify_lock()
    OUT.mkdir(parents=True,exist_ok=False)
    shutil.copy2(LOCK,OUT/"PROTOCOL_LOCK.json")
    command(["check_protocol.py"],"0",OUT/"protocol.log","protocol_check")
    assert json.loads((ROOT/"protocol_checks.json").read_text())["passed"]
    command(["check_baseline.py"],"0",OUT/"baseline_check.log","baseline_check")
    jobs=[]
    for seed in SEEDS:
        for method in ["A","B"]:
            dest=OUT/"audit"/str(seed)/method
            jobs.append(("audit_{}_{}".format(seed,method),trainer(method,seed,dest,5),OUT/("audit_{}_{}.log".format(seed,method))))
    stage(jobs)
    for seed in SEEDS:
        base=OUT/"audit"/str(seed)
        result=compare(base/"A",base/"B",5)
        atomic_json(OUT/("audit_"+str(seed)+".json"),result)
        assert result["passed"]
    # Existing seed1341 is reused only after checking its complete provenance and audit.
    old=ROOT/"runs/official_seed1341_100ep"
    for m in "ABC":
        prov=json.loads((old/m/"provenance.json").read_text())
        for n,h in prov["code"].items():assert file_hash(ROOT/Path(n).name)==h
        for n,h in prov["inputs"].items():
            path=ROOT/"preprocessing/official_ilda.npz" if n=="ilda_cache" else DEFAULT_DATA/n
            assert file_hash(path)==h
        update("formal_1341_"+m,status="reused_verified",path=str(old/m))
    jobs=[]
    for seed in SEEDS:
        if seed!=1341:
            for m in "ABC":
                jobs.append(("formal_{}_{}".format(seed,m),trainer(m,seed,run_dir(seed,m)),OUT/("formal_{}_{}.log".format(seed,m))))
        jobs.append(("formal_{}_MLUDA_full".format(seed),["full_mluda.py","--seed",str(seed),"--out",str(run_dir(seed,"MLUDA_full"))],OUT/("formal_{}_MLUDA_full.log".format(seed))))
    stage(jobs)
    for seed in SEEDS:
        report=compare(run_dir(seed,"A"),run_dir(seed,"B"),100)
        atomic_json(OUT/("formal_audit_"+str(seed)+".json"),report)
        assert report["passed"]
    jobs=[]
    for seed in SEEDS:
        for m in ["A","B","C","MLUDA_full"]:
            if seed==1341 and m!="MLUDA_full":continue
            if m=="MLUDA_full":args=["full_mluda.py","--seed",str(seed),"--out",str(run_dir(seed,m)),"--test-only"]
            else:args=["final_test.py","--run",str(run_dir(seed,m))]
            jobs.append(("test_{}_{}".format(seed,m),args,OUT/("test_{}_{}.log".format(seed,m))))
    stage(jobs);aggregate()
    atomic_json(OUT/"complete.json",dict(passed=True,seeds=SEEDS,methods=["A","B","C","MLUDA_full"]))
if __name__=="__main__":main()
