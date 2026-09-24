"""Six authorized mechanism experiments, two GPU workers, then final testing."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
from data import file_hash
from runtime import atomic_json

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUT = ROOT / "runs/formal"
SEEDS = [1341, 1174, 1370]
METHODS = ["C_state_only", "OT_pull"]
GUARD = threading.Lock()
STATE = {}


def verify():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for name, expected in manifest["code"].items():
        if file_hash(ROOT / name) != expected:
            raise RuntimeError("Candidate code changed: " + name)
    parent = REPO / "official_aligned/PROTOCOL_LOCK.json"
    if file_hash(parent) != manifest["parent_protocol_sha256"]:
        raise RuntimeError("Parent protocol changed")
    locked = json.loads(parent.read_text())
    for name, expected in locked["code"].items():
        if file_hash(parent.parent / name) != expected:
            raise RuntimeError("Round1 code changed: " + name)
    for name, expected in locked["inputs"].items():
        if file_hash(Path(name)) != expected:
            raise RuntimeError("Input changed: " + name)
    return manifest


def update(key, **values):
    with GUARD:
        STATE.setdefault(key, {}).update(values, updated_unix=time.time())
        atomic_json(OUT / "status.json", STATE)


def command(key, arguments, gpu):
    verify()
    while True:
        answer = subprocess.check_output([
            "nvidia-smi", "-i", gpu,
            "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        free, utilization = [int(x.strip()) for x in answer.strip().split(",")]
        if free >= 12000 and utilization <= 10:
            break
        update(key, status="waiting_for_gpu", gpu=gpu, free_mib=free, utilization=utilization)
        time.sleep(20)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with (OUT / (key + ".log")).open("w") as handle:
        process = subprocess.Popen([sys.executable, "-u"] + arguments, cwd=ROOT,
                                   env=env, stdout=handle, stderr=subprocess.STDOUT)
        update(key, status="running", pid=process.pid, gpu=gpu, arguments=arguments)
        result = process.wait()
    update(key, status="complete" if result == 0 else "failed", exit_code=result)
    if result:
        raise RuntimeError("Job failed: " + key)


def stage(jobs, gpus):
    work = queue.Queue()
    for job in jobs:
        work.put(job)
    stop = threading.Event()
    def worker(gpu):
        while not stop.is_set():
            try:
                key, arguments = work.get_nowait()
            except queue.Empty:
                return
            try:
                command(key, arguments, gpu)
            except Exception:
                stop.set()
                raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()


def directory(seed, method):
    return OUT / str(seed) / method


def verify_training():
    verify()
    checked = []
    for seed in SEEDS:
        parent = REPO / "official_aligned/runs"
        ref = (parent / "official_seed1341_100ep/A" if seed == 1341 else
               parent / "round1/formal" / str(seed) / "A")
        for method in METHODS:
            run = directory(seed, method)
            config = json.loads((run / "config.json").read_text())
            complete = json.loads((run / "training_complete.json").read_text())
            history = json.loads((run / "history.json").read_text())
            selected = json.loads((run / "best_source_val.json").read_text())
            assert config["epochs"] == config["lr_horizon"] == complete["epochs"] == 100
            assert complete["formal"] and config["selection"] == "source_val_best"
            assert config["method"] == method and config["seed"] == seed
            assert len(history) == 100 and selected == max(history, key=lambda r: r["source_val_accuracy"])
            with (run / "steps.jsonl").open() as handle:
                steps = [json.loads(line) for line in handle]
            assert [(r["epoch"], r["step"]) for r in steps] == [
                (epoch, step) for epoch in range(1, 101) for step in range(1, 39)]
            with np.load(run / "source_split.npz") as actual, np.load(ref / "source_split.npz") as expected:
                for key in ["train_centers", "train_labels", "val_centers", "val_labels", "target_centers"]:
                    assert np.array_equal(actual[key], expected[key]), (seed, method, key)
            checked.append(dict(seed=seed, method=method, steps=len(steps), selected_epoch=selected["epoch"]))
    atomic_json(OUT / "training_checks.json", dict(passed=True, runs=checked))


def aggregate():
    verify()
    first = json.loads((REPO / "results/round1/summary.json").read_text())["methods"]
    results = {}
    lines = ["# Round2 mechanism experiments", "",
             "All candidates and seeds; source-val-best; OA correct/53200; sample std ddof=1.", "",
             "| Method | Seed | OA (%) | AA (%) | Kappa | Epoch | OA delta vs A (pp) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        rows = []
        for seed in SEEDS:
            run = directory(seed, method)
            row = json.loads((run / "final_target.json").read_text())
            cm = np.asarray(row["confusion_matrix"])
            assert row["method"] == method and row["seed"] == seed
            assert row["evaluated_n"] == int(cm.sum()) == 53184 and row["dataset_n"] == 53200
            assert row["selection"] == "source_val_best"
            assert row["checkpoint_sha256"] == file_hash(run / "best_source_val.pth")
            assert np.isclose(row["oa"], np.trace(cm) / 53200)
            class_accuracy = np.diag(cm) / cm.sum(1)
            chance = cm.sum(0) @ cm.sum(1) / float(cm.sum())**2
            kappa = (np.trace(cm)/cm.sum()-chance)/(1-chance)
            assert np.allclose(row["per_class_accuracy"], class_accuracy)
            assert np.isclose(row["aa"], class_accuracy.mean()) and np.isclose(row["kappa"], kappa)
            baseline = next(r for r in first["A"]["seeds"] if r["seed"] == seed)
            row["oa_delta_vs_A_pp"] = (row["oa"]-baseline["oa"])*100
            rows.append(row)
            lines.append("| {} | {} | {:.4f} | {:.4f} | {:.6f} | {} | {:+.4f} |".format(
                method, seed, row["oa"]*100, row["aa"]*100, row["kappa"],
                row["selected_epoch"], row["oa_delta_vs_A_pp"]))
        values = {key: np.asarray([r[key] for r in rows]) for key in
                  ["oa", "aa", "kappa", "per_class_accuracy", "oa_delta_vs_A_pp"]}
        results[method] = dict(seeds=rows, mean={k: v.mean(axis=0).tolist() for k,v in values.items()},
                               std={k: v.std(axis=0, ddof=1).tolist() for k,v in values.items()})
    lines += ["", "| Method | OA mean ± std (%) | AA mean ± std (%) | Kappa mean ± std |",
              "|---|---:|---:|---:|"]
    for method, result in list(first.items()) + list(results.items()):
        cells = ["{:.4f} ± {:.4f}".format(result["mean"][k]*scale, result["std"][k]*scale)
                 for k,scale in [("oa",100), ("aa",100), ("kappa",1)]]
        lines.append("| " + method + " | " + " | ".join(cells) + " |")
    report = dict(methods=results, seeds=SEEDS, std_ddof=1, selection="source_val_best",
                  manifest_sha256=file_hash(ROOT / "MANIFEST.json"),
                  round1_summary_sha256=file_hash(REPO / "results/round1/summary.json"))
    atomic_json(OUT / "summary.json", report)
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    published = REPO / "results/round2"
    published.mkdir(exist_ok=False)
    for name in ["summary.json", "RESULTS.md", "MANIFEST.json", "training_checks.json"]:
        shutil.copy2(OUT / name, published / name)
    for seed in SEEDS:
        for method in METHODS:
            dest = published / str(seed) / method
            dest.mkdir(parents=True)
            for name in ["config.json", "provenance.json", "best_source_val.json", "final_target.json"]:
                shutil.copy2(directory(seed, method) / name, dest / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify()
    if args.check_only:
        print("Protocol/code/input hashes verified; no jobs launched.")
        return
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "MANIFEST.json", OUT / "MANIFEST.json")
    atomic_json(OUT / "plan.json", dict(seeds=SEEDS, methods=METHODS, gpus=args.gpus,
                epochs=100, controller_pid=os.getpid(), target_test="after all six training jobs pass"))
    for seed in SEEDS:
        for method in METHODS:
            update("train_{}_{}".format(seed, method), status="queued")
    try:
        update("controller", status="training", pid=os.getpid())
        stage([("train_{}_{}".format(seed, method), ["train.py", "--method", method,
               "--seed", str(seed), "--epochs", "100", "--out", str(directory(seed, method))])
               for seed in SEEDS for method in METHODS], args.gpus)
        verify_training()
        update("controller", status="final_testing")
        stage([("test_{}_{}".format(seed, method), ["final_test.py", "--run", str(directory(seed, method))])
               for seed in SEEDS for method in METHODS], args.gpus)
        aggregate()
        atomic_json(OUT / "complete.json", dict(passed=True, seeds=SEEDS, methods=METHODS))
        update("controller", status="complete")
        print("ROUND2 COMPLETE", flush=True)
    except Exception:
        update("controller", status="failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
