"""Run one round3 seed's complete fixed-protocol A/linear/Flow comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
AUDIT_KEYS = ("ce", "model_hash", "backbone_hash", "classifier_hash",
              "buffers_hash", "gradient_hash", "source_logits_hash",
              "source_input_hash", "target_input_hash", "cpu_rng_hash",
              "cuda_rng_hash")


def compare_A(seed, run):
    old = REPO / f"official_aligned/runs/round1/formal/{seed}/A/steps.jsonl"
    fresh = run / "steps.jsonl"
    with old.open() as left, fresh.open() as right:
        count = 0
        for count, (a, b) in enumerate(zip(left, right), 1):
            a, b = json.loads(a), json.loads(b)
            for key in AUDIT_KEYS:
                if a[key] != b[key]:
                    raise AssertionError(f"A drift: seed={seed}, step={count}, key={key}")
    if count != 3800 or sum(1 for _ in old.open()) != sum(1 for _ in fresh.open()):
        raise AssertionError("A comparison requires exactly 3800 matching steps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[1174, 1370], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    methods = [("A", "A"), ("flow_transport", "flow_cpu"),
               ("linear_transport", "linear_cpu")]
    for method, tag in methods:
        run = ROOT / "runs" / f"formal_{tag}_{args.seed}"
        if run.exists():
            raise FileExistsError(run)
        command = [sys.executable, str(ROOT / "train.py"), "--method", method,
                   "--seed", str(args.seed), "--epochs", "100", "--out",
                   str(run), "--device", "cuda:0"]
        log = ROOT / "runs" / f"formal_{tag}_{args.seed}.log"
        start = time.time()
        with log.open("w") as handle:
            subprocess.run(command, cwd=REPO, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        complete = json.loads((run / "training_complete.json").read_text())
        history = json.loads((run / "history.json").read_text())
        assert complete["formal"] and complete["epochs"] == len(history) == 100
        if method == "A":
            compare_A(args.seed, run)
        print(json.dumps(dict(seed=args.seed, method=method,
                              seconds=round(time.time()-start, 1),
                              source_val_last=history[-1]["source_val_accuracy"])),
              flush=True)


if __name__ == "__main__":
    main()
