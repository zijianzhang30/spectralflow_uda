"""Evaluate a completed seed under both predeclared checkpoint rules."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[1174, 1341, 1370], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    for tag in ("A", "flow_cpu", "linear_cpu"):
        run = ROOT / "runs" / f"formal_{tag}_{args.seed}"
        for selection in ("fixed_epoch_100", "source_val_best"):
            result = run / f"final_target_{selection}.json"
            if not result.exists():
                command = [sys.executable, str(ROOT / "final_test.py"),
                           "--run", str(run), "--selection", selection,
                           "--device", "cuda:0"]
                subprocess.run(command, cwd=REPO, env=env, check=True,
                               stdout=subprocess.DEVNULL)
            metrics = json.loads(result.read_text())
            print(json.dumps(dict(seed=args.seed, method=metrics["method"],
                                  selection=selection, epoch=metrics["selected_epoch"],
                                  oa=round(metrics["oa"]*100, 4),
                                  aa=round(metrics["aa"]*100, 4))), flush=True)


if __name__ == "__main__":
    main()
