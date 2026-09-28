"""Run label-free teacher target inference as soon as a seed finishes training."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1622, 1322, 1256))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run = HERE / 'hypersigma_runs' / f'seed_{args.seed}'
    selected = run / 'selected.json'
    deadline = time.monotonic() + 7200
    while not selected.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f'Teacher seed {args.seed} did not finish within two hours')
        time.sleep(15)
    subprocess.run([sys.executable, '-u', str(HERE / 'freeze_hypersigma_prior.py'),
                    '--seed', str(args.seed), '--device', args.device], check=True)


if __name__ == '__main__':
    main()
