"""Queue the fixed prior controls after each teacher prediction is frozen."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (1256, 1622, 1322)


def main():
    pending = set(SEEDS)
    deadline = time.monotonic() + 10800
    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError(f'Prior controls pending: {sorted(pending)}')
        for seed in sorted(pending):
            run = HERE / 'hypersigma_runs' / f'seed_{seed}'
            if not (run / 'prior.json').exists():
                continue
            print(f'Freezing prior controls for seed {seed}', flush=True)
            subprocess.run([sys.executable, '-u', str(HERE / 'freeze_prior_controls.py'),
                            '--seed', str(seed)], check=True)
            pending.remove(seed)
        if pending:
            time.sleep(15)
    print('All three prior-control prediction files are frozen.', flush=True)


if __name__ == '__main__':
    main()
