"""Start the authorized finite round2 queue in a durable background session."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    from run_formal import verify, OUT
    verify()
    if OUT.exists():
        raise RuntimeError("Formal output already exists; inspect it instead of relaunching")
    receipt = ROOT / "launch_receipt.json"
    # Exclusive create prevents duplicate controller launches.
    with receipt.open("x") as handle:
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        with (ROOT / "formal_controller.log").open("x") as log:
            process = subprocess.Popen([sys.executable, "-u", str(ROOT / "run_formal.py"),
                                        "--gpus", "0", "1"], cwd=ROOT, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(pid=process.pid, started_unix=time.time(), gpus=[0,1],
                      methods=["C_state_only", "OT_pull"], seeds=[1341,1174,1370],
                      jobs=6, epochs_per_job=100)
        json.dump(record, handle, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
