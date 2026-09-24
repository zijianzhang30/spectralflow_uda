# SpectralFlow-UDA

The main experiment protocol is **official_aligned/**. Root-level Python trainers are retained only as the historical raw/all-pixel variant.

## First formal study

Houston13 -> Houston18, seeds **1341 / 1174 / 1370**, 100 epochs.
All methods select the earliest maximum **source-validation OA** checkpoint; target metrics never select a checkpoint.

- A: DCRN convolutional backbone + classifier, source CE.
- B: A + class-wise cosine OT / detached Flow.
- C: A + source-side Flow Matching gradients, target detached.
- MLUDA_full: independent full official MLUDA comparator, including its original adaptation losses.

A/B/C share the official target/augmentation BN update schedule. A is therefore a CE-loss control with transductive BN, not a target-free source-only baseline.
All four methods share the same pinned official ILDA cache and source/target sampling protocol.

| Resource | Purpose |
|---|---|
| [Main protocol](official_aligned/MAIN_PROTOCOL.md) | Locked study definition and comparison rules |
| [Machine-readable lock](official_aligned/PROTOCOL_LOCK.json) | Parameters, code hashes and input hashes |
| [Alignment details](official_aligned/PROTOCOL_ALIGNMENT.md) | Exact matches and explicit implementation differences |
| [Study controller](official_aligned/run_round1.py) | Protocol checks, audits, training, testing and aggregation |
| Local results under `results/` | A/B/C and full MLUDA, individual seeds and mean/std; generated records stay outside Git |

Run from official_aligned/ with the existing environment:

    python run_round1.py

This refuses existing round1 outputs. It verifies the lock before each job, runs protocol checks and A/B five-epoch audits for all three seeds, and checks all 3800 A/B steps per seed before target testing.
It reuses the already completed seed1341 A/B/C runs only after verifying their code and input hashes.
It schedules GPUs0/1/4/5 with a 12000 MiB free-memory threshold; GPUs2/6 and memory-constrained GPUs3/7 are excluded.

Live state: official_aligned/runs/round1/status.json.
Final reports: official_aligned/runs/round1/RESULTS.md and summary.json, also exported to results/round1/.
The report includes per-seed OA/AA/Kappa, seven class accuracies, selected epochs, and mean +/- sample standard deviation (ddof=1).

Data, caches, model weights, experiment results and runtime logs are not committed. Dataset paths and the pinned input hashes are recorded in the lock.
The aligned ILDA cache is generated with official_aligned/prepare_ilda.py; historical raw cache preparation differs.
Requirements for ILDA include OpenCV contrib (ximgproc), scikit-image and scikit-learn.

## Scope

This is our own UDA algorithm under an MLUDA-aligned benchmark setup, not a reproduction of the full MLUDA network.
The official comparator alone uses MBCA, SCL, LMMD and its pseudo-label contrastive objective.
A/B/C do not use SceneShift, reliability, semantic OT, hard routing, intra loss or PCGrad.
Source-val best is the deliberate checkpoint-selection exception to the released official final-epoch evaluator.

The earlier root documentation is preserved in [docs/RAW_PROTOCOL.md](docs/RAW_PROTOCOL.md).
