# SpectralFlow-UDA: main experiment protocol

Start with [MAIN_PROTOCOL.md](MAIN_PROTOCOL.md) and [PROTOCOL_LOCK.json](PROTOCOL_LOCK.json).

Run `python run_round1.py` for the locked three-seed A/B/C and full-MLUDA comparison. It checks the protocol, runs fresh five-epoch audits for all seeds, verifies full-training A/B equality, evaluates source-val-best checkpoints and writes mean/std reports.

See [PROTOCOL_ALIGNMENT.md](PROTOCOL_ALIGNMENT.md) for detailed official-code differences.
