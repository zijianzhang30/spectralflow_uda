# Source class-mixture BN diagnostic (pre-result lock)

Question: Can source class composition alone move the frozen DCRN decision
boundary after a BN recalibration pass? This is a mechanistic source-only
diagnostic, not a UDA method or target performance claim.

- Use the paired CE-DCRN source-val-best checkpoints for seeds 202601, 202602,
  202603, their recorded official-ILDA source splits, and no target samples or
  target labels.
- Freeze all trainable parameters. Copy each original checkpoint separately for
  every condition. One train-mode, no-gradient forward pass updates existing BN
  buffers only; then use eval mode on the same held-out source validation set.
- Batch size 32, 1,260 calibration samples, sequential DataLoader order, including
  the last partial batch. This matches the earlier source-only BN audit.
- Conditions: original checkpoint (no calibration); balanced source calibration
  (180 unique train samples from each class); class-1-majority, class-6-majority,
  and class-7-majority calibration (630 samples from the majority class, 105
  from each other class). Minority samples are without replacement. Majority
  samples are sampled with replacement from the same 180-source-item pool.
- Run three independently seeded draws/orders per calibrated condition. Exact
  counts are fixed; no hyperparameter or condition selection based on outcome.
- Report overall and per-class source-validation recall, confusion matrices,
  true-class logit margins, and BN-buffer displacement by layer. Verify that
  every parameter tensor is bitwise unchanged. No target score is computed.

Interpretation: If skewed source composition degrades source-validation class
geometry versus balanced calibration across seeds and draws, then class mixture
is sufficient to disturb BN inference here. If not, target-BN failure cannot be
explained by composition alone; conditional/domain shift or other BN dynamics
remain. This experiment cannot establish the actual cause of target failure.
