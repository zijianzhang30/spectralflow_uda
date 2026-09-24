# Round 9: no-Flow target refinement, locked before formal runs

Question: Does target consistency refinement improve class correspondence
before Flow? The Houston13→Houston18 source split, shared official ILDA cache,
official target training order, 7×7 patches, batch 32, 38 updates/epoch,
100 epochs, SGD/LR schedule, and official target evaluation remain locked.

Arms:

- A: source CE only; same six student forward passes and BN update order as
  the existing CE control. This run must reproduce the round3 A predictions.
- B: A + target KL(EMA-teacher unaltered target view || student official
  radiation-noise target view). Teacher predictions are detached.
- C: B with only per-sample entropy weighting of that same KL. The raw weight
  is `max(1-H(q_teacher)/log(7), 0.1)`, then divided by its batch mean and
  detached. No gate, hard target pseudo-label, prototype, LMMD, or Flow.

B and C both use EMA momentum 0.99. EMA updates parameters after each student
optimizer step; teacher BN buffers copy the student buffers then. Teacher is
eval-mode for weak predictions and does not alter student BN. Both arms use
the same consistency multiplier `min(epoch/10,1)` with maximum 1.0. The
student receives gradients through the target radiation-noise pass in B/C;
the remaining pass sequence is the same across arms. Source validation picks
the secondary checkpoint with earliest-epoch tie rule. Fixed epoch 100 is
the primary checkpoint. No Houston18 target labels enter training, model
selection, or hyperparameter choice. Its GT is used only to reproduce official
target sample order and later evaluation/diagnostics.

Seeds: 1341, 1174, 1370. All nine formal runs are fresh 100-epoch runs.
The 1-epoch seed-1341 audit found 38/38 A model, BN, gradient, input,
CPU-RNG and source-logit hashes identical to the round8 CE backbone path;
CUDA RNG hashes are device-dependent and were not compared across devices.
B/C have 38/38 matched source/target inputs and RNG, and C's batch-mean
weight is 1 within floating-point tolerance. Synthetic checks verify the KL
direction, teacher detachment, student gradient, and EMA update.

Report OA/AA/Kappa/per-class accuracy only after each training run completes;
then run the frozen-feature correspondence screen without fitting to target
GT. Improvement in a target-labeled diagnostic is exploratory development
evidence, not an untouched unsupervised model-selection result.
