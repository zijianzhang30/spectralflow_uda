# Round 4: target classification consistency with an explicit Flow control

This is a new algorithm probe after round3's learned transport-view CE failed
its fixed-epoch-100 primary comparison. Earlier code, checkpoints, and reports
remain untouched. Houston13→Houston18 data, ILDA, official source/target
sampling and order, batch32, 7×7 patches, radiation/flip code, 38 updates per
epoch, 100-epoch SGD schedule, six shared-BN student passes, and official test
metric definitions remain fixed.

The core change is **direct target classification training**: a raw target
prediction supplies a stop-gradient pseudo label when its maximum probability
is at least 0.95; the same target patch with the official radiation-noise
augmentation receives CE. The weight is `0.5 × clip((epoch−20)/20, 0, 1)`.
No target class label enters the loss. Flip views retain their original
BN-only role; they are not used as paired supervision because the official
`flipud` reverses the batch axis.

| Method | Target consistency selection | Flow |
|---|---|---|
| `A` | None | None |
| `target_consistency` | Raw-target confidence ≥0.95 | Detached FM sidecar |
| `flow_gate_consistency` | Same confidence **and** classifier agrees after approximate inverse Flow step | Same detached FM |

The control and Flow-gated method both train the same Flow on detached
source/target endpoints. Thus a difference between them is attributable to
using Flow to select pseudo-labelled target samples, not to an extra FM loss.
The inverse check uses `z_t − v(z_t,t=1,pseudo_class)` and requires its
source-classifier argmax to equal the raw pseudo label. The inverse step is a
heuristic; FM training alone does not guarantee it is a valid inverse map.

Primary checkpoint rule is fixed epoch100; source-val-best is secondary and
reported separately. This is a predeclared first probe, not a Houston-target
hyperparameter search. A result above 80% on one seed is insufficient;
compare all three seeds, OA/AA/Kappa, per-class accuracy, and a future
independent scene before making a publication claim.

`check_objective.py` confirms that FM cannot update the encoder, pseudo
labels are detached, target-noise CE updates the encoder/classifier, and the
Flow gate selects a subset of the confidence control. A real-data 38-step A
audit is exactly equal to the locked historical CE control in every common
input, loss, gradient, parameter, BN buffer, and RNG hash.

Round3's target-only cumulative BN calibration diagnostic sharply reduced A
target OA on all three seeds, so this round keeps the shared BN protocol and
does not silently add test-time calibration.
