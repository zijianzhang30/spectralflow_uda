# Houston class-relation pilot: prespecified comparison

Date: 2026-09-29. This lock precedes new target prediction/GT scoring.

## Question

Does a learned class-anchored token add value beyond a source prototype relation under the same MLUDA training recipe? The existing official MLUDA and four generic-token runs are reference controls. The generic-token model has a different architecture and parameter count, so its historical result is descriptive, not a capacity-matched causal control. The two new arms differ only in seven trainable 288-dimensional class-token offsets; both have the same prototype retrieval, source-only reliability, residual update, and original MLUDA objective.

## Data and training

- Houston13→Houston18, official ILDA input, official seed 1341 source/target center arrays, 7×7 patches, batch 32, 100 epochs and 38 steps/epoch.
- Original full MLUDA objective and epoch-wise SGD schedule. Start both arms from the same seeded base MLUDA initialization; no pretraining or HyperSIGMA.
- At initialization and after each epoch, recompute one normalized source prototype per class from **all 180 labeled source-training centers per class** in the current pre-MBCA feature space. Bank extraction uses evaluation mode and restores RNG states. The source validation labels are excluded from the bank and reliability computation.
- Class reliability is source-training leave-one-out nearest-prototype recall, clipped below at 0.25. It multiplies the target residual's magnitude; lower reliability does not redistribute attention mass to competing classes.
- Target retrieval uses cosine softmax over source prototypes at temperature 0.2. Prototype-only content is the fixed source prototype. Class-token content is the normalized sum of that prototype and a learned class offset. Source branch is unchanged. The target residual gate starts near 0.02.

## Readout and decision

Primary: paired fixed epoch 100 OA and AA, then every class's recall, especially classes 6 and 7. Secondary: source-val-best checkpoint using the official source-branch validation metric. This selection can be insensitive to the new target-only branch, so it is not the sole decision criterion. Also inspect source-bank class reliability and token gradient norms.

Freeze both arms' target prediction arrays and checkpoint hashes before opening target GT. This is an exploratory one-seed Houston pilot, not a generalization claim. Do not tune temperature, gate, reliability floor, or number of tokens using this target GT. If class tokens do not improve both OA and AA beyond prototype-only, stop the class-token addition. If promising, add a parameter-matched generic control, then repeat fixed settings on official seeds 1174/1370 and at least one other cross-scene benchmark before any paper claim.
