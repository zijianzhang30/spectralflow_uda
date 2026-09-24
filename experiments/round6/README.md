# Filtered target-to-source Flow, seed1341 screen

Only pairing changes relative to round5. Within each source class, eligible
target samples must have that predicted argmax and confidence at least 0.95.
Skip a class with no eligible target. An entirely empty batch performs no
Flow optimizer step (including weight decay). Each eligible class retains 32
OT sampled pairs, reg0.05 and 100 Sinkhorn iterations. Log pair count per step.

Keep the detached class-agnostic reverse FM, backbone CE, shared BN schedule,
official sampling/preprocessing/augmentation, SGD and 100 epoch budget.
Inference transports all target features with four steps as in round5; no
inference confidence gate is added. Fixed100 is primary, source-val-best
secondary. Compare to both raw A and unfiltered reverse Flow, on seed1341.

This probes whether more reliable endpoints help. Filtering may exclude
classes and does not guarantee correct labels. The earlier 86.7% pair purity
was a post-hoc diagnostic on eval features of one checkpoint, not a verified
training-time purity or a new target OA. Houston has been used for iterative
development and requires independent-scene confirmation for final claims.
