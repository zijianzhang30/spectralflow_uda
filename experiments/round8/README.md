# Round8: prototype-weighted OT mass

This isolated experiment changes only the class-wise target mass used by the
round6 target→source Flow OT sampler. Candidate membership remains the
classifier's `argmax(q)==c and max(q)>=0.95`; 32 pairs are sampled per
present source/target class, using the same cosine cost, balanced Sinkhorn,
regularization 0.05, and 100 iterations. For eligible class `c`, target mass
is proportional to `q_c * p_c`, where `p` is the softmax of cosine similarity
to detached current-batch source-class prototypes, temperature 0.05. Absent
source classes are masked in that softmax. There is no new gradient to the
backbone, no class condition at inference, and no target label in the loss.

The frozen Houston13→Houston18 preprocessing, 7×7 patches, target sample
order, augmentation, 38 updates/epoch, SGD schedule, 100 epochs, test batching
and official metric definitions are copied from round6. Fixed epoch100 is
primary; source-val-best is secondary. The original round6 runs serve as
matched classifier-q controls. New runs snapshot all training/evaluation code
and hash their data inputs.

The read-only checkpoint comparison in
`../../investigations/round6_diagnostics/matching_*_fixed_epoch_100.json`
motivated this specific rule. On 200 fixed batches per seed, using batch
source prototypes and the same candidate gate, sampled pair purity changed
from 87.14→87.87%, 78.89→81.51%, and 84.57→84.99%. Candidate counts and
class coverage were identical. These are post-hoc development diagnostics,
not training-time purity or evidence of target OA improvement. In particular,
the absent class-7 candidates remain absent.

The independent gradient/sampling objective check passes. A one-epoch real
data audit on seed1341 completed. Relative to the first 38 steps of the
round6 seed1341 formal run, source/target inputs, model/backbone/classifier
parameters, BN buffers, gradients, and CPU/CUDA RNG hashes had zero
differences. OT pair counts also matched; FM values changed on 23/38 steps.

All three 100-epoch runs and both predeclared evaluations are complete. See
`RESULTS.md` and `summary.json`: the full 3,800-step isolation audit passes
for every seed, but prototype mass gives no material OA improvement over
round6.
