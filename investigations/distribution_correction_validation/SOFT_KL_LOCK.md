# Soft-KL correction screen, frozen before calculation

This is an **exploratory diagnostic** on an already-inspected Houston18 scene.
The three 202601/202602/202603 student checkpoints, teacher checkpoints,
official-order centers, student probabilities and all-53,200 teacher priors
are frozen in `runs/correction_*/predictions_before_gt.npz`. No backbone,
teacher, classifier, OT or Flow retraining occurs. No Houston18 GT selects a
new method or a value of lambda.

For evaluated rows `q_i` and frozen prior `p`, solve the convex objective

`mean_i KL(r_i || q_i) + lambda * KL(mean_i r_i || p)`.

The first KL is explicitly the **sample mean**, not the sum. The sole
correction-strength parameter is lambda. Report the complete, predeclared
grid `lambda = [0, 0.25, 0.5, 1, 2, 4, 8, 16, infinity]` in this order.
Lambda 0 is exactly the raw DCRN probabilities; infinity is exactly the
previous hard-KL projection. No global alpha, classwise alpha, confidence
filter, or entropy-based teacher weight is added. Finite-lambda solutions
use six class biases, with gauge b[0]=0, BFGS dual optimization, marginal
stationarity residual below 1e-5, and objective no worse than the original
student probabilities. Save every corrected array and solver record before
opening target GT in the new screen process.

Post-hoc diagnostics for every grid point: official OA (`correct / 53200`
from 53,184 evaluated samples), AA, Kappa, seven recalls, class-7 recall,
soft class mass, candidate coverage and correct class-7 candidate recall at
the fixed 0.8 measurement gate, and first-200-official-batch expected OT
purity with frozen source/target DCRN features, cosine cost, 0.05 Sinkhorn
temperature and 100 iterations. A and infinity must match the previous
audits exactly. Report each seed and mean ± sample SD; do not report a
Houston18 GT-selected "best lambda" or attach Flow on this evidence alone.
