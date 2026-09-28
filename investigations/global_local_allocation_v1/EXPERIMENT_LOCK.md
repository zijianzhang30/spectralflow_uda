# Frozen DCRN + HyperSIGMA allocation, V1

This is a no-training, exploratory Houston test on the already frozen DCRN and HyperSIGMA probabilities for seeds 202601, 202602, 202603. The rule below is fixed before generating V1 predictions or opening target labels in this experiment. Earlier Houston audits motivated the hypothesis, so Houston is **not** an independent validation set.

For DCRN probabilities `q[i,c]` and HyperSIGMA scene prior `pi[c]`, define a label-free class support floor `L[c] = mean_i q[i,c]^2`. Form `pi_eff` as the KL projection of `pi` onto the simplex with `pi_eff[c] >= L[c]`: `pi_eff[c] = max(L[c], t*pi[c])`, with the unique `t` making its sum one. This limits suppression of a class for which the local model supplies concentrated evidence. The floor is a heuristic, not a guaranteed correct class prevalence.

For sample `i`, set `w[i] = 1 + top1(q[i]) - top2(q[i])`. Compute V1 by minimizing

`mean_i w[i] KL(r[i] || q[i]) + KL(mean_i r[i] || pi_eff)`

over per-sample probability vectors `r`. Thus confident local predictions resist change more strongly. The global KL coefficient is exactly 1, inherited from the earlier soft-KL experiment. There are no fit parameters, target labels, per-seed choices, or grid search.

Frozen comparisons: raw DCRN, prior hard projection, prior soft-KL with coefficient 1, V1, class-only (same `pi_eff`, all weights 1), sample-only (original `pi`, confidence weights). The last two are explanatory ablations, not candidates selected by Houston labels. Evaluate official OA (correct / 53200), AA, Kappa, per-class recall, and flip directions. A positive Houston signal requires V1 mean OA above soft-KL 1, mean AA at least soft-KL 1, and no seed's class-7 recall below its raw DCRN value by more than 5 points. A result here is only a feasibility signal; a paper claim requires new cross-scene benchmarks and independently fixed choices.
