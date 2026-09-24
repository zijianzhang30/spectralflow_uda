# Distribution correction validation lock (2026-09-24)

This lock was written before training or evaluating the three validation seeds.
Houston18 labels remain available solely to reproduce official GT>0 center order
and for the final audit. Prior exploratory Houston18 results motivated this
algorithm, so this is a **new-seed stability check**, not an untouched-domain
generalization test.

## Fixed runs and algorithm

- New seeds: **202601, 202602, 202603**. No seed replacement based on results.
- Student: Round-9 CE-only A, 100 epochs, exact official ILDA, official
  Houston13 180/class source split, 7x7 patches, batch 32, 38 updates/epoch,
  official augmentation/BN pass order, SGD and LR schedule as in
  `experiments/round9/train.py`. Select earliest source-val accuracy maximum.
- Foundation teacher: same-seed, source-only HyperSIGMA, raw 48-band 33x33
  patches, 20 stage-1 epochs plus 20 full fine-tuning epochs and fixed
  optimization settings from `hypersigma_teacher_gate/train_matched.py`.
  Select the source-val-best checkpoint across the two stages; on ties prefer
  stage 1. No Houston18 labels in teacher fitting or selection.
- Predict both models over all **53,200** official GT>0 target centers in the
  saved official order. Teacher prior is `pi = mean(q_F, axis=0)` over all
  53,200. Student `q` is retained for the first **53,184** official test
  centers. The 16 dropped test centers still contribute to the unlabeled prior.
- B is the sole correction: minimize mean KL(r_i || q_i) over the 53,184
  evaluated rows subject to `mean(r_i) = pi`. Dual class biases are found by
  BFGS, with maximum absolute marginal residual < **1e-5**. No confidence
  threshold, class-specific tuning or fitting to target GT.
- Compare raw A and corrected B on the same source-val-best checkpoint. The
  output probabilities, teacher prior, selected checkpoint hashes and
  projection metadata must be saved before target GT is opened.

## Fixed audit and progression decision

- Official test is first 53,184 centers (batch 32, drop_last), official OA is
  `correct / 53200`; AA, Kappa and per-class accuracy use the evaluated
  confusion matrix. Report each seed and mean ± sample SD (ddof=1).
- OT diagnostic uses a **fixed 0.8 candidate gate for measurement only**,
  first 200 official-order full target batches, same cyclic source batches,
  frozen DCRN source/target features, cosine cost, Sinkhorn temperature 0.05
  and 100 iterations. Report candidate coverage, per-class correct candidate
  recall (especially class 7), and expected OT pair purity. Target labels
  enter these audit metrics only, never the correction or pairing.
- Enter a Flow experiment only if B, versus A, achieves all of: positive
  mean OA difference; mean AA drop no worse than 1 percentage point; mean
  class-7 recall drop no worse than 5 points with no seed worse than 10
  points; mean overall candidate coverage drop no worse than 5 points;
  mean class-7 correct candidate recall drop no worse than 5 points; and
  positive mean expected OT purity difference. These are progression gates,
  not a claim of statistical significance or domain generalization.
- If the gate passes, the next experiment will use the identical teacher
  prior, student source-val checkpoint and projection for paired correction-
  only versus correction-plus-target-to-source Flow. The Flow objective,
  training duration and selection rule must be locked before that training.

The prior exploratory screen on seeds 1341/1174/1370 is development evidence
and is excluded from this validation summary.
