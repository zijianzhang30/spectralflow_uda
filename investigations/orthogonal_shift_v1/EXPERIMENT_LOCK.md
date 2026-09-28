# Class-prior-inexplicable spectral shift: frozen-model diagnostic

This is a pre-result Houston13→Houston18 development diagnostic, not a claimed
paper method. Use the existing CE-DCRN (`A`) source-val-best checkpoints for
seeds 202601, 202602, 202603. Keep all weights and BatchNorm buffers in eval
mode. No training, pseudo-label, Flow, HyperSIGMA prior, target-label-based
choice, or checkpoint change is allowed.

Use the official ILDA cache and exact stored source splits/target center order.
Compute 48-band mean **center** spectra over the 1,260 balanced labeled source
training centers (`mu_s`) and all 53,200 unlabeled target centers (`mu_t`). Let
`sigma_s` be the per-band population SD over the source training center
spectra, floored at `1e-6`. Work in coordinates divided by `sigma_s` when
forming the class-difference subspace. The seven source class mean spectra use
180 centers per class. Define `U` as the column span of the six differences
`(mu_s^c - mu_s^7)/sigma_s`; compute an SVD and retain singular values greater
than `1e-8` times the largest singular value. The projection is Euclidean in
these standardized coordinates. Convert resulting shifts back to the official
ILDA input scale.

Let `Delta = mu_t - mu_s`, `delta_parallel = sigma_s * P_U(Delta/sigma_s)`, and
`delta_perp = Delta - delta_parallel`. Evaluate exactly five conditions:

1. Original: no correction, original source BN.
2. Global: subtract `Delta` from every pixel spectrum of every target patch.
3. Orthogonal: subtract only `delta_perp`.
4. Parallel: subtract only `delta_parallel`.
5. Target BN: use the existing frozen target-only BN state/predictions from
   `investigations/target_bn_recal_v1` with no input correction.

Apply each correction to **all 7×7 pixels** in a patch; no clipping, scaling,
or additional normalization. Run target inference in the official order,
batch 32, `drop_last=True`, yielding 53,184 predictions. Recompute Original in
the same inference script as the three corrections and verify it against the
existing frozen original predictions. Save all predictions, input-shift norms,
checkpoint/split/code hashes, and model parameter/buffer hash assertions before
opening Houston18 GT. After all three seeds are frozen, score official OA as
`correct/53200` and AA/Kappa/per-class recall over 53,184 predicted centers.
Report means and sample SD over seeds and pairwise differences. Do not use
Houston18 GT to revise the projection, choose an alpha, or choose a method.

The causal claim is deliberately narrow: `delta_perp` is not explainable by a
change in source class proportions **if source class-conditional means are
unchanged**. It is not automatically the true domain shift. Better target
performance would support this decomposition as a useful diagnostic; no gain
would show that this single global additive correction is insufficient.
