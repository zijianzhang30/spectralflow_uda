# Smooth positive band-wise transform feasibility study (pre-result lock)

Question: within one fixed, low-capacity diagonal affine family, can we reduce
class-prior-aware unlabeled spectral discrepancy while preserving source
classification? This is a Houston development diagnostic, not a new trained
backbone or a target-GT-selected method.

Data/model: official ILDA; CE-DCRN source-val-best checkpoints for seeds
202601/202602/202603; 1,260 balanced labeled source-train centers; 53,200
official unlabeled target centers. Source-train labels may enter fitting.
Source-val labels may only be read after each seed's five candidate transforms
have been selected and frozen. Houston18 target GT may only be read after all
three seeds' candidate transforms, source-val audits, and target predictions
are saved and hashed. Original frozen DCRN weights never change.

Transform: for each of the 48 official-ILDA bands and all 7x7 patch pixels,
`T_b(x)=a_b*x_b+b_b`, `a_b=exp(log(2)*tanh((B*u)_b))`, and
`b_b=3*sigma_s,b*tanh((B*v)_b)`. `B` comprises the first six orthonormal DCT-II
basis vectors over band index, multiplied by `sqrt(48)`. Thus each of `u,v`
has six coefficients, the band profiles are smooth, `0.5<=a_b<=2`, and
`|b_b|<=3*sigma_s,b`. `sigma_s` is source-train center-spectrum population SD
floored at `1e-6`. Initial `u=v=0` is the identity.

Spectral discrepancy: fixed random Fourier-feature approximation to a
three-bandwidth Gaussian-kernel MMD on source-SD-standardized **center
spectra**. Bandwidth base is the median distance of 2,048 pairs sampled from
the source training centers only; multipliers are `[0.5,1,2]`; each kernel
uses 256 fixed features (768 total), with random seed `seed+2718`. Source
class feature means use all 180 labeled train centers per class. Optimize
`pi>=0, sum pi=1` as a convex simplex least-squares problem for each `T`;
there is no target pseudo-label or assumed target class proportion. Fit `T`
using a fixed 8,192-center unlabeled target subset sampled once per seed
without replacement with RNG seed `seed+31415`. The identity-plus-optimized-
pi discrepancy is an explicit baseline and normalizes the optimization loss.

Source compatibility: make a frozen copy with original BN and a second frozen
copy after the already audited one-pass source-only BN recalibration (1,260
train patches, batch 32, including final partial batch). In fitting, use a
fixed source-train subset of 32 centers/class (seed `seed+1618`) and a
continuous, per-class sigmoid of negative true-class logit margin, scaled by
the identity margins' median absolute value for each model. The optimizer's
soft constraints are relative to identity on this subset for both models.
Evaluate every saved candidate on **all 1,260 source-train** patches under
both BN copies; an epsilon-feasible candidate must have no class recall drop
greater than epsilon versus identity in either copy. The epsilon grid is
`[0,0.01,0.02,0.05,0.10]` (fractions). Source-val is not used to accept or
reject candidates.

Search: independently initialize identity for each epsilon. Run 80 Adam
steps (`lr=0.05`) on normalized spectral discrepancy plus an augmented
Lagrangian for the 14 soft source constraints. Dual variables start at 0,
increase by `10*positive_violation` each step, and are capped at 100; the
quadratic penalty coefficient is 100. Re-solve pi at each step using 100
projected-gradient simplex iterations; use 1,000 iterations for saved
candidate scoring. Save checkpoints at steps 0,10,...,80 and exact
source-train recall for each. Pool all 45 saved candidates within a seed;
for each epsilon choose the lowest full-target-subset discrepancy among all
epsilon-feasible candidates, breaking ties by earlier search epsilon/step.
This nested selection yields a monotone **best-found** feasibility curve; it
does not prove a global optimum or nonexistence of other transforms.

After selection/freeze: report the identity-plus-pi baseline, identity with
the original seven-class-balanced source mixture, the selected candidate's
discrepancy/pi/source-train recall for each epsilon, source-val per-class
recall and mean correct-class margin for both BN copies, and the complete
48-band a/b profiles. Run official target inference through the **original-BN**
checkpoint for identity and all selected candidates: 7x7, batch 32,
drop_last=True, 53,184 predictions. Save arrays/hashes before opening target
GT. Score OA as correct/53,200 and AA/Kappa/per-class recall on 53,184 only
after all seeds are frozen. Show the entire epsilon curve; do not choose a
single epsilon based on Houston target GT. If source-val behaves differently
from source-train, report it without retroactively changing candidates.
