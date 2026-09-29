# Source class structure versus generic source context

Date: 2026-09-29. This is a follow-up after inspecting the one-seed prototype pilot. The variants and decision rule below are fixed before scoring the new target prediction files.

## Main test

On official Houston13→Houston18 seeds 1341, 1174, and 1370, compare full MLUDA, a class-prototype residual, and a **parameter-matched global-source residual**. The two residual arms use identical source/target splits, official ILDA input, batch 32, 7×7 patches, 100 epochs, full MLUDA objective, epoch-wise SGD, gate initialization, MLP parameter count, bank refresh schedule, and source-only reliability computation.

- **Prototype:** seven labeled source class means in the 288-D pre-MBCA feature space; target retrieves over them.
- **Global:** the same normalized mean of all source training features copied into seven anchor slots. Retrieval is therefore class-agnostic; source-derived context, added capacity, and training perturbation remain.

The original seed-1341 prototype run in `../class_relation_v1/formal_prototype_1341` is reused only if a new one-epoch prototype smoke check reproduces its original epoch-1 metrics exactly. Both arms are trained on seeds 1174 and 1370; the global arm is trained on all three seeds. The historical official MLUDA runs provide the original baseline.

## Scoring and interpretation

Primary metric: paired fixed-epoch-100 OA and AA, including class-6/7 recall. Secondary: each arm's source-val-selected checkpoint under the unchanged official rule. Freeze all new target prediction arrays and checkpoint hashes before opening their target GT. Report per-seed results, paired mean and sample SD. Do not tune temperature, reliability floor, gate, architecture, or checkpoint selection on these scores.

If Prototype and Global are similar, the prior one-seed gain is insufficient evidence for class-aware transfer. Even if Prototype wins, the existing 1,024-sample inference bypass found almost no direct effect of the residual, so a method claim still needs a direct mechanism check and another cross-scene benchmark. This experiment does not itself establish SOTA or generalization.
