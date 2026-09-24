# Paired full-MLUDA comparator lock (2026-09-24)

This comparator was added after the correction-only validation because the
new validation seeds did not yet have full-MLUDA results. The three seeds
are fixed at 202601, 202602 and 202603; no seed or checkpoint is selected
using target performance.

Run the existing, unchanged `official_aligned/full_mluda.py` for 100 epochs
on each seed. It uses the archived full MLUDA DSANSS architecture, exact
official objective, official shared ILDA cache, 180 labeled source
samples/class, 53,200 official-order target GT>0 centers, 7x7 patches,
batch 32, 38 updates/epoch and the official SGD/LR/augmentation schedule.
Use the earliest source-val-best checkpoint and evaluate once with the
selected-epoch source reference batch. As in Round 1, target GT is used only
for official center ordering and final metrics. The OA denominator is
53,200 despite 53,184 test predictions; AA/Kappa use evaluated samples.

Before viewing target metrics, verify each full-MLUDA source/target split is
byte-identical to the matching new-seed DCRN student split. Report each seed
and mean ± sample SD for OA, AA, Kappa and class-7 recall, compared with
the already frozen correction A/B outcomes. This is a **matched-seed
comparator under our source-val-best rule**, not a claim to reproduce the
paper's ten-run, final-epoch aggregate. The upstream Houston config lists
1174,1370,1417,1418,1421,1535,1546,1599,1610,1631; it does not list
the three new validation seeds.
