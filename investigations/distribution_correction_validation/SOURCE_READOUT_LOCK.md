# Frozen-feature source-validated readout screen

This is a development experiment on already-inspected Houston18. Reuse the three source-val-best CE-only DCRN checkpoints 202601/202602/202603, official ILDA and source splits. Freeze the entire DCRN; extract 288-dimensional features in eval mode from the 1,260 labeled source-train, all labeled source-val, and first 53,184 official-order unlabeled target centers. Do not read target GT before all candidate target probabilities, source-val choices and checkpoint hashes are saved.

Candidates use only labeled source data:

1. `original`: exact stored DCRN target probability (no calibration).
2. `classifier_calibrated`: DCRN classifier logits divided by a scalar temperature chosen on source-val.
3. `prototype`: cosine similarity to the seven source-train class-mean feature prototypes, divided by a source-val-chosen temperature.
4. `ridge`: L2-normalized frozen features, seven-class one-hot ridge fit on source-train, with ridge penalty and score temperature chosen on source-val.

For calibrated classifier and ridge, temperature grid is `[0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 4]`; for cosine prototypes use `[0.02, 0.05, 0.1, 0.2, 0.5, 1]`. Ridge penalties are `[0.01, 0.1, 1, 10]`. All candidate combinations use **source-val macro NLL** (unweighted mean of the seven classwise NLLs) for choice; ties follow the listed grid order. A source-val-choice variant selects among calibrated classifier, prototype and ridge by the same macro NLL; ties favor calibrated classifier. The exact original is reported separately and is not eligible for that calibration selection. No fusion weight, target prior, target confidence gate or Flow is fitted here.

After saving predictions, score each method on official OA, AA, Kappa and seven recalls, especially class 7. Also score the same fixed 0.8 measurement gate: true-class-7 wrong-argmax, correct-but-below-gate and retained fractions; full-scene soft-assignment precision and expected correctness. Source-val choice is reported even if target GT later shows another candidate better. No target-GT selection or final-method claim follows from this single-scene screen.
