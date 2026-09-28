# Frozen MLUDA–HyperSIGMA global/local premise audit

Date: 2026-09-28. This is a post-hoc diagnostic on the already inspected Houston13→Houston18 development scene. It does not select a method or estimate unseen-scene performance.

Use only the frozen source-val-best MLUDA raw and fixed soft-KL strength-1 probabilities, and the same-seed frozen HyperSIGMA 53,200-position probabilities/scene prior, for seeds 202601/202602/202603. Check file hashes against their original manifests and match center order. No training, new projection, confidence threshold selection, or target-label feedback into predictions.

For each seed report:

1. HyperSIGMA 53,200-center soft prior and true 53,200-center class prevalence; MLUDA 53,184-center soft marginal and true 53,184-center prevalence. Compare total variation and each signed class error within the matching denominator. The balanced source-train marginal (1/7 per class) is a fixed reference, evaluated against both target denominators.
2. MLUDA raw versus its already frozen soft-KL result: official OA, AA, per-class recall, changed predictions, correct→wrong, wrong→correct and wrong→different-wrong counts. Partition flip counts by true class and original MLUDA maximum-probability bins [0,.5), [.5,.7), [.7,.9), [.9,1]. Report original raw accuracy within each bin to distinguish confidence from correctness.
3. Report all three seeds individually and aggregate means where useful. Highlight classes 6 and 7. Do not choose a lambda, cutoff, class weight or candidate rule from this audit.

All target GT access is confined to the audit after verifying frozen prediction hashes. This audit can falsify or refine the proposed global/local evidence story on MLUDA, but independent scene validation remains required.
