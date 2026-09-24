# Main protocol: Houston formal round1

Status: configuration frozen before the three-seed formal study. No algorithm hyperparameters changed.
The authoritative machine-readable definition is PROTOCOL_LOCK.json. run_round1.py refuses code/input drift.

## Locked data and training

- Houston13 -> Houston18, seven classes.
- Source: official per-class shuffle,180 train/class (1260 total),1270 held-out source validation centers.
- Official ILDA/GuidedPGC preprocessing shared across all methods; PCA2, epsilon.009, guided filter radius1.
- ILDA preprocessing seed1341 is fixed once, independently of training seed. Official code does not seed this preprocessing stage.
- Target training:53200 GT>0 centers, exact official class-wise then overall shuffled order following the source split RNG draws.
- Target class labels are read for exact official sample ordering; never used as training supervision for A/B/C or checkpoint selection.
- Zero-padded7x7 patches,48 bands,batch32,shuffle/drop_last train loaders; source iterator created before target.
- Radiation noise and flip copied verbatim; preserve actual batch/channel flip axes of the official implementation.
- Six train-mode backbone passes per step in A/B/C; target and augmented views update BN but are no_grad.
- 38 updates/epoch,100epochs. SGD lr.01,momentum.9,weight_decay.0005,optimizer recreated each epoch.
- Learning rate .01/(1+10*(epoch-1)/100)^.75. Audit epochs do not shorten the LR horizon.
- Seeds1341,1174,1370, common source/target split and preprocessing across methods.

## Methods

A: DCRN convolutional branches and classifier with source CE. No attention.
B: A plus detached class-wise cosine OT/Flow; endpoints detached, Flow trains itself.
C: A plus source-side FM backprop through interpolation and displacement; target detached.
OT cost1-cos; target mass soft classifier q. No target pseudo-label CE.
Unchanged reg.05,100Sinkhorn iterations,32pairs/present class,FM weight1,existing Flow MLP.
No MBCA/SCL/LMMD/SceneShift/reliability/semantic OT/intra loss/PCGrad in A/B/C.
The shared target/augmented BN schedule makes A transductive; do not relabel it as target-free source-only.

MLUDA_full is a separate comparator, not a module added to SpectralFlow.
Its network and objective come from archived TGRS_MLUDA-2024-main code.
full_mluda_objective.py has an AST-identical official training objective.
Native global pooling forwards are preserved; deterministic equivalent backward adapters are checked against native forward and gradients.
Other than selection/evaluation isolation and deterministic pooling backward, the full official loss and augmentation logic are retained.

## Selection and evaluation

Every epoch: source validation, earliest strict maximum OA wins. A/B/C use model(x), MLUDA uses model(x,x)[3] because its network requires a pair.
Validation preserves module modes and all main RNG states.
Target-best is not computed in this study. No target accuracy is consulted during training.
After100epochs: load best_source_val.pth and evaluate once.
Target order is the official seed-dependent training dataset order; batch32,drop_last=True,53184 predictions.
Official OA uses correct/53200; oa_evaluated separately uses correct/53184. AA,Kappa,seven class accuracies use actual predictions.
A/B/C test a target patch alone. Full MLUDA uses paired forward with the final source minibatch saved at the selected epoch.
This extends the official final-epoch last-source-batch evaluation to the source-val-selected epoch without selecting the reference using target accuracy.

## Gates and provenance

1. Rerun protocol checks for seeds1341,1174,1370 against official source/target patch arrays, sampler and augmentation.
2. Run full-MLUDA objective/pooling checks and a one-epoch smoke test.
3. Fresh A/B five-epoch hash audits for EACH seed:190steps,parameter/buffer/gradient/CE/input/logit/main-RNG equality, source validation and selected model equality.
4. Only after all pass: formal three-seed A/B/C and full MLUDA.
5. Check A/B equality over all3800steps for each seed before final target tests.
6. Aggregate all12 results; assert100epochs,source-val selection,shared cache/splits,official test sample counts.

Completed seed1341 A/B/C runs are reused after byte-level code and data verification. They are the same locked protocol.
Historical full-MLUDA source-val experiments are NOT pooled into this report: they used row-major/full-sample target testing and an older ILDA cache.
Full MLUDA is therefore rerun for allthree seeds with this study's shared cache.

Report mean +/- sample std(ddof=1), individual selected epochs, seven class accuracies for each seed and their mean/std.
The seed1341 target outcomes were already observed before this study; this is not an untouched test set. No hyperparameters were tuned in this round.

## Files

run_round1.py: memory-aware four-GPU controller; outputs runs/round1/status.json.
full_mluda.py: independent comparator trainer and final evaluator.
PROTOCOL_LOCK.json: fixed settings and SHA256 code/input identity.
Final artifacts: runs/round1/RESULTS.md,summary.json and results/round1/ under repository root.

## Completed result

All12 method/seed combinations completed100 epochs and source-val-best testing. All three190-step and all three3800-step A/B audits pass. See ../results/round1/RESULTS.md and VERIFIED.md.
Mean OA: A/B74.0320%, C72.0025%, fullMLUDA78.6836%. No algorithm parameters were changed in this study.
