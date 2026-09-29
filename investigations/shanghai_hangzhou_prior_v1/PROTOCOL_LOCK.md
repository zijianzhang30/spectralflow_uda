# Shanghai→Hangzhou independent global-prior gate

Written before reading any Hangzhou target-label values or target model scores. The file header shows `DataCube1` 1600×230×198, `DataCube2` 590×230×198, and three-class `gt1`/`gt2`. Source GT counts were checked to ensure at least 180 training centers per class. The source and target cubes are in `/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Shanghai-Hangzhou/DataCube.mat`.

## Official local baseline

- Reproduce `/home/zhangzj26/TGRS_MLUDA-2024/MLUDA_sh.py` and `config_SH2HZ.py` on the **first three distinct upstream seeds 1341, 1535, 1631**, in that order. Upstream lists seed 1341 twice; the repeated entry is excluded from this technical gate.
- Use `utils.cubeData`: independently standardize each band of each full source/target cube. Apply the upstream `UtilsCMS.ILDA(source,target,PCA=2,radius=0.00009)` before `utils.set_seed(seed)`.
- Use 180 random labeled source centers per class, exact source and target classwise shuffle then global shuffle, `HalfWidth=0` (1×1 input), batch 32 with `drop_last=True`, 100 epochs, constant SGD LR 0.0003, momentum 0.9, weight decay 0.0005, optimizer recreated each epoch, full upstream DSANSS model and loss, final-epoch checkpoint. Inference uses the final source minibatch paired with target batches, as upstream.
- The archived upstream sampler reads target GT to identify evaluation centers and classwise shuffle them. This discloses center membership and order, but no target label may enter training losses, checkpoint selection, prior fitting, or method choices. Freeze checkpoints, centers, source split, target probabilities, and file hashes before computing any target metric. OA divides by all official target centers; AA/Kappa use the `drop_last` predicted centers. Compare three-seed baseline with the paper's SH2HZ result only as a technical gate, not a ten-seed reproduction.
- The [paper's Table VI](https://cfcys.github.io/paper/MLUDA__Eng_Final.pdf) reports **92.15 ± 0.94% OA** and **94.30 ± 0.63% AA** across ten runs. Its Kappa entry is also printed as **92.15%**, identical to OA; this is suspect for a multiclass problem and is not a validation target until checked against the paper's underlying predictions. Compute Kappa from the saved confusion matrix instead.

## Independent HyperSIGMA scene prior

- Build a fresh `SSFusionFramework(img_size=33,in_channels=198,patch_size=2,classes=3,model_size='base')` from original spatial/spectral pretrained weights. The 198-channel, three-class forward and compatibility load have passed a one-sample CPU smoke. It is not either fine-tuned Houston or Pavia teacher.
- Use the same per-scene standardized 198-band cubes **before ILDA**. Use each official source seed's exact 180/class training centers; select the first 30/class from the remaining official source permutation as source-only validation. Record possible patch overlap. Train stage 1 for 20 epochs and full fine-tuning for 20 epochs with the same AdamW/cosine schedule as `investigations/hypersigma_teacher_gate/train_matched.py`. Choose the first highest source-val-accuracy checkpoint within each stage and the better stage (tie→stage 1). Hangzhou labels do not select the stage/checkpoint.
- Freeze teacher probabilities for every official target evaluation center in the exact baseline order, using streamed 33×33 patches. Define the teacher prior as the mean of all teacher probabilities, including target centers omitted by local `drop_last`.

## Frozen controls and decision

- Before opening target scores, freeze full-MLUDA raw, fixed soft-KL correction (`strength=1.0`), and hard prior projection with the existing convex solvers; no weight, class threshold, BN rule, checkpoint, or model may be selected from Hangzhou target GT. Report three paired seeds with OA/AA/Kappa, every class recall, teacher-prior TV versus true target prevalence, and full MLUDA marginal TV on its evaluated centers.
- A promising cross-scene premise requires the teacher prior to beat the MLUDA marginal on most seeds and prior correction to help without unacceptable class loss. If not, stop this fixed-prior line. This run tests a premise and controls, **not** an untrained or unimplemented new global-to-local allocation network. Houston and Pavia have already been inspected; after this Hangzhou audit, a new rule would need a further untouched scene for final validation.
