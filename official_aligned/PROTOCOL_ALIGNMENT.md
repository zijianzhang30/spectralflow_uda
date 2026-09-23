# SpectralFlow-UDA: official MLUDA protocol alignment

This directory is a separate experiment version. The parent raw experiment and its saved code remain unchanged.
Reference: /home/zhangzj26/TGRS_MLUDA-2024-main, archived under reference/.
MLUDA_hu.py SHA256: 7adfac14955c7792e89842c302bf72f68685950f3a378840fe2e83e8d283020c.
Its Git blob matches the upstream GitHub contents API: 478db8ddff50057dafd5e67b0bbf8fbc2a594db1.
This is protocol-aligned SpectralFlow, NOT reproduction of the full MLUDA method.

## Explicit selection exception

Official MLUDA_hu.py evaluates at epoch % epochs == 0. With epochs=100, it evaluates at epoch 100.
Its best comparison uses target accuracy; checkpoint saving is commented out. It does not implement source-val best.
Our agreed exception: train 100 epochs, validate source each epoch, select earliest maximum source validation OA.
Target performance never selects checkpoints. Source validation preserves training modes and Python/NumPy/Torch RNG.
Historical wrapper source-val results must not be attributed to the released official selector.

## Alignment table

| Item | Official | This version |
|---|---|---|
| Domain / labels | Houston13 to Houston18, seven classes | Same files, hashes recorded |
| Source split | Shuffle within each class, 180/class, shuffle train and leftover validation | Same order; 1260 train, 1270 held-out validation |
| Input | ori_data without extra standardization, then ILDA | Fresh official ILDA cache, shared across A/B/C |
| ILDA | UtilsCMS.py GuidedPGC: PCA=2, epsilon=.009, filter radius=1 | Verbatim function bodies; preprocessing seed pinned to 1341 |
| Patch | Zero padding, 7x7, spectral dimension 48 | Same; lazy patches numerically compared against official arrays |
| Target sample set | All nonzero target GT positions | Same 53200 positions |
| Target order | Per-class shuffle then overall shuffle after source splitting | Same continued NumPy sequence, exact patches/order verified |
| Target label access | Labels read for sample preparation and evaluation | Labels read for identical class-wise sampling; never supplied to training loss |
| Train loaders | batch32, shuffle, drop_last, default global Torch generator | Same sampler behavior and source-then-target iterator order |
| Updates | range(1,len(source_loader)): 38/epoch | Same |
| SGD | lr=.01, momentum=.9, weight_decay=.0005; recreated each epoch | Same for backbone; separate optimizer for our Flow |
| LR | .01/(1+10*(epoch-1)/100)^.75 | Same; five-epoch audit retains horizon100 |
| Views | Original, radiation noise, official flip for source/target | Same augmentation functions, draw order and six backbone BN passes |
| Source loss | CE on original source plus official adaptation losses | Original source CE plus our optional FM |
| Target/augmented BN | Train mode, buffers updated | Same train-mode BN update order in A/B/C |
| Test order / batching | Seeded target order, no shuffle, drop_last=True | Same 53184 evaluated; dropped16 centers saved |
| OA | correct / full dataset size 53200 despite drop_last | oa reproduces official; oa_evaluated separately uses53184 |
| AA / Kappa / classes | Computed on predicted samples | Same definitions, checked against sklearn |
| Checkpoint | Default final epoch | Intentional source-val-best exception |
| Test network input | source and target through MBCA | Target alone through our attention-free network; intentional algorithm difference |

## Method definitions and remaining differences

All methods share one trainer, same initialization, batches, augmentation draws, six BN updates per optimizer step, validation and final evaluation.

- A: source CE, with the same target and augmented-view BN updates as B/C.
- B: A plus class-wise cosine OT and detached Flow. Both feature endpoints detached. Flow trains itself.
- C: A plus class-wise cosine OT and source-side FM gradients, target detached.
- Cost: 1-cos(z_s,z_t); soft classifier q is target OT mass, not target CE supervision.
- Existing OT/Flow settings unchanged: regularization .05, 100 Sinkhorn iterations, 32 pairs per present class, FM weight1.
- No MBCA, channel/spatial attention, SCL, LMMD, SceneShift, reliability, hard routing, intra loss or PCGrad.
- The reused backbone is the DCRN convolutional branches and 288-to-7 classifier. Global mean pooling remains deterministic; it is not the entire official attention-equipped network.
- Source/target and augmentation forwards have the same BN order, but target and augmented passes use no_grad. Official augmented-view gradient losses are intentionally excluded.
- A is a CE-loss control with transductive BN, NOT a purely source-only baseline. Raw parent A is different.
- Official flip_augmentation applies np.fliplr/np.flipud to [N,C,H,W], so it reverses channel/batch axes, not H/W. This version preserves those actual axes for reproduction; no silent correction.
- Official ILDA runs before the per-run seed. We pin preprocessing seed1341 and record package versions/cache hashes. A fresh cache is not assumed bitwise identical to historical unseeded ILDA.
- Removing official modules changes model initialization RNG consumption and hence absolute sampler sequence relative to a full MLUDA model. Sampler implementation matches; identical seed does not imply identical full-method trajectories.
- Deterministic Torch algorithms are enabled for causal hash audits; the official default does not enable all these flags.

ILDA is MLUDA/GuidedPGC preprocessing, not a novel part of SpectralFlow. Results must explicitly say "official ILDA input".
Using target GT for exact official sample ordering is an explicit change from the parent's all-pixel, target-label-free sampling.
Do not describe this configuration as label-free data preparation.

## Verification and execution

prepare_ilda.py generates preprocessing/official_ilda.npz and its provenance manifest from verbatim archived functions.
check_protocol.py compares all source and target patches/order, full epoch sampler output and augmentation values/RNG for seeds1341,1174,1370.
protocol_checks.json records results. Tests do not train on target labels.

run_alignment_audit.py runs A/B five epochs on GPUs0/1 and a one-epoch C smoke test on GPU4, checking available memory first.
runs/audit/audit.json must report passed=true before formal experiments.
The audit compares all190 optimizer steps: feature/classifier parameters, buffers, gradients, CE, inputs, logits, main RNG; validation and selected checkpoints must match.
No target metric is computed during the audit.
Formal100-epoch experiments have not been launched by this audit controller.

Commands (activate the existing project environment first):

    python prepare_ilda.py
    python check_protocol.py
    python run_alignment_audit.py
    python train.py --method A --ilda-cache preprocessing/official_ilda.npz --out runs/formal/A
    python final_test.py --run runs/formal/A

prepare_ilda.py and run_alignment_audit.py refuse existing output directories to avoid overwriting evidence.
Use the same shared cache and seed for A/B/C. Never mix this version with parent raw results in one ablation.

## Completed verification and first formal run

Protocol checks passed for all three seeds. A/B five-epoch audit passed: 190 steps, zero mismatches, equal source validation and selected model. C one-epoch smoke passed. See protocol_checks.json and runs/audit/complete.json.

run_formal.py gates on these checks and unchanged audited code/cache, then runs seed1341 A/B/C for 100 epochs on GPUs0/1/4. It checks the entire 3800-step A/B audit before final source-val-best target evaluation. Live state: runs/official_seed1341_100ep/status.json.
