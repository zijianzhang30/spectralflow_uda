> Current official-aligned experiments: [official_aligned/](official_aligned/). See [protocol alignment](official_aligned/PROTOCOL_ALIGNMENT.md) and [seed1341 results](results/official_seed1341_100ep/summary.json). The root trainer below is the earlier raw/all-pixel version.

# SpectralFlow-UDA

Working method name: SpectralFlow-UDA (spectral feature flow matching for unsupervised domain adaptation). This is a provisional project name, not a novelty or name-availability claim.
Workspace: /home/zhangzj26/spectralflow_uda. Houston13 -> Houston18 is the first benchmark; the current data adapter remains Houston-specific. Future datasets require their own explicit protocol/split adapters, not reused Houston defaults.
The former houston_clean_uda path is a compatibility symlink for already-running audits and immutable provenance records. New work should use spectralflow_uda.

Standalone workspace. No imports from scene_semantic_uda. No dynamic training-code
replacement. Target labels are not available to train.py/data.py; only
final_test.py loads them, after completed 100-epoch training.

## Scope and protocol truth

The archived local official MLUDA entry is NOT a source-val-best, target-label-free
training protocol. It constructs its target loader using GT>0, evaluates a
drop_last subset, and does not select checkpoints by held-out source validation.
We explicitly prioritize the user's requested source-val-best / final-target-test
rules. This is an aligned benchmark/training recipe with documented deviations,
not a claim of bitwise reproduction of the full official method.

| Item | Local official MLUDA | This workspace |
|---|---|---|
| Images | Houston13/18 ori_data, 48 bands | Same files, no added normalization |
| Source split | 180/class, seeded class shuffle | Same centers/order, tested for 3 seeds |
| Validation | Remaining source samples built but unused by official entry | All remaining 1270; source accuracy; earliest exact tie |
| Patch | Zero-padded 7x7 | Same |
| Train batch | 32, shuffle, drop_last | Same |
| Iterations | range(1,len(loader)): 38 steps | Same, deliberate preservation |
| RNG | Global generators | Separate source/target loader and Flow generators |
| Target train sampling | Target GT>0 | Entire 200340-pixel scene, no GT/mask |
| Optimizer | SGD .01, momentum .9, wd .0005; recreated each epoch | Same for active model parameters; separate Flow optimizer |
| LR | .01/(1+10*(epoch-1)/100)^.75 | Same 100-epoch horizon, including short audits |
| Budget | 100 epochs | 100 formal; 5 only for audit |
| Selection | Official entry lacks source-val-best | Source-val OA only, earliest tie; no target diagnostics |
| Test | Official drop_last, 53184 predictions | Full 53200 labelled centers, drop_last=False |
| Metrics | Confusion/sklearn OA/AA/Kappa | Same formulas; tested against sklearn |
| Input preprocessing | ILDA invoked before patches | Raw default; explicitly separate official_ilda option |

Source validation uses model(x) in eval mode. There is no target context or MBCA.
Target final test uses the same model(x), complete sample count as denominator,
7-class order corresponding to GT labels 1..7. No target-best model selection.

The source/validation center sets do not overlap. Like the original pixel split,
nearby 7x7 patches can overlap spatially; this is not a spatially disjoint split.

## Model and objectives

model.py directly implements the original DCRN spectral (192) and spatial (96)
convolution/residual branches, concatenated and global-mean pooled to 288, with a
Linear(288,7) classifier. There is no MBCA, channel/spatial attention, contrastive
head, domain discriminator or unused residual module. Original conv/BN ordering
and convolution initialization are preserved. Global mean is explicit PyTorch
reduction (not a custom backward); pre-attention features match the archived
reference within tested floating-point tolerance.

- A: source CE only.
- B: A + class-wise cosine OT/Flow; source and target features detached in FM.
  Flow only trains itself. Model parameters AND BN buffers must match A.
- C: A + class-wise cosine OT/Flow; source gradient through interpolation AND
  velocity displacement; target features detached. No gradient through OT pairs/q.

FM loss is mean squared velocity error with coefficient 1. Conditional MLP:
288+7+1 -> 288 -> 288 -> 288, SiLU. Class-wise cost 1-cosine, Sinkhorn reg=.05,
100 iterations, 32 sampled pairs per present source class, class-normalized
detached target softmax probabilities as OT mass. Soft q is NOT a target CE
label, hard routing or a separate pseudo-label adaptation loss. This assumption
was surfaced to the user; alternatives require a new documented algorithm.
No SceneShift, reliability, semantic cost, intra loss or PCGrad.

Target features are extracted with model.eval(), no_grad, restored modes/RNG.
They do not update BN statistics. Source train forward updates BN normally.
A draws the same unlabelled target batches but does not forward them.
Flow initialization uses fork_rng and sampling uses an independent CPU Generator.
There is no dropout in the retained convolutional backbone.

## ILDA separation

Archived UtilsCMS.py describes ILDA/TotalAdaption as image-level transfer and its
header identifies GuidedPGC. It is not required to read the Houston arrays.
Raw is the primary method and never executes or imports ILDA/GuidedPGC.

Because removing ILDA changes official network inputs, an explicit
official_ilda comparison is provided via an immutable image-only cache.
preprocessing/manifest.json records the cache hash and raw input file hashes.
The cache is a copied existing realization, not proof of a fresh exact
recomputation from an upstream release. No ILDA algorithm is reused in the UDA
training objective. Never merge raw and official_ilda scores.

## Execution

Use /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python (environment only).

    python tests/contracts.py
    python run_audit.py

    python train.py --method A --protocol raw --seed 1341 --out runs/raw/A/1341
    python train.py --method B --protocol raw --seed 1341 --out runs/raw/B/1341
    python train.py --method C --protocol raw --seed 1341 --out runs/raw/C/1341

For official_ilda, additionally specify --protocol official_ilda
--ilda-cache preprocessing/official_ilda.npz and use a separate output root.

    python final_test.py --run runs/raw/A/1341

No automatic 100-epoch jobs have been authorized/launched by workspace setup.
Final target testing refuses incomplete/short runs and checks training code/data
hashes, selected epoch, and checkpoint metrics. Outputs:
config.json, provenance.json, source_split.npz, steps.jsonl, history.json,
best_source_val.pth/json, last.pth, training_complete.json, final_target.json.
There is no automatic resume entry; last.pth is retained for inspection.
Do not change source files during a run.

reference/ archives the server's local official sources and hashes for review,
not training imports. They include prior local edits; no upstream release
identity is assumed. tests/contracts.py imports that archived network only to
verify conv features and extracts the source split function for comparison.
The runtime trainer contains no exec or sys.path injection.

## Verification completed (2026-09-23)

- Archived convolutional feature parity, 3 official source splits and patches, sklearn metrics, gradient routing and evaluation-state contracts passed.
- raw A/B: 5 epochs, 190 steps, all step hashes/source validation/selected model identical.
- official_ilda A/B: independently 5 epochs, 190 steps, same checks passed.
- raw C: one-epoch real training smoke passed.
- All four A/B best checkpoints independently reloaded; source-validation loss/accuracy exactly reproduced.
- No target labels were opened by these training/audit checks. No formal 100-epoch training or target test has been launched.

Evidence: contracts.log, runs/audit/complete.json, runs/audit/checkpoint_recheck.json, audit.log.

## First formal run started (2026-09-23)

User authorized a first effect check. run_first_experiment.py launches raw/no-ILDA A/B/C, seed1341, 100epochs with horizon100, on GPUs0/1/4 respectively. Output: runs/raw_seed1341_100ep. All training must complete and the 3800-step A/B hash audit must pass before final_test.py opens target labels. No intermediate target accuracy is computed. summary.json will contain the three selected-checkpoint final target results. This is a single-seed first check, not a multi-seed claim.
