# Round 3: transport views must help classification

This is a **new, isolated algorithm experiment**. The frozen round1 and round2
code/results are unchanged. Houston13→Houston18 data, source split, ILDA cache,
target center order, 7×7 patch, batch32, official augmentation, 38 updates per
epoch, 100-epoch SGD schedule, and target metric definitions remain the same.
All methods use the same six shared-BN backbone passes per step. This round
does **not** silently change BN routing or fix the official flip axes.

## Hypothesis and controls

The old FM loss learned a field that was unused by the classifier at test time;
its source-side gradient was often dominated by the displacement target path.
Round3 trains the Flow field on **detached** OT-matched source/target features,
then applies source-label CE to a transported feature view. The target feature
and target soft class probabilities remain detached. The Flow field is used to
generate training views only; test-time inference still uses the backbone and
classifier. The view CE does not backpropagate into Flow, so Flow cannot learn a
classifier-specific shortcut instead of matching the displacement.

| Method | Extra view CE | Flow loss | Inference |
|---|---|---|---|
| `A` | None | None | Backbone/classifier |
| `linear_transport` | `classifier(z_s + 0.5 stopgrad(z_t-z_s))` | Detached FM | Backbone/classifier |
| `flow_transport` | `classifier(z_s + 0.5 stopgrad(v(z_s,t=0,c)))` | Detached FM | Backbone/classifier |

OT pairs are sampled exactly as in round2, per present source class. Both
transport methods use the same pairs, Flow architecture, FM weight 1, and
auxiliary CE weight `0.2 × min(epoch/20, 1)`; only the view displacement differs.
Both views have the same source-side Jacobian from CE. These values are a
predeclared first probe, **not** tuned against Houston target labels.
The tiny float64 Sinkhorn systems run on CPU after GPU cosine-cost computation
to avoid GPU launch overhead. In 100 seeded GPU comparisons, CPU and GPU
solvers produced identical sampled OT indices and sampler RNG states.

`flow_transport > linear_transport` would support a contribution from the
learned field beyond ordinary paired interpolation. `linear_transport > A`
alone would support transport-style feature augmentation, not a Flow benefit.
If both hurt, this hypothesis fails for the chosen implementation and weight.

## Selection and evaluation

Primary: fixed epoch100 (`last.pth`), as in the released Houston training
entry. Secondary: highest source-val OA with earliest tie, reported in a
separate table. Do not pick a better rule per seed or method. Target evaluation
is permitted only after a formal 100-epoch run; target labels never enter
training or selection. The official test order/drop-last and OA denominator
are preserved by `final_test.py`.

## Validation before target evaluation

`check_objective.py` checks target/soft-label detachment, classifier/source
gradients, absence of auxiliary CE gradient into Flow, and identical OT/FM RNG
and FM value between transport variants. A real-data one-epoch audit of seed
1341 reproduces all 38 step-level CE, input, logit, gradient, parameter, BN
buffer, and RNG hashes from the frozen A audit. The two transport audits have
the same first-step inputs, RNG hashes, and FM loss.

Full claims require 100 epochs for `A`, `linear_transport`, and
`flow_transport` on all three seeds (1341, 1174, 1370), with both predeclared
selection tables. A single-seed target OA above 80% is not a stable result.

Example commands (from repository root, with the MLUDA virtual environment):

```bash
CUDA_VISIBLE_DEVICES=0 /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python3 experiments/round3/train.py --method flow_transport --seed 1341 --out experiments/round3/runs/formal_flow_1341 --device cuda:0
CUDA_VISIBLE_DEVICES=0 /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python3 experiments/round3/final_test.py --run experiments/round3/runs/formal_flow_1341 --selection fixed_epoch_100 --device cuda:0
```

The trainer writes snapshots and SHA256 hashes of code and input files into
each run. The final evaluator refuses modified code or data. Round1's
`PROTOCOL_LOCK.json` remains untouched.

The early `formal_flow_1341` and `formal_linear_1341` directories contain
interrupted GPU-Sinkhorn trials; they lack `training_complete.json` and are
excluded from reporting. Completed accelerated trials use `formal_flow_cpu_*`
and `formal_linear_cpu_*`.
