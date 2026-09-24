# Round7: classification gradient into Flow

Question: does source-label classification of Flow-transported features help
when its gradient reaches the Flow parameters? Round3 classified transported
features, but detached the Flow displacement before the auxiliary CE.

The only new treatment is `flow_transport_coupled`: remove that displacement
detach. Keep detached FM endpoints, class-wise OT pairs, the same source-side
feature and classifier CE path, half-step transported view, FM weight 1, and
auxiliary CE weight `0.2 * min(epoch/20, 1)`.

Controls are `flow_transport_detached` (round3 Flow treatment) and
`linear_transport` (round3 paired displacement without a learned Flow view).
The pre-existing round3 three-seed results use the same frozen Houston protocol
and are historical controls. This round's paired gradient check verifies that
all three variants produce equal FM values and sampler RNG states; coupled and
detached Flow also produce identical auxiliary CE values before an optimizer
step, differing only in the CE-to-Flow gradient.

Primary selection is fixed epoch100, matching rounds3–6. Source-val-best is
reported separately. Target GT is not used in training or checkpoint choice.
No target-best checkpoint will be reported as a formal result. The method uses
Flow to generate source-labelled training views; raw backbone/classifier
predictions are used for target testing. A result above the CE control but not
above linear transport would not establish a Flow-specific advantage.

The protocol lock and round1–6 files are unchanged. The copied data,
preprocessing, backbone, augmentation, evaluation, and runtime code match
round3. Each run snapshots and hashes its code and input data.

Validation: `check_objective.py` passes on CPU. A real GPU one-epoch audit for
seed1341 completed with finite CE/FM/transport CE and source validation; it is
not target-tested.

The three formal GPU runs are complete. `RESULTS.md` and `summary.json` report
the paired comparison. CE coupling produced no material fixed100 OA change
relative to detached Flow and did not surpass linear transport.
