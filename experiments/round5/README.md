# Round 5: class-agnostic target→source Flow

This isolated experiment tests the direction suggested by the user. The
source classifier is trained exactly as in the CE control. A separate Flow is
trained with detached endpoints, using the existing class-wise OT pairing but
learning a **class-agnostic velocity from target features to source features**.
At target test time, four predeclared Euler steps transport each target
feature through that Flow before the unchanged classifier. The same run also
reports raw logits from the identical checkpoint.

The transport changes the actual target prediction path and needs no target
pseudo label or class condition at inference. This isolates whether a learned
reverse mapping helps a source-trained classifier. The Flow receives only FM
gradients; the main model receives only source CE gradients. If the step-wise
audit remains exact, raw target predictions should reproduce the frozen A
control, and any difference comes from test-time feature transport.

The locked Houston13→Houston18 sampling, ILDA, 7×7 patches, batch32, official
augmentation and BN update order, 38 updates/epoch, 100-epoch SGD schedule,
and target test batching/metrics are unchanged. Fixed epoch100 is primary;
source-val-best is secondary. The four inference steps were chosen before
seeing round5 target scores and are not tuned per seed. Target labels are used
only for official position order and final scoring.

`check_objective.py` validates direction and detached endpoint gradients. On
the real seed1341 one-epoch audit, all 38 common model/input/gradient/RNG
hashes match the frozen CE control. The previous `investigations/
reverse_transport/` diagnostic showed that simply running the existing
class-conditioned source→target field backward changed OA by −2.27, +0.07,
and +0.90 pp across three seeds; this round trains the reverse direction
explicitly and removes the class condition.
