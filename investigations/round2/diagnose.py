"""Read-only round1 post-hoc analysis; CPU probes never open target GT.

Uses existing published confusion matrices for retrospective error attribution.
Fresh probes use saved centers, source labels and the shared image cache only.
Not a training replay: fixed diagnostic batches, selected checkpoints, train BN.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "official_aligned"))
from model import Backbone
from flow import Flow, ot_pairs
from data import Patches


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir(seed, method):
    base = ROOT / "official_aligned/runs"
    if seed == 1341:
        return base / "official_seed1341_100ep" / method
    return base / "round1/formal" / str(seed) / method


def flat_grad(loss, params):
    values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                      for g, p in zip(values, params)])


def probe(seed, method, source, target, batches):
    run = run_dir(seed, method)
    path = run / "best_source_val.pth"
    before = digest(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = Backbone()
    model.load_state_dict(checkpoint["model"])
    model.train()
    flow = Flow() if method == "C" else None
    if flow is not None:
        flow.load_state_dict(checkpoint["flow"])
    with np.load(run / "source_split.npz") as split:
        sd = Patches(source, split["train_centers"], split["train_labels"])
        td = Patches(target, split["target_centers"])
    rng = np.random.RandomState(seed + 50000)
    source_order, target_order = rng.permutation(len(sd)), rng.permutation(len(td))
    generator = torch.Generator().manual_seed(seed + 60000)
    params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    rows = []
    for batch in range(batches):
        # Reset BN before each independent checkpoint-local probe.
        model.load_state_dict(checkpoint["model"])
        ids = source_order[batch * 32:(batch + 1) * 32]
        samples = [sd[int(i)] for i in ids]
        x = torch.stack([s[0] for s in samples])
        labels = torch.tensor([s[1] for s in samples])
        tx = torch.stack([td[int(i)] for i in target_order[batch * 32:(batch + 1) * 32]])
        zs, logits = model(x)
        ce = F.cross_entropy(logits, labels)
        with torch.no_grad():
            zt, tl = model(tx)
            q = tl.softmax(1)
        row = dict(ce=float(ce.detach()), source_norm=float(zs.detach().norm(dim=1).mean()),
                   target_norm=float(zt.norm(dim=1).mean()),
                   target_confidence=float(q.max(1).values.mean()),
                   target_entropy=float(-(q * q.clamp_min(1e-12).log()).sum(1).mean()),
                   target_predicted_counts=torch.bincount(q.argmax(1), minlength=7).tolist())
        if flow is not None:
            si, ti, classes = ot_pairs(zs.detach(), zt, labels, q, generator)
            start, end = zs[si], zt[ti]
            t = torch.rand((len(si), 1), generator=generator)
            state = (1-t)*start + t*end
            displacement = end-start
            prediction = flow(state, t, classes)
            fm = F.mse_loss(prediction, displacement)
            state_only = F.mse_loss(prediction, displacement.detach())
            displacement_only = F.mse_loss(prediction.detach(), displacement)
            gc = flat_grad(ce, params)
            gf = flat_grad(fm, params)
            gs = flat_grad(state_only, params)
            gd = flat_grad(displacement_only, params)
            torch.testing.assert_close(gf, gs+gd, atol=2e-5, rtol=2e-4)
            present = labels.unique()
            support = []
            for c in present:
                weights = q[:, c] / q[:, c].sum()
                support.append(dict(class_id=int(c)+1, max_q=float(q[:, c].max()),
                                    effective_target_n=float(1/weights.square().sum())))
            row.update(fm=float(fm.detach()), zero_velocity_mse=float(displacement.detach().square().mean()),
                       ce_gradient_norm=float(gc.norm()), fm_gradient_norm=float(gf.norm()),
                       fm_ce_gradient_ratio=float(gf.norm()/gc.norm().clamp_min(1e-12)),
                       fm_ce_cosine=float(F.cosine_similarity(gc, gf, dim=0)),
                       state_gradient_norm=float(gs.norm()), displacement_gradient_norm=float(gd.norm()),
                       state_ce_cosine=float(F.cosine_similarity(gc, gs, dim=0)),
                       displacement_ce_cosine=float(F.cosine_similarity(gc, gd, dim=0)),
                       target_support=support)
        rows.append(row)
    assert digest(path) == before
    return dict(seed=seed, method=method, epoch=checkpoint["epoch"], checkpoint_sha256=before,
                classifier_weight_norm=float(model.classifier.weight.norm().detach()), batches=rows,
                means={k: float(np.mean([r[k] for r in rows])) for k,v in rows[0].items()
                       if isinstance(v, (int, float))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.batches <= 39:
        parser.error("batches must be 1..39")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    published = ROOT / "results/round1/summary.json"
    summary = json.loads(published.read_text())
    retrospective = []
    for seed in [1341, 1174, 1370]:
        rows = {m: next(r for r in summary["methods"][m]["seeds"] if r["seed"] == seed)
                for m in ["A", "C", "MLUDA_full"]}
        a, c = [np.asarray(rows[m]["confusion_matrix"]) for m in ["A", "C"]]
        assert np.array_equal(a.sum(1), c.sum(1))
        difference = c-a
        errors = difference.copy()
        np.fill_diagonal(errors, 0)
        top = sorted(np.ndindex(7, 7), key=lambda ij: int(errors[ij]), reverse=True)[:5]
        history = json.loads((run_dir(seed, "C") / "history.json").read_text())
        retrospective.append(dict(seed=seed, class_counts=a.sum(1).tolist(),
            c_minus_a_correct_per_class=np.diag(difference).tolist(),
            c_minus_a_oa_pp_per_class=(np.diag(difference)/53200*100).tolist(),
            increased_confusions=[dict(true_class=i+1, predicted_class=j+1, extra_errors=int(errors[i,j]))
                                  for i,j in top if errors[i,j]>0],
            c_training_windows={str(start): {k: float(np.mean([h[k] for h in history[start-1:start+9]]))
                                 for k in ["ce", "fm", "source_val_accuracy"]} for start in [1, 46, 91]}))
    cache = ROOT / "official_aligned/preprocessing/official_ilda.npz"
    with np.load(cache) as f:
        source, target = f["s"].astype(np.float32), f["t"].astype(np.float32)
    probes = []
    for seed in [1341, 1174, 1370]:
        for method in ["A", "C"]:
            result = probe(seed, method, source, target, args.batches)
            probes.append(result)
            print(seed, method, json.dumps(result["means"]), flush=True)
    output = dict(scope="Post-hoc published errors plus selected-checkpoint train-mode CPU probes; not a replay or causal proof",
                  target_gt_opened=False, summary_sha256=digest(published), cache_sha256=digest(cache),
                  script_sha256=digest(Path(__file__)), batches_per_checkpoint=args.batches,
                  retrospective=retrospective, probes=probes)
    (Path(__file__).parent / "diagnostics.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
