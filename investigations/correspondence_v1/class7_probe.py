"""Post-hoc class-7 confidence and source-prototype audit on frozen features."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from screen import REPO, Backbone, Patches, load_images, source_prototypes, choose_temperature, seed_everything, atomic_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(2)
    run = REPO/"experiments/round6/runs"/f"formal_{args.seed}"
    cfg = json.loads((run/"config.json").read_text())
    cp = torch.load(run/"last.pth", map_location="cpu", weights_only=False)
    assert cp["epoch"] == 100
    model = Backbone().to(args.device).eval()
    model.load_state_dict(cp["model"])
    source, target = load_images(Path(cfg["data"]), cfg["protocol"], Path(cfg["ilda_cache"]))
    with np.load(run/"source_split.npz") as split:
        trc, trl = split["train_centers"].copy(), split["train_labels"].copy()
        vac, val = split["val_centers"].copy(), split["val_labels"].copy()
        tgc = split["target_centers"].copy()
    gt = hdf5storage.loadmat(str(Path(cfg["data"])/"Houston18_7gt.mat"))["map"]
    target_labels = gt[tgc[:, 0], tgc[:, 1]].astype(np.int64)-1
    protos = source_prototypes(model, source, trc, trl, args.device)
    temperature, _ = choose_temperature(model, source, vac, val, protos, args.device)
    loader = DataLoader(Patches(target, tgc, target_labels), batch_size=32,
                        shuffle=False, drop_last=False)
    ys, qpreds, ppreds, qmaxs, pmaxs, q7s, p7s = [], [], [], [], [], [], []
    for x, y in loader:
        z, logits = model(x.to(args.device))
        q = logits.softmax(1)
        p = ((F.normalize(z, dim=1) @ F.normalize(protos, dim=1).T)/temperature).softmax(1)
        qc, qy = q.max(1)
        pc, py = p.max(1)
        ys.extend(y.tolist())
        qpreds.extend(qy.cpu().tolist())
        ppreds.extend(py.cpu().tolist())
        qmaxs.extend(qc.cpu().tolist())
        pmaxs.extend(pc.cpu().tolist())
        q7s.extend(q[:, 6].cpu().tolist())
        p7s.extend(p[:, 6].cpu().tolist())
    y, qy, py, qc, pc, q7, p7 = map(np.asarray,
                                     (ys, qpreds, ppreds, qmaxs, pmaxs, q7s, p7s))
    assert len(y) == 53200 and (y==6).sum()==6365
    correct_q7 = (y==6)&(qy==6)
    correct_p7 = (y==6)&(py==6)
    qconf = qc[correct_q7]
    pconf = pc[correct_p7]
    output = {"seed": args.seed, "target_gt_use": "post-hoc audit only",
              "temperature": temperature, "class7_n": int((y==6).sum()),
              "classifier_ungated_recall": float(correct_q7.sum()/(y==6).sum()),
              "prototype_ungated_recall": float(correct_p7.sum()/(y==6).sum()),
              "classifier_correct_class7_confidence_quantiles": np.quantile(qconf, [0,.25,.5,.75,.9,.95,1]).tolist(),
              "prototype_correct_class7_confidence_quantiles": np.quantile(pconf, [0,.25,.5,.75,.9,.95,1]).tolist() if len(pconf) else None,
              "classifier_class7_recall_by_threshold": {str(t): float(((correct_q7)&(qc>=t)).sum()/(y==6).sum()) for t in (0,.5,.7,.8,.9,.95)},
              "prototype_class7_recall_by_threshold": {str(t): float(((correct_p7)&(pc>=t)).sum()/(y==6).sum()) for t in (0,.5,.7,.8,.9,.95)},
              "classifier_misclassified_class7_to_pred_class": np.bincount(qy[(y==6)&(qy!=6)], minlength=7).tolist(),
              "prototype_misclassified_class7_to_pred_class": np.bincount(py[(y==6)&(py!=6)], minlength=7).tolist(),
              "q_p_both_correct_class7": int((correct_q7&correct_p7).sum()),
              "q_correct_p_wrong_class7": int((correct_q7&~correct_p7).sum()),
              "q_wrong_p_correct_class7": int((~correct_q7&correct_p7).sum())}
    atomic_json(Path(__file__).resolve().parent/f"class7_{args.seed}.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
