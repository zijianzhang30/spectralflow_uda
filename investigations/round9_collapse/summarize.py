"""Validate exact replays and summarize gradient/teacher-collapse diagnostics."""
import json
from pathlib import Path
from statistics import mean

import hdf5storage
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEEDS = (1341,1174,1370)
METHODS = ("A","B","C")
EPOCHS = (1,5,10,20,50,100)


def read(path):
    return json.loads(path.read_text())


def main():
    gt = hdf5storage.loadmat("/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/Houston18_7gt.mat")["map"]
    output = {"seeds":list(SEEDS),"probe_n":2048,"selected_epochs":list(EPOCHS),
              "probe_true_fraction":{},"trajectory":{},"gradients":{}}
    for seed in SEEDS:
        formal = REPO/"experiments/round9/runs"/f"formal_A_{seed}"
        with np.load(formal/"source_split.npz") as split:
            centers = split["target_centers"][:2048].copy()
        labels = gt[centers[:,0],centers[:,1]].astype(np.int64)
        counts = np.bincount(labels,minlength=8)[1:]
        assert counts.sum()==2048
        output["probe_true_fraction"][str(seed)] = (counts/2048).tolist()
    for method in METHODS:
        by_seed = {}
        for seed in SEEDS:
            run = HERE/"runs"/f"replay_{method}_{seed}"
            audit = read(run/"replay_audit.json")
            assert audit["steps"]==3800 and audit["epochs"]==100
            assert all(v==0 for v in audit["formal_step_hash_differences"].values())
            if method!="A":
                assert audit["teacher_checkpoint_exact_match"]
            trajectory = read(run/("student_trajectory.json" if method=="A" else "teacher_trajectory.json"))
            assert len(trajectory)==100 and [r["epoch"] for r in trajectory]==list(range(1,101))
            key = "student_soft_mass" if method=="A" else "teacher_soft_mass"
            hard = "student_hard_fraction" if method=="A" else "teacher_hard_fraction"
            entropy = "student_mean_entropy" if method=="A" else "teacher_mean_entropy"
            assert all(abs(sum(row[key])-1)<1e-6 for row in trajectory)
            assert all(abs(sum(row[hard])-1)<1e-6 for row in trajectory)
            by_seed[str(seed)] = {str(epoch): {"class6_mass":trajectory[epoch-1][key][5],
                                         "class7_mass":trajectory[epoch-1][key][6],
                                         "class6_hard":trajectory[epoch-1][hard][5],
                                         "class7_hard":trajectory[epoch-1][hard][6],
                                         "entropy":trajectory[epoch-1][entropy]}
                                  for epoch in EPOCHS}
        output["trajectory"][method] = by_seed
    for selection in ("fixed_epoch_100","source_val_best"):
        by_method = {}
        for method in ("B","C"):
            by_seed = {}
            for seed in SEEDS:
                suffix = "" if selection=="fixed_epoch_100" else "_sourcebest"
                row = read(HERE/f"grad_{method}_{seed}{suffix}.json")
                assert row["batches"]==20 and row["true7_sample_n"]>=60
                assert row["method"]==method and row["selection"]==selection
                by_seed[str(seed)] = {
                    "checkpoint_epoch":row.get("checkpoint_epoch",100),
                    "global_ratio_median":row["global_CE_vs_KL_backbone"]["ratio_median"],
                    "global_cosine_mean":row["global_CE_vs_KL_backbone"]["cosine_mean"],
                    "global_conflict_fraction":row["global_CE_vs_KL_backbone"]["cosine_negative_fraction"],
                    "class7_gradient_cosine_mean":row["class7_CE_vs_KL_backbone"]["cosine_mean"],
                    "true7_margin_harmful_fraction":row["true7_margin_harmful_KL_fraction"],
                    "teacher_q6_on_true7":row["teacher_q6_on_true7_mean"],
                    "teacher_q7_on_true7":row["teacher_q7_on_true7_mean"],
                    "teacher_pred6_on_true7":row["teacher_pred6_on_true7_fraction"],
                    "true7_weight_mean":row["true7_weight_mean"]}
            by_method[method] = by_seed
        output["gradients"][selection] = by_method
    (HERE/"summary.json").write_text(json.dumps(output,indent=2)+"\n")
    print("probe target true class6/class7 fractions")
    for seed in SEEDS:
        p = output["probe_true_fraction"][str(seed)]
        print(seed,round(100*p[5],1),round(100*p[6],1))
    print("epoch 10/50/100 class7 soft mass (%)")
    for method in METHODS:
        for seed in SEEDS:
            print(method,seed,[round(100*output["trajectory"][method][str(seed)][str(e)]["class7_mass"],1)
                               for e in (10,50,100)])
    print("fixed100 backbone CE/KL gradient median norm ratio, cosine, teacher q6/q7 on true class7")
    for method in ("B","C"):
        for seed in SEEDS:
            r = output["gradients"]["fixed_epoch_100"][method][str(seed)]
            print(method,seed,round(r["global_ratio_median"],2),round(r["global_cosine_mean"],3),
                  round(r["teacher_q6_on_true7"],3),round(r["teacher_q7_on_true7"],3),
                  round(r["true7_weight_mean"],3))
    print("PASS: 9 exact 3800-step replays, 900 epoch probes, 12 gradient probes")


if __name__=="__main__":
    main()
