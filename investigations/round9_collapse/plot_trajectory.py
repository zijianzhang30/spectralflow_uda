"""Publication-style fixed-probe class-6/7 soft-mass trajectories."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SEEDS = (1341,1174,1370)
METHODS = ("A","B","C")
COLORS = {"A":"#3d6aaf","B":"#d1782e","C":"#a63c58"}


def main():
    summary = json.loads((HERE/"summary.json").read_text())
    fig, axes = plt.subplots(2,3,figsize=(11.4,5.5),sharex=True,sharey="row")
    for j,seed in enumerate(SEEDS):
        prior = summary["probe_true_fraction"][str(seed)]
        for i,(class_id,label) in enumerate(((5,"Class 6"),(6,"Class 7"))):
            ax = axes[i,j]
            ax.axhline(prior[class_id]*100,color="0.45",linestyle=":",linewidth=1.25,
                       label="True fraction (post-hoc)" if j==0 else None)
            for method in METHODS:
                run = HERE/"runs"/f"replay_{method}_{seed}"
                file = "student_trajectory.json" if method=="A" else "teacher_trajectory.json"
                key = "student_soft_mass" if method=="A" else "teacher_soft_mass"
                rows = json.loads((run/file).read_text())
                ax.plot([r["epoch"] for r in rows],
                        [100*r[key][class_id] for r in rows],
                        color=COLORS[method],linewidth=1.6,label=("A student" if method=="A" else f"{method} teacher") if j==0 else None)
            ax.set_title(f"Seed {seed}" if i==0 else "")
            ax.set_xlim(1,100)
            ax.set_ylim((0,85) if i==0 else (0,32))
            ax.grid(alpha=.18)
            if j==0:
                ax.set_ylabel(f"{label} soft mass (%)")
            if i==1:
                ax.set_xlabel("Epoch")
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=4,bbox_to_anchor=(.5,1.04),frameon=False)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(HERE/"teacher_mass_trajectory.png",dpi=180,bbox_inches="tight")
    print(HERE/"teacher_mass_trajectory.png")


if __name__=="__main__":
    main()
