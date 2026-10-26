"""Figure and table for research/results/agent_realsys.csv (real-server agent-overload runs)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "results", "agent_realsys.csv")
COLORS = {"none": "#e34948", "shed": "#eb6834", "cap": "#4a3aa7", "little": "#008300"}


def main(csv_path: str = CSV, out_fig: str = os.path.join(HERE, "figures", "figA9_agent_realsys.png")) -> None:
    df = pd.read_csv(csv_path)
    df["family"] = df.policy.str.split(":").str[0]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for ax, (metric, label) in zip(axes, [("goodput_norm", "goodput (completed steps / measured capacity)"),
                                          ("lost", "fraction of tasks lost mid-way"),
                                          ("wasted_steps", "fraction of finished steps wasted")]):
        for pol, g in df.groupby("policy"):
            fam = pol.split(":")[0]
            s = g.groupby("load")[metric].agg(["median", "min", "max"])
            ax.plot(s.index, s["median"], marker="o", color=COLORS.get(fam, "#555"), label=pol,
                    linestyle="--" if fam in ("shed", "cap") else "-")
            ax.fill_between(s.index, s["min"], s["max"], color=COLORS.get(fam, "#555"), alpha=0.12, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df.load.unique()))
        ax.set_xticklabels([f"{v:g}" for v in sorted(df.load.unique())])
        ax.set_xlabel("offered load (task arrivals / task capacity)")
        ax.set_ylabel(label)
        ax.grid(True, color="#ddd")
    axes[0].legend(frameon=False, fontsize=8)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150, facecolor="white", bbox_inches="tight")
    t = df.groupby(["policy", "load"]).goodput_norm.median().unstack().round(2)
    print(t.to_string())
    print("figure:", out_fig)


if __name__ == "__main__":
    main()
