"""Paired-by-seed regret tables and figure for the trust-gate grid (results/gate.csv)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
KEY = ["scenario", "staleness", "desync", "seed"]
FAMILIES = {
    "linear model": ["greedy", "gate_greedy_p2c", "gate_greedy_jsq", "mix25_greedy_p2c", "mix50_greedy_p2c", "mix75_greedy_p2c"],
    "quadratic model (confidently wrong)": ["greedy_q2", "gate_greedy_q2_p2c", "mix50_greedy_q2_p2c"],
    "ewma model (mediocre)": ["greedy_ewma", "gate_greedy_ewma_p2c"],
}
LABEL = {"greedy": "ungated", "gate_greedy_p2c": "hard gate -> P2C", "gate_greedy_jsq": "hard gate -> JSQ",
         "mix25_greedy_p2c": "mix p=.25 -> P2C", "mix50_greedy_p2c": "mix p=.5 -> P2C", "mix75_greedy_p2c": "mix p=.75 -> P2C",
         "greedy_q2": "ungated", "gate_greedy_q2_p2c": "hard gate -> P2C", "mix50_greedy_q2_p2c": "mix p=.5 -> P2C",
         "greedy_ewma": "ungated", "gate_greedy_ewma_p2c": "hard gate -> P2C", "p2c_learned": "learned P2C (no gate)"}


def paired(df, pol, fb):
    base = df[(df.policy == fb) & (df.policy_kwargs.isna())][KEY + ["p99_ms"]].rename(columns={"p99_ms": "fb"})
    d = df[(df.policy == pol) & (df.policy_kwargs.isna())][KEY + ["p99_ms"]].merge(base, on=KEY)
    d["regret"] = (d.p99_ms - d.fb) / d.fb
    return d


def summarize(d):
    r = d.regret
    return dict(n=len(r), mean=r.mean(), p90=r.quantile(0.9), worse=(r > 0).mean(), worse_2x=(r > 1.0).mean())


def main():
    df = pd.read_csv(os.path.join(HERE, "results", "gate.csv"))
    out = ["# Trust gate: regret of the model-driven router against a non-learned fallback, paired by seed\n",
           "Regret = (P99 of policy - P99 of fallback) / P99 of fallback, per seed and cell (16 routers, 10 seeds). "
           "`mean`, `p90` (worst decile) and the share of runs worse than the fallback are over all seeds and the 8 cells "
           "(steady/gray x 50/200 ms x synchronised/not) unless a cell is named.\n"]
    for fb in ("p2c", "jsq"):
        out += [f"\n## Fallback: {fb.upper()}\n", "| model | policy | mean | p90 | worse than fallback | worse by >2x |", "|:---|:---|:---:|:---:|:---:|:---:|"]
        for fam, pols in FAMILIES.items():
            for pol in pols + (["p2c_learned"] if fam == "linear model" else []):
                d = paired(df, pol, fb)
                if d.empty:
                    continue
                s = summarize(d)
                out.append(f"| {fam} | {LABEL.get(pol, pol)} | {s['mean']:+.2f} | {s['p90']:+.2f} | {s['worse']:.0%} | {s['worse_2x']:.0%} |")
    out += ["\n## Per cell (fallback P2C): mean regret [p90]\n", "| cell | ungated | hard gate | mix .5 | ungated q2 | gated q2 | mix .5 q2 |", "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"]
    cols = ["greedy", "gate_greedy_p2c", "mix50_greedy_p2c", "greedy_q2", "gate_greedy_q2_p2c", "mix50_greedy_q2_p2c"]
    for scen in ("steady", "gray"):
        for st in (0.05, 0.2):
            for ds in (False, True):
                row = []
                for pol in cols:
                    d = paired(df, pol, "p2c")
                    d = d[(d.scenario == scen) & (d.staleness == st) & (d.desync == ds)]
                    row.append(f"{d.regret.mean():+.2f} [{d.regret.quantile(.9):+.2f}]" if len(d) else "-")
                out.append(f"| {scen} {int(st * 1000)} ms {'unsync' if ds else 'sync'} | " + " | ".join(row) + " |")
    # window sensitivity (hard gate, P2C fallback), synchronised cells
    sens = df[df.policy_kwargs.fillna("").str.startswith("w=")].copy()
    if len(sens):
        sens["w"] = sens.policy_kwargs.str.extract(r"w=(\d+)").astype(int)
        base = df[(df.policy == "p2c") & (df.policy_kwargs.isna())][KEY + ["p99_ms"]].rename(columns={"p99_ms": "fb"})
        s2 = sens.merge(base, on=KEY)
        s2["regret"] = (s2.p99_ms - s2.fb) / s2.fb
        w50 = paired(df, "gate_greedy_p2c", "p2c")
        w50 = w50[~w50.desync].assign(w=50)
        allw = pd.concat([s2[["w", "scenario", "staleness", "regret"]], w50[["w", "scenario", "staleness", "regret"]]])
        t = allw.groupby(["scenario", "staleness", "w"]).regret.agg(["mean", lambda x: x.quantile(.9)]).round(2)
        out += ["\n## Window sensitivity (hard gate to P2C, synchronised refresh): mean regret / p90\n", t.to_string()]
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, (fam, pols) in zip(axes, FAMILIES.items()):
        for i, pol in enumerate(pols):
            d = paired(df, pol, "p2c")
            c = {"ungated": "#e34948"}.get(LABEL[pol], "#2a78d6" if "hard" in LABEL[pol] else "#eda100")
            ax.scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-.18, .18, len(d)), d.regret, s=5, color=c, alpha=.35)
            ax.plot([i - .3, i + .3], [d.regret.mean()] * 2, color="black", linewidth=2)
        ax.axhline(0, color="#555", linewidth=.8)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels([LABEL[p] for p in pols], rotation=25, ha="right", fontsize=8)
        ax.set_title(fam, fontsize=10, loc="left")
        ax.grid(True, color="#ddd")
    axes[0].set_ylabel("paired regret vs. P2C fallback (symlog)")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "figT1_trust_gate_regret.png"), dpi=150, facecolor="white", bbox_inches="tight")
    with open(os.path.join(HERE, "results", "summary_trust.md"), "w") as f:
        f.write("\n".join(out))
    print("\n".join(out[:40]))


if __name__ == "__main__":
    main()
