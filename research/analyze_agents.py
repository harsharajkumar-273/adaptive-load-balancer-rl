"""
Figures and tables for the agent-overload study (research/agent_sim.py).

    python -m research.analyze_agents

Reads results/agents*.csv, writes figures/figA*.png and results/summary_agents.md.
"""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.agent_sim import AgentScenario, simulate_agents

HERE = os.path.dirname(__file__)
RESULTS, FIGURES = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9})
TEXT, MUTED, GRID = "#1b1b1b", "#555555", "#dddddd"

HEAD = {  # policy -> (label, color, linestyle, marker)
    "none": ("no admission control", "#e34948", "-", "o"),
    "prog": ("serve by progress (program-level FCFS)", "#eda100", "-", "s"),
    "shed|q=16": ("per-request shedding (Q=16)", "#eb6834", "--", "v"),
    "cap|n=64": ("task cap (N=64)", "#4a3aa7", "--", "^"),
    "task|length=mean|h=20": ("task work budget (H=20 s)", "#2a78d6", "-", "D"),
    "little|df=0.1": ("Little's-law cap (D = 0.1 x timeout)", "#008300", "-", "o"),
    "little|d=5|ident=infer|p_id=0.7": ("Little's-law cap, task inferred (70%)", "#008300", ":", "x"),
}


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _md(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    rows = ["| " + " | ".join([frame.index.name or ""] + [str(c) for c in cols]) + " |",
            "|:---|" + ":---:|" * len(cols)]
    for idx, r in frame.iterrows():
        rows.append("| " + " | ".join([str(idx)] + [("%.2f" % v) if isinstance(v, (int, float, np.floating)) else str(v)
                                                      for v in r]) + " |")
    return "\n".join(rows)


def _line(ax, df, policy, x, metric):
    label, color, ls, marker = HEAD[policy]
    g = df[df.policy == policy].groupby(x)[metric]
    med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
    ax.plot(med.index, med.values, color=color, linestyle=ls, marker=marker, markersize=5, linewidth=1.8, label=label)
    ax.fill_between(med.index, lo.values, hi.values, color=color, alpha=0.10, linewidth=0)


def overload_figure(df: pd.DataFrame, out: List[str], fname: str = "figA1_overload.png", tag: str = "Overload sweep") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.0))
    for ax, (metric, ylabel) in zip(axes, [("goodput_norm", "goodput (completed steps / capacity)"),
                                           ("lost", "fraction of tasks lost mid-way"),
                                           ("wasted", "fraction of server time wasted")]):
        for pol in HEAD:
            _line(ax, df, pol, "load", metric)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df.load.unique()))
        ax.set_xticklabels([f"{v:g}" for v in sorted(df.load.unique())])
        _style(ax, "offered load (task arrivals / task capacity)", ylabel)
    axes[0].axhline(1.0, color=MUTED, linewidth=0.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.tight_layout()
    _save(fig, fname)
    for metric, name in [("goodput_norm", "goodput (completed steps / capacity)"), ("lost", "tasks lost mid-way"),
                         ("wasted", "server time wasted"), ("p99_task_s", "P99 task latency (s) of completed tasks"),
                         ("completed", "fraction of arriving tasks completed")]:
        t = df.groupby(["policy", "load"])[metric].median().unstack()
        order = [p for p in HEAD if p in t.index] + [p for p in t.index if p not in HEAD]
        out += [f"## {tag}: {name} (median over seeds)\n", _md(t.loc[order]), ""]


SENSITIVITY = [("shed|q=", [4, 16, 64], "per-request queue limit Q"), ("cap|n=", [16, 32, 64, 128], "task cap N"),
               ("task|length=mean|h=", [5, 10, 20, 40, 80], "work budget H (s)"),
               ("little|d=", [1, 2.5, 5, 10], "wait target D (s)"), ("little|df=", [0.05, 0.1, 0.2], "D / timeout")]


def _param_policy(prefix, v):
    return f"{prefix}{v:g}" if isinstance(v, float) else f"{prefix}{v}"


def sensitivity_figure(df: pd.DataFrame, out: List[str], load: float = 2.4) -> None:
    d = df[df.load == load]
    fig, axes = plt.subplots(1, 5, figsize=(17.0, 3.6), sharey=True)
    rows = []
    for ax, (prefix, values, xlabel) in zip(axes, SENSITIVITY):
        meds, los, his = [], [], []
        for v in values:
            vals = d[d.policy == _param_policy(prefix, v)].goodput_norm
            meds.append(vals.median()); los.append(vals.quantile(0.25)); his.append(vals.quantile(0.75))
            rows.append((prefix.rstrip("=").split("|")[0] + " " + xlabel, v, vals.median()))
        pos = range(len(values))
        color = "#008300" if prefix.startswith("little") else "#4a3aa7"
        ax.plot(pos, meds, color=color, marker="o", linewidth=1.8)
        ax.fill_between(pos, los, his, color=color, alpha=0.12, linewidth=0)
        ax.set_xticks(list(pos)); ax.set_xticklabels([f"{v:g}" for v in values])
        ax.axhline(1.0, color=MUTED, linewidth=0.8)
        _style(ax, xlabel, "goodput" if ax is axes[0] else "")
    fig.suptitle(f"Sensitivity to the one tuning parameter (load {load:g}x capacity)", color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, "figA2_sensitivity.png")
    out += [f"## Parameter sensitivity at load {load:g} (median goodput)\n",
            "| rule | value | goodput |\n|:---|:---:|:---:|\n" + "\n".join(f"| {a} | {b:g} | {c:.2f} |" for a, b, c in rows), ""]


def surge_figure(df: pd.DataFrame, out: List[str]) -> None:
    pols = ["none", "prog", "shed|q=16", "cap|n=64", "task|length=mean|h=20", "little|df=0.1"]
    factors = sorted(df.surge_factor.unique())
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    w = 0.8 / len(factors)
    for i, f in enumerate(factors):
        meds = [df[(df.policy == p) & (df.surge_factor == f)].recovery_completed.median() for p in pols]
        los = [df[(df.policy == p) & (df.surge_factor == f)].recovery_completed.quantile(0.25) for p in pols]
        his = [df[(df.policy == p) & (df.surge_factor == f)].recovery_completed.quantile(0.75) for p in pols]
        x = np.arange(len(pols)) + i * w
        ax.bar(x, meds, w * 0.9, color=["#9bb7e6", "#2a78d6"][i % 2], label=f"{f:g}x surge for 60 s")
        ax.errorbar(x, meds, yerr=[np.array(meds) - np.array(los), np.array(his) - np.array(meds)], fmt="none",
                    ecolor=MUTED, linewidth=1)
    ax.set_xticks(np.arange(len(pols)) + w * (len(factors) - 1) / 2)
    ax.set_xticklabels([HEAD[p][0].replace(" (", "\n(") for p in pols], fontsize=7)
    ax.legend(frameon=False)
    _style(ax, "", "tasks completed after the surge (base load 0.8x)")
    fig.tight_layout()
    _save(fig, "figA3_surge.png")
    t = df.groupby(["policy", "surge_factor"]).recovery_completed.median().unstack()
    out += ["## Recovery after a surge: fraction of tasks arriving after the surge that complete (median)\n",
            _md(t.loc[[p for p in HEAD if p in t.index] + [p for p in t.index if p not in HEAD]]), ""]


FAMILIES = {"shed": "shed|q=", "cap": "cap|n=", "task work budget": "task|length=mean|h="}


def regime_figure(df: pd.DataFrame, out: List[str], keys=None, fname: str = "figA4_regimes.png",
                  tag: str = "synthetic workloads at 2.4x overload") -> None:
    keys = keys or ["mean_steps", "tool_sec", "timeout", "kv_tasks"]
    sel = df[df.seed < 5].groupby(keys + ["policy"]).goodput_norm.median().unstack()
    ev = df[df.seed >= 5].groupby(keys + ["policy"]).goodput_norm.median().unstack()
    best = ev.max(axis=1)
    frac = ev.div(best, axis=0)
    cols: Dict[str, pd.Series] = {"no admission control": frac["none"], "serve by progress": frac["prog"]}
    for name, prefix in FAMILIES.items():
        members = [c for c in sel.columns if str(c).startswith(prefix)]
        pick = sel[members].idxmax(axis=1)                                 # tuned on seeds 0-4 per regime
        cols[f"{name} (tuned per regime)"] = pd.Series([frac.loc[k, pick.loc[k]] for k in frac.index], index=frac.index)
        # one value for all regimes: the member that is best on average (tuned once)
        once = sel[members].div(sel.max(axis=1), axis=0).mean().idxmax()
        cols[f"{name} (one value for all regimes)"] = frac[once]
    cols["Little's-law cap (D = 0.1 x timeout, never tuned)"] = frac["little|df=0.1"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    palette = {"no admission control": "#e34948", "serve by progress": "#eda100"}
    for name, series in cols.items():
        r = np.sort(series.to_numpy())
        color = palette.get(name, "#008300" if name.startswith("Little") else
                            "#4a3aa7" if name.startswith("cap") else "#2a78d6" if name.startswith("task") else "#eb6834")
        ls = ":" if "one value" in name else "-"
        ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=color, linestyle=ls, linewidth=2.2 if name.startswith("Little") else 1.4,
                label=name)
    _style(ax, "goodput relative to the best policy in that regime", "fraction of regimes ≤ x")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    _save(fig, fname)
    rows = ["| policy | median | 10th pct | worst | regimes within 0.9 of best |", "|:---|:---:|:---:|:---:|:---:|"]
    for name, series in cols.items():
        rows.append(f"| {name} | {series.median():.2f} | {series.quantile(0.1):.2f} | {series.min():.2f} | "
                    f"{(series >= 0.9).mean() * 100:.0f}% |")
    out += [f"## Goodput relative to the best policy per regime, {len(frac)} regimes, {tag} "
            "(parameters chosen on seeds 0-4, evaluation on seeds 5-9)\n", "\n".join(rows), ""]
    for fam, prefix in FAMILIES.items():
        members = [c for c in sel.columns if str(c).startswith(prefix)]
        counts = sel[members].idxmax(axis=1).str.replace(prefix, "").value_counts()
        out += [f"- best {fam} value per regime: " + ", ".join(f"{k}: {v}" for k, v in counts.items()), ""]
    out += ["", "KV-bound regimes (kv_tasks=16) cap every policy; ratios are relative to the best policy, not to capacity.", ""]


def information_figure(df: pd.DataFrame, out: List[str]) -> None:
    pols = [("little|df=0.1", "Little cap\n(no length)"), ("task|length=mean|h=20", "work budget\n(mean length)"),
            ("task|length=pred|sigma=0.5|h=20", "work budget\n(noisy length)"),
            ("task|length=oracle|h=20", "work budget\n(true length)")]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, (metric, ylabel) in zip(axes, [("goodput_norm", "goodput (steps / capacity)"),
                                           ("long_completed", "completion of long tasks (> 2x mean)")]):
        for i, load in enumerate([1.6, 4.0]):
            v = [df[(df.policy == p) & (df.load == load)][metric].median() for p, _ in pols]
            ax.bar(np.arange(len(pols)) + i * 0.4, v, 0.38, color=["#9bb7e6", "#2a78d6"][i], label=f"load {load:g}x")
        ax.set_xticks(np.arange(len(pols)) + 0.2)
        ax.set_xticklabels([n for _, n in pols], fontsize=7.5)
        _style(ax, "", ylabel)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "figA5_information.png")
    t = df[df.load.isin([1.6, 4.0])].groupby(["policy", "load"])[["goodput_norm", "goodput_tasks", "long_completed", "completed"]].median()
    sub = t.loc[[p for p, _ in pols]]
    out += ["## Value and cost of length information (median)\n", "| policy | load | work goodput | task goodput | long-task completion | tasks completed |",
            "|:---|:---:|:---:|:---:|:---:|:---:|"]
    for (pol, load), r in sub.iterrows():
        out.append(f"| {pol} | {load:g} | {r.goodput_norm:.2f} | {r.goodput_tasks:.2f} | {r.long_completed:.2f} | {r.completed:.2f} |")
    out.append("")


def retries_table(df: pd.DataFrame, out: List[str]) -> None:
    d = df[df.load == 2.4]
    for pol in ("none", "little|df=0.1"):
        t = d[d.policy == pol].groupby(["retries", "backoff"]).goodput_norm.median().unstack()
        t.index.name = "retries"
        out += [f"## Retry sensitivity, {pol}, load 2.4: goodput by retry budget (rows) and backoff base (cols)\n", _md(t), ""]


def fluid_check(out: List[str]) -> None:
    """Closed network of N tasks: measured mean wait against max(0, N s / C - s - tau)."""
    pts = []
    for tool in (1.0, 3.0, 10.0):
        for n in (16, 32, 48, 64, 96, 128):
            sc = AgentScenario(load=6.0, tool_sec=tool, kv_tasks=100000, rebuild=0.0, timeout=10000.0, retries=0,
                               step_cv=0.0)
            m = float(np.mean([simulate_agents(sc, f"cap|n={n}", seed=k)["mean_wait_s"] for k in range(3)]))
            pts.append((tool, n, m, max(0.0, n * sc.step_sec / sc.slots - sc.step_sec - tool)))
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for tool, color in zip((1.0, 3.0, 10.0), ("#2a78d6", "#008300", "#eb6834")):
        p = [(a, b) for t, _, a, b in pts if t == tool]
        ax.scatter([b for _, b in p], [a for a, _ in p], color=color, label=f"tool time {tool:g} s", s=22)
    lim = max(max(b for *_, b in pts), 1) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=0.8)
    ax.legend(frameon=False)
    _style(ax, "fluid prediction N s / C - s - tau (s)", "measured mean queueing wait (s)")
    fig.tight_layout()
    _save(fig, "figA6_fluid.png")
    err = max(abs(a - b) for _, _, a, b in pts)
    out += ["## Closed-network wait: measured vs. fluid prediction N s/C - s - tau\n",
            "| tool s | N | measured | fluid |\n|:---:|:---:|:---:|:---:|\n"
            + "\n".join(f"| {t:g} | {n} | {m:.2f} | {f:.2f} |" for t, n, m, f in pts),
            f"\nLargest absolute deviation: {err:.2f} s.", ""]


def main() -> None:
    out: List[str] = ["# Agent-overload study: results\n", "Median over 10 seeds unless stated; goodput = completed steps / "
                      "no-failure step capacity.\n"]
    paths = {k: os.path.join(RESULTS, f"{k}.csv") for k in ("agents", "agents_surge", "agents_regimes", "agents_retries")}
    if os.path.exists(paths["agents"]):
        df = pd.read_csv(paths["agents"])
        overload_figure(df, out); sensitivity_figure(df, out); information_figure(df, out)
    if os.path.exists(paths["agents_surge"]):
        surge_figure(pd.read_csv(paths["agents_surge"]), out)
    if os.path.exists(paths["agents_regimes"]):
        regime_figure(pd.read_csv(paths["agents_regimes"]), out)
    tr = os.path.join(RESULTS, "agents_trace.csv")
    if os.path.exists(tr):
        t = pd.read_csv(tr)
        overload_figure(t[t.kv_tasks == 64], out, "figA7_trace_overload.png", "Trace-calibrated overload sweep (KV 64)")
        for kv in (16, 256):
            sub = t[t.kv_tasks == kv].groupby(["policy", "load"]).goodput_norm.median().unstack()
            out += [f"## Trace-calibrated goodput, KV capacity {kv} tasks (median)\n",
                    _md(sub.loc[[p for p in HEAD if p in sub.index]]), ""]
    trr = os.path.join(RESULTS, "agents_trace_regimes.csv")
    if os.path.exists(trr):
        regime_figure(pd.read_csv(trr), out, ["timeout", "kv_tasks", "load"], "figA8_trace_regimes.png",
                      "trace-calibrated tasks")
    if os.path.exists(paths["agents_retries"]):
        retries_table(pd.read_csv(paths["agents_retries"]), out)
    fluid_check(out)
    with open(os.path.join(RESULTS, "summary_agents.md"), "w") as f:
        f.write("\n".join(out))
    print("wrote", os.path.join(RESULTS, "summary_agents.md"))


if __name__ == "__main__":
    main()
