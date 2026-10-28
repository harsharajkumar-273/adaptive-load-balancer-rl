"""
Turns research/results/*.csv into figures (research/figures/) and a summary
markdown table (research/results/summary.md).

    python -m research.analyze

Latency tails are heavy (herding episodes produce multi-second outliers), so
every statistic is the median over seeds with the interquartile range, not the mean.
"""
from __future__ import annotations

import os

from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

# Fixed colour per policy (reference categorical palette, light mode, 8 slots).
# Beyond eight series we use composite encoding instead of new hues:
# "+lc" variants share their base colour (dashed); learned power-of-d
# variants share P2C-learned's green and differ by line style and marker.
COLORS = {
    "wrandom": "#8a8984",
    "jsq": "#2a78d6",
    "sed": "#eda100",
    "p2c": "#1baf7a",
    "greedy": "#eb6834",
    "ts": "#4a3aa7",
    "ts_epoch": "#e87ba4",
    "p2c_learned": "#008300",
    "prequal": "#e34948",
}
LABELS = {
    "wrandom": "weighted random",
    "jsq": "JSQ",
    "sed": "SED",
    "p2c": "P2C",
    "greedy": "learned greedy",
    "ts": "Thompson (per request)",
    "ts_epoch": "Thompson (150 ms epochs)",
    "p2c_learned": "P2C over learned scores",
    "p3c_learned": "P3C over learned scores",
    "awf": "age-aware split (nominal rates)",
    "awf_learned": "age-aware split (learned rates)",
    "p2c_learned_probe": "learned P2C, probing (2 probes)",
    "p3c_learned_probe": "learned P3C, probing (3 probes)",
    "prequal": "Prequal (3 probes)",
    "p2c_probe": "P2C, probing (2 probes)",
    "p3c_probe": "P3C, probing (3 probes)",
    "sed3_probe": "SED over 3 probes (true nominal speeds)",
}
# Classic probing power-of-d shares the colour of its non-probing counterpart.
CLASSIC_PROBING = {  # variant -> (base colour key, linestyle, marker)
    "p2c_probe": ("p2c", ":", "s"),
    "p3c_probe": ("p2c", ":", "^"),
    "sed3_probe": ("sed", ":", "^"),
}
# Age-aware stochastic splits share the colour of the policy that uses the same information.
AGE_AWARE = {"awf": ("sed", "-.", "D"), "awf_learned": ("p2c_learned", "-.", "D")}
POWER_OF_D = {  # variant -> (linestyle, marker)
    "p2c_learned": ("-", "o"),
    "p3c_learned": ("-.", "D"),
    "p2c_learned_probe": (":", "s"),
    "p3c_learned_probe": (":", "^"),
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

HEURISTICS = ["wrandom", "jsq", "sed", "p2c"]
LEARNED = ["greedy", "ts", "ts_epoch", "p2c_learned"]
FIXES = ["jsq", "jsq+lc", "greedy", "greedy+lc", "ts", "ts+lc", "p2c_learned", "p2c_learned+lc"]
HEADLINE = ["jsq", "sed", "p2c", "greedy", "ts", "ts_epoch", "p2c_learned", "prequal"]
PROBING = ["prequal", "p2c_probe", "p3c_probe", "sed3_probe", "p2c_learned_probe", "p3c_learned_probe"]
TABLE_ORDER = (HEURISTICS + LEARNED + ["p3c_learned"] + PROBING
               + ["jsq+lc", "greedy+lc", "ts+lc", "p2c_learned+lc"])


def style(name: str) -> dict:
    lc = name.endswith("+lc")
    base = name.replace("+lc", "")
    label = LABELS[base] + (" + local correction" if lc else "")
    if base in AGE_AWARE:
        key, ls, marker = AGE_AWARE[base]
        return dict(color=COLORS[key], linestyle="--" if lc else ls, marker=marker, label=label, linewidth=1.8,
                    markersize=5)
    if base in CLASSIC_PROBING:
        key, ls, marker = CLASSIC_PROBING[base]
        return dict(color=COLORS[key], linestyle=ls, marker=marker, label=label, linewidth=1.8, markersize=5)
    if base in POWER_OF_D:
        ls, marker = POWER_OF_D[base]
        return dict(color=COLORS["p2c_learned"], linestyle="--" if lc else ls, marker=marker,
                    label=label, linewidth=1.8, markersize=5)
    return dict(color=COLORS[base], linestyle="--" if lc else "-", label=label,
                linewidth=1.8, marker="o", markersize=5)


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def agg(df: pd.DataFrame, by: list, metric: str) -> pd.DataFrame:
    return df.groupby(by)[metric].agg(["median", q25, q75]).reset_index()


def _axes_style(ax, xlabel, ylabel, logy=True):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def staleness_label(v: float) -> str:
    return "live" if v == 0 else f"{v * 1000:g} ms"


def multi_panels(panels, x, metric, title, xlabel, ylabel, fname, xticks=None, xlog=False, logy=True,
                 categorical=False, tick_label=lambda v: f"{v:g}", ncols=None):
    """panels: list of (title, dataframe, policies). categorical=True spaces x values evenly."""
    n = len(panels)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    single = n == 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0 if single else 5.4 * ncols, 4.2 * nrows),
                             sharey=True, squeeze=False)
    axes = axes.ravel()
    for i, (ax, (panel_title, frame, policies)) in enumerate(zip(axes, panels)):
        for pol in policies:
            sub = agg(frame[frame.policy == pol], [x], metric).sort_values(x)
            if sub.empty:
                continue
            st = style(pol)
            xs = [xticks.index(v) for v in sub[x]] if categorical else sub[x]
            ax.plot(xs, sub["median"], **st)
            ax.fill_between(xs, sub["q25"], sub["q75"], color=st["color"], alpha=0.10, linewidth=0)
        ax.set_title(panel_title, color=TEXT, fontsize=11, loc="left")
        _axes_style(ax, xlabel, ylabel if i % ncols == 0 else "", logy=logy)
        if xlog:
            ax.set_xscale("log", base=2)
        if xticks is not None:
            ax.set_xticks(range(len(xticks)) if categorical else xticks)
            ax.set_xticklabels([tick_label(t) for t in xticks])
    for ax in axes[n:]:
        ax.set_visible(False)
    # One legend for the whole figure, outside the plot area.
    handles, labels = {}, []
    for ax in axes[:n]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h
                labels.append(l)
    fig.legend([handles[l] for l in labels], labels, frameon=False, fontsize=8, labelcolor=TEXT,
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    fig.suptitle(title, color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, fname)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def frontier_figure(df: pd.DataFrame) -> str:
    """P99 vs probes per request: Prequal's curve against learned power-of-d probing."""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    titles = {"steady": "Steady state (ρ=0.9)", "gray": "Gray failure (ρ=0.75)"}
    for ax, scen in zip(axes, ["steady", "gray"]):
        sub = df[df.scenario == scen]
        kw = sub.policy_kwargs.fillna("")
        curves = [
            ("prequal", sub[sub.policy == "prequal"], style("prequal") | {"label": "Prequal (async probes)"}),
            ("learned", sub[(sub.policy == "p2c_learned") & kw.str.contains("probe=True")],
             style("p2c_learned_probe") | {"label": "learned power-of-d, probing (d = 2, 3, 4)"}),
            ("jsq_d", sub[(sub.policy == "probe_d") & ~kw.str.contains("score=sed")],
             style("p3c_probe") | {"label": "JSQ(d), probing (d = 2, 3, 4)"}),
            ("sed_d", sub[(sub.policy == "probe_d") & kw.str.contains("score=sed")],
             style("sed3_probe") | {"label": "SED(d) with true nominal speeds, probing"}),
        ]
        for _, frame, st in curves:
            g = frame.groupby("policy_kwargs").agg(x=("probes_per_request", "median"), m=("p99_ms", "median"),
                                                   lo=("p99_ms", q25), hi=("p99_ms", q75)).sort_values("x")
            ax.plot(g.x, g.m, **st)
            ax.fill_between(g.x, g.lo, g.hi, color=st["color"], alpha=0.10, linewidth=0)
        for pol in ["p2c", "p2c_learned", "ts+lc"]:
            frame = sub[(sub.policy == pol) & sub.policy_kwargs.isna()]
            if frame.empty:
                continue
            st = style(pol)
            ax.plot([0], [frame.p99_ms.median()], linestyle="none", marker=st["marker"], markersize=8,
                    color=st["color"], label=st["label"] + " (no probes)")
        ax.set_title(titles[scen], color=TEXT, fontsize=11, loc="left")
        _axes_style(ax, "probes per request", "P99 latency (ms, log)" if scen == "steady" else "")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=TEXT, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    fig.suptitle("Probe cost vs. tail latency (16 routers, 50 ms staleness for snapshot-based policies)",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    path = os.path.join(FIGURES, "fig7_probe_frontier.png")
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


# Real-system strategy name -> simulator policy name.
REAL_TO_SIM = {
    "least_conn": "jsq", "sed": "sed", "p2c": "p2c",
    "greedy_learned": "greedy", "ts_learned": "ts", "lin_ts": "ts_epoch",
    "p2c_learned": "p2c_learned", "p3c_learned": "p3c_learned",
    "least_conn+lc": "jsq+lc", "greedy_learned+lc": "greedy+lc", "ts_learned+lc": "ts+lc",
    "p2c_learned+lc": "p2c_learned+lc",
    "prequal": "prequal", "p2c_probe": "p2c_probe", "p3c_probe": "p3c_probe", "sed3_probe": "sed3_probe",
    "p2c_learned_probe": "p2c_learned_probe", "p3c_learned_probe": "p3c_learned_probe",
}
REAL_TIME_SCALE = 10


def _spearman(a, b) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def real_system_section(out, real: pd.DataFrame, sim_main: pd.DataFrame, sim_probing) -> None:
    """Real-system tables, sim-vs-real scatter, and per-cell rank agreement."""
    real = real.copy()
    _section(out, "NOTE", f"Real-system data are partial ({len(real)} of 2,160 planned runs); not used in the paper.")
    real["policy"] = real.strategy.map(REAL_TO_SIM)
    real["staleness"] = (real.state_refresh_sec / REAL_TIME_SCALE).round(4)
    real["desync"] = ~real.aligned.astype(bool)
    real.loc[real.staleness == 0, "desync"] = False
    real["num_dispatchers"] = real.num_gateways
    real["p99_sim_units"] = real.p99_ms / REAL_TIME_SCALE

    sim = sim_main.copy()
    if sim_probing is not None:
        probe = sim_probing.copy()
        # Probing policies ignore staleness/refresh: replicate them into every cell.
        cells = sim[["scenario", "staleness", "desync", "num_dispatchers"]].drop_duplicates()
        probe = probe.drop(columns=["staleness", "desync"]).merge(cells, on=["scenario", "num_dispatchers"])
        sim = pd.concat([sim, probe], ignore_index=True)
    keys = ["scenario", "staleness", "desync", "num_dispatchers", "policy"]
    real_med = real.groupby(keys).p99_sim_units.median().rename("real")
    sim_med = sim.groupby(keys).p99_ms.median().rename("sim")
    both = pd.concat([real_med, sim_med], axis=1, join="inner").reset_index()
    if both.empty:
        return

    # Rank agreement within each cell (same scenario, staleness, refresh mode, gateways).
    rows = []
    for cell, g in both.groupby(["scenario", "staleness", "desync", "num_dispatchers"]):
        if len(g) >= 4:
            rows.append((*cell, len(g), _spearman(g.real, g.sim)))
    cells = pd.DataFrame(rows, columns=["scenario", "staleness", "desync", "gateways", "policies", "spearman"])
    lines = ["| scenario | staleness (sim s) | unsync refresh | routers | policies | Spearman ρ |",
             "|:---|:---:|:---:|:---:|:---:|:---:|"]
    lines += [f"| {r.scenario} | {r.staleness:g} | {r.desync} | {r.gateways} | {r.policies} | {r.spearman:.2f} |"
              for r in cells.itertuples()]
    summary = (f"Median per-cell Spearman ρ = {cells.spearman.median():.2f} over {len(cells)} cells; "
               f"{(cells.spearman >= 0.7).mean() * 100:.0f}% of cells have ρ ≥ 0.7. "
               f"Pooled over all {len(both)} matched configurations: ρ = {_spearman(both.real, both.sim):.2f}.")
    _section(out, "Real system vs. simulator: rank agreement of policies by P99", summary + "\n\n" + "\n".join(lines))

    # Scatter: real (in simulator time units) vs. simulator, per matched configuration.
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    families = {"heuristic": (HEURISTICS + ["jsq+lc"], "#2a78d6", "o"),
                "learned": (["greedy", "ts", "ts_epoch", "greedy+lc", "ts+lc"], "#eb6834", "^"),
                "learned power-of-d": (["p2c_learned", "p3c_learned", "p2c_learned+lc"], "#008300", "s"),
                "probing": (PROBING, "#e34948", "D")}
    for label, (pols, color, marker) in families.items():
        sub = both[both.policy.isin(pols)]
        ax.scatter(sub.sim, sub.real, s=18, color=color, marker=marker, alpha=0.7, label=label, linewidths=0)
    lo = max(1.0, min(both.sim.min(), both.real.min()) * 0.8)
    hi = max(both.sim.max(), both.real.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--", label="real = simulator")
    _axes_style(ax, "simulator P99 (ms)", "real system P99 ÷ 10 (ms)")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    ax.set_title(f"PARTIAL DATA ({len(real)} of 2,160 runs): real system vs. simulator", color=TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig10_real_vs_sim.png"), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # Real-system P99 tables at 16 routers.
    order = [p for p in REAL_TO_SIM if p in set(real.strategy)]
    for scen in sorted(real.scenario.unique()):
        for phase, name in ((False, "synchronised"), (True, "unsynchronised")):
            sub = real[(real.scenario == scen) & (real.num_gateways == 16)
                       & ((real.desync == phase) | (real.state_refresh_sec == 0))]
            if sub.empty:
                continue
            _section(out, f"Real system, {scen}, {name} refresh, 16 routers: P99 (ms) vs. refresh period (s)",
                     table(sub, "strategy", "state_refresh_sec", "p99_ms", order, sorted(sub.state_refresh_sec.unique())))
    cens = real.groupby("strategy").censored.sum()
    cens = cens[cens > 0]
    if len(cens):
        _section(out, "Real system: censored requests (still waiting after the 120 s drain; tails are lower bounds)",
                 "\n".join(f"- {k}: {int(v)}" for k, v in cens.items()))



# ----------------------------------------------------------------------------
# LLM serving, model ablation, theory validation
# ----------------------------------------------------------------------------

LLM_STYLE = {  # policy -> (colour, linestyle, marker, label)
    "jsq": ("#2a78d6", "-", "o", "JSQ (cache-oblivious)"),
    "p2c": ("#1baf7a", "-", "o", "P2C (cache-oblivious)"),
    "hash": ("#8a8984", "-", "o", "static hash by prefix"),
    "affinity": ("#eb6834", "-", "o", "prefix affinity"),
    "affinity_thr1": ("#eda100", ":", "v", "affinity + load threshold T=1"),
    "affinity_thr2": ("#eda100", ":", "^", "affinity + load threshold T=2"),
    "affinity_thr4": ("#eda100", "-", "o", "affinity + load threshold T=4"),
    "affinity_thr8": ("#eda100", "--", "s", "affinity + load threshold T=8"),
    "affinity_thr16": ("#eda100", "-.", "D", "affinity + load threshold T=16"),
    "affinity_thr32": ("#eda100", (0, (1, 3)), "<", "affinity + load threshold T=32"),
    "greedy_learned": ("#e87ba4", "-", "o", "learned greedy"),
    "ts_learned": ("#4a3aa7", "-", "o", "learned Thompson"),
    "p2c_learned": ("#008300", ":", "s", "learned P2C over any 2 replicas"),
    "ring2_learned": ("#008300", "-", "o", "learned ring-2"),
    "ring3_learned": ("#008300", "--", "D", "learned ring-3"),
    "affinity_p2c_learned": ("#e34948", "-", "o", "learned P2C: one holder + random"),
    "ring2_learned_c": ("#00a0a0", "-", "s", "learned ring-2 + request-cost feature"),
    "ring3_learned_c": ("#00a0a0", "--", "s", "learned ring-3 + request-cost feature"),
    "greedy_learned_c": ("#eb6834", ":", "s", "learned greedy + request-cost feature"),
    "affinity_p2c_learned_c": ("#e34948", ":", "s", "learned P2C holder + request-cost feature"),
}
LLM_HEAD = ["jsq", "hash", "affinity", "affinity_thr4", "greedy_learned", "ring2_learned", "ring3_learned",
            "affinity_p2c_learned"]
LLM_REGIME_NAMES = {(1.1, 75.0): "moderate skew (zipf 1.1, 75 req/s)", (1.4, 90.0): "hot prefix (zipf 1.4, 90 req/s)"}
LLM_THRESHOLDS = [f"affinity_thr{t}" for t in (1, 2, 4, 8, 16, 32)]


def _llm_line(ax, frame, policy, x, xs, metric="p99_ms"):
    color, ls, marker, label = LLM_STYLE[policy]
    sub = agg(frame[frame.policy == policy], [x], metric).sort_values(x)
    if sub.empty:
        return
    pos = [xs.index(v) for v in sub[x]]
    ax.plot(pos, sub["median"], color=color, linestyle=ls, marker=marker, label=label, linewidth=1.8, markersize=5)
    ax.fill_between(pos, sub["q25"], sub["q75"], color=color, alpha=0.10, linewidth=0)


def llm_staleness_figure(df: pd.DataFrame, k: int = 16) -> None:
    d = df[(df.num_dispatchers == k) & (~df.desync)]
    st = sorted(d.staleness.unique())
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), sharey=True)
    for r, ((zipf, rate), name) in enumerate(LLM_REGIME_NAMES.items()):
        for c, (scen, scen_name) in enumerate((("llm", "healthy replicas"), ("llm_gray", "one replica slows to 25% at t=20 s"))):
            ax = axes[r, c]
            frame = d[(d.zipf == zipf) & (d.rate == rate) & (d.scenario == scen)]
            for pol in LLM_HEAD:
                _llm_line(ax, frame, pol, "staleness", st)
            ax.set_title(f"{name}; {scen_name}", color=TEXT, fontsize=10, loc="left")
            _axes_style(ax, "shared-state refresh period" if r == 1 else "", "P99 TTFT (ms, log)" if c == 0 else "")
            ax.set_xticks(range(len(st)))
            ax.set_xticklabels([staleness_label(v) for v in st])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, labelcolor=TEXT, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    fig.suptitle(f"LLM serving: prefix-cache routers, {k} routers, 8 prefill replicas", color=TEXT, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig11_llm_staleness.png"), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def regret_tables(df: pd.DataFrame):
    """
    Regret of every policy against the best fixed-threshold router in each of the
    workload regimes. Held-out: the threshold T is chosen on seeds 0-4 and every
    policy is evaluated on seeds 5-9. In-sample: T is chosen on the evaluation
    seeds themselves (favours the thresholds).
    """
    keys = ["zipf", "rate", "cache_size", "staleness", "scenario"]
    sel = df[df.seed < 5].groupby(keys + ["policy"]).p99_ms.median().unstack()
    ev = df[df.seed >= 5].groupby(keys + ["policy"]).p99_ms.median().unstack()
    best_t = sel[LLM_THRESHOLDS].idxmin(axis=1)
    ref_held = pd.Series([ev.loc[k, best_t.loc[k]] for k in ev.index], index=ev.index)
    held = ev.div(ref_held, axis=0)
    insample = ev.div(ev[LLM_THRESHOLDS].min(axis=1), axis=0)
    return held, insample, best_t


def llm_regret_figure(held: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    n = len(held)
    for pol in LLM_THRESHOLDS:       # fixed thresholds: thin, one hue (they are one family)
        r = np.sort(held[pol].to_numpy())
        ax.plot(r, np.arange(1, n + 1) / n, color="#eda100", linewidth=1.0, alpha=0.8,
                label="fixed threshold T (1, 2, 4, 8, 16, 32)" if pol == LLM_THRESHOLDS[0] else None)
    for pol in ["affinity", "greedy_learned", "ring2_learned", "ring3_learned", "affinity_p2c_learned"]:
        color, ls, marker, label = LLM_STYLE[pol]
        r = np.sort(held[pol].to_numpy())
        ax.plot(r, np.arange(1, n + 1) / n, color=color, linestyle=ls, linewidth=2.0, label=label)
    ax.set_xscale("log")
    ax.axvline(1.0, color=MUTED, linewidth=0.8)
    _axes_style(ax, "P99 relative to the best tuned threshold in that regime (log)", "fraction of regimes ≤ x", logy=False)
    ax.set_xlim(0.6, 200)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    ax.set_title(f"Tuning-free robustness across {n} workload regimes (threshold tuned on held-out seeds)",
                 color=TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig12_llm_regret.png"), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def llm_section(out, llm: Optional[pd.DataFrame], regimes: Optional[pd.DataFrame]) -> None:
    if llm is not None:
        llm_staleness_figure(llm)
        order = [p for p in LLM_STYLE if p in set(llm.policy)] + [p for p in llm.policy.unique() if p not in LLM_STYLE]
        for (zipf, rate), name in LLM_REGIME_NAMES.items():
            for scen, sname in (("llm", "healthy"), ("llm_gray", "gray failure")):
                for phase, pname in ((False, "synchronised"), (True, "unsynchronised")):
                    sub = llm[(llm.zipf == zipf) & (llm.rate == rate) & (llm.scenario == scen)
                              & (llm.num_dispatchers == 16) & ((llm.desync == phase) | (llm.staleness == 0))]
                    _section(out, f"LLM serving, {name}, {sname}, {pname} refresh, 16 routers: P99 TTFT (ms)",
                             table(sub, "policy", "staleness", "p99_ms", order, sorted(sub.staleness.unique())))
                    if not phase and scen == "llm":
                        _section(out, f"LLM serving, {name}, healthy, 16 routers: cache hit rate (median)",
                                 table(sub.assign(hit_pct=sub.hit_rate * 100), "policy", "staleness", "hit_pct",
                                       order, sorted(sub.staleness.unique())))
            sub = llm[(llm.zipf == zipf) & (llm.rate == rate) & (llm.scenario == "llm") & (llm.staleness == 0.05)
                      & (~llm.desync)]
            _section(out, f"LLM serving, {name}, healthy, 50 ms staleness: P99 TTFT (ms) vs. routers",
                     table(sub, "policy", "num_dispatchers", "p99_ms", order, sorted(sub.num_dispatchers.unique())))
    if regimes is not None:
        held, insample, best_t = regret_tables(regimes)
        llm_regret_figure(held)
        pols = LLM_THRESHOLDS + ["affinity", "jsq", "greedy_learned", "ring2_learned", "ring3_learned",
                                 "affinity_p2c_learned", "p2c_learned", "ring2_learned+lc", "greedy_learned+lc"]
        rows = ["| policy | median | 90th pct | worst | regimes within 1.5× |", "|:---|:---:|:---:|:---:|:---:|"]
        for pol in pols:
            if pol not in held:
                continue
            r = held[pol]
            rows.append(f"| {pol} | {r.median():.2f}× | {r.quantile(0.9):.2f}× | {r.max():.1f}× | "
                        f"{(r <= 1.5).mean() * 100:.0f}% |")
        _section(out, f"LLM serving: P99 relative to the best tuned threshold, {len(held)} regimes "
                      f"(threshold chosen on seeds 0-4, policies evaluated on seeds 5-9)", "\n".join(rows))
        rows = ["| policy | median | 90th pct | worst |", "|:---|:---:|:---:|:---:|"]
        for pol in pols:
            if pol in insample:
                r = insample[pol]
                rows.append(f"| {pol} | {r.median():.2f}× | {r.quantile(0.9):.2f}× | {r.max():.1f}× |")
        _section(out, "Same, with the threshold chosen in hindsight on the evaluation seeds (favours thresholds)",
                 "\n".join(rows))
        counts = best_t.str.replace("affinity_thr", "T=").value_counts()
        _section(out, "Which fixed threshold was best (held-out selection): no single value is safe",
                 "\n".join(f"- {k}: {v} regimes" for k, v in counts.items()))


TRACE_HEAD = ["jsq", "affinity", "affinity_thr4", "greedy_learned", "ring2_learned", "ring2_learned_c"]
TRACE_KEYS = ["trace", "speedup", "staleness", "scenario"]


def trace_regret(df: pd.DataFrame, metric: str = "p99_slowdown"):
    """Same held-out threshold selection as regret_tables, over the trace cells."""
    sel = df[df.seed < 5].groupby(TRACE_KEYS + ["policy"])[metric].median().unstack()
    ev = df[df.seed >= 5].groupby(TRACE_KEYS + ["policy"])[metric].median().unstack()
    best_t = sel[LLM_THRESHOLDS].idxmin(axis=1)
    ref = pd.Series([ev.loc[k, best_t.loc[k]] for k in ev.index], index=ev.index)
    return ev.div(ref, axis=0), best_t


def trace_figure(df: pd.DataFrame, metric: str = "p99_slowdown") -> None:
    traces = [t for t in ("toolagent", "conversation") if t in set(df.trace)]
    speeds = sorted(df.speedup.unique())
    fig, axes = plt.subplots(len(traces), 2, figsize=(10.5, 3.9 * len(traces)), sharey="row", squeeze=False)
    for r, trace in enumerate(traces):
        for c, (scen, name) in enumerate((("trace", "healthy replicas"), ("trace_gray", "one replica slows to 25%"))):
            ax = axes[r, c]
            frame = df[(df.trace == trace) & (df.scenario == scen) & (df.staleness == 0.2)]
            for pol in TRACE_HEAD:
                _llm_line(ax, frame, pol, "speedup", speeds, metric)
            ax.set_title(f"{trace} trace; {name}", color=TEXT, fontsize=10, loc="left")
            _axes_style(ax, "load (trace time compression ×)" if r == len(traces) - 1 else "",
                        "P99 slowdown (TTFT / unloaded prefill, log)" if c == 0 else "")
            ax.set_xticks(range(len(speeds)))
            ax.set_xticklabels([f"{v:g}" for v in speeds])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, labelcolor=TEXT, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.suptitle("Replay of real Mooncake traces (16 routers, 200 ms staleness, 8 replicas)", color=TEXT, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig16_trace.png"), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def trace_section(out, df: Optional[pd.DataFrame]) -> None:
    if df is None:
        return
    trace_figure(df)
    order = [p for p in LLM_STYLE if p in set(df.policy)] + [p for p in df.policy.unique() if p not in LLM_STYLE]
    for trace in sorted(df.trace.unique()):
        for scen, sname in (("trace", "healthy"), ("trace_gray", "gray failure")):
            for st in sorted(df.staleness.unique()):
                sub = df[(df.trace == trace) & (df.scenario == scen) & (df.staleness == st)]
                for metric, mname in (("p99_slowdown", "P99 slowdown"), ("p99_wait_ms", "P99 queueing wait (ms)")):
                    _section(out, f"Trace {trace}, {sname}, {st * 1000:g} ms staleness: {mname} vs. load (×)",
                             table(sub, "policy", "speedup", metric, order, sorted(sub.speedup.unique())))
        sub = df[(df.trace == trace) & (df.scenario == "trace") & (df.staleness == 0.2)]
        _section(out, f"Trace {trace}, healthy, 200 ms: cache hit rate (median)",
                 table(sub.assign(hit_pct=sub.hit_rate * 100), "policy", "speedup", "hit_pct", order,
                       sorted(sub.speedup.unique())))
    held, best_t = trace_regret(df)
    pols = LLM_THRESHOLDS + ["affinity", "jsq", "hash", "greedy_learned", "ring2_learned", "ring3_learned", "ring2_learned_c", "ring3_learned_c", "greedy_learned_c", "affinity_p2c_learned_c",
                             "affinity_p2c_learned", "p2c_learned", "ring2_learned+lc", "greedy_learned+lc"]
    rows = ["| policy | median | 90th pct | worst | cells within 1.5× |", "|:---|:---:|:---:|:---:|:---:|"]
    for pol in pols:
        if pol in held:
            r = held[pol]
            rows.append(f"| {pol} | {r.median():.2f}× | {r.quantile(0.9):.2f}× | {r.max():.1f}× | "
                        f"{(r <= 1.5).mean() * 100:.0f}% |")
    _section(out, f"Trace replay: P99 slowdown relative to the best tuned threshold, {len(held)} cells "
                  "(threshold chosen on seeds 0-4, evaluated on seeds 5-9)", "\n".join(rows))
    counts = best_t.str.replace("affinity_thr", "T=").value_counts()
    _section(out, "Trace replay: which fixed threshold was best", "\n".join(f"- {k}: {v} cells" for k, v in counts.items()))


MODEL_FAMILIES = [("greedy", ["greedy", "greedy_q2", "greedy_ewma", "greedy_rbf", "greedy_mlp"], "learned greedy"),
                  ("linucb", ["linucb", None, "linucb_ewma", "linucb_rbf", None], "LinUCB (optimistic)"),
                  ("p2c", ["p2c_learned", "p2c_learned_q2", "p2c_learned_ewma", "p2c_learned_rbf", "p2c_learned_mlp"], "learned P2C"),
                  ("p3c", ["p3c_learned", "p3c_learned_q2", "p3c_learned_ewma", "p3c_learned_rbf", "p3c_learned_mlp"], "learned P3C")]
MODEL_COLORS = {"greedy": "#eb6834", "linucb": "#e87ba4", "p2c": "#008300", "p3c": "#4a3aa7"}
MODEL_FEATURES = ["q", "q²", "q + own latency", "q, non-linear (RBF)", "small neural net (MLP)"]


def models_section(out, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.6), sharey=True)
    cells = [("steady", 0.05), ("steady", 0.2), ("gray", 0.05), ("gray", 0.2)]
    for ax, (scen, st) in zip(axes, cells):
        frame = df[(df.scenario == scen) & (df.staleness == st) & (~df.desync)]
        for fam, pols, label in MODEL_FAMILIES:
            xs, med, lo, hi = [], [], [], []
            for i, pol in enumerate(pols):
                if pol is None:
                    continue
                v = frame[frame.policy == pol].p99_ms
                if v.empty:
                    continue
                xs.append(i); med.append(v.median()); lo.append(v.quantile(0.25)); hi.append(v.quantile(0.75))
            ax.plot(xs, med, color=MODEL_COLORS[fam], marker="o", linewidth=1.8, markersize=5, label=label)
            ax.fill_between(xs, lo, hi, color=MODEL_COLORS[fam], alpha=0.10, linewidth=0)
        ax.set_xticks(range(5))
        ax.set_xticklabels(MODEL_FEATURES, fontsize=8, rotation=20, ha="right")
        ax.set_title(f"{scen}, {staleness_label(st)} staleness", color=TEXT, fontsize=10, loc="left")
        _axes_style(ax, "latency-model features", "P99 latency (ms, log)" if ax is axes[0] else "")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, labelcolor=TEXT, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    fig.suptitle("The conclusions do not depend on the latency model (16 routers, synchronised refresh)",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig14_models.png"), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    order = [p for _, pols, _ in MODEL_FAMILIES for p in pols if p] + ["ts"]
    for scen in ("steady", "gray"):
        for phase, name in ((False, "synchronised"), (True, "unsynchronised")):
            sub = df[(df.scenario == scen) & (df.desync == phase)]
            _section(out, f"Model ablation, {scen}, {name} refresh, 16 routers: P99 (ms)",
                     table(sub, "policy", "staleness", "p99_ms", [p for p in order if p in set(sub.policy)],
                           sorted(sub.staleness.unique())))


# Largest probability that any single server is chosen, for policies where it is known.
def predicted_caps(n_servers: int, mus) -> Dict[str, float]:
    return {"wrandom": max(mus) / sum(mus), "jsq": 1.0, "sed": 1.0,
            "p2c": 2.0 / n_servers, "p2c_learned": 2.0 / n_servers, "p3c_learned": 3.0 / n_servers}


def candidate_mass_llm(zipf: float, d: int, n: int = 8, num_prefixes: int = 64) -> float:
    """kappa_max: the largest total popularity of prefixes whose d-replica ring includes one replica."""
    from research.llm_sim import LLMScenario, prefix_probabilities, rendezvous_order
    probs = prefix_probabilities(LLMScenario(zipf=zipf, num_prefixes=num_prefixes))
    mass = np.zeros(n)
    for c in range(num_prefixes):
        for s in rendezvous_order(c, n)[:d]:
            mass[s] += probs[c]
    return float(mass.max())


def theory_section(out, main: pd.DataFrame, llm: Optional[pd.DataFrame]) -> None:
    from research.sim import DEFAULT_MUS
    if "window_share_mean" not in main:
        return
    caps = predicted_caps(len(DEFAULT_MUS), DEFAULT_MUS)
    sub = main[(main.scenario == "steady") & (main.num_dispatchers == 16) & (main.staleness == 0.2) & (~main.desync)]
    sub = sub.dropna(subset=["window_share_mean"])
    pols = [p for p in ["wrandom", "jsq", "sed", "p2c", "p2c_learned", "p3c_learned", "greedy", "ts", "ts_epoch",
                        "awf", "awf_learned"] if p in set(sub.policy)]
    rows = ["| policy | predicted cap | measured mean share | measured 95th pct |", "|:---|:---:|:---:|:---:|"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for i, pol in enumerate(pols):
        v = sub[sub.policy == pol]
        mean, p95 = v.window_share_mean.median(), v.window_share_p95.median()
        cap = caps.get(pol)
        rows.append(f"| {pol} | {'–' if cap is None else f'{cap:.3f}'} | {mean:.2f} | {p95:.2f} |")
        st = style(pol)
        ax.barh(i, mean, color=st["color"], height=0.6)
        if cap is not None:
            ax.plot([cap, cap], [i - 0.4, i + 0.4], color=TEXT, linewidth=2.0)
    ax.set_yticks(range(len(pols)))
    ax.set_yticklabels([LABELS.get(p, p) for p in pols], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(1.0 / len(DEFAULT_MUS), color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlim(0, 1.05)
    _axes_style(ax, "busiest server's share of a 200 ms window (bar: measured; tick: cap, Prop. 1 gives 1.0, Prop. 2 gives d/N)",
                "", logy=False)
    ax.set_title("Burst concentration: measured vs. predicted caps (16 routers, 200 ms stale state)",
                 color=TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig15_burst_share.png"), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    _section(out, "Burst concentration (queue model): busiest server's share of a 200 ms window, 16 routers, "
                  "synchronised stale state", "\n".join(rows))
    if llm is not None and "window_share_mean" in llm:
        rows = ["| policy | κ_max (cap) | measured mean share |", "|:---|:---:|:---:|"]
        sub = llm[(llm.zipf == 1.1) & (llm.rate == 75.0) & (llm.scenario == "llm") & (llm.num_dispatchers == 16)
                  & (llm.staleness == 0.2) & (~llm.desync)]
        n = 8
        for pol, kappa in [("hash", candidate_mass_llm(1.1, 1)), ("ring2_learned", candidate_mass_llm(1.1, 2)),
                           ("ring3_learned", candidate_mass_llm(1.1, 3)), ("p2c_learned", 2.0 / n),
                           ("p3c_learned", 3.0 / n), ("jsq", 1.0), ("affinity", None), ("greedy_learned", None)]:
            v = sub[sub.policy == pol]
            if v.empty:
                continue
            rows.append(f"| {pol} | {'–' if kappa is None else f'{kappa:.2f}'} | {v.window_share_mean.median():.2f} |")
        _section(out, "Burst concentration (LLM scenario, moderate skew): κ_max = largest popularity mass of "
                      "prefixes whose candidate set includes one replica, 16 routers, 200 ms windows", "\n".join(rows))


def awf_figure(df: pd.DataFrame) -> None:
    d = df[df.num_dispatchers == 16]
    panels = []
    for scen, sname in (("steady", "steady"), ("gray", "gray failure")):
        for phase, pname in ((False, "synchronised"), (True, "unsynchronised")):
            frame = d[(d.scenario == scen) & ((d.desync == phase) | (d.staleness == 0))]
            panels.append((f"{sname}, {pname} refresh", frame,
                           ["sed", "jsq", "p2c_learned", "p3c_learned", "awf", "awf_learned"]))
    multi_panels(panels, "staleness", "p99_ms",
                 "Age-aware stochastic split vs. power-of-d over learned scores (16 routers)",
                 "shared-state refresh period", "P99 latency (ms, log)", "fig13_age_aware.png",
                 xticks=sorted(d.staleness.unique()), categorical=True, tick_label=staleness_label, ncols=2)


def fmt(med, lo, hi):
    return f"{med:,.0f} [{lo:,.0f}–{hi:,.0f}]"


def table(df, rows_by, cols_by, metric, row_order, col_order, digits=0):
    g = agg(df, [rows_by, cols_by], metric)
    header = f"| {rows_by} | " + " | ".join(f"{cols_by}={c:g}" for c in col_order) + " |"
    lines = [header, "|:---|" + ":---:|" * len(col_order)]
    for r in row_order:
        cells = []
        for c in col_order:
            hit = g[(g[rows_by] == r) & (g[cols_by] == c)]
            if hit.empty:
                cells.append("–")
            else:
                h = hit.iloc[0]
                cells.append(fmt(h["median"], h["q25"], h["q75"]) if digits == 0
                             else f"{h['median']:.1f} [{h['q25']:.1f}–{h['q75']:.1f}]")
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _section(out, title, body):
    out.append(f"## {title}\n")
    out.append(body)
    out.append("")


def main():
    out = []
    k_ref, st_ref = 16, 0.05

    main_csv = os.path.join(RESULTS, "main.csv")
    if os.path.exists(main_csv):
        df = pd.read_csv(main_csv)
        # Live state (staleness 0) has no refresh phase; show it in both panels.
        live = df[df.staleness == 0.0]
        df = pd.concat([df, live.assign(desync=True)], ignore_index=True)
        steady, gray = df[df.scenario == "steady"], df[df.scenario == "gray"]
        st_ticks = sorted(df.staleness.unique())
        st_axis = dict(xticks=st_ticks, categorical=True, tick_label=staleness_label)

        def by_phase(frame, policies):
            return [("Synchronised refresh", frame[~frame.desync], policies),
                    ("Unsynchronised refresh", frame[frame.desync], policies)]

        multi_panels(by_phase(steady[steady.num_dispatchers == k_ref], HEADLINE), "staleness", "p99_ms",
                     f"P99 latency vs. state staleness ({k_ref} routers, ρ=0.9)",
                     "shared-state refresh period", "P99 latency (ms, log)", "fig1_p99_vs_staleness.png", **st_axis)
        multi_panels(by_phase(steady[steady.staleness == st_ref],
                              ["jsq", "p2c", "greedy", "ts", "ts_epoch", "p2c_learned", "prequal"]),
                     "num_dispatchers", "p99_ms",
                     f"P99 latency vs. number of routers ({st_ref * 1000:.0f} ms staleness, ρ=0.9)",
                     "independent routers K", "P99 latency (ms, log)", "fig2_p99_vs_dispatchers.png",
                     xticks=sorted(df.num_dispatchers.unique()), xlog=True)
        multi_panels(by_phase(steady[steady.num_dispatchers == k_ref],
                              ["jsq", "sed", "p2c", "greedy", "ts", "p2c_learned", "prequal"]),
                     "staleness", "fano",
                     f"Herding index vs. staleness ({k_ref} routers); 1 = independent arrivals",
                     "shared-state refresh period", "Fano factor of per-server arrivals (log)",
                     "fig3_herding_vs_staleness.png", **st_axis)
        multi_panels([("Synchronised refresh", steady[(steady.num_dispatchers == k_ref) & ~steady.desync],
                       FIXES + ["prequal"])],
                     "staleness", "p99_ms",
                     f"Mitigations: local correction and power-of-two over learned scores ({k_ref} routers)",
                     "shared-state refresh period", "P99 latency (ms, log)", "fig4_mitigations.png", **st_axis)
        multi_panels(by_phase(gray[gray.num_dispatchers == k_ref],
                              ["jsq", "sed", "p2c", "greedy", "ts", "p2c_learned", "p2c_learned+lc",
                               "prequal", "p3c_learned_probe"]),
                     "staleness", "p99_ms",
                     f"Gray failure: fastest server drops to 20% speed at t=12 s ({k_ref} routers, ρ=0.75)",
                     "shared-state refresh period", "P99 latency (ms, log)", "fig5_gray_failure.png", **st_axis)

        for scen, sub in (("Steady state", steady), ("Gray failure", gray)):
            for phase, name in ((False, "synchronised"), (True, "unsynchronised")):
                ph = sub[sub.desync == phase]
                _section(out, f"{scen}, {name} refresh: P99 (ms), K={k_ref}, median [IQR] over seeds",
                         table(ph[ph.num_dispatchers == k_ref], "policy", "staleness", "p99_ms", TABLE_ORDER, st_ticks))
                _section(out, f"{scen}, {name} refresh: P99 (ms) vs. routers, staleness={st_ref}s",
                         table(ph[ph.staleness == st_ref], "policy", "num_dispatchers", "p99_ms", TABLE_ORDER,
                               sorted(ph.num_dispatchers.unique())))
        for phase, name in ((False, "synchronised"), (True, "unsynchronised")):
            ph = steady[(steady.desync == phase) & (steady.num_dispatchers == k_ref)]
            _section(out, f"Steady state, {name} refresh: herding index (Fano factor), K={k_ref}",
                     table(ph, "policy", "staleness", "fano", TABLE_ORDER, st_ticks, digits=1))

    if os.path.exists(main_csv):
        mdf = pd.read_csv(main_csv)
        mdf = pd.concat([mdf, mdf[mdf.staleness == 0.0].assign(desync=True)], ignore_index=True)
        if "awf" in set(mdf.policy):
            awf_figure(mdf)
            for scen in ("steady", "gray"):
                for phase, name in ((False, "synchronised"), (True, "unsynchronised")):
                    sub = mdf[(mdf.scenario == scen) & (mdf.desync == phase) & (mdf.num_dispatchers == 16)]
                    _section(out, f"Age-aware split, {scen}, {name} refresh, 16 routers: P99 (ms)",
                             table(sub, "policy", "staleness", "p99_ms",
                                   ["sed", "jsq", "p2c_learned", "p3c_learned", "awf", "awf_learned"],
                                   sorted(sub.staleness.unique())))
        llm_df = pd.read_csv(os.path.join(RESULTS, "llm.csv")) if os.path.exists(os.path.join(RESULTS, "llm.csv")) else None
        theory_section(out, pd.read_csv(main_csv), llm_df)
    llm_path, regimes_path = os.path.join(RESULTS, "llm.csv"), os.path.join(RESULTS, "llm_regimes.csv")
    if os.path.exists(llm_path) or os.path.exists(regimes_path):
        llm_section(out, pd.read_csv(llm_path) if os.path.exists(llm_path) else None,
                    pd.read_csv(regimes_path) if os.path.exists(regimes_path) else None)
    trace_path = os.path.join(RESULTS, "trace.csv")
    if os.path.exists(trace_path):
        cost_path = os.path.join(RESULTS, "trace_cost.csv")
        trace_df = pd.read_csv(trace_path)
        if os.path.exists(cost_path):
            trace_df = pd.concat([trace_df, pd.read_csv(cost_path)], ignore_index=True)
        trace_section(out, trace_df)
    models_path = os.path.join(RESULTS, "models.csv")
    if os.path.exists(models_path):
        models_section(out, pd.read_csv(models_path))

    frontier_csv = os.path.join(RESULTS, "frontier.csv")
    if os.path.exists(frontier_csv):
        df = pd.read_csv(frontier_csv)
        frontier_figure(df)
        df["setting"] = df.policy + " " + df.policy_kwargs.fillna("")
        for scen in ("steady", "gray"):
            sub = df[df.scenario == scen]
            g = sub.groupby("setting").agg(probes=("probes_per_request", "median"),
                                           p99=("p99_ms", "median"),
                                           q25=("p99_ms", q25), q75=("p99_ms", q75)).reset_index()
            g = g.sort_values(["probes", "p99"])
            rows = ["| setting | probes / request | P99 (ms) [IQR] |", "|:---|:---:|:---:|"]
            rows += [f"| {r.setting.strip()} | {r.probes:.2f} | {fmt(r.p99, r.q25, r.q75)} |" for r in g.itertuples()]
            _section(out, f"Probe-cost frontier ({scen}, K={k_ref}, staleness {st_ref}s, synchronised)", "\n".join(rows))

    probing_csv = os.path.join(RESULTS, "probing.csv")
    if os.path.exists(probing_csv):
        df = pd.read_csv(probing_csv)
        pols = [p for p in TABLE_ORDER if p in set(df.policy)]
        multi_panels([("Steady state (ρ=0.9)", df[df.scenario == "steady"], pols),
                      ("Gray failure (ρ=0.75)", df[df.scenario == "gray"], pols)],
                     "num_dispatchers", "p99_ms", "Probing policies vs. number of routers",
                     "independent routers K", "P99 latency (ms, log)", "fig9_probing_vs_gateways.png",
                     xticks=sorted(df.num_dispatchers.unique()), xlog=True)
        for scen in ("steady", "gray"):
            sub = df[df.scenario == scen]
            _section(out, f"Probing policies ({scen}): P99 (ms) vs. routers",
                     table(sub, "policy", "num_dispatchers", "p99_ms", pols, sorted(sub.num_dispatchers.unique())))

    robust_csv = os.path.join(RESULTS, "robust.csv")
    if os.path.exists(robust_csv):
        df = pd.read_csv(robust_csv)
        pols = [p for p in TABLE_ORDER if p in set(df.policy)]
        panels = []
        for burst in sorted(df.burstiness.unique()):
            for phase, name in ((False, "sync"), (True, "unsync")):
                panels.append((f"{'bursty (ρ=0.75, ±25%)' if burst else 'Poisson (ρ=0.9)'}, {name} refresh",
                               df[(df.burstiness == burst) & (df.desync == phase)],
                               ["jsq", "p2c", "ts+lc", "p2c_learned", "prequal", "sed3_probe", "p3c_learned_probe"]))
        multi_panels(panels, "service_cv", "p99_ms",
                     f"Robustness: service-time variability and bursty arrivals ({k_ref} routers, 50 ms staleness)",
                     "service-time coefficient of variation", "P99 latency (ms, log)", "fig8_robustness.png",
                     xticks=sorted(df.service_cv.unique()), ncols=2)
        for burst in sorted(df.burstiness.unique()):
            for phase, name in ((False, "synchronised"), (True, "unsynchronised")):
                sub = df[(df.burstiness == burst) & (df.desync == phase)]
                _section(out, f"Robustness: burstiness={burst:g}, {name} refresh: P99 (ms) vs. service CV",
                         table(sub, "policy", "service_cv", "p99_ms", pols, sorted(sub.service_cv.unique())))

    epoch_csv = os.path.join(RESULTS, "epoch.csv")
    if os.path.exists(epoch_csv):
        df = pd.read_csv(epoch_csv)
        df["control_interval"] = df.policy_kwargs.str.extract(r"control_interval=([\d.]+)").astype(float)
        _section(out, "ts_epoch: P99 (ms) vs. control interval, K=16",
                 table(df, "staleness", "control_interval", "p99_ms",
                       sorted(df.staleness.unique()), sorted(df.control_interval.unique())))

    load_csv = os.path.join(RESULTS, "load.csv")
    if os.path.exists(load_csv):
        df = pd.read_csv(load_csv)
        pols = [p for p in TABLE_ORDER if p in set(df.policy)]
        multi_panels([("All policies", df, pols)], "rho", "p99_ms",
                     "P99 latency vs. utilisation (K=16, 50 ms staleness)",
                     "utilisation ρ", "P99 latency (ms, log)", "fig6_load.png", xticks=sorted(df.rho.unique()))
        _section(out, "P99 (ms) vs. utilisation, K=16, staleness=0.05s",
                 table(df, "policy", "rho", "p99_ms", pols, sorted(df.rho.unique())))

    tuning_csv = os.path.join(RESULTS, "tuning.csv")
    if os.path.exists(tuning_csv):
        df = pd.read_csv(tuning_csv)
        df["setting"] = df.policy + " " + df.policy_kwargs
        _section(out, "Hyperparameter sensitivity: P99 (ms), K=16",
                 table(df, "setting", "staleness", "p99_ms", sorted(df.setting.unique()), sorted(df.staleness.unique())))

    real_csv = os.path.join(RESULTS, "realsys.csv")
    if os.path.exists(real_csv) and os.path.exists(main_csv):
        real_system_section(out, pd.read_csv(real_csv), pd.read_csv(main_csv), pd.read_csv(probing_csv)
                            if os.path.exists(probing_csv) else None)

    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write("# Results summary (generated by `python -m research.analyze`)\n\n" + "\n".join(out) + "\n")
    print("wrote", os.path.join(RESULTS, "summary.md"))


if __name__ == "__main__":
    main()
