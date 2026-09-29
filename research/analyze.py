"""
Turns research/results/*.csv into figures (research/figures/) and a summary
markdown table (research/results/summary.md).

    python -m research.analyze

Latency tails are heavy (herding episodes produce multi-second outliers), so
every statistic is the median over seeds with the interquartile range, not the mean.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
               loc="center left", bbox_to_anchor=(1.0, 0.5))
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
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=TEXT, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Probe cost vs. tail latency (16 gateways, 50 ms staleness for snapshot-based policies)",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    path = os.path.join(FIGURES, "fig7_probe_frontier.png")
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


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
                     f"P99 latency vs. state staleness ({k_ref} gateways, ρ=0.9)",
                     "shared-state refresh period", "P99 latency (ms, log)", "fig1_p99_vs_staleness.png", **st_axis)
        multi_panels(by_phase(steady[steady.staleness == st_ref],
                              ["jsq", "p2c", "greedy", "ts", "ts_epoch", "p2c_learned", "prequal"]),
                     "num_dispatchers", "p99_ms",
                     f"P99 latency vs. number of gateways ({st_ref * 1000:.0f} ms staleness, ρ=0.9)",
                     "independent gateways K", "P99 latency (ms, log)", "fig2_p99_vs_dispatchers.png",
                     xticks=sorted(df.num_dispatchers.unique()), xlog=True)
        multi_panels(by_phase(steady[steady.num_dispatchers == k_ref],
                              ["jsq", "sed", "p2c", "greedy", "ts", "p2c_learned", "prequal"]),
                     "staleness", "fano",
                     f"Herding index vs. staleness ({k_ref} gateways); 1 = independent arrivals",
                     "shared-state refresh period", "Fano factor of per-server arrivals (log)",
                     "fig3_herding_vs_staleness.png", **st_axis)
        multi_panels([("Synchronised refresh", steady[(steady.num_dispatchers == k_ref) & ~steady.desync],
                       FIXES + ["prequal"])],
                     "staleness", "p99_ms",
                     f"Mitigations: local correction and power-of-two over learned scores ({k_ref} gateways)",
                     "shared-state refresh period", "P99 latency (ms, log)", "fig4_mitigations.png", **st_axis)
        multi_panels(by_phase(gray[gray.num_dispatchers == k_ref],
                              ["jsq", "sed", "p2c", "greedy", "ts", "p2c_learned", "p2c_learned+lc",
                               "prequal", "p3c_learned_probe"]),
                     "staleness", "p99_ms",
                     f"Gray failure: fastest server drops to 20% speed at t=12 s ({k_ref} gateways, ρ=0.75)",
                     "shared-state refresh period", "P99 latency (ms, log)", "fig5_gray_failure.png", **st_axis)

        for scen, sub in (("Steady state", steady), ("Gray failure", gray)):
            for phase, name in ((False, "synchronised"), (True, "unsynchronised")):
                ph = sub[sub.desync == phase]
                _section(out, f"{scen}, {name} refresh: P99 (ms), K={k_ref}, median [IQR] over seeds",
                         table(ph[ph.num_dispatchers == k_ref], "policy", "staleness", "p99_ms", TABLE_ORDER, st_ticks))
                _section(out, f"{scen}, {name} refresh: P99 (ms) vs. gateways, staleness={st_ref}s",
                         table(ph[ph.staleness == st_ref], "policy", "num_dispatchers", "p99_ms", TABLE_ORDER,
                               sorted(ph.num_dispatchers.unique())))
        for phase, name in ((False, "synchronised"), (True, "unsynchronised")):
            ph = steady[(steady.desync == phase) & (steady.num_dispatchers == k_ref)]
            _section(out, f"Steady state, {name} refresh: herding index (Fano factor), K={k_ref}",
                     table(ph, "policy", "staleness", "fano", TABLE_ORDER, st_ticks, digits=1))

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
                     "num_dispatchers", "p99_ms", "Probing policies vs. number of gateways",
                     "independent gateways K", "P99 latency (ms, log)", "fig9_probing_vs_gateways.png",
                     xticks=sorted(df.num_dispatchers.unique()), xlog=True)
        for scen in ("steady", "gray"):
            sub = df[df.scenario == scen]
            _section(out, f"Probing policies ({scen}): P99 (ms) vs. gateways",
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
                     f"Robustness: service-time variability and bursty arrivals ({k_ref} gateways, 50 ms staleness)",
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
    if os.path.exists(real_csv):
        df = pd.read_csv(real_csv)
        df["policy"] = df.strategy
        df["setting"] = "K=" + df.num_gateways.astype(str)
        for k in sorted(df.num_gateways.unique()):
            sub = df[df.num_gateways == k]
            _section(out, f"Real system, {k} gateway(s): P99 (ms) vs. Redis refresh period (s)",
                     table(sub, "strategy", "state_refresh_sec", "p99_ms", list(dict.fromkeys(sub.strategy)),
                           sorted(sub.state_refresh_sec.unique())))

    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write("# Results summary (generated by `python -m research.analyze`)\n\n" + "\n".join(out) + "\n")
    print("wrote", os.path.join(RESULTS, "summary.md"))


if __name__ == "__main__":
    main()
