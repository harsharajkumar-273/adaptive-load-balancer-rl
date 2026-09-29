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

# Fixed colour per policy (reference categorical palette, light mode).
# "+lc" variants share their base colour and are drawn dashed.
COLORS = {
    "wrandom": "#8a8984",
    "jsq": "#2a78d6",
    "sed": "#eda100",
    "p2c": "#1baf7a",
    "greedy": "#eb6834",
    "ts": "#4a3aa7",
    "ts_epoch": "#e87ba4",
    "p2c_learned": "#008300",
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
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

HEURISTICS = ["wrandom", "jsq", "sed", "p2c"]
LEARNED = ["greedy", "ts", "ts_epoch", "p2c_learned"]
FIXES = ["jsq", "jsq+lc", "greedy", "greedy+lc", "ts", "ts+lc", "p2c_learned"]


def style(name: str) -> dict:
    base = name.replace("+lc", "")
    label = LABELS[base] + (" + local correction" if name.endswith("+lc") else "")
    return dict(color=COLORS[base], linestyle="--" if name.endswith("+lc") else "-",
                label=label, linewidth=1.8, marker="o", markersize=5)


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


def line_panels(df, x, metric, panels, title, xlabel, ylabel, fname, xticks=None, xlog=False, logy=True,
                categorical=False, tick_label=lambda v: f"{v:g}"):
    """One panel per (title, policies). categorical=True spaces the x values evenly."""
    single = len(panels) == 1
    fig, axes = plt.subplots(1, len(panels), figsize=(9.0 if single else 5.2 * len(panels), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (panel_title, policies) in zip(axes, panels):
        for pol in policies:
            sub = agg(df[df.policy == pol], [x], metric).sort_values(x)
            if sub.empty:
                continue
            st = style(pol)
            xs = [xticks.index(v) for v in sub[x]] if categorical else sub[x]
            ax.plot(xs, sub["median"], **st)
            ax.fill_between(xs, sub["q25"], sub["q75"], color=st["color"], alpha=0.12, linewidth=0)
        ax.set_title(panel_title, color=TEXT, fontsize=11, loc="left")
        _axes_style(ax, xlabel, ylabel if ax is axes[0] else "", logy=logy)
        if xlog:
            ax.set_xscale("log", base=2)
        if xticks is not None:
            ax.set_xticks(range(len(xticks)) if categorical else xticks)
            ax.set_xticklabels([tick_label(t) for t in xticks])
        if single:
            ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        else:
            ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.suptitle(title, color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, fname)
    fig.savefig(path, dpi=150, facecolor="white")
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


def main():
    out = []
    policies_all = HEURISTICS + LEARNED + ["jsq+lc", "greedy+lc", "ts+lc"]

    main_csv = os.path.join(RESULTS, "main.csv")
    if os.path.exists(main_csv):
        df = pd.read_csv(main_csv)
        steady, gray = df[df.scenario == "steady"], df[df.scenario == "gray"]
        k_ref, st_ref = 16, 0.05
        st_ticks = sorted(df.staleness.unique())
        st_axis = dict(xticks=st_ticks, categorical=True, tick_label=staleness_label)

        line_panels(steady[steady.num_dispatchers == k_ref], "staleness", "p99_ms",
                    [("Heuristics", HEURISTICS), ("Learned (P2C for reference)", LEARNED + ["p2c"])],
                    f"P99 latency vs. state staleness ({k_ref} dispatchers, ρ=0.9)",
                    "shared-state refresh period", "P99 latency (ms, log)", "fig1_p99_vs_staleness.png",
                    **st_axis)
        line_panels(steady[steady.staleness == st_ref], "num_dispatchers", "p99_ms",
                    [("Heuristics", HEURISTICS), ("Learned (P2C for reference)", LEARNED + ["p2c"])],
                    f"P99 latency vs. number of dispatchers ({st_ref * 1000:.0f} ms staleness, ρ=0.9)",
                    "independent dispatchers K", "P99 latency (ms, log)", "fig2_p99_vs_dispatchers.png",
                    xticks=sorted(df.num_dispatchers.unique()), xlog=True)
        line_panels(steady[steady.num_dispatchers == k_ref], "staleness", "fano",
                    [("Heuristics", HEURISTICS), ("Learned (P2C for reference)", LEARNED + ["p2c"])],
                    f"Herding index vs. staleness ({k_ref} dispatchers) — 1 = independent arrivals",
                    "shared-state refresh period", "Fano factor of per-server arrivals (log)",
                    "fig3_herding_vs_staleness.png", **st_axis)
        line_panels(steady[steady.num_dispatchers == k_ref], "staleness", "p99_ms",
                    [("Mitigations (P2C for reference)", FIXES + ["p2c"])],
                    f"Mitigations: local correction and P2C over learned scores ({k_ref} dispatchers)",
                    "shared-state refresh period", "P99 latency (ms, log)", "fig4_mitigations.png",
                    **st_axis)
        line_panels(gray[gray.num_dispatchers == k_ref], "staleness", "p99_ms",
                    [("Heuristics", HEURISTICS), ("Learned", LEARNED + ["ts+lc", "p2c"])],
                    f"Gray failure: fastest server drops to 20% speed at t=12 s ({k_ref} dispatchers, ρ=0.75)",
                    "shared-state refresh period", "P99 latency (ms, log)", "fig5_gray_failure.png",
                    **st_axis)

        for scen, sub in (("Steady state", steady), ("Gray failure", gray)):
            out.append(f"## {scen}: P99 latency (ms), K={k_ref}, median [IQR] over seeds\n")
            out.append(table(sub[sub.num_dispatchers == k_ref], "policy", "staleness", "p99_ms",
                             policies_all, st_ticks))
            out.append("")
            out.append(f"## {scen}: P99 latency (ms) vs. dispatchers, staleness={st_ref}s\n")
            out.append(table(sub[sub.staleness == st_ref], "policy", "num_dispatchers", "p99_ms",
                             policies_all, sorted(sub.num_dispatchers.unique())))
            out.append("")
        out.append(f"## Steady state: herding index (Fano factor), K={k_ref}\n")
        out.append(table(steady[steady.num_dispatchers == k_ref], "policy", "staleness", "fano",
                         policies_all, st_ticks, digits=1))
        out.append("")

    epoch_csv = os.path.join(RESULTS, "epoch.csv")
    if os.path.exists(epoch_csv):
        df = pd.read_csv(epoch_csv)
        df["control_interval"] = df.policy_kwargs.str.extract(r"control_interval=([\d.]+)").astype(float)
        out.append("## ts_epoch: P99 (ms) vs. control interval, K=16\n")
        out.append(table(df, "staleness", "control_interval", "p99_ms",
                         sorted(df.staleness.unique()), sorted(df.control_interval.unique())))
        out.append("")

    load_csv = os.path.join(RESULTS, "load.csv")
    if os.path.exists(load_csv):
        df = pd.read_csv(load_csv)
        pols = [p for p in policies_all if p in set(df.policy)]
        line_panels(df, "rho", "p99_ms", [("All policies", pols)],
                    "P99 latency vs. utilisation (K=16, 50 ms staleness)",
                    "utilisation ρ", "P99 latency (ms, log)", "fig6_load.png",
                    xticks=sorted(df.rho.unique()))
        out.append("## P99 (ms) vs. utilisation, K=16, staleness=0.05s\n")
        out.append(table(df, "policy", "rho", "p99_ms", pols, sorted(df.rho.unique())))
        out.append("")

    tuning_csv = os.path.join(RESULTS, "tuning.csv")
    if os.path.exists(tuning_csv):
        df = pd.read_csv(tuning_csv)
        df["setting"] = df.policy + " " + df.policy_kwargs
        out.append("## Hyperparameter sensitivity: P99 (ms), K=16\n")
        out.append(table(df, "setting", "staleness", "p99_ms",
                         sorted(df.setting.unique()), sorted(df.staleness.unique())))
        out.append("")

    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write("# Results summary (generated by `python -m research.analyze`)\n\n" + "\n".join(out) + "\n")
    print("wrote", os.path.join(RESULTS, "summary.md"))


if __name__ == "__main__":
    main()
