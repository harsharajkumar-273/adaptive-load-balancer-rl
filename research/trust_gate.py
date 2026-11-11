"""
Offline replay for the trust-gate study (research/TRUST_GATE.md).

collect: runs the *ungated* learned-greedy routers (mode="off", which still grades and logs every step) and saves
         the per-step grades: router, time, model_ok, baseline_ok, realised latency (latency-scale units).
replay:  re-evaluates the hard-gate rule on those two aligned boolean lists for any window / tolerance without
         re-running the simulator, and asks: of the worst 1% of latencies the model produced, how many happened
         while the gate had withdrawn trust (caught), and how much of all traffic would the gate have taken away?

Regret needs the simulator (a gated router changes the load others see); replay only measures the gate's signal.

    python -m research.trust_gate collect      # about 1-2 minutes
    python -m research.trust_gate replay       # writes results/summary_trust_replay.md
"""
from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

from research.run import _scenario
from research.sim import simulate

HERE = os.path.dirname(__file__)
STEPS = os.path.join(HERE, "results", "trust_steps.npz")
OUT = os.path.join(HERE, "results", "summary_trust_replay.md")
CELLS = [("steady", 0.05), ("steady", 0.2), ("gray", 0.05), ("gray", 0.2)]
MODELS = {"linear": "gate_off_greedy", "quadratic": "gate_off_greedy_q2", "ewma": "gate_off_greedy_ewma"}


def collect(seeds=range(3)) -> None:
    cols = {k: [] for k in ("model", "scen", "stale", "seed", "router", "t", "model_ok", "base_ok", "y")}
    for model, policy in MODELS.items():
        for scen, st in CELLS:
            for seed in seeds:
                keep: list = []
                simulate(_scenario(scen, staleness=st, num_dispatchers=16, desync=False), policy, seed=seed,
                         policy_kwargs={"log_steps": True}, keep=keep)
                for r, d in enumerate(keep):
                    n = len(d.log)
                    if n == 0:
                        continue
                    a = np.asarray(d.log, dtype=float)
                    cols["model"] += [model] * n; cols["scen"] += [scen] * n; cols["stale"] += [st] * n
                    cols["seed"] += [seed] * n; cols["router"] += [r] * n
                    for k, j in (("t", 0), ("model_ok", 1), ("base_ok", 2), ("y", 3)):
                        cols[k] += a[:, j].tolist()
                print("collected", model, scen, st, seed, flush=True)
    np.savez_compressed(STEPS, **{k: np.asarray(v) for k, v in cols.items()})
    print("wrote", STEPS)


def gate_states(model_ok, base_ok, w, min_grades=20):
    min_grades = min(min_grades, w)
    """Gate state *before* each step: trusted iff, over the last w graded steps, model_ok beats baseline_ok."""
    mo, bo = deque(maxlen=w), deque(maxlen=w)
    out = np.zeros(len(model_ok), dtype=bool)
    sm = sb = 0
    for i, (m, b) in enumerate(zip(model_ok, base_ok)):
        out[i] = len(mo) >= min_grades and sm > sb
        if len(mo) == w:
            sm -= mo[0]; sb -= bo[0]
        mo.append(m); bo.append(b); sm += m; sb += b
    return out


def replay() -> None:
    z = np.load(STEPS)
    lines = ["# Trust gate: offline replay of the two grade lists (ungated routers, 3 seeds per cell)\n",
             "Gate rule: trust iff over the last `w` graded steps the model is within tolerance more often than the baseline "
             "(persistence: the server's smoothed own latency). `caught` = share of the worst 1% of latencies that occurred "
             "while the gate had withdrawn trust; `taken` = share of all steps the gate would have taken away. A useless gate "
             "has caught = taken.\n"]
    for model in MODELS:
        lines += [f"\n## {model} model\n", "| cell | w | model ok | baseline ok | taken | caught | lift |", "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"]
        for scen, st in CELLS:
            sel = (z["model"] == model) & (z["scen"] == scen) & (np.isclose(z["stale"], st))
            if not sel.any():
                continue
            y, mo, bo, rt = z["y"][sel], z["model_ok"][sel], z["base_ok"][sel], (z["seed"][sel] * 100 + z["router"][sel])
            order = np.lexsort((z["t"][sel], rt))
            y, mo, bo, rt = y[order], mo[order], bo[order], rt[order]
            thresh = np.percentile(y, 99)
            for w in (10, 25, 50, 100, 200):
                trusted = np.zeros(len(y), dtype=bool)
                for r in np.unique(rt):
                    m = rt == r
                    trusted[m] = gate_states(mo[m], bo[m], w)
                tail = y >= thresh
                taken = float((~trusted).mean())
                caught = float((~trusted[tail]).mean())
                lines.append(f"| {scen} {int(st * 1000)} ms | {w} | {mo.mean():.2f} | {bo.mean():.2f} | {taken:.2f} | "
                             f"{caught:.2f} | {caught / max(taken, 1e-9):.2f}x |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    {"collect": collect, "replay": replay}[sys.argv[1]]()
