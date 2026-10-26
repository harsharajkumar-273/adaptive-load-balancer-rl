"""
Experiment runner: expands an experiment grid, runs it in parallel, writes CSV.

    python -m research.run main          # staleness x dispatchers x refresh phase x policy (steady + gray)
    python -m research.run robust        # heavy-tailed service x bursty arrivals
    python -m research.run frontier      # P99 vs probes/request: Prequal vs learned power-of-d probing
    python -m research.run probing       # probing policies (Prequal, power-of-d) vs number of gateways
    python -m research.run models        # richer latency models and LinUCB: do conclusions hold?
    python -m research.run llm           # LLM serving: prefix-cache routers, 2 workload regimes
    python -m research.run llm_regimes   # LLM serving: 108 regimes, tuned-threshold regret
    python -m research.run epoch         # control-interval ablation for ts_epoch
    python -m research.run load          # utilisation sweep
    python -m research.run tuning        # hyperparameter sensitivity (learned policies and Prequal)
    python -m research.run all
    python -m research.run main --quick  # small smoke-test grid

Results go to research/results/<experiment>.csv (one row per run).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import time
from multiprocessing import Pool
from typing import Dict, Iterable, List, Tuple

import math

from research.llm_sim import LLMScenario, THRESHOLDS, simulate_llm
from research.trace_sim import TraceScenario, simulate_trace
from research.agent_sim import AgentScenario, simulate_agents
from research.policies import ALL_POLICIES
from research.sim import Scenario, simulate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

Job = Tuple[str, Scenario, str, int, Dict]


def _scenario(scen: str, **kw) -> Scenario:
    if scen == "gray":
        # rho=0.75 -> 0.89 utilisation after the fastest server drops to 20% speed.
        return Scenario(rho=0.75, gray_server=0, **kw)
    return Scenario(**kw)


def _grid_main(quick: bool) -> Iterable[Job]:
    staleness = [0.0, 0.05] if quick else [0.0, 0.01, 0.05, 0.2]
    dispatchers = [1, 16] if quick else [1, 4, 16, 64]
    seeds = range(2 if quick else 10)
    for scen, st, k, desync, pol, seed in itertools.product(
            ["steady", "gray"], staleness, dispatchers, [False, True], ALL_POLICIES, seeds):
        if desync and st == 0.0:
            continue  # identical to synchronised live state
        yield ("main", _scenario(scen, staleness=st, num_dispatchers=k, desync=desync), pol, seed, {})


ROBUST_POLICIES = ["jsq", "p2c", "greedy", "ts", "p2c_learned", "p2c_learned+lc", "ts+lc", "awf", "awf_learned",
                   "prequal", "p2c_learned_probe", "p3c_learned_probe", "p3c_probe", "sed3_probe"]


def _grid_probing(quick: bool) -> Iterable[Job]:
    """Probing policies do not read the snapshot, so staleness/refresh phase are irrelevant."""
    seeds = range(2 if quick else 10)
    policies = ["prequal", "p2c_probe", "p3c_probe", "sed3_probe", "p2c_learned_probe", "p3c_learned_probe"]
    for scen, k, pol, seed in itertools.product(["steady", "gray"], [1, 4, 16, 64], policies, seeds):
        yield ("probing", _scenario(scen, staleness=0.05, num_dispatchers=k), pol, seed, {})


def _grid_robust(quick: bool) -> Iterable[Job]:
    seeds = range(2 if quick else 10)
    # Bursty arrivals run at rho=0.75 with +/-25% rate swings so the peak (0.94)
    # stays below capacity; Poisson rows keep the default rho=0.9.
    for cv, burst, desync, pol, seed in itertools.product(
            [1.0, 2.0, 4.0], [0.0, 0.25], [False, True], ROBUST_POLICIES, seeds):
        rho = 0.75 if burst else 0.9
        sc = Scenario(rho=rho, staleness=0.05, num_dispatchers=16, desync=desync, service_cv=cv, burstiness=burst)
        yield ("robust", sc, pol, seed, {})


def _grid_frontier(quick: bool) -> Iterable[Job]:
    seeds = range(2 if quick else 10)
    for scen, seed in itertools.product(["steady", "gray"], seeds):
        sc = _scenario(scen, staleness=0.05, num_dispatchers=16)
        for r in [0.5, 1.0, 2.0, 3.0, 5.0]:
            # Below one probe per query, responses must be reusable or the pool runs dry.
            yield ("frontier", sc, "prequal", seed, {"r_probe": r, "reuse_budget": max(1, math.ceil(1.5 / r))})
        for d in [2, 3, 4]:
            yield ("frontier", sc, "p2c_learned", seed, {"d": d, "probe": True})
            yield ("frontier", sc, "probe_d", seed, {"d": d})
            yield ("frontier", sc, "probe_d", seed, {"d": d, "score": "sed"})
        for pol in ["p2c", "p2c_learned", "p2c_learned+lc", "ts+lc", "awf", "awf_learned"]:
            yield ("frontier", sc, pol, seed, {})


def _grid_epoch(quick: bool) -> Iterable[Job]:
    seeds = range(2 if quick else 10)
    for st, ci, seed in itertools.product([0.0, 0.05], [0.01, 0.05, 0.15, 0.5], seeds):
        sc = Scenario(staleness=st, num_dispatchers=16)
        yield ("epoch", sc, "ts_epoch", seed, {"control_interval": ci})


def _grid_load(quick: bool) -> Iterable[Job]:
    seeds = range(2 if quick else 10)
    policies = ["wrandom", "jsq", "p2c", "greedy", "ts", "p2c_learned", "greedy+lc", "ts+lc",
                "p2c_learned+lc", "awf", "awf_learned", "prequal", "p3c_learned_probe"]
    for rho, pol, seed in itertools.product([0.5, 0.7, 0.8, 0.9, 0.95], policies, seeds):
        sc = Scenario(rho=rho, staleness=0.05, num_dispatchers=16)
        yield ("load", sc, pol, seed, {})


def _grid_tuning(quick: bool) -> Iterable[Job]:
    seeds = range(2 if quick else 5)
    sweeps = [
        ("ts", "explore", [1.0, 3.0, 10.0, 30.0]),
        ("ts_epoch", "temperature", [0.5, 2.0, 5.0, 20.0]),
        ("greedy", "gamma", [0.98, 0.995, 0.999]),
        ("ts", "gamma", [0.98, 0.995, 0.999]),
        ("prequal", "q_rif", [0.75, 0.84, 0.95, 0.99]),
        ("prequal", "reuse_budget", [1, 2, 4]),
        ("prequal", "max_age", [0.05, 0.5, 2.0]),
    ]
    for (pol, param, values), st, seed in itertools.product(sweeps, [0.0, 0.05], seeds):
        for v in values:
            sc = Scenario(staleness=st, num_dispatchers=16)
            yield ("tuning", sc, pol, seed, {param: v})


LLM_REGIMES = {"moderate": dict(zipf=1.1, rate=75.0), "hot_prefix": dict(zipf=1.4, rate=90.0)}
LLM_MAIN_POLICIES = ["jsq", "p2c", "hash", "affinity", "affinity_thr4", "affinity_thr16", "greedy_learned",
                     "ts_learned", "p2c_learned", "p3c_learned", "ring2_learned", "ring3_learned",
                     "affinity_p2c_learned", "affinity+lc", "greedy_learned+lc", "ring2_learned+lc"]


def _grid_llm(quick: bool) -> Iterable[Job]:
    seeds = range(2 if quick else 10)
    ks = [1, 16] if quick else [1, 4, 16, 64]
    for (name, regime), k, st, desync, gray, pol, seed in itertools.product(
            LLM_REGIMES.items(), ks, [0.0, 0.05, 0.2], [False, True], [False, True], LLM_MAIN_POLICIES, seeds):
        if desync and st == 0.0:
            continue
        sc = LLMScenario(num_dispatchers=k, staleness=st, desync=desync,
                         slow_replica=0 if gray else None, **regime)
        yield ("llm", sc, pol, seed, {})


def _grid_llm_regimes(quick: bool) -> Iterable[Job]:
    """108 workload regimes; thresholds are chosen on seeds 0-4 and every policy is evaluated on seeds 5-9."""
    seeds = range(5, 7) if quick else range(10)
    policies = ([f"affinity_thr{t}" for t in THRESHOLDS]
                + ["affinity", "jsq", "greedy_learned", "ring2_learned", "ring3_learned",
                   "affinity_p2c_learned", "p2c_learned", "ring2_learned+lc", "greedy_learned+lc"])
    zipfs, rates, caches = ([1.1], [75.0], [24]) if quick else ([0.8, 1.1, 1.4], [60.0, 75.0, 90.0], [16, 24, 48])
    for z, rate, c, st, gray, pol, seed in itertools.product(zipfs, rates, caches, [0.05, 0.2], [False, True],
                                                             policies, seeds):
        sc = LLMScenario(zipf=z, rate=rate, cache_size=c, staleness=st, slow_replica=0 if gray else None)
        yield ("llm_regimes", sc, pol, seed, {})


TRACE_POLICIES = (["jsq", "p2c", "hash", "affinity"] + [f"affinity_thr{t}" for t in THRESHOLDS]
                  + ["greedy_learned", "ts_learned", "p2c_learned", "ring2_learned", "ring3_learned",
                     "affinity_p2c_learned", "greedy_learned+lc", "ring2_learned+lc"])


def _grid_trace(quick: bool) -> Iterable[Job]:
    """Replay of real Mooncake traces: load (trace time compression) x staleness x gray failure."""
    seeds = range(2 if quick else 10)
    speedups = [6.0, 10.0] if quick else [4.0, 6.0, 8.0, 10.0, 12.0]
    traces = ["toolagent"] if quick else ["toolagent", "conversation"]
    for trace, sp, st, gray, pol, seed in itertools.product(traces, speedups, [0.05, 0.2], [False, True],
                                                            TRACE_POLICIES, seeds):
        sc = TraceScenario(trace=trace, speedup=sp, staleness=st, prefill_tokens_per_sec=50000.0,
                           max_requests=4000 if quick else 12000, slow_replica=0 if gray else None)
        yield ("trace", sc, pol, seed, {})


def _grid_trace_cost(quick: bool) -> Iterable[Job]:
    """Same cells as `trace`, for routers whose learned score also sees the request's uncached share."""
    pols = ["greedy_learned_c", "ring2_learned_c", "ring3_learned_c", "affinity_p2c_learned_c"]
    for exp, sc, pol, seed, kw in _grid_trace(quick):
        if pol == "jsq":
            for p in pols:
                yield (exp, sc, p, seed, kw)


AGENT_POLICIES = (["none", "prog"] + [f"shed|q={q}" for q in (4, 16, 64)] + [f"cap|n={n}" for n in (16, 32, 64, 128)]
                  + [f"task|length=mean|h={h}" for h in (5, 10, 20, 40, 80)]
                  + [f"task|length=oracle|h={h}" for h in (10, 20, 40)]
                  + [f"task|length=pred|sigma=0.5|h={h}" for h in (10, 20, 40)]
                  + [f"little|d={d}" for d in (1, 2.5, 5, 10)] + [f"little|df={f}" for f in (0.05, 0.1, 0.2)]
                  + [f"little|d=5|ident=infer|p_id={p}" for p in (0.9, 0.7, 0.5)]
                  + ["little|d=5|fifo=1", "little|d=5|hint=1"])


def _grid_agents(quick: bool) -> Iterable[Job]:
    seeds = range(2 if quick else 10)
    loads = [1.6, 3.0] if quick else [0.6, 0.9, 1.2, 1.6, 2.4, 4.0]
    for load, pol, seed in itertools.product(loads, AGENT_POLICIES, seeds):
        yield ("agents", AgentScenario(load=load), pol, seed, {})


def _grid_agents_surge(quick: bool) -> Iterable[Job]:
    seeds = range(2 if quick else 10)
    for factor, pol, seed in itertools.product([3.0, 5.0], AGENT_POLICIES, seeds):
        yield ("agents_surge", AgentScenario(load=0.8, surge_factor=factor), pol, seed, {})


REGIME_POLICIES = (["none", "prog"] + [f"shed|q={q}" for q in (4, 16, 64)] + [f"cap|n={n}" for n in (16, 32, 64, 128)]
                   + [f"task|length=mean|h={h}" for h in (5, 10, 20, 40, 80)] + [f"little|d={d}" for d in (2, 5, 10)] + [f"little|df={f}" for f in (0.05, 0.1, 0.2)])


def _grid_agents_regimes(quick: bool) -> Iterable[Job]:
    """81 workload regimes at 2.4x overload; fixed parameters are chosen on seeds 0-4, all policies evaluated on 5-9."""
    seeds = range(5, 7) if quick else range(10)
    steps, tools, touts, kvs = ([6.0], [3.0], [30.0], [64]) if quick else ([3.0, 6.0, 12.0], [1.0, 3.0, 10.0], [15.0, 30.0, 60.0], [16, 64, 256])
    for m, tool, to, kv, pol, seed in itertools.product(steps, tools, touts, kvs, REGIME_POLICIES, seeds):
        yield ("agents_regimes", AgentScenario(load=2.4, mean_steps=m, tool_sec=tool, timeout=to, kv_tasks=kv), pol, seed, {})


def _grid_agents_trace(quick: bool) -> Iterable[Job]:
    """Task lengths and tool times resampled from real coding-agent sessions (TraceLab)."""
    seeds = range(2 if quick else 10)
    loads = [1.6] if quick else [0.6, 0.9, 1.2, 1.6, 2.4, 4.0]
    for load, kv, pol, seed in itertools.product(loads, [16, 64, 256], AGENT_POLICIES, seeds):
        yield ("agents_trace", AgentScenario.from_trace(load=load, kv_tasks=kv), pol, seed, {})


def _grid_agents_trace_regimes(quick: bool) -> Iterable[Job]:
    seeds = range(5, 7) if quick else range(10)
    for to, kv, load, pol, seed in itertools.product([15.0, 30.0, 60.0], [16, 64, 256], [1.6, 3.0],
                                                     REGIME_POLICIES, seeds):
        yield ("agents_trace_regimes", AgentScenario.from_trace(load=load, kv_tasks=kv, timeout=to), pol, seed, {})


def _grid_agents_retries(quick: bool) -> Iterable[Job]:
    seeds = range(2 if quick else 10)
    pols = ["none", "prog", "shed|q=16", "cap|n=64", "little|d=5"]
    for retries, backoff, load, pol, seed in itertools.product([0, 1, 3, 8], [0.5, 2.0, 8.0], [1.2, 2.4], pols, seeds):
        yield ("agents_retries", AgentScenario(load=load, retries=retries, backoff=backoff), pol, seed, {})


MODEL_POLICIES = ["greedy", "greedy_q2", "greedy_ewma", "linucb", "linucb_ewma", "ts",
                  "p2c_learned", "p2c_learned_q2", "p2c_learned_ewma",
                  "p3c_learned", "p3c_learned_q2", "p3c_learned_ewma",
                  "greedy_rbf", "linucb_rbf", "p2c_learned_rbf", "p3c_learned_rbf"]


def _grid_models(quick: bool) -> Iterable[Job]:
    seeds = range(2 if quick else 10)
    for scen, st, desync, pol, seed in itertools.product(
            ["steady", "gray"], [0.05, 0.2], [False, True], MODEL_POLICIES, seeds):
        yield ("models", _scenario(scen, staleness=st, num_dispatchers=16, desync=desync), pol, seed, {})


GRIDS = {"main": _grid_main, "models": _grid_models, "llm": _grid_llm, "llm_regimes": _grid_llm_regimes, "trace": _grid_trace, "trace_cost": _grid_trace_cost, "agents": _grid_agents, "agents_surge": _grid_agents_surge,
         "agents_regimes": _grid_agents_regimes, "agents_retries": _grid_agents_retries, "agents_trace": _grid_agents_trace,
         "agents_trace_regimes": _grid_agents_trace_regimes, "robust": _grid_robust, "frontier": _grid_frontier, "probing": _grid_probing,
         "epoch": _grid_epoch, "load": _grid_load, "tuning": _grid_tuning}


def _run(job: Job) -> Dict:
    exp, sc, pol, seed, kw = job
    fn = (simulate_llm if isinstance(sc, LLMScenario) else simulate_trace if isinstance(sc, TraceScenario)
          else simulate_agents if isinstance(sc, AgentScenario) else simulate)
    row = fn(sc, pol, seed=seed, policy_kwargs=kw)
    row["experiment"] = exp
    row["policy_kwargs"] = ";".join(f"{k}={v}" for k, v in sorted(kw.items()))
    return row


def run_experiment(name: str, quick: bool = False, workers: int = os.cpu_count() or 1) -> str:
    jobs: List[Job] = list(GRIDS[name](quick))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{name}{'_quick' if quick else ''}.csv")
    started = time.time()
    rows = []
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_run, jobs, chunksize=4), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(jobs):
                print(f"[{name}] {i}/{len(jobs)} runs, {time.time() - started:.0f}s", flush=True)
    keys = ("scenario", "staleness", "num_dispatchers", "desync", "rho", "service_cv", "burstiness",
            "zipf", "rate", "cache_size", "policy", "policy_kwargs", "seed")
    rows.sort(key=lambda r: tuple(str(r.get(k, "")) if isinstance(r.get(k, ""), str) else r.get(k, 0) for k in keys))
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{name}] wrote {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=list(GRIDS) + ["all"])
    parser.add_argument("--quick", action="store_true", help="small grid for smoke testing")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    for name in (list(GRIDS) if args.experiment == "all" else [args.experiment]):
        run_experiment(name, quick=args.quick, workers=args.workers)
