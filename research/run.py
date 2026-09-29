"""
Experiment runner: expands an experiment grid, runs it in parallel, writes CSV.

    python -m research.run main          # staleness x dispatchers x refresh phase x policy (steady + gray)
    python -m research.run robust        # heavy-tailed service x bursty arrivals
    python -m research.run frontier      # P99 vs probes/request: Prequal vs learned power-of-d probing
    python -m research.run probing       # probing policies (Prequal, power-of-d) vs number of gateways
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


ROBUST_POLICIES = ["jsq", "p2c", "greedy", "ts", "p2c_learned", "p2c_learned+lc", "ts+lc",
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
        for pol in ["p2c", "p2c_learned", "p2c_learned+lc", "ts+lc"]:
            yield ("frontier", sc, pol, seed, {})


def _grid_epoch(quick: bool) -> Iterable[Job]:
    seeds = range(2 if quick else 10)
    for st, ci, seed in itertools.product([0.0, 0.05], [0.01, 0.05, 0.15, 0.5], seeds):
        sc = Scenario(staleness=st, num_dispatchers=16)
        yield ("epoch", sc, "ts_epoch", seed, {"control_interval": ci})


def _grid_load(quick: bool) -> Iterable[Job]:
    seeds = range(2 if quick else 10)
    policies = ["wrandom", "jsq", "p2c", "greedy", "ts", "p2c_learned", "greedy+lc", "ts+lc",
                "p2c_learned+lc", "prequal", "p3c_learned_probe"]
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


GRIDS = {"main": _grid_main, "robust": _grid_robust, "frontier": _grid_frontier, "probing": _grid_probing,
         "epoch": _grid_epoch, "load": _grid_load, "tuning": _grid_tuning}


def _run(job: Job) -> Dict:
    exp, sc, pol, seed, kw = job
    row = simulate(sc, pol, seed=seed, policy_kwargs=kw)
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
    rows.sort(key=lambda r: (r["scenario"], r["staleness"], r["num_dispatchers"], r["desync"], r["rho"],
                             r["service_cv"], r["burstiness"], r["policy"], r["policy_kwargs"], r["seed"]))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
