"""
Real-system validation with the same experimental axes as the simulator.

K gateways (src/gateway.py, hosted by src/gateway_pool.py) share one Redis,
8 FIFO backend processes (src/backend_node.py --service-rate) serve requests,
and an open-loop Poisson load generator spreads requests over the gateways.
The lin_ts strategy also runs the production control-plane agent
(src/agent_main.py).

Time scale: 10x slower than research/sim.py, so single-threaded Python
processes keep headroom. Backend rates are 20/20/15/15/10/10/5/5 req/s (total
100), and every duration is scaled by 10: 10 ms simulator staleness is 0.1 s
here, the 150 ms control interval is 1.5 s, and the gray failure starts 40%
into the run.

Grid (mirrors research/run.py "main" and "probing"):
  scenario   steady (rho = 0.9) | gray (rho = 0.75, backend 0 at 20% speed)
  refresh    0 | 0.1 / 0.5 / 2.0 s, synchronised or not
  gateways   1, 4, 16, 64
  strategies 12 snapshot-based + 6 probing (probing ignores refresh)
  seeds      3

    redis-server --daemonize yes
    python -m research.realsys                  # full grid, resumable
    python -m research.realsys --lanes 2        # 2 isolated experiments at once (own cores each)
    python -m research.realsys --quick          # smoke test

Results are appended to research/results/realsys.csv as each run finishes.
Re-running skips configurations that are already in the file.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List, Tuple

import aiohttp
import httpx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "research", "results", "realsys.csv")

TIME_SCALE = 10  # relative to research/sim.py
MUS = [20.0, 20.0, 15.0, 15.0, 10.0, 10.0, 5.0, 5.0]
MAX_GATEWAYS_PER_PROCESS = 8

SNAPSHOT_STRATEGIES = [
    "least_conn", "sed", "p2c",
    "greedy_learned", "ts_learned", "lin_ts", "p2c_learned", "p3c_learned",
    "least_conn+lc", "greedy_learned+lc", "ts_learned+lc", "p2c_learned+lc",
]
PROBING_STRATEGIES = ["prequal", "p2c_probe", "p3c_probe", "sed3_probe", "p2c_learned_probe", "p3c_learned_probe"]
# Simulator staleness 0 / 10 / 50 / 200 ms, scaled.
REFRESH = [0.0, 0.01 * TIME_SCALE, 0.05 * TIME_SCALE, 0.2 * TIME_SCALE]
GATEWAYS = [1, 4, 16, 64]
SCENARIOS = {"steady": 0.9, "gray": 0.75}
GRAY_FACTOR = 0.2
GRAY_AT_FRACTION = 0.4   # simulator: t=12 s of a 30 s run
CONTROL_INTERVAL_SEC = 0.15 * TIME_SCALE
SETUP_SEC = 15.0         # launch-to-load delay (fixed, so the gray failure lands at a known time)
# After the load stops, wait at most this long for outstanding requests. Any
# still waiting are recorded as *censored* at the time waited so far, so tail
# percentiles of collapsed runs are lower bounds (see the "censored" column).
DRAIN_TIMEOUT_SEC = 120.0

FIELDS = ["scenario", "strategy", "state_refresh_sec", "aligned", "num_gateways", "rho", "seed",
          "requests", "censored", "errors", "p50_ms", "p99_ms", "p999_ms", "mean_ms", "fano", "fallback_frac", "modes",
          "duration_sec", "warmup_sec", "lane"]

Config = Tuple[str, str, float, bool, int, int]   # scenario, strategy, refresh, aligned, K, seed


def grid(seeds: int) -> List[Config]:
    configs = []
    for seed in range(seeds):   # seed-major: a full first pass finishes before any repeats
        for scen, k in itertools.product(SCENARIOS, GATEWAYS):
            for strategy in SNAPSHOT_STRATEGIES:
                for refresh in REFRESH:
                    for aligned in ([False] if refresh == 0 else [True, False]):
                        configs.append((scen, strategy, refresh, aligned, k, seed))
            for strategy in PROBING_STRATEGIES:
                configs.append((scen, strategy, 0.0, False, k, seed))
    return configs


class Lane:
    """
    An isolated experiment slot: its own Redis database, port ranges and CPU
    cores. Every run executes in its own process (so lanes never share a GIL)
    and every process of a lane is pinned to the lane's cores with taskset.
    """

    def __init__(self, idx: int, cores: str = ""):
        self.idx = idx
        self.cores = cores
        self.redis_db = 1 + idx
        self.backend_base = 8100 + 100 * idx
        self.gateway_base = 9100 + 200 * idx


def _lane_cores(n_lanes: int) -> List[str]:
    cpus = sorted(os.sched_getaffinity(0))
    if n_lanes <= 1 or len(cpus) < n_lanes:
        return [""] * n_lanes
    per = len(cpus) // n_lanes
    return [",".join(str(c) for c in cpus[i * per:(i + 1) * per]) for i in range(n_lanes)]


_LIVE: set = set()          # every child process not yet reaped, across lanes
_LIVE_LOCK = threading.Lock()


def _spawn(cmd: List[str], env: Dict[str, str]) -> subprocess.Popen:
    p = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **env},
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with _LIVE_LOCK:
        _LIVE.add(p)
    return p


def _stop(procs) -> None:
    for p in procs:
        if getattr(p, "own_group", False):
            try:  # a whole run: its process group holds every child it started
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
        with _LIVE_LOCK:
            _LIVE.discard(p)


async def _wait_ready(urls: List[str], timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=1.0) as client:
        for url in urls:
            while True:
                try:
                    await client.get(url)
                    break
                except httpx.HTTPError:
                    if time.time() > deadline:
                        raise RuntimeError(f"{url} did not come up")
                    await asyncio.sleep(0.2)


async def _load(gateways: List[str], rate: float, duration: float, warmup: float, seed: int):
    rng = random.Random(seed)
    results = []
    # aiohttp: finding a free connection is O(1) even with thousands open
    # (httpx's pool scans every connection per request, which would inflate
    # client-side latency exactly when many requests are outstanding).
    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=240)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=None)) as client:

        async def one(url: str, t_send: float):
            node = mode = None
            try:
                async with client.get(url) as resp:
                    await resp.read()
                    ok = resp.status == 200
                    node = resp.headers.get("x-routed-to")
                    mode = resp.headers.get("x-routing-mode")
            except aiohttp.ClientError as e:
                ok = False
                mode = f"client_error:{type(e).__name__}"
            except asyncio.CancelledError:
                results.append((t_send, time.perf_counter() - t_send, False, None, "censored"))
                raise
            results.append((t_send, time.perf_counter() - t_send, ok, node, mode))

        tasks = []
        start = time.perf_counter()
        next_t = start
        while next_t - start < warmup + duration:
            next_t += rng.expovariate(rate)
            delay = next_t - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(f"{rng.choice(gateways)}/play-video", time.perf_counter())))
        _, still_waiting = await asyncio.wait(tasks, timeout=DRAIN_TIMEOUT_SEC)
        for task in still_waiting:
            task.cancel()
        await asyncio.gather(*still_waiting, return_exceptions=True)
    return [(t - start, lat, ok, node, mode) for t, lat, ok, node, mode in results if t - start >= warmup]


def _fano(measured, n: int) -> float:
    """Herding index as in the simulator: Fano factor of per-backend arrivals in bins."""
    bin_sec = 0.05 * TIME_SCALE
    counts: Dict[Tuple[int, int], int] = {}
    bins = set()
    for t, _, _, node, _ in measured:
        if not node:
            continue
        b = int(t / bin_sec)
        bins.add(b)
        s = int(node.split("-")[1]) - 1
        counts[(b, s)] = counts.get((b, s), 0) + 1
    if not bins:
        return float("nan")
    weights = np.asarray(MUS) / sum(MUS)
    total = 0.0
    for s in range(n):
        x = np.array([counts.get((b, s), 0) for b in sorted(bins)], dtype=float)
        if x.mean() > 0:
            total += weights[s] * x.var() / x.mean()
    return float(total)


def run_one(cfg: Config, lane: Lane, duration: float, warmup: float) -> Dict:
    scen, strategy, refresh, aligned, k, seed = cfg
    rho = SCENARIOS[scen]
    subprocess.run(["redis-cli", "-n", str(lane.redis_db), "flushdb"], check=True, stdout=subprocess.DEVNULL)
    env = {
        "REDIS_DB": str(lane.redis_db),
        "BACKEND_PORT_BASE": str(lane.backend_base),
        "NUM_INSTANCES": str(len(MUS)),
        "MAX_INSTANCES": str(len(MUS)),
        "NOMINAL_RATES": ",".join(f"{m:g}" for m in MUS),
        "ROUTING_STRATEGY": strategy,
        "STATE_REFRESH_SEC": str(refresh),
        "STATE_REFRESH_ALIGNED": "true" if aligned else "false",
        "CPU_MASK": "false",
        "GATEWAY_TIMEOUT_SEC": "600",          # no timeout-driven retries (they cause retry storms)
        "LEARNED_LATENCY_SCALE_SEC": str(len(MUS) / sum(MUS)),
        "GATEWAY_SEED": str(seed),
        "CONTROL_PLANE_INTERVAL_SEC": str(CONTROL_INTERVAL_SEC),
        "STALENESS_THRESHOLD_SEC": "1e9",      # measure the LinTS policy itself, not the breaker fallback
        "PREQUAL_MAX_AGE_SEC": str(0.5 * TIME_SCALE),
    }
    launch = time.time()
    load_start = launch + SETUP_SEC
    gray_epoch = load_start + GRAY_AT_FRACTION * (warmup + duration)
    gateway_ports = [lane.gateway_base + g for g in range(k)]
    n_procs = min(k, MAX_GATEWAYS_PER_PROCESS)
    groups = [gateway_ports[i::n_procs] for i in range(n_procs)]

    procs: List[subprocess.Popen] = []
    try:
        for i, mu in enumerate(MUS):
            cmd = [sys.executable, "src/backend_node.py", "--port", str(lane.backend_base + i),
                   "--node-idx", str(i), "--service-rate", str(mu)]
            if scen == "gray" and i == 0:
                cmd += ["--slowdown-epoch", str(gray_epoch), "--slowdown-factor", str(GRAY_FACTOR)]
            procs.append(_spawn(cmd, env))
        for group in groups:
            procs.append(_spawn([sys.executable, "-m", "src.gateway_pool", "--ports", ",".join(map(str, group))], env))
        if strategy == "lin_ts":
            procs.append(_spawn([sys.executable, "-m", "src.agent_main", "--no-checkpoint"], env))

        gateways = [f"http://127.0.0.1:{p}" for p in gateway_ports]
        asyncio.run(_wait_ready([f"http://127.0.0.1:{lane.backend_base + i}/docs" for i in range(len(MUS))]
                                + [f"{g}/docs" for g in gateways]))
        time.sleep(2.5)  # let heartbeats (and the agent's first weights) reach Redis
        wait = load_start - time.time()
        if wait < 0:
            raise RuntimeError(f"start-up took longer than {SETUP_SEC}s")
        time.sleep(wait)
        measured = asyncio.run(_load(gateways, rho * sum(MUS), duration, warmup, seed))
    finally:
        _stop(procs)
    lat_ms = np.array([m[1] for m in measured]) * 1000.0
    modes: Dict[str, int] = {}
    for m in measured:
        modes[m[4] or "none"] = modes.get(m[4] or "none", 0) + 1
    fallback = sum(c for mode, c in modes.items() if any(w in mode for w in ("failed", "empty", "error", "none")))
    censored = modes.get("censored", 0)
    return {
        "scenario": scen, "strategy": strategy, "state_refresh_sec": refresh, "aligned": aligned,
        "num_gateways": k, "rho": rho, "seed": seed,
        "requests": len(measured), "censored": censored,
        "errors": sum(1 for m in measured if not m[2] and m[4] != "censored"),
        "p50_ms": float(np.percentile(lat_ms, 50)), "p99_ms": float(np.percentile(lat_ms, 99)),
        "p999_ms": float(np.percentile(lat_ms, 99.9)), "mean_ms": float(lat_ms.mean()),
        "fano": _fano(measured, len(MUS)), "fallback_frac": fallback / max(1, len(measured)),
        "modes": ";".join(f"{k}={v}" for k, v in sorted(modes.items())),
        "duration_sec": duration, "warmup_sec": warmup, "lane": lane.idx,
    }


def _key(row) -> Config:
    return (row["scenario"], row["strategy"], float(row["state_refresh_sec"]),
            str(row["aligned"]) == "True", int(row["num_gateways"]), int(row["seed"]))


def _done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {_key(r) for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="a few short runs to check the setup")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--lanes", type=int, default=2, help="isolated experiments run in parallel")
    parser.add_argument("--duration", type=float, default=90.0, help="measured seconds per run")
    parser.add_argument("--warmup", type=float, default=15.0)
    parser.add_argument("--run-one", help=argparse.SUPPRESS)  # internal: one run, JSON in/out
    args = parser.parse_args()

    # Turn SIGTERM into an exception so run_one's `finally` stops child processes.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    if args.run_one:
        job = json.loads(args.run_one)
        row = run_one(tuple(job["cfg"]), Lane(job["lane"], job["cores"]), job["duration"], job["warmup"])
        print(json.dumps(row), flush=True)
        return

    out = RESULTS
    if args.quick:
        out = RESULTS.replace(".csv", "_quick.csv")
        # A healthy run next to a collapsing one (lane isolation check), then 64 gateways.
        configs = [("steady", "least_conn", 0.0, False, 4, 0), ("steady", "lin_ts", 0.0, False, 4, 0),
                   ("gray", "p3c_learned_probe", 0.0, False, 64, 0), ("steady", "least_conn", 0.0, False, 64, 0)]
        args.duration, args.warmup = 30.0, 6.0
    else:
        configs = grid(args.seeds)
    done = _done(out)
    todo = [c for c in configs if c not in done]
    print(f"{len(configs)} configurations, {len(done)} already done, {len(todo)} to run "
          f"on {args.lanes} lane(s)", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_lock = threading.Lock()
    queue_lock = threading.Lock()
    pending = list(todo)
    counter = {"n": 0}

    def worker(lane: Lane):
        while True:
            with queue_lock:
                if not pending:
                    return
                cfg = pending.pop(0)
            job = json.dumps({"cfg": cfg, "lane": lane.idx, "cores": lane.cores,
                              "duration": args.duration, "warmup": args.warmup})
            cmd = [sys.executable, "-m", "research.realsys", "--run-one", job]
            if lane.cores:
                cmd = ["taskset", "-c", lane.cores] + cmd
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    start_new_session=True)
            proc.own_group = True
            with _LIVE_LOCK:
                _LIVE.add(proc)
            out_text, err_text = proc.communicate()
            with _LIVE_LOCK:
                _LIVE.discard(proc)
            if proc.returncode != 0:  # keep the grid going; a re-run retries this config
                print(f"[lane {lane.idx}] FAILED {cfg}: {err_text.strip().splitlines()[-1:]}", flush=True)
                continue
            row = json.loads(out_text.strip().splitlines()[-1])
            with write_lock:
                new = not os.path.exists(out)
                with open(out, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                counter["n"] += 1
                print(f"[{counter['n']}/{len(todo)}] lane {lane.idx} {cfg[0]:6s} {cfg[1]:18s} "
                      f"refresh={cfg[2]:<4} aligned={str(cfg[3]):5s} K={cfg[4]:<2} seed={cfg[5]}: "
                      f"p50={row['p50_ms']:.0f}ms p99={row['p99_ms']:.0f}ms fano={row['fano']:.1f} "
                      f"errors={row['errors']}", flush=True)

    cores = _lane_cores(args.lanes)
    threads = [threading.Thread(target=worker, args=(Lane(i, cores[i]),), daemon=True) for i in range(args.lanes)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=1.0)
    except KeyboardInterrupt:
        with _LIVE_LOCK:
            live = list(_LIVE)
        _stop(live)
        print("interrupted: child processes stopped; finished runs are saved", flush=True)
        raise
    print("wrote", out)


if __name__ == "__main__":
    main()
