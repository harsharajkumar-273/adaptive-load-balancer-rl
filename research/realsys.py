"""
Real-system validation: K gateway processes (src/gateway.py) sharing one Redis,
8 FIFO backend processes (src/backend_node.py --service-rate), and an
open-loop Poisson load generator.

The setup mirrors the simulator, slowed down 5x so Python processes can keep
up: backend rates 40/40/30/30/20/20/10/10 req/s (total 200), load rho * 200.
Simulator staleness 50 ms / 200 ms corresponds to 250 ms / 1 s here.

    redis-server --daemonize yes
    python -m research.realsys                 # full grid -> research/results/realsys.csv
    python -m research.realsys --quick         # one short run per strategy

Gateways poll Redis every STATE_REFRESH_SEC seconds; they start a few hundred
milliseconds apart, so their refresh phases are not synchronised.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import os
import random
import subprocess
import sys
import time
from typing import Dict, List

import httpx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "research", "results", "realsys.csv")

MUS = [40.0, 40.0, 30.0, 30.0, 20.0, 20.0, 10.0, 10.0]
BACKEND_PORT = 8001
GATEWAY_PORT = 9001
STRATEGIES = ["least_conn", "p2c", "greedy_learned", "p2c_learned"]


def _spawn(cmd: List[str], env: Dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **env},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def _wait_ready(urls: List[str], timeout: float = 20.0) -> None:
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
    limits = httpx.Limits(max_connections=4000, max_keepalive_connections=1000)
    async with httpx.AsyncClient(timeout=180.0, limits=limits) as client:

        async def one(url: str, t_send: float):
            try:
                resp = await client.get(url)
                ok = resp.status_code == 200
            except httpx.HTTPError:
                ok = False
            results.append((t_send, time.perf_counter() - t_send, ok))

        tasks = []
        start = time.perf_counter()
        next_t = start
        while next_t - start < warmup + duration:
            next_t += rng.expovariate(rate)
            delay = next_t - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(f"{rng.choice(gateways)}/play-video", time.perf_counter())))
        await asyncio.gather(*tasks)
    measured = [(lat, ok) for t, lat, ok in results if t - start >= warmup]
    return measured


def run_one(strategy: str, refresh: float, k: int, rho: float, seed: int,
            duration: float, warmup: float) -> Dict:
    subprocess.run(["redis-cli", "flushall"], check=True, stdout=subprocess.DEVNULL)
    procs = []
    try:
        for i, mu in enumerate(MUS):
            procs.append(_spawn([sys.executable, "src/backend_node.py", "--port", str(BACKEND_PORT + i),
                                 "--node-idx", str(i), "--service-rate", str(mu)], {}))
        gw_env = {"ROUTING_STRATEGY": strategy, "STATE_REFRESH_SEC": str(refresh), "CPU_MASK": "false",
                  "GATEWAY_TIMEOUT_SEC": "120",  # no timeout-driven retries (they cause retry storms)
                  "NUM_INSTANCES": str(len(MUS)), "LEARNED_LATENCY_SCALE_SEC": str(len(MUS) / sum(MUS))}
        gateways = []
        for g in range(k):
            port = GATEWAY_PORT + g
            procs.append(_spawn([sys.executable, "-m", "uvicorn", "src.gateway:app", "--port", str(port),
                                 "--log-level", "warning"], gw_env))
            gateways.append(f"http://127.0.0.1:{port}")
            time.sleep(0.137)  # stagger start-up -> unsynchronised refresh phases
        asyncio.run(_wait_ready([f"http://127.0.0.1:{BACKEND_PORT + i}/docs" for i in range(len(MUS))]
                                + [f"{g}/docs" for g in gateways]))
        time.sleep(2.5)  # let heartbeats populate Redis
        measured = asyncio.run(_load(gateways, rho * sum(MUS), duration, warmup, seed))
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
    lat_ms = np.array([lat for lat, ok in measured]) * 1000.0
    return {
        "strategy": strategy, "state_refresh_sec": refresh, "num_gateways": k, "rho": rho, "seed": seed,
        "requests": len(measured), "errors": sum(1 for _, ok in measured if not ok),
        "p50_ms": float(np.percentile(lat_ms, 50)), "p99_ms": float(np.percentile(lat_ms, 99)),
        "mean_ms": float(lat_ms.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=6.0)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--rho", type=float, default=0.8,
                        help="nominal load; Python overheads make the effective load a few %% higher")
    args = parser.parse_args()

    if args.quick:
        grid = [(s, 0.25, 4, 0) for s in STRATEGIES]
        args.duration, args.warmup = 15.0, 4.0
    else:
        grid = list(itertools.product(STRATEGIES, [0.0, 0.25, 1.0], [1, 8], range(args.seeds)))
    rows = []
    for i, (strategy, refresh, k, seed) in enumerate(grid, 1):
        row = run_one(strategy, refresh, k, args.rho, seed, args.duration, args.warmup)
        rows.append(row)
        print(f"[{i}/{len(grid)}] {strategy:15s} refresh={refresh:<5} K={k} seed={seed}: "
              f"p50={row['p50_ms']:.0f}ms p99={row['p99_ms']:.0f}ms errors={row['errors']}", flush=True)
    out = RESULTS.replace(".csv", "_quick.csv") if args.quick else RESULTS
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
