"""
Discrete-event simulator for K independent dispatchers routing to N servers.

Model
-----
* N servers, each a single FIFO queue with exponential service at rate mu_i.
* Poisson arrivals at rate lambda = rho * sum(mu); each arrival lands on one
  of K dispatchers uniformly at random (horizontally scaled gateways behind
  an L4 balancer).
* Shared state: every `staleness` seconds a snapshot of all queue lengths is
  published (think: backends heartbeat into Redis, gateways read it). All
  dispatchers see the same snapshot. staleness == 0 means perfect live state.
* Each dispatcher observes the latency of *its own* requests only, at their
  completion time (feedback is naturally delayed by the response time).
* Optional gray failure: from `gray_start` onwards one server serves at
  `gray_factor` x its nominal rate. Nothing announces it; it shows up only
  through queue lengths and latencies.

Because queues are FIFO single-server, a request's departure time is fixed at
arrival: depart = max(arrival, previous departure) + service. That keeps the
simulator exact and fast without a per-server event loop.
"""
from __future__ import annotations

import heapq
import math
import random
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

from research.policies import make_policy

# Default fleet: heterogeneous capacities, total 1000 req/s.
DEFAULT_MUS = (200.0, 200.0, 150.0, 150.0, 100.0, 100.0, 50.0, 50.0)

# Width of the bins used for the arrival-burstiness (herding) metric.
BURST_BIN_SEC = 0.05


@dataclass(frozen=True)
class Scenario:
    mus: tuple = DEFAULT_MUS
    rho: float = 0.9
    duration: float = 30.0
    warmup: float = 3.0
    staleness: float = 0.05
    num_dispatchers: int = 8
    gray_server: Optional[int] = None
    gray_factor: float = 0.2
    gray_start: float = 12.0

    def post_failure_utilisation(self) -> float:
        """Offered load / capacity after the gray failure (== rho without one)."""
        capacity = sum(self.mus)
        if self.gray_server is not None:
            capacity -= self.mus[self.gray_server] * (1.0 - self.gray_factor)
        return self.rho * sum(self.mus) / capacity

    def to_row(self) -> Dict:
        row = asdict(self)
        row["mus"] = "/".join(f"{m:g}" for m in self.mus)
        row["scenario"] = "gray" if self.gray_server is not None else "steady"
        return row


def _service_rate(sc: Scenario, server: int, t: float) -> float:
    mu = sc.mus[server]
    if sc.gray_server == server and t >= sc.gray_start:
        mu *= sc.gray_factor
    return mu


def simulate(sc: Scenario, policy: str, seed: int = 0, policy_kwargs: Optional[Dict] = None) -> Dict:
    """Runs one simulation and returns summary metrics (post-warmup)."""
    policy_kwargs = policy_kwargs or {}
    if sc.post_failure_utilisation() >= 1.0:
        raise ValueError(f"Scenario is overloaded after the gray failure "
                         f"(utilisation {sc.post_failure_utilisation():.2f}); no policy can be stable.")
    n = len(sc.mus)
    lam = sc.rho * sum(sc.mus)
    rng = np.random.default_rng(seed)

    # Pre-draw the exogenous randomness (same for every policy given a seed).
    n_req = int(lam * sc.duration * 1.1) + 100
    arrivals = np.cumsum(rng.exponential(1.0 / lam, n_req))
    n_req = int(np.searchsorted(arrivals, sc.duration))
    arrivals = arrivals[:n_req].tolist()
    unit_service = rng.exponential(1.0, n_req).tolist()
    dispatcher_of = rng.integers(0, sc.num_dispatchers, n_req).tolist()

    mean_service = n / sum(sc.mus)
    dispatchers = [
        make_policy(policy, n=n, nominal_mus=sc.mus, latency_scale=mean_service,
                    rng=random.Random(seed * 100_003 + d), np_rng=np.random.default_rng([seed, d]),
                    **policy_kwargs)
        for d in range(sc.num_dispatchers)
    ]

    in_system = [deque() for _ in range(n)]   # departure times, FIFO order
    last_departure = [0.0] * n
    feedback = []                              # (depart, req_id, dispatcher, server, latency, feature)

    snapshot = [0] * n
    snapshot_version = 0
    last_snapshot_t = -1.0

    latencies = []
    joined_queue = []
    burst_bins = []
    burst_servers = []

    for j in range(n_req):
        t = arrivals[j]

        # 1. Deliver completed-request feedback to the dispatcher that sent it.
        while feedback and feedback[0][0] <= t:
            dep, _, d, s, lat, feat = heapq.heappop(feedback)
            dispatchers[d].observe(s, lat, feat, dep)

        # 2. Publish a new shared snapshot if a refresh boundary has passed.
        if sc.staleness > 0:
            tk = math.floor(t / sc.staleness) * sc.staleness
            if tk > last_snapshot_t:
                for q in in_system:
                    while q and q[0] <= tk:
                        q.popleft()
                snapshot = [len(q) for q in in_system]
                snapshot_version += 1
                last_snapshot_t = tk

        for q in in_system:
            while q and q[0] <= t:
                q.popleft()
        live = [len(q) for q in in_system]
        if sc.staleness == 0:
            snapshot = live
            snapshot_version += 1

        # 3. Route.
        d = dispatcher_of[j]
        s, feat = dispatchers[d].choose(t, snapshot, snapshot_version)

        # 4. Serve (FIFO): departure time is known now.
        start = max(t, last_departure[s])
        dep = start + unit_service[j] / _service_rate(sc, s, start)
        last_departure[s] = dep
        in_system[s].append(dep)
        lat = dep - t
        heapq.heappush(feedback, (dep, j, d, s, lat, feat))

        if t >= sc.warmup:
            latencies.append(lat)
            joined_queue.append(live[s])
            burst_bins.append(int(t / BURST_BIN_SEC))
            burst_servers.append(s)

    return _summarize(sc, policy, seed, latencies, joined_queue, burst_bins, burst_servers)


def _summarize(sc, policy, seed, latencies, joined_queue, burst_bins, burst_servers) -> Dict:
    n = len(sc.mus)
    lat_ms = np.asarray(latencies) * 1000.0
    jq = np.asarray(joined_queue)

    # Herding metric: Fano factor (variance / mean) of per-server arrival
    # counts in short bins. Independent (Poisson-like) routing gives ~1;
    # dispatchers stampeding onto the same server give >> 1.
    bins = np.asarray(burst_bins)
    bins -= bins.min()
    counts = np.bincount(bins * n + np.asarray(burst_servers), minlength=(bins.max() + 1) * n)
    counts = counts.reshape(-1, n)
    means = counts.mean(axis=0)
    fano = np.where(means > 0, counts.var(axis=0) / np.maximum(means, 1e-9), np.nan)
    weights = np.asarray(sc.mus) / sum(sc.mus)

    row = sc.to_row()
    row.update({
        "policy": policy,
        "seed": seed,
        "requests": int(len(lat_ms)),
        "mean_ms": float(lat_ms.mean()),
        "p50_ms": float(np.percentile(lat_ms, 50)),
        "p99_ms": float(np.percentile(lat_ms, 99)),
        "p999_ms": float(np.percentile(lat_ms, 99.9)),
        "joined_queue_mean": float(jq.mean()),
        "joined_queue_p99": float(np.percentile(jq, 99)),
        "fano": float(np.nansum(fano * weights)),
    })
    return row
