"""
Discrete-event simulator for K independent dispatchers routing to N servers.

Model
-----
* N servers, each a single FIFO queue. Service times have mean 1/mu_i and are
  exponential (service_cv == 1) or lognormal with the given coefficient of
  variation (service_cv > 1: heavy-tailed).
* Arrivals at mean rate lambda = rho * sum(mu): Poisson, or a two-state
  Markov-modulated Poisson process (burstiness b: rates lambda*(1 +/- b),
  exponential sojourns with mean burst_period). Each arrival lands on one of K
  dispatchers uniformly at random (horizontally scaled gateways behind an L4
  balancer).
* Shared state: every `staleness` seconds a snapshot of all queue lengths is
  published (backends heartbeat into Redis, gateways read it). With
  desync=False every dispatcher refreshes at the same instants; with
  desync=True each dispatcher has its own random refresh phase.
  staleness == 0 means perfect live state.
* Each dispatcher observes the latency of *its own* requests only, at their
  completion time (feedback is naturally delayed by the response time).
* Probing policies (Prequal) can query a server's live requests-in-flight (RIF)
  and its latency estimate *at that RIF*: an EWMA of recent latency per queue
  slot, latency / (RIF at arrival + 1), times (current RIF + 1). Every probe
  is counted.
* Optional gray failure: from `gray_start` onwards one server serves at
  `gray_factor` x its nominal rate. Nothing announces it; it shows up only
  through queue lengths and latencies.

Because queues are FIFO single-server, a request's departure time is fixed at
arrival: depart = max(arrival, previous departure) + service. Per-server sorted
logs of arrival and departure times give the queue length at any past instant
with two binary searches, so the simulator is exact and fast.
"""
from __future__ import annotations

import heapq
import math
import random
from bisect import bisect_right
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from research.policies import make_policy

# Default fleet: heterogeneous capacities, total 1000 req/s.
DEFAULT_MUS = (200.0, 200.0, 150.0, 150.0, 100.0, 100.0, 50.0, 50.0)

# Width of the bins used for the arrival-burstiness (herding) metric.
BURST_BIN_SEC = 0.05

# Smoothing of each server's own per-slot latency estimate (reported to probes).
SERVER_LATENCY_EWMA = 0.1


@dataclass(frozen=True)
class Scenario:
    mus: tuple = DEFAULT_MUS
    rho: float = 0.9
    duration: float = 30.0
    warmup: float = 3.0
    staleness: float = 0.05
    num_dispatchers: int = 8
    desync: bool = False
    service_cv: float = 1.0
    burstiness: float = 0.0
    burst_period: float = 0.5
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


def _draw_arrivals(sc: Scenario, lam: float, rng: np.random.Generator) -> np.ndarray:
    if sc.burstiness <= 0:
        n = int(lam * sc.duration * 1.2) + 100
        arr = np.cumsum(rng.exponential(1.0 / lam, n))
        return arr[arr < sc.duration]
    if not 0 < sc.burstiness < 1:
        raise ValueError("burstiness must be in [0, 1)")
    chunks, t, high = [], 0.0, bool(rng.integers(0, 2))
    while t < sc.duration:
        end = min(sc.duration, t + rng.exponential(sc.burst_period))
        rate = lam * (1 + sc.burstiness if high else 1 - sc.burstiness)
        k = rng.poisson(rate * (end - t))
        chunks.append(np.sort(rng.uniform(t, end, k)))
        t, high = end, not high
    return np.concatenate(chunks)


def _draw_service(sc: Scenario, n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-mean service requirements."""
    if sc.service_cv == 1.0:
        return rng.exponential(1.0, n)
    sigma2 = math.log(1.0 + sc.service_cv ** 2)
    return rng.lognormal(-sigma2 / 2.0, math.sqrt(sigma2), n)


class ClusterView:
    """What probing policies may observe: live RIF and each server's latency estimate."""

    def __init__(self, n: int, arrival_log: List[List[float]], departure_log: List[List[float]]):
        self._arr = arrival_log
        self._dep = departure_log
        self.per_slot_latency = [0.0] * n
        self.probes = 0

    def queue_at(self, s: int, t: float) -> int:
        return bisect_right(self._arr[s], t) - bisect_right(self._dep[s], t)

    def record_completion(self, s: int, latency: float, rif_at_arrival: int) -> None:
        sample = latency / (rif_at_arrival + 1)
        est = self.per_slot_latency[s]
        self.per_slot_latency[s] = sample if est == 0.0 else est + SERVER_LATENCY_EWMA * (sample - est)

    def probe(self, s: int, t: float):
        """Returns (RIF, estimated latency at that RIF)."""
        self.probes += 1
        rif = self.queue_at(s, t)
        return rif, self.per_slot_latency[s] * (rif + 1)


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
    arrivals = _draw_arrivals(sc, lam, rng).tolist()
    n_req = len(arrivals)
    unit_service = _draw_service(sc, n_req, rng).tolist()
    dispatcher_of = rng.integers(0, sc.num_dispatchers, n_req).tolist()
    phase_rng = np.random.default_rng([seed, 7919])
    phases = (phase_rng.uniform(0, sc.staleness, sc.num_dispatchers).tolist()
              if sc.desync and sc.staleness > 0 else [0.0] * sc.num_dispatchers)

    arrival_log: List[List[float]] = [[] for _ in range(n)]
    departure_log: List[List[float]] = [[] for _ in range(n)]
    view = ClusterView(n, arrival_log, departure_log)

    mean_service = n / sum(sc.mus)
    dispatchers = [
        make_policy(policy, n=n, nominal_mus=sc.mus, latency_scale=mean_service,
                    rng=random.Random(seed * 100_003 + d), np_rng=np.random.default_rng([seed, d]),
                    view=view, **policy_kwargs)
        for d in range(sc.num_dispatchers)
    ]

    last_departure = [0.0] * n
    feedback = []                              # (depart, req_id, dispatcher, server, latency, feature, rif)

    snap_time = [None] * sc.num_dispatchers    # refresh instant of each dispatcher's snapshot
    snap = [[0] * n for _ in range(sc.num_dispatchers)]
    snap_version = [0] * sc.num_dispatchers
    snap_cache: Dict[float, List[int]] = {}    # shared snapshots when refreshes are synchronised

    latencies = []
    joined_queue = []
    burst_bins = []
    burst_servers = []

    for j in range(n_req):
        t = arrivals[j]

        # 1. Deliver completed-request feedback to the dispatcher that sent it,
        #    and let the server update its own latency estimate.
        while feedback and feedback[0][0] <= t:
            dep, _, d, s, lat, feat, rif = heapq.heappop(feedback)
            dispatchers[d].observe(s, lat, feat, dep)
            view.record_completion(s, lat, rif)

        # 2. Refresh this dispatcher's view of the shared state if due.
        d = dispatcher_of[j]
        if sc.staleness > 0:
            phase = phases[d]
            tk = math.floor((t - phase) / sc.staleness) * sc.staleness + phase
            if tk != snap_time[d]:
                cached = snap_cache.get(tk)
                if cached is None:
                    cached = [view.queue_at(s, tk) for s in range(n)]
                    if not sc.desync:
                        snap_cache = {tk: cached}
                snap[d] = cached
                snap_time[d] = tk
                snap_version[d] += 1
        else:
            snap[d] = [view.queue_at(s, t) for s in range(n)]
            snap_version[d] += 1

        # 3. Route.
        s, feat = dispatchers[d].choose(t, snap[d], snap_version[d])

        # 4. Serve (FIFO): departure time is known now.
        queue_joined = view.queue_at(s, t)
        t_at_server = t + dispatchers[d].added_delay
        start = max(t_at_server, last_departure[s])
        dep = start + unit_service[j] / _service_rate(sc, s, start)
        last_departure[s] = dep
        arrival_log[s].append(t_at_server)
        departure_log[s].append(dep)
        lat = dep - t
        heapq.heappush(feedback, (dep, j, d, s, lat, feat, queue_joined))

        if t >= sc.warmup:
            latencies.append(lat)
            joined_queue.append(queue_joined)
            burst_bins.append(int(t / BURST_BIN_SEC))
            burst_servers.append(s)

    row = _summarize(sc, policy, seed, latencies, joined_queue, burst_bins, burst_servers)
    row["probes_per_request"] = view.probes / max(1, n_req)
    return row


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
