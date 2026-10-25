"""
LLM-serving scenario: K independent routers send requests to N prefill replicas
that keep prefix (KV) caches. Same structure as research/sim.py, with two changes.

* Service time depends on cache state. A request has a shared prefix (system
  prompt / document, one of P classes with Zipf popularity) and a unique
  suffix. A replica that has the prefix cached prefills only the suffix;
  otherwise it prefills prefix + suffix. TTFT = queueing delay + prefill time.
  Replicas are single-server FIFO queues (prefill workers of a
  prefill/decode-disaggregated deployment), each with an LRU prefix cache.
* Routers see a *snapshot* of every replica's queue length AND its set of cached
  prefixes, refreshed every `staleness` seconds (or live if 0), i.e. the
  cache metadata is as stale as the load signal, as in routers that replicate
  radix-tree metadata asynchronously.

Because replicas are FIFO, a request's cache hit/miss and departure time are
fixed at arrival: the replica's logical LRU state after all earlier arrivals is
exactly its state when this request starts service.

Metrics: TTFT percentiles, cache hit rate, herding index (Fano factor of
per-replica arrivals in short bins), and the share of traffic to the busiest
replica.
"""
from __future__ import annotations

import heapq
import math
import random
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.learned_routing import LatencyModel
from research.sim import window_burst_stats

BIN_SEC = 0.05
MEAN_BLOCKS = 16.0   # typical request length in KV blocks (trace mode); only scales a feature


@dataclass(frozen=True)
class LLMScenario:
    n: int = 8
    rate: float = 75.0              # total requests per second
    duration: float = 40.0
    warmup: float = 10.0
    staleness: float = 0.05         # snapshot refresh period (0 = live state)
    num_dispatchers: int = 16
    desync: bool = False
    num_prefixes: int = 64
    zipf: float = 1.1
    cache_size: int = 24            # prefixes per replica (LRU)
    prefill_sec_per_token: float = 1.0 / 8000.0
    overhead_sec: float = 0.010
    suffix_mean_tokens: float = 300.0
    suffix_cv: float = 0.8
    prefix_tokens_min: float = 500.0
    prefix_tokens_max: float = 4000.0
    slow_replica: Optional[int] = None   # gray failure: this replica prefills slower
    slow_factor: float = 0.25            # ... at this fraction of its speed
    slow_start: float = 20.0

    def to_row(self) -> Dict:
        row = asdict(self)
        row["scenario"] = "llm_gray" if self.slow_replica is not None else "llm"
        return row


# ----------------------------------------------------------------------------
# Workload
# ----------------------------------------------------------------------------

def prefix_probabilities(sc: LLMScenario) -> np.ndarray:
    w = 1.0 / np.arange(1, sc.num_prefixes + 1) ** sc.zipf
    return w / w.sum()


def prefix_lengths(sc: LLMScenario, rng: np.random.Generator) -> np.ndarray:
    """Prefix lengths, log-uniform in [min, max] tokens (independent of popularity)."""
    u = rng.uniform(math.log(sc.prefix_tokens_min), math.log(sc.prefix_tokens_max), sc.num_prefixes)
    return np.exp(u)


def _lead_fraction(blocks: Sequence[int], present) -> float:
    """Fraction of `blocks` forming a cached leading run; 0 if only the universal first block matches."""
    m = 0
    for b in blocks:
        if not present(b):
            break
        m += 1
    return m / len(blocks) if m >= 2 else 0.0


def rendezvous_order(prefix: int, n: int) -> List[int]:
    """Replicas ranked for a prefix by a fixed pseudo-random hash (identical at every router)."""
    return sorted(range(n), key=lambda s: random.Random(prefix * 1_000_003 + s).random())


# ----------------------------------------------------------------------------
# Routers
# ----------------------------------------------------------------------------

class Router:
    """One independent router. Sees (queue lengths, cached-prefix sets) from the shared snapshot."""

    def __init__(self, n: int, rng: random.Random, np_rng: np.random.Generator, latency_scale: float,
                 local_correction: bool = False, **_):
        self.n, self.rng, self.np_rng = n, rng, np_rng
        self.latency_scale = latency_scale
        self.local_correction = local_correction
        self._version = -1
        self._sent = [0] * n
        self._sent_prefixes: List[set] = [set() for _ in range(n)]
        self.snapshot_age = 0.0
        self.cost_feature = False
        self._rank: Dict[int, List[int]] = {}

    def ring(self, prefix: int) -> List[int]:
        if prefix not in self._rank:
            self._rank[prefix] = rendezvous_order(prefix, self.n)
        return self._rank[prefix]

    def choose(self, t: float, prefix: int, q: List[int], caches: Sequence, version: int, blocks=None):
        """With `blocks` (trace mode) a request is a chain of KV blocks and `hit` is the cached
        fraction of the leading run (0 unless more than the universal first block matches);
        otherwise `prefix` is a single prefix id and `hit` is membership."""
        if blocks is None:
            hit = [prefix in c for c in caches]
        else:
            hit = [_lead_fraction(blocks, c.__contains__) for c in caches]
        if self.local_correction:
            if version != self._version:
                self._version = version
                self._sent = [0] * self.n
                self._sent_prefixes = [set() for _ in range(self.n)]
            q = [qi + si for qi, si in zip(q, self._sent)]
            if blocks is None:
                hit = [h or (prefix in sp) for h, sp in zip(hit, self._sent_prefixes)]
            else:
                hit = [max(h, _lead_fraction(blocks, sp.__contains__)) for h, sp in zip(hit, self._sent_prefixes)]
        self._req_blocks = len(blocks) if blocks is not None else 0
        s = self.pick(t, prefix, q, hit)
        if self.local_correction:
            self._sent[s] += 1
            self._sent_prefixes[s].update(blocks) if blocks is not None else self._sent_prefixes[s].add(prefix)
        x = [1.0, float(q[s]), float(hit[s])]
        if self.cost_feature and blocks is not None:
            x.append((1.0 - float(hit[s])) * len(blocks) / MEAN_BLOCKS)   # uncached share of this request
        return s, x

    def pick(self, t, prefix, q, hit) -> int:
        raise NotImplementedError

    def observe(self, server: int, latency: float, x: List[float], t: float) -> None:
        pass

    def _argmin(self, idx: Sequence[int], key) -> int:
        best = min(key(i) for i in idx)
        ties = [i for i in idx if key(i) == best]
        return ties[0] if len(ties) == 1 else self.rng.choice(ties)


class JSQ(Router):
    """Cache-oblivious join-shortest-queue."""
    def pick(self, t, prefix, q, hit):
        return self._argmin(range(self.n), lambda i: q[i])


class P2C(Router):
    """Cache-oblivious power-of-two choices."""
    def pick(self, t, prefix, q, hit):
        a, b = self.rng.sample(range(self.n), 2)
        return a if q[a] < q[b] else b if q[b] < q[a] else self.rng.choice((a, b))


class Hash(Router):
    """Static prefix -> replica (consistent / rendezvous hashing): perfect locality, no load awareness."""
    def pick(self, t, prefix, q, hit):
        return self.ring(prefix)[0]


class Affinity(Router):
    """
    Prefix-cache-aware routing as in common LLM routers: among replicas that
    (per the snapshot) hold the prefix, take the least loaded; if none holds it
    use the least loaded overall. With threshold T, fall back to the least
    loaded replica when the holder's queue exceeds the minimum by more than T.
    """
    def __init__(self, *a, threshold: Optional[int] = None, **kw):
        super().__init__(*a, **kw)
        self.threshold = threshold

    def pick(self, t, prefix, q, hit):
        overall = self._argmin(range(self.n), lambda i: q[i])
        holders = [i for i in range(self.n) if hit[i]]
        if not holders:
            return overall
        best = self._argmin(holders, lambda i: q[i])
        if self.threshold is not None and q[best] > q[overall] + self.threshold:
            return overall
        return best


class LearnedBase(Router):
    """
    Bayesian latency model on features [1, queue, cache_hit]. With pooled=True
    (default; replicas are homogeneous here) one model is shared by all
    replicas, so it learns n times faster from the small share of traffic each
    router sees; pooled=False keeps one model per replica.
    """

    def __init__(self, *a, gamma: float = 0.995, explore: float = 1.0, d: int = 2, pooled: bool = True,
                 cost_feature: bool = False, **kw):
        super().__init__(*a, **kw)
        self.pooled = pooled
        self.cost_feature = cost_feature
        self.dim = 4 if cost_feature else 3
        self._req_blocks = 0
        self.model = LatencyModel(1 if pooled else self.n, gamma=gamma, dim=self.dim)
        self.explore, self.d = explore, d

    def features(self, q, hit) -> np.ndarray:
        rows = [[1.0, float(q[i]), float(hit[i])] for i in range(self.n)]
        if self.cost_feature:
            rows = [r + [(1.0 - r[2]) * self._req_blocks / MEAN_BLOCKS] for r in rows]
        return np.array(rows)

    def scores(self, q, hit) -> np.ndarray:
        X = self.features(q, hit)
        if self.pooled:
            return X @ self.model.theta[0]
        return self.model.predict_mean(X)

    def sampled_scores(self, q, hit) -> np.ndarray:
        X = self.features(q, hit)
        if self.pooled:
            theta = self.model.theta[0] + self.model.chol[0] @ (self.np_rng.standard_normal(self.dim) * self.explore)
            return X @ theta
        return self.model.predict_sample(X, self.np_rng, self.explore)

    def observe(self, server, latency, x, t):
        self.model.update(0 if self.pooled else server, x, latency / self.latency_scale)


class GreedyLearned(LearnedBase):
    def pick(self, t, prefix, q, hit):
        scores = self.scores(q, hit).tolist()
        return self._argmin(range(self.n), lambda i: scores[i])


class TSLearned(LearnedBase):
    def pick(self, t, prefix, q, hit):
        return int(np.argmin(self.sampled_scores(q, hit)))


class PdLearned(LearnedBase):
    """Power-of-d: d candidates sampled uniformly from all replicas, ranked by the learned score."""
    def pick(self, t, prefix, q, hit):
        cands = self.rng.sample(range(self.n), min(self.d, self.n))
        scores = self.scores(q, hit)
        return self._argmin(cands, lambda i: scores[i])


class RingLearned(LearnedBase):
    """
    Power-of-d restricted to the prefix's d rendezvous-hash replicas: locality is
    structural (a prefix only ever lives on d replicas), the learned score picks
    among them.
    """
    def pick(self, t, prefix, q, hit):
        cands = self.ring(prefix)[:self.d]
        scores = self.scores(q, hit)
        return self._argmin(cands, lambda i: scores[i])


class AffinityPd(LearnedBase):
    """
    Power-of-d with locality-aware candidates: one candidate drawn uniformly from
    the replicas that (per the snapshot) hold the prefix, if any, plus d-1
    drawn uniformly from all replicas; ranked by the learned score. Any
    replica is a candidate with probability at most 1/|holders| + (d-1)/N.
    """
    def pick(self, t, prefix, q, hit):
        holders = [i for i in range(self.n) if hit[i]]
        cands = set()
        if holders:
            cands.add(self.rng.choice(holders))
        while len(cands) < min(self.d, self.n):
            cands.add(self.rng.randrange(self.n))
        scores = self.scores(q, hit)
        return self._argmin(sorted(cands), lambda i: scores[i])


ROUTERS = {
    "jsq": (JSQ, {}),
    "p2c": (P2C, {}),
    "hash": (Hash, {}),
    "affinity": (Affinity, {}),
    **{f"affinity_thr{t}": (Affinity, {"threshold": t}) for t in (1, 2, 4, 8, 16, 32)},
    "greedy_learned": (GreedyLearned, {}),
    "ts_learned": (TSLearned, {}),
    "p2c_learned": (PdLearned, {"d": 2}),
    "p3c_learned": (PdLearned, {"d": 3}),
    "ring2_learned": (RingLearned, {"d": 2}),
    "ring3_learned": (RingLearned, {"d": 3}),
    "affinity_p2c_learned": (AffinityPd, {"d": 2}),
    # trace mode only: the learned score also sees this request's uncached share
    "greedy_learned_c": (GreedyLearned, {"cost_feature": True}),
    "ring2_learned_c": (RingLearned, {"d": 2, "cost_feature": True}),
    "ring3_learned_c": (RingLearned, {"d": 3, "cost_feature": True}),
    "affinity_p2c_learned_c": (AffinityPd, {"d": 2, "cost_feature": True}),
    "affinity_p3c_learned": (AffinityPd, {"d": 3}),
}
THRESHOLDS = [1, 2, 4, 8, 16, 32]
LLM_POLICIES = [p for p in ROUTERS if not p.endswith("_c")] + [p + "+lc" for p in ["affinity", "greedy_learned", "affinity_p2c_learned", "ring2_learned", "ring3_learned"]]


def make_router(name: str, **kw) -> Router:
    base, _, suffix = name.partition("+")
    if base not in ROUTERS or suffix not in ("", "lc"):
        raise ValueError(f"Unknown LLM policy {name!r}")
    cls, fixed = ROUTERS[base]
    return cls(local_correction=(suffix == "lc"), **{**kw, **fixed})


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------

class _Replica:
    """Logical LRU prefix cache with a history of its *membership* for snapshots."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.lru: "OrderedDict[int, None]" = OrderedDict()
        self.times: List[float] = [0.0]
        self.sets: List[frozenset] = [frozenset()]
        self.arrivals: List[float] = []
        self.departures: List[float] = []
        self.last_departure = 0.0

    def touch(self, prefix: int, t: float) -> bool:
        """Registers an arrival; returns True on a cache hit."""
        if prefix in self.lru:
            self.lru.move_to_end(prefix)
            return True
        self.lru[prefix] = None
        if len(self.lru) > self.capacity:
            self.lru.popitem(last=False)
        self.times.append(t)
        self.sets.append(frozenset(self.lru))
        return False

    def cached_at(self, t: float) -> frozenset:
        return self.sets[bisect_right(self.times, t) - 1]

    def queue_at(self, t: float) -> int:
        return bisect_right(self.arrivals, t) - bisect_right(self.departures, t)


def simulate_llm(sc: LLMScenario, policy: str, seed: int = 0, policy_kwargs: Optional[Dict] = None) -> Dict:
    policy_kwargs = policy_kwargs or {}
    rng = np.random.default_rng(seed)
    n = sc.n

    pref_len = prefix_lengths(sc, np.random.default_rng([seed, 17]))
    probs = prefix_probabilities(sc)
    n_req = rng.poisson(sc.rate * sc.duration)
    arrivals = np.sort(rng.uniform(0, sc.duration, n_req)).tolist()
    prefix_of = rng.choice(sc.num_prefixes, n_req, p=probs).tolist()
    sigma2 = math.log(1.0 + sc.suffix_cv ** 2)
    suffix = rng.lognormal(math.log(sc.suffix_mean_tokens) - sigma2 / 2, math.sqrt(sigma2), n_req).tolist()
    dispatcher_of = rng.integers(0, sc.num_dispatchers, n_req).tolist()
    phase_rng = np.random.default_rng([seed, 7919])
    phases = (phase_rng.uniform(0, sc.staleness, sc.num_dispatchers).tolist()
              if sc.desync and sc.staleness > 0 else [0.0] * sc.num_dispatchers)

    latency_scale = sc.overhead_sec + sc.suffix_mean_tokens * sc.prefill_sec_per_token
    routers = [make_router(policy, n=n, rng=random.Random(seed * 100_003 + d),
                           np_rng=np.random.default_rng([seed, d]), latency_scale=latency_scale, **policy_kwargs)
               for d in range(sc.num_dispatchers)]
    replicas = [_Replica(sc.cache_size) for _ in range(n)]

    snap_time: List[Optional[float]] = [None] * sc.num_dispatchers
    snap = [None] * sc.num_dispatchers
    version = [0] * sc.num_dispatchers
    shared: Dict[float, Tuple[List[int], List[frozenset]]] = {}
    feedback: list = []

    ttft, hits, bins, servers, times = [], [], [], [], []

    for j in range(n_req):
        t = arrivals[j]
        while feedback and feedback[0][0] <= t:
            dep, _, d, s, lat, x = heapq.heappop(feedback)
            routers[d].observe(s, lat, x, dep)

        d = dispatcher_of[j]
        if sc.staleness > 0:
            tk = math.floor((t - phases[d]) / sc.staleness) * sc.staleness + phases[d]
            if tk != snap_time[d]:
                state = shared.get(tk)
                if state is None:
                    state = ([r.queue_at(tk) for r in replicas], [r.cached_at(tk) for r in replicas])
                    if not sc.desync:
                        shared = {tk: state}
                snap[d], snap_time[d] = state, tk
                version[d] += 1
            routers[d].snapshot_age = t - tk
        else:
            snap[d] = ([r.queue_at(t) for r in replicas], [r.cached_at(t) for r in replicas])
            version[d] += 1
            routers[d].snapshot_age = 0.0

        prefix = prefix_of[j]
        s, x = routers[d].choose(t, prefix, snap[d][0], snap[d][1], version[d])

        rep = replicas[s]
        live_q = rep.queue_at(t)
        hit = rep.touch(prefix, t)
        tokens = suffix[j] + (0.0 if hit else pref_len[prefix])
        per_token = sc.prefill_sec_per_token
        start = max(t, rep.last_departure)
        if sc.slow_replica == s and start >= sc.slow_start:
            per_token /= sc.slow_factor
        dep = start + sc.overhead_sec + tokens * per_token
        rep.last_departure = dep
        rep.arrivals.append(t)
        rep.departures.append(dep)
        lat = dep - t
        heapq.heappush(feedback, (dep, j, d, s, lat, x))

        if t >= sc.warmup:
            ttft.append(lat)
            hits.append(hit)
            bins.append(int(t / BIN_SEC))
            servers.append(s)
            times.append(t)

    row = _summarize(sc, policy, seed, ttft, hits, bins, servers)
    # Windows need enough arrivals for the share to mean anything: at least 0.25 s here.
    window = max(sc.staleness, 0.25)
    row["window_share_mean"], row["window_share_p95"] = window_burst_stats(times, servers, n, window)
    return row


def _summarize(sc, policy, seed, ttft, hits, bins, servers) -> Dict:
    n = sc.n
    lat = np.asarray(ttft) * 1000.0
    b = np.asarray(bins)
    b -= b.min()
    counts = np.bincount(b * n + np.asarray(servers), minlength=(b.max() + 1) * n).reshape(-1, n)
    mean = counts.mean(axis=0)
    fano = np.where(mean > 0, counts.var(axis=0) / np.maximum(mean, 1e-9), np.nan)
    share = np.bincount(servers, minlength=n) / len(servers)
    row = sc.to_row()
    row.update({
        "policy": policy, "seed": seed, "requests": int(len(lat)),
        "mean_ms": float(lat.mean()), "p50_ms": float(np.percentile(lat, 50)),
        "p99_ms": float(np.percentile(lat, 99)), "p999_ms": float(np.percentile(lat, 99.9)),
        "hit_rate": float(np.mean(hits)), "fano": float(np.nanmean(fano)),
        "max_share": float(share.max()),
    })
    return row
