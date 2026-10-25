"""
Trace-driven LLM-serving simulation: replays a real request trace (Mooncake,
FAST'25: Kimi production traffic; https://github.com/kvcache-ai/Mooncake) through
the routers of research/llm_sim.py.

A trace request is a chain of 512-token KV blocks (`hash_ids`; equal ids mean
equal content). Each replica keeps an LRU cache of blocks; a request reuses the
*leading* run of blocks the replica holds and prefills the rest. Routers see
queue lengths and cached-block membership from a snapshot refreshed every
`staleness` seconds, exactly as in llm_sim.py. The first block of every request
is a universal system prompt and is ignored when judging locality.

What is real: arrival timestamps (compressed by `speedup`), input lengths and
block-sharing structure. What is modelled: replicas (FIFO prefill workers at a
fixed token rate; no decode), the routers' snapshot, and the load level.
"""
from __future__ import annotations

import heapq
import json
import math
import os
import random
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from research.llm_sim import BIN_SEC, _summarize, make_router
from research.sim import window_burst_stats

BLOCK_TOKENS = 512
DATA = os.path.join(os.path.dirname(__file__), "data")
TRACES = {"toolagent": "mooncake_toolagent.jsonl", "conversation": "mooncake_conversation.jsonl",
          "synthetic": "mooncake_synthetic.jsonl"}


@dataclass(frozen=True)
class TraceScenario:
    trace: str = "toolagent"
    n: int = 8
    speedup: float = 3.0            # trace time is divided by this (raises load)
    max_requests: int = 12000
    warmup_frac: float = 0.05
    staleness: float = 0.05
    num_dispatchers: int = 16
    desync: bool = False
    cache_blocks: int = 2000        # KV blocks per replica (LRU)
    prefill_tokens_per_sec: float = 20000.0
    overhead_sec: float = 0.010
    slow_replica: Optional[int] = None
    slow_factor: float = 0.25
    ring_depth: int = 1             # ring key = blocks[ring_depth] (clamped); deeper = finer-grained keys
    load_signal: str = "count"      # "count" = queued requests, "work" = queued prefill work (mean-request units)
    cost_feature: bool = False      # learned routers also see this request's uncached share
    slow_frac: float = 0.4          # gray failure starts at this fraction of the run

    def to_row(self) -> Dict:
        row = asdict(self)
        row["scenario"] = "trace_gray" if self.slow_replica is not None else "trace"
        return row


@lru_cache(maxsize=None)
def load_trace(name: str, max_requests: int):
    reqs = []
    with open(os.path.join(DATA, TRACES[name])) as f:
        for line in f:
            r = json.loads(line)
            reqs.append((r["timestamp"] / 1000.0, int(r["input_length"]), tuple(r["hash_ids"])))
            if len(reqs) >= max_requests:
                break
    return reqs


class _View:
    """Cache membership of one replica as of time `t` (lazy: no copying)."""
    __slots__ = ("rep", "t")

    def __init__(self, rep, t):
        self.rep, self.t = rep, t

    def __contains__(self, block) -> bool:
        tg = self.rep.toggles.get(block)
        return tg is not None and bisect_right(tg, self.t) % 2 == 1


class _Replica:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.lru: "OrderedDict[int, None]" = OrderedDict()
        self.toggles: Dict[int, List[float]] = {}
        self.arrivals: List[float] = []
        self.departures: List[float] = []
        self.last_departure = 0.0

    def lead(self, blocks) -> int:
        m = 0
        for b in blocks:
            if b not in self.lru:
                break
            m += 1
        return m

    def insert(self, blocks, t: float) -> None:
        for b in blocks:
            if b in self.lru:
                self.lru.move_to_end(b)
            else:
                self.lru[b] = None
                self.toggles.setdefault(b, []).append(t)
        while len(self.lru) > self.capacity:
            old, _ = self.lru.popitem(last=False)
            self.toggles[old].append(t)

    def queue_at(self, t: float) -> int:
        return bisect_right(self.arrivals, t) - bisect_right(self.departures, t)

    def backlog_at(self, t: float) -> float:
        """Seconds until the replica would be idle if nothing else arrived (FIFO)."""
        i = bisect_right(self.arrivals, t)
        return max(0.0, self.departures[i - 1] - t) if i else 0.0


def ring_key(blocks, depth: int = 1) -> int:
    """Rendezvous key: the block at `depth` (default: first block after the universal system prompt)."""
    return blocks[min(depth, len(blocks) - 1)]


def simulate_trace(sc: TraceScenario, policy: str, seed: int = 0, policy_kwargs: Optional[Dict] = None) -> Dict:
    policy_kwargs = {**(policy_kwargs or {}), **({"cost_feature": True} if sc.cost_feature else {})}
    reqs = load_trace(sc.trace, sc.max_requests)
    n_req = len(reqs)
    rng = np.random.default_rng(seed)
    arrivals = [r[0] / sc.speedup for r in reqs]
    horizon = arrivals[-1]
    warmup = horizon * sc.warmup_frac
    dispatcher_of = rng.integers(0, sc.num_dispatchers, n_req).tolist()
    phase_rng = np.random.default_rng([seed, 7919])
    phases = (phase_rng.uniform(0, sc.staleness, sc.num_dispatchers).tolist()
              if sc.desync and sc.staleness > 0 else [0.0] * sc.num_dispatchers)

    mean_tokens = float(np.mean([r[1] for r in reqs]))
    latency_scale = sc.overhead_sec + 0.25 * mean_tokens / sc.prefill_tokens_per_sec
    n = sc.n
    routers = [make_router(policy, n=n, rng=random.Random(seed * 100_003 + d),
                           np_rng=np.random.default_rng([seed, d]), latency_scale=latency_scale, **policy_kwargs)
               for d in range(sc.num_dispatchers)]
    replicas = [_Replica(sc.cache_blocks) for _ in range(n)]
    per_token = 1.0 / sc.prefill_tokens_per_sec
    slow_start = horizon * sc.slow_frac
    unit = sc.overhead_sec + mean_tokens / sc.prefill_tokens_per_sec    # one mean request, no cache

    def load(r, tt):
        return r.queue_at(tt) if sc.load_signal == "count" else r.backlog_at(tt) / unit

    snap_time: List[Optional[float]] = [None] * sc.num_dispatchers
    snap = [None] * sc.num_dispatchers
    version = [0] * sc.num_dispatchers
    shared: Dict[float, tuple] = {}
    feedback: list = []
    ttft, hits, bins, servers, times, waits, slows = [], [], [], [], [], [], []

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
                    state = ([load(r, tk) for r in replicas], [_View(r, tk) for r in replicas])
                    if not sc.desync:
                        shared = {tk: state}
                snap[d], snap_time[d] = state, tk
                version[d] += 1
            routers[d].snapshot_age = t - tk
        else:
            snap[d] = ([load(r, t) for r in replicas], [_View(r, t) for r in replicas])
            version[d] += 1
            routers[d].snapshot_age = 0.0

        _, in_len, blocks = reqs[j]
        s, x = routers[d].choose(t, ring_key(blocks, sc.ring_depth), snap[d][0], snap[d][1], version[d], blocks=blocks)

        rep = replicas[s]
        m = rep.lead(blocks)
        rep.insert(blocks, t)
        tokens = max(in_len - m * BLOCK_TOKENS, 64)
        pt = per_token / sc.slow_factor if (sc.slow_replica == s and t >= slow_start) else per_token
        start = max(t, rep.last_departure)
        dep = start + sc.overhead_sec + tokens * pt
        rep.last_departure = dep
        rep.arrivals.append(t)
        rep.departures.append(dep)
        lat = dep - t
        heapq.heappush(feedback, (dep, j, d, s, lat, x))

        if t >= warmup:
            ttft.append(lat)
            hits.append(m * BLOCK_TOKENS / max(in_len, 1))
            bins.append(int(t / BIN_SEC))
            servers.append(s)
            times.append(t)
            waits.append(start - t)
            slows.append(lat / (sc.overhead_sec + in_len * per_token))

    row = _summarize(sc, policy, seed, ttft, hits, bins, servers)
    row["p99_wait_ms"] = float(np.percentile(waits, 99)) * 1000.0
    row["mean_wait_ms"] = float(np.mean(waits)) * 1000.0
    row["p99_slowdown"] = float(np.percentile(slows, 99))
    row["mean_slowdown"] = float(np.mean(slows))
    row["window_share_mean"], row["window_share_p95"] = window_burst_stats(times, servers, n, max(sc.staleness, 0.25))
    return row
