# src/probing.py
"""
Probing-based routing for the gateway.

Backends expose GET /probe -> {"rif": requests in flight,
"latency_est": expected latency at that RIF (seconds)}.

PROBING_STRATEGIES
  prequal            Prequal-style (Wydrowski et al., NSDI 2024): every request
                     fires r_probe *asynchronous* probes to random backends;
                     responses are pooled (max size, reuse budget, max age) and
                     the hot-cold lexicographic (HCL) rule picks the backend:
                     among "cold" entries (RIF <= q_rif quantile of the pool)
                     take the lowest latency estimate, else the lowest RIF.
  p2c_probe          classic power-of-d over d synchronous live probes
  p3c_probe            (shortest RIF)
  sed3_probe         power-of-3, shortest expected delay with nominal rates
  p2c_learned_probe  learned power-of-d over d synchronous live probes
  p3c_learned_probe    (lowest predicted latency from the gateway's model)

This mirrors research/policies.py (Prequal, ProbeD, P2CLearned(probe=True)).
"""
import asyncio
import random
from typing import Dict, List, Optional, Tuple

import httpx

PROBING_STRATEGIES = ["prequal", "p2c_probe", "p3c_probe", "sed3_probe",
                      "p2c_learned_probe", "p3c_learned_probe"]

# strategy -> (d, scorer) for synchronous power-of-d probing
POWER_OF_D_PROBING = {
    "p2c_probe": (2, "rif"),
    "p3c_probe": (3, "rif"),
    "sed3_probe": (3, "sed"),
    "p2c_learned_probe": (2, "learned"),
    "p3c_learned_probe": (3, "learned"),
}


async def probe(client: httpx.AsyncClient, url: str, timeout: float) -> Optional[Tuple[int, float]]:
    try:
        resp = await client.get(f"{url}/probe", timeout=timeout)
        data = resp.json()
        return int(data["rif"]), float(data["latency_est"])
    except (httpx.HTTPError, KeyError, ValueError):
        return None


class PrequalPool:
    """Probe pool and HCL selection (one per gateway)."""

    def __init__(self, n_backends: int, rng: random.Random, r_probe: float = 3.0, pool_size: int = 16,
                 reuse_budget: int = 1, max_age: float = 5.0, q_rif: float = 0.84):
        self.n = n_backends
        self.rng = rng
        self.r_probe = r_probe
        self.pool_size = pool_size
        self.reuse_budget = reuse_budget
        self.max_age = max_age
        self.q_rif = q_rif
        self._pool: List[list] = []   # [probe_time, server, rif, latency, uses]

    def probe_targets(self, n_backends: int) -> List[int]:
        self.n = n_backends
        k = int(self.r_probe) + (1 if self.rng.random() < self.r_probe % 1 else 0)
        return self.rng.sample(range(self.n), min(k, self.n))

    def add(self, t: float, server: int, rif: int, latency: float) -> None:
        # A newer probe of the same backend replaces the old one.
        self._pool = [e for e in self._pool if e[1] != server]
        self._pool.append([t, server, rif, latency, 0])
        if len(self._pool) > self.pool_size:
            self._pool.sort(key=lambda e: e[0])
            self._pool = self._pool[-self.pool_size:]

    def select(self, t: float) -> Optional[Tuple[int, int]]:
        self._pool = [e for e in self._pool if t - e[0] <= self.max_age and e[4] < self.reuse_budget]
        if not self._pool:
            return None
        rifs = sorted(e[2] for e in self._pool)
        threshold = rifs[min(len(rifs) - 1, int(self.q_rif * len(rifs)))]
        cold = [e for e in self._pool if e[2] <= threshold]
        if cold:
            entry = min(cold, key=lambda e: (e[3], self.rng.random()))
        else:
            entry = min(self._pool, key=lambda e: (e[2], self.rng.random()))
        entry[2] += 1   # account for the request about to be sent
        entry[4] += 1
        return entry[1], entry[2]

    def spawn_probes(self, client: httpx.AsyncClient, urls: Dict[int, str], timeout: float,
                     now_fn, n_backends: int) -> None:
        """Fire-and-forget probes; responses land in the pool when they arrive."""
        for s in self.probe_targets(n_backends):
            asyncio.ensure_future(self._probe_into_pool(client, s, urls[s], timeout, now_fn))

    async def _probe_into_pool(self, client, s, url, timeout, now_fn):
        result = await probe(client, url, timeout)
        if result is not None:
            self.add(now_fn(), s, result[0], result[1])
