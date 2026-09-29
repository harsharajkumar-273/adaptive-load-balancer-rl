"""
Dispatcher policies for the multi-dispatcher simulator.

Every dispatcher is an independent instance (one per gateway replica). It sees:
  * the shared, possibly stale, queue-length snapshot, and
  * the latencies of requests *it* routed, when they complete.

Heuristic baselines
  wrandom      capacity-weighted random (oblivious; uses nominal capacities)
  jsq          join-shortest-queue on the shared snapshot
  sed          shortest-expected-delay: argmin (q+1)/mu_nominal on the snapshot
  p2c          power-of-two-choices on the snapshot

Learned policies (per-dispatcher Bayesian linear model of latency per server,
latency ~ a_s + b_s * q_s, discounted so it tracks non-stationarity)
  greedy       argmin of posterior-mean predicted latency
  ts           Thompson sampling: argmin of a posterior sample, drawn per request
  ts_epoch     posterior sample + softmax weights recomputed every control
               interval, then weighted-random routing in between. This mirrors
               the production agent in src/agent.py.
  p2c_learned  sample two servers, pick the lower posterior-mean prediction

Probing baseline (ignores the shared snapshot)
  prequal      Prequal-style (Wydrowski et al., NSDI 2024): each query triggers
               r_probe probes of random servers returning live requests-in-flight
               (RIF) and the server's latency estimate; responses are pooled
               with a reuse budget and a maximum age; selection uses the
               hot-cold lexicographic (HCL) rule with RIF quantile q_rif.

Any policy name suffixed with "+lc" adds local correction: the dispatcher adds
the requests it has itself sent to each server since the current snapshot was
published, which the shared snapshot cannot yet reflect.
"""
from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np

from src.learned_routing import LatencyModel  # shared with the real gateway

HEURISTICS = ["wrandom", "jsq", "sed", "p2c"]
LEARNED = ["greedy", "ts", "ts_epoch", "p2c_learned"]
PROBING = ["prequal", "p2c_probe", "p3c_probe", "sed3_probe"]
# Named variants: (base policy, fixed kwargs).
VARIANTS = {
    "p2c_probe": ("probe_d", {"d": 2}),
    "p3c_probe": ("probe_d", {"d": 3}),
    "sed3_probe": ("probe_d", {"d": 3, "score": "sed"}),
    "p3c_learned": ("p2c_learned", {"d": 3}),
    "p2c_learned_probe": ("p2c_learned", {"probe": True}),
    "p3c_learned_probe": ("p2c_learned", {"d": 3, "probe": True}),
}
ALL_POLICIES = (HEURISTICS + LEARNED + PROBING + list(VARIANTS)
                + [p + "+lc" for p in ["jsq", "greedy", "ts", "p2c_learned"]])


class Dispatcher:
    # Extra latency this dispatcher adds to every request it routes (e.g. a
    # synchronous probe round-trip). Charged to the request by the simulator.
    added_delay = 0.0

    def __init__(self, n: int, nominal_mus: Sequence[float], rng: random.Random,
                 local_correction: bool = False, **_):
        self.n = n
        self.nominal_mus = list(nominal_mus)
        self.rng = rng
        self.local_correction = local_correction
        self._version = -1
        self._sent_since_snapshot = [0] * n

    def effective_queues(self, snapshot: List[int], version: int) -> List[int]:
        if not self.local_correction:
            return snapshot
        if version != self._version:
            self._version = version
            self._sent_since_snapshot = [0] * self.n
        return [q + c for q, c in zip(snapshot, self._sent_since_snapshot)]

    def choose(self, t: float, snapshot: List[int], version: int) -> Tuple[int, float]:
        q = self.effective_queues(snapshot, version)
        s = self.pick(t, q)
        if self.local_correction:
            self._sent_since_snapshot[s] += 1
        return s, float(q[s])

    def pick(self, t: float, q: List[int]) -> int:
        raise NotImplementedError

    def observe(self, server: int, latency: float, feature: float, t: float) -> None:
        pass

    def _argmin_random_tie(self, values) -> int:
        best = min(values)
        ties = [i for i, v in enumerate(values) if v == best]
        return ties[0] if len(ties) == 1 else self.rng.choice(ties)


class WeightedRandom(Dispatcher):
    def pick(self, t, q):
        return self.rng.choices(range(self.n), weights=self.nominal_mus, k=1)[0]


class JSQ(Dispatcher):
    def pick(self, t, q):
        return self._argmin_random_tie(q)


class SED(Dispatcher):
    def pick(self, t, q):
        return self._argmin_random_tie([(qi + 1) / mu for qi, mu in zip(q, self.nominal_mus)])


class ProbeD(Dispatcher):
    """
    Classic power-of-d with live probes (no learning): sample d servers, read
    their live queue lengths (d synchronous probes), pick the shortest queue
    (score="queue") or the shortest expected delay with nominal speeds
    (score="sed"). The non-learned control for learned power-of-d probing.
    """

    def __init__(self, n, nominal_mus, rng, view=None, d: int = 2, score: str = "queue",
                 probe_rtt: float = 0.0005, **kw):
        super().__init__(n, nominal_mus, rng, **kw)
        if view is None:
            raise ValueError("probing policies need a ClusterView")
        self.view, self.d, self.score = view, d, score
        self.added_delay = probe_rtt

    def choose(self, t, snapshot, version):
        cands = self.rng.sample(range(self.n), min(self.d, self.n))
        live = {s: self.view.probe(s, t)[0] for s in cands}
        if self.score == "sed":
            key = lambda s: ((live[s] + 1) / self.nominal_mus[s], self.rng.random())
        else:
            key = lambda s: (live[s], self.rng.random())
        s = min(cands, key=key)
        return s, float(live[s])

    def pick(self, t, q):  # not used: choose() is overridden
        raise NotImplementedError


class P2C(Dispatcher):
    def pick(self, t, q):
        a, b = self.rng.sample(range(self.n), 2)
        if q[a] == q[b]:
            return self.rng.choice((a, b))
        return a if q[a] < q[b] else b


class Learned(Dispatcher):
    def __init__(self, n, nominal_mus, rng, latency_scale: float, np_rng: np.random.Generator,
                 gamma: float = 0.995, sigma2: float = 1.0, explore: float = 1.0,
                 control_interval: float = 0.15, temperature: float = 5.0, **kw):
        # temperature=5 was the best of {0.5, 2, 5, 20} in a small sweep (see research/README.md).
        super().__init__(n, nominal_mus, rng, **kw)
        self.model = LatencyModel(n, sigma2=sigma2, gamma=gamma)
        self.latency_scale = latency_scale
        self.np_rng = np_rng
        self.explore = explore
        self.control_interval = control_interval
        self.temperature = temperature
        self._epoch_end = -1.0
        self._weights = [1.0] * n

    def observe(self, server, latency, feature, t):
        self.model.update(server, feature, latency / self.latency_scale)


class Greedy(Learned):
    def pick(self, t, q):
        return self._argmin_random_tie(self.model.predict_mean(q).tolist())


class ThompsonSampling(Learned):
    def pick(self, t, q):
        return int(np.argmin(self.model.predict_sample(q, self.np_rng, self.explore)))


class ThompsonEpoch(Learned):
    def pick(self, t, q):
        if t >= self._epoch_end:
            scores = -self.model.predict_sample(q, self.np_rng, self.explore) / self.temperature
            w = np.exp(scores - scores.max())
            self._weights = (w / w.sum()).tolist()
            self._epoch_end = t + self.control_interval
        return self.rng.choices(range(self.n), weights=self._weights, k=1)[0]


class P2CLearned(Learned):
    """
    Power-of-d choices over posterior-mean predicted latency (d=2 by default).
    With probe=True the d candidates' queue lengths are read live (d probes per
    request) instead of from the shared snapshot. These probes are synchronous,
    so one probe round-trip is added to the request's latency (Prequal's probes
    are asynchronous and add none).
    """

    def __init__(self, *args, d: int = 2, probe: bool = False, view=None, probe_rtt: float = 0.0005, **kw):
        super().__init__(*args, **kw)
        self.d = d
        self.probe = probe
        self.view = view
        self.added_delay = probe_rtt if probe else 0.0
        if probe and view is None:
            raise ValueError("probe=True needs a ClusterView")

    def choose(self, t, snapshot, version):
        if not self.probe:
            return super().choose(t, snapshot, version)
        th = self.model.theta
        cands = self.rng.sample(range(self.n), min(self.d, self.n))
        live = {s: self.view.probe(s, t)[0] for s in cands}
        s = min(cands, key=lambda c: (th[c, 0] + th[c, 1] * live[c], self.rng.random()))
        return s, float(live[s])

    def pick(self, t, q):
        th = self.model.theta
        cands = self.rng.sample(range(self.n), min(self.d, self.n))
        return min(cands, key=lambda s: (th[s, 0] + th[s, 1] * q[s], self.rng.random()))


class Prequal(Dispatcher):
    """
    Prequal-style probing dispatcher. q_rif and r_probe are within the ranges
    reported by the paper (q_rif 0.75-0.99, r_probe > 1 safe). reuse_budget=1
    was the best setting in our sweep (research/README.md), i.e. the baseline is
    tuned in its favour; the sensitivity experiment reports the others.
    """

    def __init__(self, n, nominal_mus, rng, view=None, r_probe: float = 3.0, pool_size: int = 16,
                 reuse_budget: int = 1, max_age: float = 0.5, q_rif: float = 0.84,
                 probe_rtt: float = 0.0005, **kw):
        super().__init__(n, nominal_mus, rng, **kw)
        if view is None:
            raise ValueError("prequal needs a ClusterView to probe")
        self.view = view
        self.r_probe = r_probe
        self.pool_size = pool_size
        self.reuse_budget = reuse_budget
        self.max_age = max_age
        self.q_rif = q_rif
        self.probe_rtt = probe_rtt
        self._pending = []   # [ready_time, server, rif, latency]
        self._pool = []      # [probe_time, server, rif, latency, uses]

    def _issue_probes(self, t):
        k = int(self.r_probe) + (1 if self.rng.random() < self.r_probe % 1 else 0)
        for s in self.rng.sample(range(self.n), min(k, self.n)):
            rif, lat = self.view.probe(s, t)
            self._pending.append([t + self.probe_rtt, s, rif, lat])

    def _refresh_pool(self, t):
        still = []
        for ready, s, rif, lat in self._pending:
            if ready <= t:
                # A newer probe of the same server replaces the old one.
                self._pool = [e for e in self._pool if e[1] != s]
                self._pool.append([ready, s, rif, lat, 0])
            else:
                still.append([ready, s, rif, lat])
        self._pending = still
        self._pool = [e for e in self._pool if t - e[0] <= self.max_age and e[4] < self.reuse_budget]
        if len(self._pool) > self.pool_size:
            self._pool.sort(key=lambda e: e[0])
            self._pool = self._pool[-self.pool_size:]

    def choose(self, t, snapshot, version):
        self._issue_probes(t)
        self._refresh_pool(t)
        if not self._pool:
            s = self.rng.randrange(self.n)
            return s, 0.0
        rifs = sorted(e[2] for e in self._pool)
        threshold = rifs[min(len(rifs) - 1, int(self.q_rif * len(rifs)))]
        cold = [e for e in self._pool if e[2] <= threshold]
        if cold:
            entry = min(cold, key=lambda e: (e[3], self.rng.random()))
        else:
            entry = min(self._pool, key=lambda e: (e[2], self.rng.random()))
        entry[2] += 1   # account for the request we are about to send
        entry[4] += 1
        return entry[1], float(entry[2])

    def pick(self, t, q):  # not used: choose() is overridden
        raise NotImplementedError


_REGISTRY = {
    "wrandom": WeightedRandom,
    "jsq": JSQ,
    "sed": SED,
    "p2c": P2C,
    "greedy": Greedy,
    "ts": ThompsonSampling,
    "ts_epoch": ThompsonEpoch,
    "p2c_learned": P2CLearned,
    "prequal": Prequal,
    "probe_d": ProbeD,
}


def make_policy(name: str, **kwargs) -> Dispatcher:
    base, _, suffix = name.partition("+")
    if base in VARIANTS:
        base, fixed = VARIANTS[base]
        kwargs = {**kwargs, **fixed}
    if base not in _REGISTRY or suffix not in ("", "lc"):
        raise ValueError(f"Unknown policy {name!r}; choose from {ALL_POLICIES}")
    cls = _REGISTRY[base]
    if not issubclass(cls, Learned):
        for k in ("latency_scale", "np_rng", "gamma", "sigma2", "explore", "control_interval", "temperature",
                  *(("d", "probe") if cls is not ProbeD else ())):
            kwargs.pop(k, None)
    if cls is ProbeD:
        kwargs.pop("probe", None)
    elif "score" in kwargs:
        raise ValueError(f"'score' only applies to probe_d, not {name!r}")
    if cls not in (Prequal, P2CLearned, ProbeD):
        kwargs.pop("view", None)
    if cls in (Prequal, ProbeD) and suffix:
        raise ValueError(f"local correction does not apply to {base}")
    return cls(local_correction=(suffix == "lc"), **kwargs)
