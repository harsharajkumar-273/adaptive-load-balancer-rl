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

Any policy name suffixed with "+lc" adds local correction: the dispatcher adds
the requests it has itself sent to each server since the current snapshot was
published, which the shared snapshot cannot yet reflect.
"""
from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

import numpy as np

HEURISTICS = ["wrandom", "jsq", "sed", "p2c"]
LEARNED = ["greedy", "ts", "ts_epoch", "p2c_learned"]
ALL_POLICIES = HEURISTICS + LEARNED + [p + "+lc" for p in ["jsq", "greedy", "ts"]]


class Dispatcher:
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


class P2C(Dispatcher):
    def pick(self, t, q):
        a, b = self.rng.sample(range(self.n), 2)
        if q[a] == q[b]:
            return self.rng.choice((a, b))
        return a if q[a] < q[b] else b


class LatencyModel:
    """
    Independent discounted Bayesian linear regressions, one per server:
        y = latency / latency_scale ~ theta_s . [1, q]
    Prior N(0, sigma2/lam I). Sufficient statistics are discounted by `gamma`
    on every update of that server so the model tracks drift.
    """

    def __init__(self, n: int, lam: float = 1.0, sigma2: float = 1.0, gamma: float = 0.995):
        self.n, self.lam, self.sigma2, self.gamma = n, lam, sigma2, gamma
        self.A = np.zeros((n, 2, 2))
        self.b = np.zeros((n, 2))
        self.theta = np.zeros((n, 2))
        self.chol = np.tile(np.eye(2) * math.sqrt(sigma2 / lam), (n, 1, 1))

    def update(self, s: int, q: float, y: float) -> None:
        x = np.array([1.0, q])
        self.A[s] = self.gamma * self.A[s] + np.outer(x, x)
        self.b[s] = self.gamma * self.b[s] + y * x
        p = self.A[s] + self.lam * np.eye(2)
        det = p[0, 0] * p[1, 1] - p[0, 1] * p[1, 0]
        inv = np.array([[p[1, 1], -p[0, 1]], [-p[1, 0], p[0, 0]]]) / det
        self.theta[s] = inv @ self.b[s]
        cov = self.sigma2 * inv
        l11 = math.sqrt(cov[0, 0])
        l21 = cov[1, 0] / l11
        l22 = math.sqrt(max(cov[1, 1] - l21 * l21, 1e-12))
        self.chol[s] = [[l11, 0.0], [l21, l22]]

    def predict_mean(self, q: Sequence[float]) -> np.ndarray:
        return self.theta[:, 0] + self.theta[:, 1] * np.asarray(q, dtype=float)

    def predict_sample(self, q: Sequence[float], np_rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        z = np_rng.standard_normal((self.n, 2)) * scale
        th = self.theta + np.einsum("nij,nj->ni", self.chol, z)
        return th[:, 0] + th[:, 1] * np.asarray(q, dtype=float)


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
    def pick(self, t, q):
        a, b = self.rng.sample(range(self.n), 2)
        th = self.model.theta
        pa = th[a, 0] + th[a, 1] * q[a]
        pb = th[b, 0] + th[b, 1] * q[b]
        if pa == pb:
            return self.rng.choice((a, b))
        return a if pa < pb else b


_REGISTRY = {
    "wrandom": WeightedRandom,
    "jsq": JSQ,
    "sed": SED,
    "p2c": P2C,
    "greedy": Greedy,
    "ts": ThompsonSampling,
    "ts_epoch": ThompsonEpoch,
    "p2c_learned": P2CLearned,
}


def make_policy(name: str, **kwargs) -> Dispatcher:
    base, _, suffix = name.partition("+")
    if base not in _REGISTRY or suffix not in ("", "lc"):
        raise ValueError(f"Unknown policy {name!r}; choose from {ALL_POLICIES}")
    cls = _REGISTRY[base]
    if not issubclass(cls, Learned):
        for k in ("latency_scale", "np_rng", "gamma", "sigma2", "explore", "control_interval", "temperature"):
            kwargs.pop(k, None)
    return cls(local_correction=(suffix == "lc"), **kwargs)
