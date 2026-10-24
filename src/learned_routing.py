# src/learned_routing.py
"""
Per-gateway learned latency model and the routing rules built on it.

Each gateway process keeps its own model, trained only on the requests it
proxied: for every backend s,

    latency / latency_scale  ~  theta_s . [1, q]

where q is the queue length the gateway believed the backend had when it
routed the request. This is the same model the simulator in research/ uses.

Strategies
  greedy_learned  send to the backend with the lowest predicted latency
  ts_learned      Thompson sampling: lowest latency under a posterior sample,
                  drawn per request
  p2c_learned     sample two backends, send to the lower predicted latency
                  (the herding-resistant variant; see research/README.md)
  p3c_learned     the same with three sampled backends
"""
import math
import random
import threading
from typing import List, Optional, Sequence

import numpy as np

LEARNED_STRATEGIES = ["greedy_learned", "ts_learned", "p2c_learned", "p3c_learned"]
_SAMPLE_SIZE = {"p2c_learned": 2, "p3c_learned": 3}


class LatencyModel:
    """
    Independent discounted Bayesian linear regressions, one per server:
        y = latency / latency_scale ~ theta_s . x
    with feature vector x = [1, q] (dim=2, the default; q is the queue length) or
    any dim-dimensional vector whose first entry is 1 (e.g. [1, q, cache_hit]).
    Prior N(0, sigma2/lam I). Sufficient statistics are discounted by `gamma`
    on every update of that server so the model tracks drift.

    Feature arguments: for dim=2 a scalar q is accepted everywhere (and a list of
    q's for predict_*); for any dim, predict_* also accept an (n, dim) matrix with
    one feature row per server, and update() accepts a length-dim vector.
    """

    def __init__(self, n: int, lam: float = 1.0, sigma2: float = 1.0, gamma: float = 0.995, dim: int = 2):
        self.n, self.lam, self.sigma2, self.gamma, self.dim = 0, lam, sigma2, gamma, dim
        self.A = np.zeros((0, dim, dim))
        self.b = np.zeros((0, dim))
        self.theta = np.zeros((0, dim))
        self.chol = np.zeros((0, dim, dim))
        self.ensure_size(n)

    def ensure_size(self, n: int) -> None:
        """Adds untrained servers when the fleet grows (existing ones keep their fit)."""
        extra = n - self.n
        if extra <= 0:
            return
        d = self.dim
        self.A = np.concatenate([self.A, np.zeros((extra, d, d))])
        self.b = np.concatenate([self.b, np.zeros((extra, d))])
        self.theta = np.concatenate([self.theta, np.zeros((extra, d))])
        prior_chol = np.eye(d) * math.sqrt(self.sigma2 / self.lam)
        self.chol = np.concatenate([self.chol, np.tile(prior_chol, (extra, 1, 1))])
        self.n = n

    def _row(self, v) -> np.ndarray:
        if self.dim == 2 and np.ndim(v) == 0:
            return np.array([1.0, v])
        return np.asarray(v, dtype=float)

    def update(self, s: int, v, y: float) -> None:
        x = self._row(v)
        self.A[s] = self.gamma * self.A[s] + np.outer(x, x)
        self.b[s] = self.gamma * self.b[s] + y * x
        if self.dim == 2:   # closed form: this runs on every request
            p = self.A[s] + self.lam * np.eye(2)
            det = p[0, 0] * p[1, 1] - p[0, 1] * p[1, 0]
            inv = np.array([[p[1, 1], -p[0, 1]], [-p[1, 0], p[0, 0]]]) / det
            self.theta[s] = inv @ self.b[s]
            cov = self.sigma2 * inv
            l11 = math.sqrt(cov[0, 0])
            l21 = cov[1, 0] / l11
            l22 = math.sqrt(max(cov[1, 1] - l21 * l21, 1e-12))
            self.chol[s] = [[l11, 0.0], [l21, l22]]
        else:
            p = self.A[s] + self.lam * np.eye(self.dim)
            inv = np.linalg.inv(p)
            self.theta[s] = inv @ self.b[s]
            cov = self.sigma2 * (inv + inv.T) / 2.0
            self.chol[s] = np.linalg.cholesky(cov + 1e-12 * np.eye(self.dim))

    def predict(self, s: int, v) -> float:
        return float(self.theta[s] @ self._row(v))

    def _matrix(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1 and self.dim == 2:      # list of queue lengths, one per server
            return np.stack([np.ones_like(X), X], axis=1)
        return X

    def predict_mean(self, X) -> np.ndarray:
        return np.einsum("nd,nd->n", self.theta, self._matrix(X))

    def predict_sample(self, X, np_rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        z = np_rng.standard_normal((self.n, self.dim)) * scale
        th = self.theta + np.einsum("nij,nj->ni", self.chol, z)
        return np.einsum("nd,nd->n", th, self._matrix(X))


class LearnedRouter:
    """Thread-safe wrapper used by the gateway: pick a backend, then learn from the outcome."""

    def __init__(self, latency_scale_sec: float = 0.025, rng: Optional[random.Random] = None,
                 np_rng: Optional[np.random.Generator] = None):
        self.latency_scale_sec = latency_scale_sec
        self.model = LatencyModel(0)
        self.rng = rng or random.Random()
        self.np_rng = np_rng or np.random.default_rng()
        self._lock = threading.Lock()

    def select(self, strategy: str, queue_list: List[int], eligible: List[int]) -> int:
        with self._lock:
            self.model.ensure_size(len(queue_list))
            if strategy == "ts_learned":
                sample = self.model.predict_sample(queue_list, self.np_rng)
                return min(eligible, key=lambda s: sample[s])
            d = _SAMPLE_SIZE.get(strategy)
            if d is not None and len(eligible) >= d:
                cands = self.rng.sample(eligible, d)
            else:
                cands = list(eligible)
            return self.best_of(cands, queue_list)

    def best_of(self, cands: Sequence[int], queue_list: Sequence[float]) -> int:
        """Candidate with the lowest posterior-mean predicted latency (random tie-break)."""
        self.model.ensure_size(len(queue_list))
        return min(cands, key=lambda s: (self.model.predict(s, queue_list[s]), self.rng.random()))

    def observe(self, s: int, latency_sec: float, queue_seen: float) -> None:
        with self._lock:
            self.model.ensure_size(s + 1)
            self.model.update(s, queue_seen, latency_sec / self.latency_scale_sec)
