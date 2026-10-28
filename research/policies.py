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
  linucb       optimistic: argmin of the lower confidence bound mean - beta * std

Model variants (any learned policy accepts features=...)
  q            [1, q]                         (default)
  q2           [1, q, q^2/10]                 (nonlinear in the queue)
  ewma         [1, q, e_s]                    (e_s: this gateway's smoothed latency
                                               for server s, in latency-scale units)

Age-aware stochastic split (uses the snapshot's age and the other gateways'
expected arrivals; inspired by stochastic coordination, Goren et al. PODC'21,
but our own construction, not their algorithm)
  awf          water-filling over nominal service rates
  awf_learned  water-filling over service rates learned from latencies

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

import math
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
    "greedy_q2": ("greedy", {"features": "q2"}),
    "greedy_ewma": ("greedy", {"features": "ewma"}),
    "p2c_learned_q2": ("p2c_learned", {"features": "q2"}),
    "p2c_learned_ewma": ("p2c_learned", {"features": "ewma"}),
    "p3c_learned_q2": ("p2c_learned", {"d": 3, "features": "q2"}),
    "p3c_learned_ewma": ("p2c_learned", {"d": 3, "features": "ewma"}),
    "linucb_ewma": ("linucb", {"features": "ewma"}),
    "greedy_mlp": ("mlp", {"d": 0}),
    "p2c_learned_mlp": ("mlp", {"d": 2}),
    "p3c_learned_mlp": ("mlp", {"d": 3}),
    "greedy_rbf": ("greedy", {"features": "rbf"}),
    "linucb_rbf": ("linucb", {"features": "rbf"}),
    "p2c_learned_rbf": ("p2c_learned", {"features": "rbf"}),
    "p3c_learned_rbf": ("p2c_learned", {"d": 3, "features": "rbf"}),
    "awf": ("awf_class", {"rates": "nominal"}),
    "awf_learned": ("awf_class", {"rates": "learned"}),
}
# Model-variant ablation (own experiment, not part of the main grid).
ABLATION_POLICIES = ["greedy_q2", "greedy_ewma", "p2c_learned_q2", "p2c_learned_ewma", "p3c_learned_q2",
                     "p3c_learned_ewma", "linucb", "linucb_ewma", "greedy_rbf", "linucb_rbf", "p2c_learned_rbf",
                     "p3c_learned_rbf", "greedy_mlp", "p2c_learned_mlp", "p3c_learned_mlp"]
ALL_POLICIES = (HEURISTICS + LEARNED + PROBING + [v for v in VARIANTS if v not in ABLATION_POLICIES]
                + [p + "+lc" for p in ["jsq", "greedy", "ts", "p2c_learned"]])


class Dispatcher:
    # Extra latency this dispatcher adds to every request it routes (e.g. a
    # synchronous probe round-trip). Charged to the request by the simulator.
    added_delay = 0.0
    # Seconds since the shared snapshot was taken (set by the simulator before choose()).
    snapshot_age = 0.0

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


RBF_CENTERS = (1.0, 3.0, 6.0, 10.0)
FEATURE_DIMS = {"q": 2, "q2": 3, "ewma": 3, "rbf": 2 + len(RBF_CENTERS)}


class Learned(Dispatcher):
    def __init__(self, n, nominal_mus, rng, latency_scale: float, np_rng: np.random.Generator,
                 gamma: float = 0.995, sigma2: float = 1.0, explore: float = 1.0,
                 control_interval: float = 0.15, temperature: float = 5.0, features: str = "q", **kw):
        # temperature=5 was the best of {0.5, 2, 5, 20} in a small sweep (see research/README.md).
        super().__init__(n, nominal_mus, rng, **kw)
        if features not in FEATURE_DIMS:
            raise ValueError(f"unknown feature set {features!r}")
        self.features = features
        self.model = LatencyModel(n, sigma2=sigma2, gamma=gamma, dim=FEATURE_DIMS[features])
        self._ewma = [0.0] * n   # own smoothed latency per server, in latency-scale units
        self.latency_scale = latency_scale
        self.np_rng = np_rng
        self.explore = explore
        self.control_interval = control_interval
        self.temperature = temperature
        self._epoch_end = -1.0
        self._weights = [1.0] * n

    def _x_row(self, s, q):
        """Feature vector of server s for queue lengths q (richer feature sets only)."""
        if self.features == "q2":
            return [1.0, float(q[s]), float(q[s]) ** 2 / 10.0]
        if self.features == "rbf":   # flexible non-linear function of the queue length
            return [1.0, float(q[s])] + [math.exp(-((float(q[s]) - c) / 2.0) ** 2 / 2.0) for c in RBF_CENTERS]
        return [1.0, float(q[s]), self._ewma[s]]

    def _X(self, q):
        """What the latency model expects: queue lengths (dim 2) or one feature row per server."""
        if self.features == "q":
            return q
        return np.array([self._x_row(s, q) for s in range(self.n)])

    def choose(self, t, snapshot, version):
        if self.features == "q":
            return super().choose(t, snapshot, version)
        q = self.effective_queues(snapshot, version)
        s = self.pick(t, q)
        if self.local_correction:
            self._sent_since_snapshot[s] += 1
        return s, self._x_row(s, q)

    def observe(self, server, latency, feature, t):
        y = latency / self.latency_scale
        self.model.update(server, feature, y)
        self._ewma[server] = y if self._ewma[server] == 0.0 else 0.9 * self._ewma[server] + 0.1 * y


class MLPLearned(Learned):
    """
    A small neural latency model instead of a linear one: one MLP over (one-hot server, queue length)
    that predicts latency in latency-scale units, trained online with Adam on minibatches drawn from a
    replay buffer of this router's own (server, queue seen, latency) samples. d=0 routes greedily to
    the lowest predicted latency; d>=2 ranks d uniformly sampled candidates (power-of-d).
    """

    def __init__(self, n, nominal_mus, rng, latency_scale: float, np_rng: np.random.Generator, d: int = 0,
                 hidden: int = 32, lr: float = 0.01, batch: int = 32, buffer: int = 2000, **kw):
        Dispatcher.__init__(self, n, nominal_mus, rng, **kw)
        self.features = "q"          # the feature handed back to observe() is the queue length seen
        self.latency_scale, self.np_rng, self.d = latency_scale, np_rng, d
        self.batch, self.buffer, self.lr = batch, buffer, lr
        self.W1 = np_rng.standard_normal((n + 1, hidden)) * 0.3
        self.b1 = np.zeros(hidden)
        self.W2 = np_rng.standard_normal((hidden, 1)) * 0.3
        self.b2 = np.ones(1)
        self._m = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]
        self._v = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]
        self._t = 0
        self._X = np.zeros((buffer, n + 1))
        self._y = np.zeros(buffer)
        self._count = 0

    def _inputs(self, servers, q):
        X = np.zeros((len(servers), self.n + 1))
        X[np.arange(len(servers)), servers] = 1.0
        X[:, -1] = np.asarray(q, dtype=float) / 10.0
        return X

    def _predict(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2)[:, 0], h

    def pick(self, t, q):
        servers = list(range(self.n)) if self.d < 2 else self.rng.sample(range(self.n), min(self.d, self.n))
        pred, _ = self._predict(self._inputs(servers, [q[s] for s in servers]))
        return servers[self._argmin_random_tie(pred.tolist())]

    def observe(self, server, latency, feature, t):
        i = self._count % self.buffer
        self._X[i] = self._inputs([server], [feature])[0]
        self._y[i] = latency / self.latency_scale
        self._count += 1
        m = min(self._count, self.buffer)
        idx = self.np_rng.integers(0, m, size=min(self.batch, m))
        X, y = self._X[idx], self._y[idx]
        pred, h = self._predict(X)
        g = (pred - y)[:, None] * (2.0 / len(y))
        grads = [X.T @ ((g @ self.W2.T) * (1 - h * h)), ((g @ self.W2.T) * (1 - h * h)).sum(0), h.T @ g, g.sum(0)]
        self._t += 1
        for k, (p, gr) in enumerate(zip((self.W1, self.b1, self.W2, self.b2), grads)):
            self._m[k] = 0.9 * self._m[k] + 0.1 * gr
            self._v[k] = 0.999 * self._v[k] + 0.001 * gr * gr
            p -= self.lr * (self._m[k] / (1 - 0.9 ** self._t)) / (np.sqrt(self._v[k] / (1 - 0.999 ** self._t)) + 1e-8)


class Greedy(Learned):
    def pick(self, t, q):
        return self._argmin_random_tie(self.model.predict_mean(self._X(q)).tolist())


class LinUCB(Learned):
    """Optimism in the face of uncertainty: argmin of mean - beta * std of the predicted latency."""

    def pick(self, t, q):
        X = np.asarray(self._X(q), dtype=float)
        if X.ndim == 1:
            X = np.stack([np.ones(self.n), X], axis=1)
        mean = np.einsum("nd,nd->n", self.model.theta, X)
        # std of the prediction x . theta under the posterior: || chol^T x ||
        std = np.sqrt((np.einsum("nij,ni->nj", self.model.chol, X) ** 2).sum(axis=1))
        return self._argmin_random_tie((mean - self.explore * std).tolist())


class ThompsonSampling(Learned):
    def pick(self, t, q):
        return int(np.argmin(self.model.predict_sample(self._X(q), self.np_rng, self.explore)))


class ThompsonEpoch(Learned):
    def pick(self, t, q):
        if t >= self._epoch_end:
            scores = -self.model.predict_sample(self._X(q), self.np_rng, self.explore) / self.temperature
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
        if self.features == "q":
            s = min(cands, key=lambda c: (th[c, 0] + th[c, 1] * live[c], self.rng.random()))
            return s, float(live[s])
        s = min(cands, key=lambda c: (float(np.dot(th[c], self._x_row(c, live))), self.rng.random()))
        return s, self._x_row(s, live)

    def _score(self, s, q):
        th = self.model.theta
        if self.features == "q":
            return th[s, 0] + th[s, 1] * q[s]
        return float(np.dot(th[s], self._x_row(s, q)))

    def pick(self, t, q):
        cands = self.rng.sample(range(self.n), min(self.d, self.n))
        return min(cands, key=lambda s: (self._score(s, q), self.rng.random()))


def waterfill(q: Sequence[float], mu: Sequence[float], lam: float, age: float) -> List[float]:
    """
    Randomised split that equalises expected delay once the arrivals the
    snapshot cannot yet reflect are accounted for.

    Fluid model: a request arriving `age` seconds after the snapshot sees
    queue Q_i = max(0, q_i + (lam * p_i - mu_i) * age) at server i (all gateways
    use the same split p, total arrival rate lam), hence expected delay
    (Q_i + 1) / mu_i. Equalising it at level L gives
        p_i = max(0, (mu_i * (L + age) - q_i - 1) / (lam * age)),
    with L chosen so that sum_i p_i = 1 (exact, by sorting the breakpoints).
    As age -> 0 this tends to shortest-expected-delay; as age -> infinity it
    tends to capacity-proportional random splitting. When the fluid model
    predicts no queueing anywhere (all servers stay idle), every split has
    the same expected delay and the formula allocates by spare capacity,
    mu_i * age - q_i - 1.
    """
    n = len(q)
    lam_age = max(lam * age, 1e-9)
    c = [(mu[i] * age - q[i] - 1.0) / lam_age for i in range(n)]
    s = [mu[i] / lam_age for i in range(n)]
    order = sorted(range(n), key=lambda i: -c[i] / s[i])   # ascending breakpoints (q+1)/mu - age
    sum_c = sum_s = 0.0
    level = None
    for k, i in enumerate(order):
        sum_c += c[i]
        sum_s += s[i]
        cand = (1.0 - sum_c) / sum_s
        nxt = order[k + 1] if k + 1 < n else None
        if cand >= -c[i] / s[i] and (nxt is None or cand < -c[nxt] / s[nxt]):
            level = cand
            break
    if level is None:   # numerical edge case: fall back to the shortest expected delay
        p = [0.0] * n
        p[min(range(n), key=lambda i: (q[i] + 1.0) / mu[i])] = 1.0
        return p
    p = [max(0.0, c[i] + s[i] * level) for i in range(n)]
    total = sum(p)
    return [x / total for x in p]


class AgeAwareSplit(Learned):
    """
    Age-aware stochastic water-filling. Needs the number of gateways K (known
    to a deployment) and estimates the total arrival rate as K times its own
    smoothed arrival rate. Service rates are the nominal ones (rates="nominal")
    or learned: the slope of the latency model is the service time per queue
    slot (rates="learned").
    """

    def __init__(self, *args, rates: str = "nominal", num_dispatchers: int = 1,
                 rate_alpha: float = 0.02, min_age: float = 1e-3, **kw):
        super().__init__(*args, **kw)
        if rates not in ("nominal", "learned"):
            raise ValueError(rates)
        self.rates = rates
        self.k = num_dispatchers
        self.rate_alpha = rate_alpha
        self.min_age = min_age
        self._last_t = None
        self._ia = None   # smoothed own inter-arrival time

    def _service_rates(self) -> List[float]:
        if self.rates == "nominal":
            return self.nominal_mus
        scale = self.latency_scale
        seen = [self.model.A[s, 0, 0] > 0 for s in range(self.n)]
        mus = []
        for s in range(self.n):
            b = self.model.theta[s, 1] * scale   # seconds of latency per queued request
            mus.append(1.0 / b if seen[s] and b > 1e-4 else None)
        known = [m for m in mus if m is not None]
        default = sum(known) / len(known) if known else self.n / max(scale, 1e-9) / self.n
        return [m if m is not None else default for m in mus]

    def choose(self, t, snapshot, version):
        if self._last_t is not None:
            dt = max(t - self._last_t, 1e-6)
            self._ia = dt if self._ia is None else (1 - self.rate_alpha) * self._ia + self.rate_alpha * dt
        self._last_t = t
        if self._ia is None:
            s = self.rng.randrange(self.n)
            return s, float(snapshot[s])
        lam = self.k / self._ia
        p = waterfill(snapshot, self._service_rates(), lam, max(self.snapshot_age, self.min_age))
        s = self.rng.choices(range(self.n), weights=p, k=1)[0]
        return s, float(snapshot[s])

    def pick(self, t, q):  # not used: choose() is overridden
        raise NotImplementedError


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
    "linucb": LinUCB,
    "prequal": Prequal,
    "probe_d": ProbeD,
    "awf_class": AgeAwareSplit,
    "mlp": MLPLearned,
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
