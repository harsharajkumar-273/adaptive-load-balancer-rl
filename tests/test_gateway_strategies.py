# tests/test_gateway_strategies.py
"""
Tests for the experiment strategies in the gateway: SED, Thompson sampling,
local correction, synchronised refresh, Prequal-style probe pool, probing
power-of-d, and independence of gateways built by create_app().
"""
import asyncio
import random
from types import SimpleNamespace

import pytest

import src.gateway as gw
from src.learned_routing import LearnedRouter
from src.probing import PrequalPool
from src.routing_strategies import RoutingEngine
from src.shared_state import DistributedStateCache


def _state(nominal=(4.0, 1.0), **extra):
    rng = random.Random(0)
    st = SimpleNamespace(
        routing_engine=RoutingEngine(apply_cpu_mask=False, learned=LearnedRouter(rng=rng),
                                     nominal_rates=list(nominal), rng=rng),
        rng=rng, sent_since_refresh=[0] * len(nominal), registry=None, client=None,
        prequal=PrequalPool(len(nominal), rng),
    )
    for k, v in extra.items():
        setattr(st, k, v)
    return SimpleNamespace(app=SimpleNamespace(state=st))


def _choose(req, strategy, queues):
    cpu = [5.0] * len(queues)
    return asyncio.run(gw.choose_backend(req, strategy, [1.0 / len(queues)] * len(queues), cpu, list(queues)))


def test_sed_uses_nominal_rates():
    # (q+1)/mu: backend 0 -> (3+1)/4 = 1.0, backend 1 -> (0+1)/1 = 1.0 ... tie; make 0 clearly better
    req = _state(nominal=(10.0, 1.0))
    idx, mode, feat = _choose(req, "sed", [3, 0])
    assert idx == 0 and mode == "baseline_shortest_expected_delay" and feat == 3.0


def test_local_correction_spreads_sends_until_refresh():
    req = _state(nominal=(1.0, 1.0))
    picks = [_choose(req, "least_conn+lc", [0, 0])[0] for _ in range(4)]
    assert sorted(picks) == [0, 0, 1, 1]
    assert req.app.state.sent_since_refresh == [2, 2]


def test_plain_strategy_does_not_count_sends():
    req = _state(nominal=(1.0, 1.0))
    _choose(req, "least_conn", [0, 0])
    assert req.app.state.sent_since_refresh == [0, 0]


def test_ts_learned_prefers_learned_fast_backend():
    req = _state(nominal=(1.0, 1.0))
    learned = req.app.state.routing_engine.learned
    for _ in range(200):
        learned.observe(0, 0.001, 0)   # fast
        learned.observe(1, 0.100, 0)   # slow
    picks = [_choose(req, "ts_learned", [0, 0])[0] for _ in range(20)]
    assert picks.count(0) >= 18


def test_probing_power_of_d_uses_live_rif(monkeypatch):
    live = {0: (7, 0.1), 1: (0, 0.1), 2: (3, 0.1)}

    async def fake_probe(client, url, timeout):
        return live[int(url.rsplit(":", 1)[1]) - gw.BACKEND_PORT_BASE]

    monkeypatch.setattr(gw, "probe", fake_probe)
    req = _state(nominal=(1.0, 1.0, 1.0))
    idx, mode, feat = _choose(req, "p3c_probe", [0, 0, 0])   # snapshot says all empty
    assert idx == 1 and mode == "p3c_probe" and feat == 0.0


def test_prequal_pool_hot_cold_rule():
    pool = PrequalPool(4, random.Random(0), q_rif=0.5, reuse_budget=1)
    pool.add(0.0, 0, rif=1, latency=0.9)   # cold, slow
    pool.add(0.0, 1, rif=2, latency=0.1)   # cold, fast
    pool.add(0.0, 2, rif=9, latency=0.01)  # hot, fastest
    assert pool.select(0.1)[0] == 1        # 0.5-quantile RIF is 2: backend 2 is hot
    # Reuse budget 1 removes backend 1. With RIFs {1, 9} the 0.5-quantile is 9,
    # so both remaining entries are cold and the lower latency estimate wins.
    assert pool.select(0.1)[0] == 2
    assert pool.select(0.1)[0] == 0
    assert pool.select(0.1) is None        # every entry used up


def test_prequal_pool_expires_old_probes():
    pool = PrequalPool(2, random.Random(0), max_age=1.0)
    pool.add(0.0, 0, rif=0, latency=0.1)
    assert pool.select(5.0) is None


def test_aligned_refresh_boundaries(monkeypatch):
    monkeypatch.setattr(gw, "STATE_REFRESH_ALIGNED", True)
    monkeypatch.setattr(gw, "STATE_REFRESH_SEC", 1.0)
    assert not gw._refresh_due(10.2, 10.9)
    assert gw._refresh_due(10.9, 11.0)
    monkeypatch.setattr(gw, "STATE_REFRESH_ALIGNED", False)
    assert not gw._refresh_due(10.9, 11.0)
    assert gw._refresh_due(10.2, 11.2)


def test_create_app_gives_independent_gateways():
    from fastapi.testclient import TestClient
    a, b = gw.create_app(1), gw.create_app(2)
    a.state.shared_cache = DistributedStateCache.in_memory(num_instances=2)
    b.state.shared_cache = DistributedStateCache.in_memory(num_instances=2)
    with TestClient(a), TestClient(b):
        assert a.state.routing_engine is not b.state.routing_engine
        assert a.state.routing_engine.learned is not b.state.routing_engine.learned
        assert a.state.prequal is not b.state.prequal


def test_strategy_validation():
    assert gw._is_valid_strategy("p2c_learned+lc")
    assert gw._is_valid_strategy("prequal")
    assert not gw._is_valid_strategy("prequal+lc")
    assert not gw._is_valid_strategy("nope")
