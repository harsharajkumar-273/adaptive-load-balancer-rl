import numpy as np

from research.trace_sim import TraceScenario, simulate_trace, load_trace, ring_key, _Replica, _View


def test_trace_loads_and_blocks_are_chains():
    reqs = load_trace("toolagent", 500)
    assert len(reqs) == 500
    assert all(len(r[2]) >= 1 and r[1] > 0 for r in reqs)
    assert all(a[0] <= b[0] for a, b in zip(reqs, reqs[1:]))


def test_replica_history_view_matches_lru_over_time():
    rep = _Replica(capacity=3)
    rep.insert([1, 2], 1.0)
    rep.insert([3, 4], 2.0)          # evicts 1 and 2's oldest
    assert 4 in _View(rep, 2.5) and 3 in _View(rep, 2.5)
    assert 1 not in _View(rep, 2.5)
    assert 1 in _View(rep, 1.5) and 3 not in _View(rep, 1.5)
    assert rep.lead([3, 4, 9]) == 2


def test_ring_key_skips_universal_block():
    assert ring_key((0, 17, 18)) == 17
    assert ring_key((5,)) == 5


def test_simulation_is_deterministic_and_sane():
    sc = TraceScenario(max_requests=1500, speedup=6.0, prefill_tokens_per_sec=50000.0)
    a = simulate_trace(sc, "ring2_learned", seed=1)
    b = simulate_trace(sc, "ring2_learned", seed=1)
    assert a["p99_ms"] == b["p99_ms"]
    assert 0.0 <= a["hit_rate"] <= 1.0 and a["p99_wait_ms"] >= 0


def test_affinity_gets_more_hits_than_cache_oblivious_jsq():
    sc = TraceScenario(max_requests=3000, speedup=4.0, prefill_tokens_per_sec=50000.0)
    assert (simulate_trace(sc, "affinity", seed=0)["hit_rate"]
            > simulate_trace(sc, "jsq", seed=0)["hit_rate"])
