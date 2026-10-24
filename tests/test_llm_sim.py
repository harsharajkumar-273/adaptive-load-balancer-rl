"""
Tests for the LLM-serving scenario simulator (research/llm_sim.py).
"""
import pytest

from research.llm_sim import (LLM_POLICIES, LLMScenario, _Replica, prefix_probabilities, rendezvous_order,
                              simulate_llm)

SHORT = dict(duration=8.0, warmup=2.0, rate=60.0)


def test_replica_lru_hits_misses_and_eviction():
    r = _Replica(capacity=2)
    assert r.touch(1, 0.1) is False        # miss
    assert r.touch(2, 0.2) is False
    assert r.touch(1, 0.3) is True         # hit; 1 is now most recent
    assert r.touch(3, 0.4) is False        # evicts 2 (least recent)
    assert r.touch(2, 0.5) is False        # 2 was evicted
    assert r.cached_at(0.05) == frozenset()
    assert r.cached_at(0.25) == frozenset({1, 2})
    assert r.cached_at(0.45) == frozenset({1, 3})


def test_replica_queue_length_from_logs():
    r = _Replica(capacity=1)
    r.arrivals += [0.0, 0.1, 0.2]
    r.departures += [0.15, 0.30, 0.45]
    assert r.queue_at(0.05) == 1
    assert r.queue_at(0.12) == 2
    assert r.queue_at(0.25) == 2
    assert r.queue_at(0.50) == 0


def test_rendezvous_order_is_a_stable_permutation():
    a = rendezvous_order(7, 8)
    assert sorted(a) == list(range(8)) and a == rendezvous_order(7, 8)
    assert rendezvous_order(8, 8) != a


def test_zipf_probabilities_sum_to_one_and_decrease():
    p = prefix_probabilities(LLMScenario())
    assert p.sum() == pytest.approx(1.0) and all(p[i] >= p[i + 1] for i in range(len(p) - 1))


def test_same_seed_is_deterministic():
    sc = LLMScenario(**SHORT)
    a, b = simulate_llm(sc, "ring3_learned", seed=2), simulate_llm(sc, "ring3_learned", seed=2)
    assert a.keys() == b.keys() and all(a[k] == b[k] for k in a)


@pytest.mark.parametrize("policy", LLM_POLICIES)
def test_every_llm_policy_runs(policy):
    r = simulate_llm(LLMScenario(**SHORT), policy, seed=0)
    assert r["requests"] > 200 and r["p50_ms"] <= r["p99_ms"] and 0.0 <= r["hit_rate"] <= 1.0


def test_cache_aware_routing_hits_the_cache_more_than_cache_oblivious():
    sc = LLMScenario(duration=15.0, warmup=5.0, rate=60.0, staleness=0.0)
    assert simulate_llm(sc, "affinity_thr4", seed=0)["hit_rate"] > simulate_llm(sc, "jsq", seed=0)["hit_rate"] + 0.1


def test_static_hashing_has_perfect_locality_but_concentrates_hot_prefixes():
    sc = LLMScenario(duration=15.0, warmup=5.0, rate=60.0, staleness=0.0, cache_size=64)
    r = simulate_llm(sc, "hash", seed=0)
    assert r["hit_rate"] > 0.98
    assert r["max_share"] > 1.0 / sc.n + 0.05          # the replica of the hottest prefix is overloaded


def test_gray_failure_scenario_runs_and_slows_the_replica():
    base = LLMScenario(**SHORT, staleness=0.0)
    slow = LLMScenario(**SHORT, staleness=0.0, slow_replica=0, slow_start=2.0, slow_factor=0.1)
    assert simulate_llm(slow, "hash", seed=0)["p99_ms"] > simulate_llm(base, "hash", seed=0)["p99_ms"]


def test_unknown_llm_policy_rejected():
    with pytest.raises(ValueError):
        simulate_llm(LLMScenario(**SHORT), "nope", seed=0)
