"""
Sanity tests for the multi-dispatcher simulator (research/).
"""
import numpy as np
import pytest

from research.policies import ALL_POLICIES, LatencyModel, make_policy
from research.sim import Scenario, simulate

SHORT = dict(duration=6.0, warmup=1.0)


def test_same_seed_is_deterministic():
    sc = Scenario(staleness=0.05, num_dispatchers=4, **SHORT)
    assert simulate(sc, "ts", seed=3) == simulate(sc, "ts", seed=3)


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_every_policy_runs(policy):
    r = simulate(Scenario(staleness=0.05, num_dispatchers=4, rho=0.7, **SHORT), policy, seed=0)
    assert r["requests"] > 1000
    assert 0 < r["p50_ms"] <= r["p99_ms"] <= r["p999_ms"]


def test_single_server_matches_mm1_theory():
    # One M/M/1 queue: mean response time = 1 / (mu - lambda).
    sc = Scenario(mus=(100.0, 100.0), rho=0.5, staleness=0.0, num_dispatchers=1, duration=200.0, warmup=5.0)
    r = simulate(sc, "wrandom", seed=1)  # random split -> two independent M/M/1 at lambda=50
    assert r["mean_ms"] == pytest.approx(1000.0 / (100.0 - 50.0), rel=0.1)


def test_stale_jsq_herds_and_p2c_does_not():
    # Classic result (Mitzenmacher 2000): stale JSQ stampedes, P2C stays robust.
    sc = Scenario(staleness=0.2, num_dispatchers=8, **SHORT)
    jsq = simulate(sc, "jsq", seed=0)
    p2c = simulate(sc, "p2c", seed=0)
    assert jsq["fano"] > 5 * p2c["fano"]
    assert jsq["p99_ms"] > p2c["p99_ms"]


def test_fresh_state_jsq_does_not_herd():
    r = simulate(Scenario(staleness=0.0, num_dispatchers=8, **SHORT), "jsq", seed=0)
    assert r["fano"] < 2.0


def test_gray_failure_slows_the_server():
    base = Scenario(rho=0.75, staleness=0.0, num_dispatchers=1, **SHORT)
    gray = Scenario(rho=0.75, staleness=0.0, num_dispatchers=1, gray_server=0, gray_start=1.0, **SHORT)
    assert simulate(gray, "wrandom", seed=0)["p99_ms"] > simulate(base, "wrandom", seed=0)["p99_ms"]


def test_latency_model_recovers_linear_relation():
    rng = np.random.default_rng(0)
    m = LatencyModel(n=1, gamma=1.0, lam=1e-3)
    for _ in range(2000):
        q = rng.integers(0, 20)
        m.update(0, q, 0.5 + 2.0 * q + rng.normal(0, 0.1))
    assert m.theta[0] == pytest.approx([0.5, 2.0], abs=0.05)


def test_local_correction_counts_own_sends_until_next_snapshot():
    import random
    d = make_policy("jsq+lc", n=2, nominal_mus=(1, 1), rng=random.Random(0))
    picks = [d.choose(0.0, [0, 0], version=1)[0] for _ in range(4)]
    assert sorted(picks) == [0, 0, 1, 1]          # alternates instead of stampeding
    s, feat = d.choose(0.0, [5, 0], version=2)    # new snapshot resets local counts
    assert s == 1 and feat == 0.0


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        make_policy("nope", n=2, nominal_mus=(1, 1), rng=None)


def test_overloaded_gray_scenario_is_rejected():
    sc = Scenario(rho=0.9, gray_server=0, **SHORT)  # capacity 1000 -> 840 < 900 offered
    with pytest.raises(ValueError):
        simulate(sc, "p2c", seed=0)


def test_desync_refresh_reduces_stale_jsq_herding():
    sync = simulate(Scenario(staleness=0.2, num_dispatchers=16, **SHORT), "jsq", seed=0)
    desync = simulate(Scenario(staleness=0.2, num_dispatchers=16, desync=True, **SHORT), "jsq", seed=0)
    assert desync["fano"] < sync["fano"] / 2


def test_heavy_tailed_service_keeps_mean_and_raises_tail():
    base = Scenario(rho=0.5, staleness=0.0, num_dispatchers=1, duration=60.0, warmup=2.0)
    heavy = Scenario(rho=0.5, staleness=0.0, num_dispatchers=1, duration=60.0, warmup=2.0, service_cv=4.0)
    a, b = simulate(base, "wrandom", seed=0), simulate(heavy, "wrandom", seed=0)
    assert b["p999_ms"] > a["p999_ms"]


def test_bursty_arrivals_keep_mean_rate():
    long = dict(duration=200.0, warmup=1.0, rho=0.5, staleness=0.0, num_dispatchers=1)
    base = simulate(Scenario(**long), "wrandom", seed=0)
    bursty = simulate(Scenario(burstiness=0.5, **long), "wrandom", seed=0)
    assert bursty["requests"] == pytest.approx(base["requests"], rel=0.05)


def test_prequal_uses_its_probe_budget_and_ignores_snapshot_staleness():
    fresh = simulate(Scenario(staleness=0.0, num_dispatchers=4, **SHORT), "prequal", seed=0)
    stale = simulate(Scenario(staleness=0.2, num_dispatchers=4, **SHORT), "prequal", seed=0)
    assert fresh["probes_per_request"] == pytest.approx(3.0, abs=0.01)
    assert fresh["p99_ms"] == stale["p99_ms"]


def test_synchronous_probing_is_charged_its_round_trip():
    sc = Scenario(rho=0.1, staleness=0.0, num_dispatchers=1, **SHORT)
    probe = simulate(sc, "p2c_learned_probe", seed=0)
    assert probe["probes_per_request"] == pytest.approx(2.0)
    assert probe["p50_ms"] >= 0.5  # at least one 0.5 ms probe round-trip
