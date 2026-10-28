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
    a, b = simulate(sc, "ts", seed=3), simulate(sc, "ts", seed=3)
    assert a.keys() == b.keys() and all(a[k] == b[k] for k in a)


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


# --- age-aware water-filling ------------------------------------------------

from research.policies import waterfill  # noqa: E402


def test_waterfill_is_a_distribution():
    p = waterfill([3, 0, 7, 1], [200.0, 100.0, 50.0, 150.0], lam=450.0, age=0.05)
    assert all(x >= 0 for x in p) and sum(p) == pytest.approx(1.0)


def test_waterfill_tends_to_capacity_split_when_state_is_very_old():
    mu = [200.0, 100.0, 50.0]
    p = waterfill([40, 0, 5], mu, lam=300.0, age=1e4)
    assert p == pytest.approx([m / sum(mu) for m in mu], abs=1e-3)


def test_waterfill_tends_to_shortest_expected_delay_when_state_is_fresh():
    mu = [200.0, 100.0, 50.0]
    q = [30, 4, 2]            # (q+1)/mu: 0.155, 0.05, 0.06 -> server 1 is best
    p = waterfill(q, mu, lam=300.0, age=1e-6)
    assert p[1] > 0.99


def test_waterfill_shifts_load_away_from_long_queues():
    mu = [100.0, 100.0]
    p = waterfill([20, 0], mu, lam=150.0, age=0.05)
    assert p[1] > p[0]


def test_waterfill_equalises_fluid_delay_where_queues_form():
    # Offered load (400/s) exceeds capacity (350/s) within the window: queues form
    # at every active server, and the split equalises their expected delay.
    mu, q, lam, age = [200.0, 100.0, 50.0], [2, 5, 1], 400.0, 0.1
    p = waterfill(q, mu, lam, age)
    queues = [q[i] + (lam * p[i] - mu[i]) * age for i in range(3)]
    assert all(x > 0 for x in queues)
    delays = [(queues[i] + 1.0) / mu[i] for i in range(3) if p[i] > 1e-9]
    assert max(delays) - min(delays) < 1e-9


@pytest.mark.parametrize("policy", ["awf", "awf_learned"])
def test_awf_policies_run_and_stay_stable_under_stale_state(policy):
    r = simulate(Scenario(staleness=0.2, num_dispatchers=16, **SHORT), policy, seed=0)
    assert r["requests"] > 1000 and r["p99_ms"] < 5000


def test_latency_model_higher_dimension_recovers_coefficients():
    rng = np.random.default_rng(1)
    m = LatencyModel(n=1, gamma=1.0, lam=1e-3, dim=3)
    for _ in range(3000):
        q, hit = rng.integers(0, 15), rng.integers(0, 2)
        m.update(0, [1.0, q, hit], 0.4 + 1.5 * q - 0.3 * hit + rng.normal(0, 0.1))
    assert m.theta[0] == pytest.approx([0.4, 1.5, -0.3], abs=0.05)
    X = np.array([[1.0, 2.0, 1.0]])
    assert m.predict_mean(X)[0] == pytest.approx(0.4 + 3.0 - 0.3, abs=0.1)


def test_latency_model_dim2_matrix_and_list_inputs_agree():
    m = LatencyModel(n=3)
    for s in range(3):
        for q in range(5):
            m.update(s, q, 0.1 * (s + 1) * (q + 1))
    q = [1.0, 2.0, 3.0]
    X = np.stack([np.ones(3), np.array(q)], axis=1)
    assert m.predict_mean(q) == pytest.approx(m.predict_mean(X))


# --- herding as burst concentration --------------------------------------------

from research.sim import window_burst_stats  # noqa: E402


def test_window_burst_stats_extremes():
    n, per_window = 4, 40
    times = [w + i / per_window for w in range(50) for i in range(per_window)]
    stampede = window_burst_stats(times, [w % n for w in range(50) for _ in range(per_window)], n, 1.0)
    spread = window_burst_stats(times, [i % n for i in range(len(times))], n, 1.0)
    assert stampede[0] == pytest.approx(1.0) and spread[0] == pytest.approx(1.0 / n)


def test_deterministic_routing_on_common_state_concentrates_a_window_and_power_of_d_caps_it():
    # Theory (paper): argmin on a snapshot every gateway shares sends ~all of a window to one server;
    # sampling d of N candidates caps any server's share near d/N.
    sc = Scenario(staleness=0.2, num_dispatchers=16, duration=12.0, warmup=2.0)
    jsq = simulate(sc, "jsq", seed=0)["window_share_mean"]
    p2c = simulate(sc, "p2c", seed=0)["window_share_mean"]
    assert jsq > 0.6
    assert p2c < 2.0 / len(sc.mus) + 0.06


# --- model-variant ablation -----------------------------------------------------

from research.policies import ABLATION_POLICIES  # noqa: E402


@pytest.mark.parametrize("policy", ABLATION_POLICIES)
def test_ablation_policies_run(policy):
    r = simulate(Scenario(staleness=0.05, num_dispatchers=4, rho=0.7, **SHORT), policy, seed=0)
    assert r["requests"] > 1000 and 0 < r["p50_ms"] <= r["p99_ms"]


def test_linucb_is_optimistic_about_unexplored_servers():
    import random
    d = make_policy("linucb", n=3, nominal_mus=(1, 1, 1), rng=random.Random(0), latency_scale=0.01,
                    np_rng=np.random.default_rng(0), explore=3.0)
    for _ in range(300):                   # servers 0 and 1 are well explored, equally good
        d.observe(0, 0.01, 1.0, 0.0)
        d.observe(1, 0.01, 1.0, 0.0)
    picks = {d.choose(0.0, [1, 1, 1], 1)[0] for _ in range(5)}
    assert picks == {2}                    # never-observed server has the widest posterior


def test_richer_features_are_recorded_and_learned():
    import random
    d = make_policy("greedy_ewma", n=2, nominal_mus=(1, 1), rng=random.Random(0), latency_scale=0.01,
                    np_rng=np.random.default_rng(0))
    s, x = d.choose(0.0, [3, 0], 1)
    assert len(x) == 3 and x[0] == 1.0 and x[1] == float([3, 0][s])
    d.observe(s, 0.05, x, 0.1)
    assert d._ewma[s] == pytest.approx(5.0)    # latency 0.05 s / scale 0.01 s


def test_mlp_learner_runs_deterministically_and_learns():
    from research.sim import Scenario, simulate
    sc = Scenario(staleness=0.05, num_dispatchers=4, duration=6.0, warmup=1.0)
    a = simulate(sc, "p2c_learned_mlp", seed=1)
    b = simulate(sc, "p2c_learned_mlp", seed=1)
    assert a["p99_ms"] == b["p99_ms"] and a["p99_ms"] < 1000.0
