import math

from research.agent_sim import AgentScenario, parse_policy, simulate_agents


def test_policy_parsing():
    p = parse_policy("task|length=pred|sigma=0.5|h=20|ident=infer|p_id=0.7|hint=1")
    assert (p.kind, p.length, p.sigma, p.h, p.ident, p.p_id, p.hint) == ("task", "pred", 0.5, 20.0, "infer", 0.7, True)


def test_underload_completes_everything():
    r = simulate_agents(AgentScenario(load=0.3), "none", seed=0)
    assert r["completed"] > 0.97 and r["lost"] == 0 and r["wasted"] < 0.02
    assert 0.2 < r["goodput_norm"] < 0.4          # offered load 0.3 -> goodput ~0.3 of capacity


def test_deterministic():
    sc = AgentScenario(load=1.6)
    a, b = simulate_agents(sc, "little|d=5", seed=3), simulate_agents(sc, "little|d=5", seed=3)
    assert a["goodput_norm"] == b["goodput_norm"] and a["p99_task_s"] == b["p99_task_s"]


def test_goodput_never_exceeds_capacity_by_much():
    for pol in ("none", "shed|q=16", "cap|n=64", "little|d=5"):
        assert simulate_agents(AgentScenario(load=3.0), pol, seed=1)["goodput_norm"] < 1.15


def test_unprotected_overload_collapses_and_task_admission_does_not():
    sc = AgentScenario(load=2.4)
    none = simulate_agents(sc, "none", seed=0)
    little = simulate_agents(sc, "little|d=5", seed=0)
    assert none["goodput_norm"] < 0.6 and none["lost"] > 0.5 and none["wasted"] > 0.4
    assert little["goodput_norm"] > 0.9 and little["lost"] < 0.02


def test_admitted_tasks_are_not_lost_when_wait_is_bounded():
    # Little's-law cap: admitted tasks see bounded waits, hence no timeouts and no retries
    r = simulate_agents(AgentScenario(load=4.0), "little|d=2.5", seed=2)
    assert r["lost"] == 0.0 and r["wasted"] < 0.05


def test_kv_limit_binds():
    big = simulate_agents(AgentScenario(load=2.4, kv_tasks=256), "little|d=5", seed=0)["goodput_norm"]
    small = simulate_agents(AgentScenario(load=2.4, kv_tasks=16), "little|d=5", seed=0)["goodput_norm"]
    assert small < 0.6 * big


def test_oracle_length_starves_long_tasks():
    sc = AgentScenario(load=3.0)
    oracle = simulate_agents(sc, "task|h=20|length=oracle", seed=0)
    mean = simulate_agents(sc, "task|h=20|length=mean", seed=0)
    assert oracle["long_completed"] < mean["long_completed"] and not math.isnan(oracle["long_completed"])
