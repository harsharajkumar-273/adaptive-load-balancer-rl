import numpy as np

from research.policies import GatedGreedy, make_policy
from research.sim import Scenario, simulate
from research.trust_gate import gate_states

SC = Scenario(staleness=0.05, num_dispatchers=4, duration=6.0, warmup=1.0)


def test_ungated_mode_reproduces_plain_greedy_exactly():
    assert simulate(SC, "gate_off_greedy", seed=2)["p99_ms"] == simulate(SC, "greedy", seed=2)["p99_ms"]


def test_gate_withdraws_and_restores_model_control():
    r = simulate(SC, "gate_greedy_p2c", seed=1)
    assert 0.0 < r["model_step_frac"] < 1.0           # starts untrusted, earns trust, loses it again


def test_mix_probability_extremes():
    assert simulate(SC, "mix25_greedy_p2c", seed=0)["model_step_frac"] < 0.5
    assert simulate(SC, "mix75_greedy_p2c", seed=0)["model_step_frac"] > 0.5


def test_short_window_can_still_open():
    r = simulate(SC, "gate_greedy_p2c", seed=1, policy_kwargs={"w": 10})
    assert r["model_step_frac"] > 0.0                  # regression: min_grades larger than w used to pin the gate shut


def test_grading_happens_before_learning():
    pol = make_policy("gate_off_greedy", n=3, nominal_mus=[1, 1, 1], rng=__import__("random").Random(0),
                      latency_scale=1.0, np_rng=np.random.default_rng(0), num_dispatchers=1)
    assert isinstance(pol, GatedGreedy)
    before = pol.model.theta[1].copy()
    # model predicted 5.0, baseline predicted 1.0, truth is 1.1 -> model wrong, baseline right
    pol.observe(1, 1.1, (3.0, 5.0, 1.0, True), 0.0)
    assert list(pol._model_ok) == [False] and list(pol._base_ok) == [True]
    assert not np.allclose(before, pol.model.theta[1])  # the update happened, after the grade


def test_gate_states_replay_matches_rule():
    model_ok = [1, 1, 1, 1, 0, 0, 0, 0]
    base_ok = [0, 0, 0, 0, 1, 1, 1, 1]
    st = gate_states(model_ok, base_ok, w=4, min_grades=4)
    assert list(st) == [False, False, False, False, True, True, False, False]
