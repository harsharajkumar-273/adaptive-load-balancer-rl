import asyncio

from research import agent_realsys as ar


def test_policy_parsing():
    assert ar.parse_policy("little") == ("little", None)
    assert ar.parse_policy("shed:16") == ("shed", 16.0)


def test_mock_end_to_end_collapse_and_protection(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "RESULTS", str(tmp_path / "out.csv"))
    args = ar.parse_args(["--backend", "mock", "--slots", "4", "--mock-step-sec", "0.1", "--loads", "3.0",
                          "--policies", "none", "little", "--reps", "1", "--duration", "40", "--warmup", "8",
                          "--calibrate-sec", "4", "--port", "8791"])
    rows = asyncio.run(ar.main_async(args))
    by = {r["policy"]: r for r in rows}
    assert by["little"]["goodput_norm"] < 1.15                       # bounded by capacity
    assert by["little"]["lost"] <= by["none"]["lost"]
    assert by["little"]["goodput_norm"] > by["none"]["goodput_norm"]
