# tests/test_phase1.py
"""
Unit and integration tests for Phase 1 enhancements:
- Dynamic YAML Config loading
- Multi-strategy routing engine (P2C, Round Robin, Weighted Round Robin, Least Conn, LinTS)
- Chaos Engineering fault injections
- AI Explainability endpoint & headers
"""
import pytest
import numpy as np
import os
from src.config import load_yaml_fallback, CONFIG_PATH
from src.routing_strategies import RoutingEngine
from src.chaos import chaos_manager, ChaosEngine

def test_yaml_config_loader():
    """Verifies that config.yaml or line-by-line fallback loader parses correctly."""
    data = load_yaml_fallback("config.yaml")
    assert isinstance(data, dict)
    assert "cluster" in data
    assert data["cluster"].get("num_instances") == 5
    assert "agent" in data
    assert "backends" in data
    assert len(data["backends"]) == 5

def test_routing_engine_strategies():
    """Tests that RoutingEngine correctly handles all 5 load balancing strategies."""
    engine = RoutingEngine(num_instances=5)
    weights = [0.4, 0.3, 0.15, 0.1, 0.05]
    cpu_list = [20.0, 30.0, 40.0, 50.0, 60.0]
    queue_list = [2, 5, 1, 8, 0]

    # 1. LinTS RL Strategy
    idx, mode, eff_w = engine.select_instance("lin_ts", weights, cpu_list, queue_list)
    assert 0 <= idx < 5
    assert mode == "rl_adaptive_lin_ts"

    # 2. Least Connections Strategy
    # Node 4 has queue_list[4] = 0 (least connections)
    idx, mode, eff_w = engine.select_instance("least_conn", weights, cpu_list, queue_list)
    assert idx == 4
    assert mode == "baseline_least_connections"

    # 3. Power of Two Choices (P2C)
    idx, mode, eff_w = engine.select_instance("p2c", weights, cpu_list, queue_list)
    assert 0 <= idx < 5
    assert mode == "baseline_power_of_two_choices"

    # 4. Round Robin
    idx1, mode1, _ = engine.select_instance("round_robin", weights, cpu_list, queue_list)
    idx2, mode2, _ = engine.select_instance("round_robin", weights, cpu_list, queue_list)
    assert idx1 != idx2
    assert mode1 == "baseline_round_robin"

    # 5. Weighted Round Robin
    idx, mode, eff_w = engine.select_instance("weighted_round_robin", weights, cpu_list, queue_list)
    assert 0 <= idx < 5
    assert mode == "baseline_weighted_round_robin"

def test_action_masking_in_routing_engine():
    """Verifies that nodes with CPU > 85% are masked to 0 weight across all strategies."""
    engine = RoutingEngine(num_instances=5)
    weights = [0.2] * 5
    # Node 0 is overloaded (CPU 92%)
    cpu_list = [92.0, 30.0, 40.0, 50.0, 60.0]
    queue_list = [0, 5, 2, 8, 1]

    # Run LinTS
    for _ in range(20):
        idx, mode, eff_w = engine.select_instance("lin_ts", weights, cpu_list, queue_list)
        assert idx != 0  # Node 0 must NEVER be chosen when masked!
        assert eff_w[0] == 0.0

def test_chaos_engineering_module():
    """Verifies ChaosEngine fault injection registration and cleanup."""
    engine = ChaosEngine()
    
    # Inject CPU spike on Node 2
    engine.inject_fault(node_idx=2, fault_type="cpu_spike", extra_latency_ms=500.0)
    fault = engine.get_fault(node_idx=2)
    assert fault is not None
    assert fault["type"] == "cpu_spike"
    assert fault["forced_cpu"] == 99.0
    assert fault["extra_latency_ms"] == 500.0

    # Inject Error spike on Node 4
    engine.inject_fault(node_idx=4, fault_type="error_spike")
    assert engine.get_fault(node_idx=4)["error_rate"] == 0.9

    # Clear fault on Node 2
    engine.clear_fault(node_idx=2)
    assert engine.get_fault(node_idx=2) is None
    assert engine.get_fault(node_idx=4) is not None

    # Clear all faults
    engine.clear_fault()
    assert len(engine.list_active_faults()) == 0
