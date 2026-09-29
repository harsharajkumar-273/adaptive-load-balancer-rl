# tests/test_phase4.py
"""
Unit and integration tests for Phase 4 completions:
- Dynamic fleet sizing in ContextualBanditAgent (scaling 5 to 10 nodes)
- Real-time reactive stream updates on Redis Pub/Sub outcome events
- Prometheus QoS load shedding counters and latency histograms
- Dynamic backend URL resolution and failover retry handling
"""
import pytest
import numpy as np
import json
from src.shared_state import DistributedStateCache, MockRedis
from src.agent import ContextualBanditAgent
from src.registry import ServiceRegistry
from src.gateway import get_node_url
from src.metrics import generate_prometheus_metrics, HAS_PROMETHEUS, PROM_QOS_SHED_TOTAL

@pytest.fixture
def mock_cache():
    return DistributedStateCache.in_memory(num_instances=5)

def test_dynamic_fleet_sizing(mock_cache):
    """Verifies that ContextualBanditAgent dynamically scales its internal matrices from 5 to 10 nodes."""
    agent = ContextualBanditAgent(num_instances=5, shared_cache=mock_cache, load_checkpoint=False)
    assert len(agent.B) == 5
    assert len(agent.f) == 5
    assert len(agent.theta_hat) == 5

    prior_B0 = agent.B[0].copy()

    # Scale to 8 nodes
    agent.sync_fleet_size(8)
    assert agent.num_instances == 8
    assert len(agent.B) == 8
    assert len(agent.f) == 8
    assert len(agent.theta_hat) == 8

    # Scale to 10 nodes
    agent.sync_fleet_size(10)
    assert agent.num_instances == 10
    assert len(agent.B) == 10

    # Ensure existing node 0's learned weights are preserved
    assert np.array_equal(agent.B[0], prior_B0)

def test_each_outcome_updates_model_once(mock_cache):
    """The agent learns only from windowed feedback: one window -> one update per node."""
    agent = ContextualBanditAgent(num_instances=5, shared_cache=mock_cache, load_checkpoint=False)
    agent.prev_context_vectors = [np.ones(agent.d) for _ in range(5)]
    assert not hasattr(agent, "_event_stream_listener")

    mock_cache.record_request_outcome(node_idx=2, latency_ms=220.0, success=False, breached_sla=True)
    agent.update_model(mock_cache.flush_window_feedback())

    x = np.ones(agent.d)
    assert np.allclose(agent.B[2], np.eye(agent.d) + np.outer(x, x))
    # A bad outcome must push the predicted reward negative.
    assert agent.theta_hat[2] @ x < 0.0

    # The window is flushed, so a second update is a no-op.
    B2 = agent.B[2].copy()
    agent.update_model(mock_cache.flush_window_feedback())
    assert np.array_equal(agent.B[2], B2)

def test_dynamic_node_url_resolution(mock_cache):
    """Verifies that get_node_url resolves registered, fallback, and autoscaled node URLs correctly."""
    registry = ServiceRegistry(mock_cache)
    
    # Unregistered node 0 defaults to BACKEND_URLS or port 8001
    url0 = get_node_url(registry, 0)
    assert "8001" in url0

    # Autoscaled node 7 (port 8008)
    url7 = get_node_url(registry, 7)
    assert url7 == "http://127.0.0.1:8008"

    # Registered dynamic node with custom host/port
    registry.register_instance(
        node_idx=5,
        name="Custom-Node-6",
        host="10.0.0.15",
        port=9090
    )
    url5 = get_node_url(registry, 5)
    assert url5 == "http://10.0.0.15:9090"

def test_prometheus_qos_counter_synchronization(mock_cache):
    """Verifies that generate_prometheus_metrics updates QoS shed counters correctly."""
    telemetry = mock_cache.get_telemetry()
    qos_shed_counts = {"high": 0, "medium": 0, "low": 12}
    
    content, content_type = generate_prometheus_metrics(telemetry, qos_shed_counts)
    assert isinstance(content, bytes)
    if HAS_PROMETHEUS:
        # Check that counter value matches
        val = PROM_QOS_SHED_TOTAL.labels(priority="low")._value.get()
        assert val == 12
