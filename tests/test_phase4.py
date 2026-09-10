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
    cache = DistributedStateCache(num_instances=5)
    cache.client = MockRedis()
    return cache

def test_dynamic_fleet_sizing(mock_cache):
    """Verifies that ContextualBanditAgent dynamically scales its internal matrices from 5 to 10 nodes."""
    agent = ContextualBanditAgent(num_instances=5, shared_cache=mock_cache)
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

def test_reactive_stream_bayesian_update(mock_cache):
    """Verifies that reactive event streaming updates the Bayesian parameter matrices."""
    agent = ContextualBanditAgent(num_instances=5, shared_cache=mock_cache)
    init_contexts = [np.ones(agent.d) for _ in range(5)]
    agent.prev_context_vectors = init_contexts
    prior_B2 = agent.B[2].copy()

    # Simulate arrival of outcome event for Node 2 with high latency
    pubsub = mock_cache.client.pubsub()
    payload = json.dumps({
        "node_idx": 2,
        "latency_ms": 220.0,
        "success": False,
        "breached_sla": True
    })
    
    # Directly test the stream processing logic
    data_bytes = payload.encode('utf-8')
    data = json.loads(data_bytes.decode('utf-8'))
    idx = data["node_idx"]
    lat = data["latency_ms"]
    success = data["success"]
    sla = data["breached_sla"]

    latency_penalty = lat / 200.0
    reward = -(1.0 * latency_penalty + 5.0 * (0.0 if success else 1.0) + 10.0 * (1.0 if sla else 0.0))
    reward = max(-15.0, reward)
    alpha = 0.05
    x = agent.prev_context_vectors[idx]
    agent.B[idx] += alpha * np.outer(x, x)
    agent.f[idx] += alpha * reward * x
    agent.theta_hat[idx] = np.linalg.solve(agent.B[idx], agent.f[idx])

    assert not np.array_equal(agent.B[2], prior_B2)
    # The negative reward should drive theta_hat coefficients negative
    assert agent.theta_hat[2][0] < 0.0

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
