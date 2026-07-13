# tests/test_load_balancer.py
"""
Unit tests for the Distributed Load Balancer system components.
"""
import pytest
import numpy as np
import time
from src.shared_state import DistributedStateCache, MockRedis
from src.agent import ContextualBanditAgent
from src.config import BACKEND_SPECS

@pytest.fixture
def mock_cache():
    """Returns a state cache forced to use MockRedis to avoid live Redis dependencies."""
    cache = DistributedStateCache(num_instances=5)
    # Force MockRedis
    cache.client = MockRedis()
    # Reset weights
    cache.set_routing_weights([0.2, 0.2, 0.2, 0.2, 0.2])
    return cache

def test_mock_redis_basic_operations():
    """Verifies that the MockRedis fallback behaves like a real Redis instance."""
    client = MockRedis()
    assert client.get("non-existent") is None
    
    client.set("test:key", "value")
    assert client.get("test:key") == b"value"
    
    # Test increments
    client.set("counter", "10")
    client.incr("counter", 5)
    assert client.get("counter") == b"15"
    
    # Test lists
    client.lpush("list-key", "a", "b", "c")
    items = client.lrange("list-key", 0, -1)
    assert items == [b"a", b"b", b"c"]

def test_heartbeat_registration(mock_cache):
    """Verifies that nodes can successfully heartbeat metrics to the cache."""
    # Register Node 1 heartbeat (CPU 45%, Queue Depth 3)
    mock_cache.record_node_heartbeat(0, 45.0, 3)
    
    metrics = mock_cache.get_instance_metrics()
    assert metrics["cpu"][0] == 45.0
    assert metrics["queue"][0] == 3
    # Check that other nodes default to healthy baseline or offline checks
    assert metrics["cpu"][1] == 100.0  # Offline since no heartbeat was recorded!
    assert metrics["queue"][1] == 999  # Forced to high queue if dead!

def test_action_masking_boundary(mock_cache):
    """Verifies that weights are overridden based on node CPU health."""
    # Scenario: Node 1 is overloaded (CPU 90%), others are healthy
    mock_cache.record_node_heartbeat(0, 90.0, 10)
    for i in range(1, 5):
        mock_cache.record_node_heartbeat(i, 40.0, 1)

    weights, _ = mock_cache.get_routing_weights()
    metrics = mock_cache.get_instance_metrics()
    
    # Gateway masking logic simulation
    cpu_list = metrics["cpu"]
    masked_weights = []
    for idx, w in enumerate(weights):
        if cpu_list[idx] > 85.0:
            masked_weights.append(0.0)
        else:
            masked_weights.append(w)
            
    assert masked_weights[0] == 0.0  # Overloaded node must be masked to 0
    assert sum(masked_weights[1:]) > 0.0

def test_bandit_math_update(mock_cache):
    """Verifies that the Thompson Sampling agent updates its matrices without exceptions."""
    agent = ContextualBanditAgent(num_instances=5, shared_cache=mock_cache)
    
    # Reset model parameters to ensure deterministic starting point for the test
    # (prevents pre-existing checkpoint files on disk from altering test assertions)
    agent.B = [np.eye(6) for _ in range(5)]
    agent.f = [np.zeros(6) for _ in range(5)]
    agent.theta_hat = [np.zeros(6) for _ in range(5)]
    
    # Simulate first loop context vector creation
    context_vectors = [np.ones(6) for _ in range(5)]
    agent.prev_context_vectors = context_vectors
    
    # Record some mock request latency feedback
    # Node 1 succeeded in 10ms, Node 5 had an SLA breach (250ms)
    mock_cache.record_request_outcome(node_idx=0, latency_ms=10.0, success=True, breached_sla=False)
    mock_cache.record_request_outcome(node_idx=4, latency_ms=250.0, success=True, breached_sla=True)
    
    # Flush feedback and run update
    feedback_data = mock_cache.flush_window_feedback()
    
    # Prior values
    prior_B0 = agent.B[0].copy()
    
    # Update model
    agent.update_model(feedback_data)
    
    # Verify updates registered
    # B0 should be updated since Node 1 handled traffic
    assert not np.array_equal(agent.B[0], prior_B0)
    # B1 should NOT be updated since Node 2 had 0 requests in the window
    assert np.array_equal(agent.B[1], np.eye(6))
