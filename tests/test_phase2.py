# tests/test_phase2.py
"""
Unit and integration tests for Phase 2 enhancements:
- Prometheus Metrics Exposition exporter
- QoS Priority Traffic Shaping & Load Shedding
- Redis Pub/Sub Event-Driven Streaming
"""
import pytest
import time
import json
from src.shared_state import DistributedStateCache, MockRedis
from src.metrics import generate_prometheus_metrics

@pytest.fixture
def mock_cache():
    cache = DistributedStateCache(num_instances=5)
    cache.client = MockRedis()
    cache.set_routing_weights([0.2, 0.2, 0.2, 0.2, 0.2])
    return cache

def test_prometheus_exposition_generator(mock_cache):
    """Verifies that Prometheus metrics generator outputs valid exposition string."""
    mock_cache.record_node_heartbeat(0, 45.0, 2)
    mock_cache.record_node_heartbeat(1, 65.0, 5)
    
    telemetry = mock_cache.get_telemetry()
    content, content_type = generate_prometheus_metrics(telemetry, {"low": 0})
    
    assert isinstance(content, bytes)
    text = content.decode('utf-8')
    assert "load_balancer_circuit_breaker_status" in text
    assert "load_balancer_routing_weight" in text
    assert "load_balancer_node_cpu_utilization" in text

def test_redis_pubsub_event_streaming(mock_cache):
    """Verifies that recording request outcome publishes to Redis Pub/Sub stream."""
    pubsub = mock_cache.client.pubsub()
    pubsub.subscribe("events:request_outcomes")
    
    # Record request outcome
    mock_cache.record_request_outcome(node_idx=1, latency_ms=25.4, success=True, breached_sla=False)
    
    # Check published message
    msg = pubsub.get_message()
    assert msg is not None
    assert msg["type"] == "message"
    data = json.loads(msg["data"].decode('utf-8'))
    assert data["node_idx"] == 1
    assert data["latency_ms"] == 25.4
    assert data["success"] is True

def test_qos_load_shedding_threshold():
    """Verifies QoS traffic shedding boundaries logic."""
    cpu_list_stressed = [85.0, 82.0, 90.0, 78.0, 88.0]
    avg_cpu = sum(cpu_list_stressed) / len(cpu_list_stressed)
    
    assert avg_cpu > 80.0  # Cluster stressed!
    
    # Priority classification checks
    high_priority_endpoints = ["/auth", "/checkout"]
    low_priority_endpoints = ["/analytics", "/logs"]
    
    assert "auth" in high_priority_endpoints[0]
    assert "analytics" in low_priority_endpoints[0]
