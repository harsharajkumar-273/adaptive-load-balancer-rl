# tests/test_metrics_and_events.py
"""
Tests for:
- Prometheus Metrics Exposition exporter
- QoS load shedding through the gateway
- Redis Pub/Sub Event-Driven Streaming
"""
import pytest
import time
import json
from src.shared_state import DistributedStateCache, MockRedis
from src.metrics import generate_prometheus_metrics

@pytest.fixture
def mock_cache():
    cache = DistributedStateCache.in_memory(num_instances=5)
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

def test_gateway_sheds_low_priority_under_load(mock_cache):
    """With average CPU > 80%, /analytics is shed with 429 before any proxying happens."""
    from fastapi.testclient import TestClient
    from src.gateway import app, qos_shed_counts

    for i in range(5):
        mock_cache.record_node_heartbeat(i, 90.0, 10)
    app.state.shared_cache = mock_cache
    before = qos_shed_counts["low"]
    with TestClient(app) as client:
        resp = client.get("/analytics")
    assert resp.status_code == 429
    assert resp.json()["status"] == "shedded"
    assert qos_shed_counts["low"] == before + 1
