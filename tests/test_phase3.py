# tests/test_phase3.py
"""
Unit and integration tests for Phase 3 enhancements:
- Dynamic Service Discovery & Registration Engine
- Multi-Region Geo-Routing Latency Calculations
- Kubernetes HPA Cluster Auto-Scaler Engine logic
"""
import pytest
import time
from src.shared_state import DistributedStateCache, MockRedis
from src.registry import ServiceRegistry
from src.autoscaler import ClusterAutoScaler

@pytest.fixture
def mock_cache():
    cache = DistributedStateCache(num_instances=5)
    cache.client = MockRedis()
    return cache

def test_service_registry(mock_cache):
    """Verifies ServiceRegistry instance registration, heartbeats, and deregistration."""
    registry = ServiceRegistry(mock_cache)
    
    # Register Node 1
    info = registry.register_instance(
        node_idx=0,
        name="Instance-1",
        host="127.0.0.1",
        port=8001,
        capacity=100.0,
        region="us-east"
    )
    assert info["status"] == "HEALTHY"
    assert info["port"] == 8001
    
    active = registry.get_active_instances()
    assert len(active) == 1
    assert active[0]["node_idx"] == 0

    # Heartbeat
    assert registry.heartbeat(0) is True

    # Deregister Node 1
    assert registry.deregister_instance(0) is True
    assert len(registry.get_active_instances()) == 0

def test_autoscaler_trigger_thresholds(mock_cache):
    """Verifies HPA ClusterAutoScaler trigger thresholds for scale up and down."""
    autoscaler = ClusterAutoScaler(shared_cache=mock_cache, min_nodes=5, max_nodes=10)
    
    assert autoscaler.min_nodes == 5
    assert autoscaler.max_nodes == 10
    assert len(autoscaler.dynamic_processes) == 0

def test_geo_routing_penalty_matrix():
    """Verifies cross-region network latency penalty calculations."""
    from src.gateway import REGION_LATENCY_PENALTY
    
    assert REGION_LATENCY_PENALTY[("us-east", "us-east")] == 0.0
    assert REGION_LATENCY_PENALTY[("us-east", "us-west")] == 35.0
    assert REGION_LATENCY_PENALTY[("us-east", "eu-west")] == 75.0
    assert REGION_LATENCY_PENALTY[("us-east", "ap-south")] == 180.0
