"""
Tests for dynamic fleet sizing, the CPU-mask toggle, and gateway geo handling.
"""
import time
import numpy as np
from src.shared_state import DistributedStateCache
from src.routing_strategies import RoutingEngine
from src.agent import ContextualBanditAgent


def test_autoscaled_nodes_join_and_leave_the_fleet():
    cache = DistributedStateCache.in_memory(num_instances=5, max_instances=10)
    for i in range(5):
        cache.record_node_heartbeat(i, 20.0, 0)
    assert len(cache.get_instance_metrics()["cpu"]) == 5

    # Autoscaler brings up nodes 5 and 6.
    cache.record_node_heartbeat(5, 10.0, 0)
    cache.record_node_heartbeat(6, 10.0, 0)
    metrics = cache.get_instance_metrics()
    assert len(metrics["cpu"]) == 7
    assert metrics["cpu"][6] == 10.0
    assert cache.num_instances == 7

    # Node 6 stops heartbeating -> fleet shrinks back.
    cache.client.set("node:6:last_seen", str(time.time() - 60))
    assert len(cache.get_instance_metrics()["cpu"]) == 6


def test_routing_engine_routes_to_autoscaled_nodes():
    engine = RoutingEngine(num_instances=5)
    cpu = [90.0] * 5 + [10.0, 10.0]      # original nodes masked, 2 new healthy nodes
    queue = [5] * 5 + [0, 0]
    weights = [0.2] * 5                  # stale 5-entry weights from the agent
    for strategy in ["lin_ts", "least_conn", "p2c", "round_robin", "weighted_round_robin"]:
        for _ in range(20):
            idx, _, eff = engine.select_instance(strategy, weights, cpu, queue)
            assert idx in (5, 6), strategy
            assert len(eff) == 7


def test_cpu_mask_can_be_disabled():
    engine = RoutingEngine(apply_cpu_mask=False)
    cpu = [99.0, 10.0]
    queue = [0, 5]
    picks = {engine.select_instance("least_conn", [0.5, 0.5], cpu, queue)[0] for _ in range(10)}
    assert picks == {0}  # without the mask, least-conn happily picks the hot node


def test_agent_grows_with_fleet_via_control_step():
    cache = DistributedStateCache.in_memory(num_instances=5, max_instances=10)
    for i in range(7):
        cache.record_node_heartbeat(i, 20.0, 0)
    agent = ContextualBanditAgent(num_instances=5, shared_cache=cache, load_checkpoint=False)
    agent._control_step()
    weights, _ = cache.get_routing_weights()
    assert agent.num_instances == 7
    assert len(weights) == 7
    assert np.isclose(sum(weights), 1.0)


def test_node_region_comes_from_registry():
    from src.gateway import get_node_region
    from src.registry import ServiceRegistry
    cache = DistributedStateCache.in_memory(num_instances=5)
    registry = ServiceRegistry(cache)
    assert get_node_region(registry, 0) == "unknown"
    registry.register_instance(node_idx=0, name="n", host="h", port=1, region="us-west")
    assert get_node_region(registry, 0) == "us-west"
