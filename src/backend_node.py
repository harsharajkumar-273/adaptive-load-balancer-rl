# src/backend_node.py
"""
Standalone Backend Microservice representing a cluster node.
Each node runs on a distinct port and registers its health/telemetry to Redis and the Service Registry.
Includes support for Chaos Engineering fault injection and dynamic registration.
"""
import sys
import os
import argparse
import asyncio
import math
import random
import time
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status, Body

# Ensure parent directory is on sys.path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BACKEND_SPECS
from src.shared_state import DistributedStateCache
from src.registry import ServiceRegistry
from src.chaos import chaos_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global node_idx, spec, registry
    if registry:
        region = "us-east" if node_idx % 2 == 0 else "us-west"
        registry.register_instance(
            node_idx=node_idx,
            name=spec.get("name", f"Instance-{node_idx+1}"),
            host="127.0.0.1",
            port=spec.get("port", 8001 + node_idx),
            capacity=spec.get("capacity", 50.0),
            base_latency_ms=spec.get("base_latency_ms", 20.0),
            region=region
        )
    hb_task = asyncio.create_task(heartbeat_loop())
    yield
    hb_task.cancel()
    if registry:
        registry.deregister_instance(node_idx)

app = FastAPI(title="Backend Cluster Microservice Node", lifespan=lifespan)

# Global state for the running node
node_idx: int = 0
spec: dict = {}
active_connections: int = 0
cpu_utilization: float = 5.0
cache: DistributedStateCache = None
registry: ServiceRegistry = None
lock = asyncio.Lock()

def calculate_cpu():
    """Simulates CPU utilization based on active connection depth & chaos injection."""
    global cpu_utilization, active_connections, spec, node_idx
    
    fault = chaos_manager.get_fault(node_idx)
    if fault and fault.get("forced_cpu") is not None:
        cpu_utilization = fault["forced_cpu"]
        return cpu_utilization

    capacity = spec["capacity"]
    multiplier = spec["cpu_multiplier"]
    
    target_cpu = (active_connections / capacity) * 100.0 * multiplier
    target_cpu = max(5.0, min(100.0, target_cpu))
    
    cpu_utilization = 0.80 * cpu_utilization + 0.20 * target_cpu
    return cpu_utilization

def calculate_latency(cpu: float) -> float:
    """Computes response latency based on queue size, CPU stress, & chaos injection."""
    global active_connections, spec, node_idx
    base = spec["base_latency_ms"]
    jitter = random.uniform(spec["jitter_range_ms"][0], spec["jitter_range_ms"][1])
    
    queue_impact = (active_connections ** 1.3) * 3.5
    
    cpu_impact = 0.0
    if cpu > 70.0:
        cpu_impact = math.exp((cpu - 70.0) / 7.0) * 8.0
        
    extra_delay = 0.0
    fault = chaos_manager.get_fault(node_idx)
    if fault and "extra_latency_ms" in fault:
        extra_delay = fault["extra_latency_ms"]

    return base + queue_impact + cpu_impact + jitter + extra_delay

def check_success(cpu: float) -> bool:
    """Evaluates HTTP error rates if CPU exceeds node threshold or under chaos."""
    global spec, node_idx
    fault = chaos_manager.get_fault(node_idx)
    if fault and fault.get("error_rate", 0.0) > 0.0:
        if random.random() < fault["error_rate"]:
            return False

    threshold = spec["error_threshold_cpu"]
    if cpu <= threshold:
        return True
    
    excess = cpu - threshold
    span = 100.0 - threshold
    error_prob = min(0.85, excess / span)
    return random.random() > error_prob

@app.get("/work")
async def do_work(response: Response):
    """
    Executes actual concurrent API task. 
    Simulates latency and load by sleeping asynchronously.
    """
    global active_connections, cpu_utilization, node_idx, cache
    
    async with lock:
        active_connections += 1
        
    cpu = calculate_cpu()
    latency_ms = calculate_latency(cpu)
    success = check_success(cpu)
    
    if cache:
        cache.record_node_heartbeat(node_idx, cpu, active_connections)
        
    await asyncio.sleep(latency_ms / 1000.0)
    
    async with lock:
        active_connections = max(0, active_connections - 1)
        
    final_cpu = calculate_cpu()
    if cache:
        cache.record_node_heartbeat(node_idx, final_cpu, active_connections)
        
    if not success:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "error", "message": f"Node {node_idx + 1} overloaded / chaos injected"}
        
    return {
        "status": "success",
        "node_idx": node_idx,
        "processed_latency_ms": round(latency_ms, 2),
        "cpu": round(final_cpu, 1),
        "active_connections": active_connections
    }

@app.post("/chaos/inject")
async def inject_chaos(
    fault_type: str = Body("cpu_spike", embed=True),
    extra_latency_ms: float = Body(300.0, embed=True)
):
    """Chaos engineering fault injection endpoint."""
    global node_idx
    if fault_type == "clear":
        chaos_manager.clear_fault(node_idx)
        return {"status": "cleared", "node_idx": node_idx}

    chaos_manager.inject_fault(node_idx, fault_type=fault_type, extra_latency_ms=extra_latency_ms)
    return {
        "status": "injected",
        "node_idx": node_idx,
        "fault_type": fault_type,
        "extra_latency_ms": extra_latency_ms
    }

async def heartbeat_loop():
    """Background task to report node statistics to Redis and Service Registry periodically."""
    global node_idx, cache, registry, active_connections
    await asyncio.sleep(1.0)
    while True:
        try:
            cpu = calculate_cpu()
            if cache:
                cache.record_node_heartbeat(node_idx, cpu, active_connections)
            if registry:
                registry.heartbeat(node_idx)
        except Exception:
            pass
        await asyncio.sleep(0.1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Backend Cluster Node")
    parser.add_argument("--port", type=int, required=True, help="Port to run the node on")
    parser.add_argument("--node-idx", type=int, required=True, help="Index of this node (0-9)")
    args = parser.parse_args()

    node_idx = args.node_idx
    
    # Handle dynamic spec generation for autoscaled nodes (node-idx >= 5)
    if node_idx < len(BACKEND_SPECS):
        spec = BACKEND_SPECS[node_idx]
    else:
        spec = {
            "id": node_idx + 1,
            "name": f"Autoscaled-Node-{node_idx+1}",
            "base_latency_ms": 15.0,
            "capacity": 100.0,
            "cpu_multiplier": 0.6,
            "error_threshold_cpu": 95.0,
            "jitter_range_ms": (0.0, 3.0),
            "port": args.port
        }
    
    cache = DistributedStateCache(num_instances=10)
    registry = ServiceRegistry(cache)
    
    print(f"[Node] Starting {spec['name']} on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
