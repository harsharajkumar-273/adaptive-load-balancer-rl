# src/backend_node.py
"""
Standalone Backend Microservice representing a cluster node.
Each node runs on a distinct port and registers its health/telemetry to Redis.
Includes support for Chaos Engineering fault injection.
"""
import sys
import os
import argparse
import asyncio
import math
import random
import time
import uvicorn
from fastapi import FastAPI, Response, status, Body

# Ensure parent directory is on sys.path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BACKEND_SPECS
from src.shared_state import DistributedStateCache
from src.chaos import chaos_manager

app = FastAPI(title="Backend Cluster Microservice Node")

# Global state for the running node
node_idx: int = 0
spec: dict = {}
active_connections: int = 0
cpu_utilization: float = 5.0
cache: DistributedStateCache = None
lock = asyncio.Lock()

def calculate_cpu():
    """Simulates CPU utilization based on active connection depth & chaos injection."""
    global cpu_utilization, active_connections, spec, node_idx
    
    # Check for Chaos Fault Injection
    fault = chaos_manager.get_fault(node_idx)
    if fault and fault.get("forced_cpu") is not None:
        cpu_utilization = fault["forced_cpu"]
        return cpu_utilization

    capacity = spec["capacity"]
    multiplier = spec["cpu_multiplier"]
    
    target_cpu = (active_connections / capacity) * 100.0 * multiplier
    target_cpu = max(5.0, min(100.0, target_cpu))
    
    # Exponential moving average to simulate CPU spin-up lag
    cpu_utilization = 0.80 * cpu_utilization + 0.20 * target_cpu
    return cpu_utilization

def calculate_latency(cpu: float) -> float:
    """Computes response latency based on queue size, CPU stress, & chaos injection."""
    global active_connections, spec, node_idx
    base = spec["base_latency_ms"]
    jitter = random.uniform(spec["jitter_range_ms"][0], spec["jitter_range_ms"][1])
    
    # Queue depth impact
    queue_impact = (active_connections ** 1.3) * 3.5
    
    # High CPU degradation
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
    
    # Heartbeat immediately to reflect load increase
    if cache:
        cache.record_node_heartbeat(node_idx, cpu, active_connections)
        
    # Simulate processing duration
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
    """Background task to report node statistics to Redis periodically."""
    global node_idx, cache, active_connections
    await asyncio.sleep(1.0)
    while True:
        try:
            cpu = calculate_cpu()
            if cache:
                cache.record_node_heartbeat(node_idx, cpu, active_connections)
        except Exception:
            pass
        await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(heartbeat_loop())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Backend Cluster Node")
    parser.add_argument("--port", type=int, required=True, help="Port to run the node on")
    parser.add_argument("--node-idx", type=int, required=True, help="Index of this node (0-4)")
    args = parser.parse_args()

    node_idx = args.node_idx
    spec = BACKEND_SPECS[node_idx]
    
    # Initialize connection to Redis
    cache = DistributedStateCache(num_instances=5)
    
    print(f"[Node] Starting {spec['name']} on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
