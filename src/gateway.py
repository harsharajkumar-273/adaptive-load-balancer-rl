# src/gateway.py
"""
Data Plane API Gateway acting as a high-performance HTTP reverse proxy
with action masking, circuit breaking, and telemetry reporting.
"""
import time
import random
import httpx
from fastapi import FastAPI, Request, Response, HTTPException, status
from src.config import STALENESS_THRESHOLD_SEC, BACKEND_URLS, SLA_LATENCY_MS
from src.shared_state import DistributedStateCache

app = FastAPI(title="Distributed AI-Driven Load Balancer Gateway")

@app.on_event("startup")
async def startup_event():
    """Initializes the connection-pooled async HTTP client for proxying."""
    # Reuse TCP connections for sub-millisecond connection overhead
    limits = httpx.Limits(max_keepalive_connections=200, max_connections=500)
    app.state.client = httpx.AsyncClient(timeout=4.0, limits=limits)
    app.state.shared_cache = DistributedStateCache(num_instances=5)

@app.on_event("shutdown")
async def shutdown_event():
    """Closes the HTTP client connections on exit."""
    await app.state.client.aclose()

@app.get("/play-video")
async def play_video(request: Request, response: Response):
    """
    HTTP Reverse Proxy endpoint.
    Selects a target backend node stochastically using weights from Redis,
    makes the real HTTP call, and records latency feedback.
    """
    shared_cache = request.app.state.shared_cache
    http_client = request.app.state.client
    
    current_time = time.time()
    
    # 1. Fetch latest weights and metrics from Redis
    weights, last_update = shared_cache.get_routing_weights()
    metrics = shared_cache.get_instance_metrics()
    cpu_list = metrics["cpu"]
    queue_list = metrics["queue"]
    
    # 2. Safety Guardrail: Circuit Breaker for Stale Cache
    is_stale = (current_time - last_update) > STALENESS_THRESHOLD_SEC
    
    if is_stale:
        shared_cache.set_circuit_breaker(True)
        # Fall back to Least Connections routing based on heartbeat queues in Redis
        min_queue = min(queue_list)
        best_indices = [idx for idx, q in enumerate(queue_list) if q == min_queue]
        chosen_idx = random.choice(best_indices)
        routing_mode = "fallback_circuit_breaker_least_conn"
    else:
        shared_cache.set_circuit_breaker(False)
        # 3. Safety Guardrail: Action Masking for High CPU (> 85%)
        masked_weights = []
        for idx, w in enumerate(weights):
            if cpu_list[idx] > 85.0:
                masked_weights.append(0.0)
            else:
                masked_weights.append(w)
                
        # Re-normalize masked weights
        total_masked_weight = sum(masked_weights)
        if total_masked_weight > 0:
            normalized_weights = [w / total_masked_weight for w in masked_weights]
            
            # Stochastic Selection
            chosen_idx = random.choices(range(len(normalized_weights)), weights=normalized_weights, k=1)[0]
            routing_mode = "rl_adaptive"
        else:
            # Emergency Fallback: If all backends are overloaded (>85% CPU)
            min_queue = min(queue_list)
            best_indices = [idx for idx, q in enumerate(queue_list) if q == min_queue]
            chosen_idx = random.choice(best_indices)
            routing_mode = "fallback_all_masked_least_conn"

    # 4. Proxy the HTTP request to the selected microservice node
    target_url = f"{BACKEND_URLS[chosen_idx]}/work"
    start_time = time.time()
    success = False
    
    try:
        # Forward request
        proxy_resp = await http_client.get(target_url)
        latency_ms = (time.time() - start_time) * 1000.0
        success = (proxy_resp.status_code == status.HTTP_200_OK)
        
        # Write custom proxy headers
        response.headers["X-Routed-To"] = f"Node-{chosen_idx + 1}"
        response.headers["X-Routing-Mode"] = routing_mode
        response.headers["X-Proxy-Latency"] = f"{latency_ms:.2f} ms"
        
        # Propagate response body and status code
        response.status_code = proxy_resp.status_code
        
        # Record outcome in Redis
        breached_sla = latency_ms > SLA_LATENCY_MS
        shared_cache.record_request_outcome(chosen_idx, latency_ms, success, breached_sla)
        
        return proxy_resp.json()
        
    except httpx.RequestError as e:
        latency_ms = (time.time() - start_time) * 1000.0
        # Write error telemetry
        shared_cache.record_request_outcome(chosen_idx, latency_ms, False, True)
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Connection failed proxying to Backend Node {chosen_idx + 1}"
        )

@app.get("/metrics")
async def get_metrics(request: Request):
    """Exposes current system telemetry metrics collected in Redis."""
    shared_cache = request.app.state.shared_cache
    return shared_cache.get_telemetry()
