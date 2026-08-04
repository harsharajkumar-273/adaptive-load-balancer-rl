# src/gateway.py
"""
Data Plane API Gateway acting as a high-performance HTTP reverse proxy
with Multi-Strategy Routing, Adaptive Traffic Shaping (QoS Load Shedding),
Prometheus Metrics Exposition (/metrics), AI Explainability (/explain-routing),
Chaos Injection, Action Masking, and Circuit Breaking.
"""
import time
import random
import httpx
from typing import Dict
from fastapi import FastAPI, Request, Response, HTTPException, status, Body, Query
from src.config import (
    STALENESS_THRESHOLD_SEC,
    BACKEND_URLS,
    SLA_LATENCY_MS,
    ROUTING_STRATEGY,
    BACKEND_SPECS,
    NUM_INSTANCES
)
from src.shared_state import DistributedStateCache
from src.routing_strategies import RoutingEngine
from src.chaos import chaos_manager
from src.metrics import generate_prometheus_metrics

app = FastAPI(title="Distributed AI-Driven Load Balancer Gateway")

# Track QoS load shedding counters
qos_shed_counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}

@app.on_event("startup")
async def startup_event():
    """Initializes connection-pooled async HTTP client & routing engine."""
    limits = httpx.Limits(max_keepalive_connections=200, max_connections=500)
    app.state.client = httpx.AsyncClient(timeout=4.0, limits=limits)
    app.state.shared_cache = DistributedStateCache(num_instances=NUM_INSTANCES)
    app.state.routing_engine = RoutingEngine(num_instances=NUM_INSTANCES)
    app.state.current_strategy = ROUTING_STRATEGY

@app.on_event("shutdown")
async def shutdown_event():
    """Closes HTTP client connections on exit."""
    await app.state.client.aclose()

async def forward_proxy_request(request: Request, response: Response, priority: str = "medium", strategy: str = None):
    """Core HTTP reverse proxy logic with QoS Traffic Shaping & Action Masking."""
    shared_cache = request.app.state.shared_cache
    http_client = request.app.state.client
    routing_engine = request.app.state.routing_engine
    
    active_strategy = strategy.lower() if strategy else request.app.state.current_strategy
    current_time = time.time()
    
    # 1. Fetch latest weights and metrics from Redis
    weights, last_update = shared_cache.get_routing_weights()
    metrics = shared_cache.get_instance_metrics()
    cpu_list = metrics["cpu"]
    queue_list = metrics["queue"]
    
    # 2. Adaptive Traffic Shaping (QoS Load Shedding under heavy cluster load)
    avg_cpu = sum(cpu_list) / max(1, len(cpu_list))
    if avg_cpu > 80.0 and priority == "low":
        qos_shed_counts["low"] += 1
        response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
        return {
            "status": "shedded",
            "priority": priority,
            "message": f"QoS Load Shedding active (Average Cluster CPU {avg_cpu:.1f}% > 80%). Low priority request shed to protect SLA."
        }

    # 3. Safety Guardrail: Circuit Breaker for Stale Cache
    is_stale = (current_time - last_update) > STALENESS_THRESHOLD_SEC
    
    if is_stale and active_strategy == "lin_ts":
        shared_cache.set_circuit_breaker(True)
        chosen_idx, routing_mode, effective_weights = routing_engine.select_instance("least_conn", weights, cpu_list, queue_list)
        routing_mode = "fallback_circuit_breaker_least_conn"
    else:
        shared_cache.set_circuit_breaker(False)
        chosen_idx, routing_mode, effective_weights = routing_engine.select_instance(active_strategy, weights, cpu_list, queue_list)

    # 4. Proxy the HTTP request to the selected microservice node
    target_url = f"{BACKEND_URLS[chosen_idx]}/work"
    start_time = time.time()
    success = False
    
    try:
        proxy_resp = await http_client.get(target_url)
        latency_ms = (time.time() - start_time) * 1000.0
        success = (proxy_resp.status_code == status.HTTP_200_OK)
        
        # Write audit headers & X-Decision-Reason
        reason = f"Strategy={routing_mode}; Target=Node-{chosen_idx + 1}; Priority={priority.upper()}; CPU={cpu_list[chosen_idx]:.1f}%; Weight={effective_weights[chosen_idx]*100:.1f}%"
        response.headers["X-Routed-To"] = f"Node-{chosen_idx + 1}"
        response.headers["X-Routing-Mode"] = routing_mode
        response.headers["X-Decision-Reason"] = reason
        response.headers["X-Proxy-Latency"] = f"{latency_ms:.2f} ms"
        
        response.status_code = proxy_resp.status_code
        
        # Record outcome in Redis (triggers Event-Driven Pub/Sub stream)
        breached_sla = latency_ms > SLA_LATENCY_MS
        shared_cache.record_request_outcome(chosen_idx, latency_ms, success, breached_sla)
        
        return proxy_resp.json()
        
    except httpx.RequestError:
        latency_ms = (time.time() - start_time) * 1000.0
        shared_cache.record_request_outcome(chosen_idx, latency_ms, False, True)
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Connection failed proxying to Backend Node {chosen_idx + 1}"
        )

# Route Endpoints with QoS Tiers
@app.get("/auth")
@app.get("/checkout")
async def high_priority_endpoint(request: Request, response: Response, strategy: str = None):
    """High Priority Endpoint (SLA protected)."""
    return await forward_proxy_request(request, response, priority="high", strategy=strategy)

@app.get("/play-video")
async def medium_priority_endpoint(request: Request, response: Response, strategy: str = None):
    """Medium Priority Endpoint."""
    return await forward_proxy_request(request, response, priority="medium", strategy=strategy)

@app.get("/analytics")
@app.get("/logs")
async def low_priority_endpoint(request: Request, response: Response, strategy: str = None):
    """Low Priority Endpoint (Subject to QoS load shedding under CPU > 80%)."""
    return await forward_proxy_request(request, response, priority="low", strategy=strategy)

@app.get("/explain-routing")
async def explain_routing(request: Request):
    """
    AI & Routing Explainability Audit Endpoint.
    Provides detailed audit trail explaining why backends are selected, expected rewards,
    candidate evaluations, feature values, and action mask states.
    """
    shared_cache = request.app.state.shared_cache
    weights, last_update = shared_cache.get_routing_weights()
    metrics = shared_cache.get_instance_metrics()
    telemetry = shared_cache.get_telemetry()
    
    cpu_list = metrics["cpu"]
    queue_list = metrics["queue"]
    p99_list = metrics["p99"]
    
    candidates = []
    max_weight = max(weights) if weights else 0.2
    
    for i in range(NUM_INSTANCES):
        cpu = cpu_list[i]
        weight = weights[i]
        is_masked = cpu > 85.0
        
        status_str = "MASKED_HIGH_CPU" if is_masked else ("TOP_CHOICE" if weight == max_weight else "ACTIVE")
        
        candidates.append({
            "node": f"Node-{i+1}",
            "name": BACKEND_SPECS[i]["name"],
            "rl_weight": round(weight, 4),
            "rl_weight_percentage": f"{weight*100:.1f}%",
            "cpu_load": f"{cpu:.1f}%",
            "queue_depth": queue_list[i],
            "recent_p99_ms": round(p99_list[i], 2),
            "status": status_str
        })
        
    return {
        "active_strategy": request.app.state.current_strategy,
        "circuit_breaker_tripped": telemetry.get("circuit_breaker_tripped", False),
        "cache_last_updated_sec_ago": round(time.time() - last_update, 3),
        "global_traffic": {
            "current_rps": round(telemetry.get("global_request_rate", 0.0), 1),
            "spike_active": telemetry.get("traffic_spike_active", False)
        },
        "qos_traffic_shaping": {
            "load_shedding_active": sum(cpu_list) / max(1, len(cpu_list)) > 80.0,
            "low_priority_shed_count": qos_shed_counts["low"]
        },
        "features_evaluated": [
            "Bias (1.0)",
            "Normalized CPU (CPU/100)",
            "Normalized Queue Depth (Queue/20)",
            "Normalized P99 Latency (P99/200)",
            "Global Traffic Rate (Rate/150)",
            "Traffic Rate Trend (d/dt / 50)"
        ],
        "candidate_evaluations": candidates
    }

@app.post("/chaos/inject")
async def inject_chaos(
    node_idx: int = Query(..., ge=0, le=4),
    fault_type: str = Body("cpu_spike", embed=True),
    extra_latency_ms: float = Body(300.0, embed=True)
):
    """Chaos Engineering Fault Injector Endpoint."""
    http_client = app.state.client
    target_url = f"{BACKEND_URLS[node_idx]}/chaos/inject"
    try:
        resp = await http_client.post(target_url, json={"fault_type": fault_type, "extra_latency_ms": extra_latency_ms})
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inject chaos on Node {node_idx + 1}: {str(e)}")

@app.post("/chaos/clear")
async def clear_chaos():
    """Clears all injected chaos faults across the cluster."""
    http_client = app.state.client
    results = []
    for idx, url in enumerate(BACKEND_URLS):
        try:
            resp = await http_client.post(f"{url}/chaos/inject", json={"fault_type": "clear"})
            results.append(resp.json())
        except Exception:
            pass
    return {"status": "all_cleared", "nodes": results}

@app.post("/strategy")
async def set_strategy(request: Request, strategy: str = Body(..., embed=True)):
    """Dynamically switches the active load balancing strategy at runtime."""
    valid_strategies = ["lin_ts", "least_conn", "p2c", "round_robin", "weighted_round_robin"]
    strat = strategy.lower()
    if strat not in valid_strategies:
        raise HTTPException(status_code=400, detail=f"Invalid strategy. Choose from: {valid_strategies}")
    
    request.app.state.current_strategy = strat
    return {"status": "updated", "current_strategy": strat}

@app.get("/metrics")
async def get_metrics(request: Request):
    """Exposes Prometheus exposition formatted metrics payload."""
    shared_cache = request.app.state.shared_cache
    telemetry = shared_cache.get_telemetry()
    content, content_type = generate_prometheus_metrics(telemetry, qos_shed_counts)
    return Response(content=content, media_type=content_type)
