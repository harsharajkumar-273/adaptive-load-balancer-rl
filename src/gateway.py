# src/gateway.py
"""
Data Plane API Gateway: an HTTP reverse proxy with multi-strategy routing,
dynamic service discovery (/registry), QoS load shedding, Prometheus metrics
(/metrics), routing explainability (/explain-routing), chaos injection, action
masking, and a staleness circuit breaker.

Geo-routing is informational only: the estimated cross-region penalty is
reported in a response header but is NOT added to the measured latency, so it
never contaminates the learning signal.

`create_app()` builds an independent gateway (its own state, learned model and
probe pool), so one process can host several gateways (see
src/gateway_pool.py). The module-level `app` is one such gateway.

Strategies (VALID_STRATEGIES, set via ROUTING_STRATEGY or POST /strategy):
snapshot-based ones read shared state from Redis every STATE_REFRESH_SEC
seconds; a "+lc" suffix adds local correction (count this gateway's own sends
since the last refresh). Probing ones (src/probing.py) query backends directly.
"""
import asyncio
import math
import random
import time
import logging
import httpx
import numpy as np
from typing import Dict, List, Any
from fastapi import APIRouter, FastAPI, Request, Response, HTTPException, status, Body, Query, Header
from src.config import (
    STALENESS_THRESHOLD_SEC,
    STATE_REFRESH_SEC,
    STATE_REFRESH_ALIGNED,
    CPU_MASK_ENABLED,
    LEARNED_LATENCY_SCALE_SEC,
    GATEWAY_TIMEOUT_SEC,
    GATEWAY_SEED,
    NOMINAL_RATES,
    BACKEND_URLS,
    BACKEND_PORT_BASE,
    SLA_LATENCY_MS,
    ROUTING_STRATEGY,
    BACKEND_SPECS,
    NUM_INSTANCES,
    PREQUAL_R_PROBE,
    PREQUAL_POOL_SIZE,
    PREQUAL_REUSE_BUDGET,
    PREQUAL_MAX_AGE_SEC,
    PREQUAL_Q_RIF,
    PROBE_TIMEOUT_SEC,
)
from src.shared_state import DistributedStateCache
from src.routing_strategies import RoutingEngine, VALID_STRATEGIES, CPU_MASK_THRESHOLD
from src.learned_routing import LEARNED_STRATEGIES, LearnedRouter
from src.probing import PROBING_STRATEGIES, POWER_OF_D_PROBING, PrequalPool, probe
from src.registry import ServiceRegistry
from src.chaos import chaos_manager
from contextlib import asynccontextmanager
from src.metrics import generate_prometheus_metrics, HAS_PROMETHEUS, PROM_REQUEST_LATENCY

logger = logging.getLogger(__name__)

qos_shed_counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}

# Cross-region network latency matrix penalty (ms)
REGION_LATENCY_PENALTY = {
    ("us-east", "us-east"): 0.0,
    ("us-east", "us-west"): 35.0,
    ("us-east", "eu-west"): 75.0,
    ("us-east", "ap-south"): 180.0,
    ("us-west", "us-west"): 0.0,
    ("us-west", "us-east"): 35.0,
    ("us-west", "eu-west"): 110.0,
    ("us-west", "ap-south"): 140.0,
}

def get_node_region(registry: ServiceRegistry, idx: int) -> str:
    """Region a node registered with, or "unknown" if it never registered."""
    if registry:
        with registry._lock:
            info = registry._instances.get(idx)
            if info:
                return info.get("region", "unknown")
    return "unknown"

def get_node_url(registry: ServiceRegistry, chosen_idx: int) -> str:
    """Dynamically resolves backend instance URL from service registry or fallback list."""
    if registry:
        with registry._lock:
            if chosen_idx in registry._instances:
                return registry._instances[chosen_idx]["url"]
    if chosen_idx < len(BACKEND_URLS):
        return BACKEND_URLS[chosen_idx]
    return f"http://127.0.0.1:{BACKEND_PORT_BASE + chosen_idx}"

def _refresh_due(last_read: float, now: float) -> bool:
    if STATE_REFRESH_ALIGNED and STATE_REFRESH_SEC > 0:
        # Every gateway refreshes at the same wall-clock boundaries.
        return math.floor(now / STATE_REFRESH_SEC) > math.floor(last_read / STATE_REFRESH_SEC)
    return now - last_read >= STATE_REFRESH_SEC

def read_cluster_state(app: FastAPI, now: float):
    """
    Returns (weights, weights_last_update, node_metrics), re-reading Redis only
    every STATE_REFRESH_SEC seconds (every request when it is 0). A refresh
    also resets the local-correction counters.
    """
    snap = app.state.state_snapshot
    if snap is None or _refresh_due(snap[0], now):
        weights, last_update = app.state.shared_cache.get_routing_weights()
        metrics = app.state.shared_cache.get_instance_metrics()
        snap = (now, weights, last_update, metrics)
        app.state.state_snapshot = snap
        app.state.sent_since_refresh = [0] * len(metrics["queue"])
    return snap[1], snap[2], snap[3]

def _gateway_rng(port_hint: int):
    if GATEWAY_SEED is None:
        return random.Random(), np.random.default_rng()
    seed = int(GATEWAY_SEED) * 100_003 + port_hint
    return random.Random(seed), np.random.default_rng(seed)

def create_app(port_hint: int = 0) -> FastAPI:
    """Builds an independent gateway. port_hint only seeds its randomness."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initializes connection-pooled async HTTP client, routing engine & service registry."""
        # Persistent upstream connections: expire before the backends' 300 s keep-alive.
        limits = httpx.Limits(max_keepalive_connections=500, max_connections=1000, keepalive_expiry=240)
        app.state.client = httpx.AsyncClient(timeout=GATEWAY_TIMEOUT_SEC, limits=limits)
        app.state.shared_cache = getattr(app.state, "shared_cache", None) or DistributedStateCache(num_instances=NUM_INSTANCES)
        rng, np_rng = _gateway_rng(port_hint)
        app.state.rng = rng
        app.state.routing_engine = RoutingEngine(
            num_instances=NUM_INSTANCES,
            apply_cpu_mask=CPU_MASK_ENABLED,
            learned=LearnedRouter(latency_scale_sec=LEARNED_LATENCY_SCALE_SEC, rng=rng, np_rng=np_rng),
            nominal_rates=NOMINAL_RATES,
            rng=rng,
        )
        app.state.prequal = PrequalPool(NUM_INSTANCES, rng, r_probe=PREQUAL_R_PROBE, pool_size=PREQUAL_POOL_SIZE,
                                        reuse_budget=PREQUAL_REUSE_BUDGET, max_age=PREQUAL_MAX_AGE_SEC,
                                        q_rif=PREQUAL_Q_RIF)
        app.state.state_snapshot = None      # (read_time, weights, last_update, metrics)
        app.state.sent_since_refresh = []
        app.state.registry = ServiceRegistry(app.state.shared_cache)
        app.state.current_strategy = ROUTING_STRATEGY
        app.state.circuit_breaker_tripped = None
        yield
        await app.state.client.aclose()

    new_app = FastAPI(title="Distributed AI-Driven Load Balancer Gateway", lifespan=lifespan)
    new_app.include_router(router)
    return new_app

router = APIRouter()

async def choose_backend(request: Request, strategy: str, weights, cpu_list, queue_list):
    """
    Returns (backend index, routing mode, feature for the learned model).
    The feature is the queue length the decision was based on.
    """
    st = request.app.state
    engine = st.routing_engine
    n = len(cpu_list)
    if strategy in PROBING_STRATEGIES:
        urls = {i: get_node_url(st.registry, i) for i in range(n)}

    if strategy == "prequal":
        now = time.time
        st.prequal.spawn_probes(st.client, urls, PROBE_TIMEOUT_SEC, now, n)
        picked = st.prequal.select(now())
        if picked is None:
            return st.rng.randrange(n), "prequal_empty_pool_random", 0.0
        return picked[0], "prequal", float(picked[1])

    if strategy in POWER_OF_D_PROBING:
        d, scorer = POWER_OF_D_PROBING[strategy]
        cands = st.rng.sample(range(n), min(d, n))
        results = await asyncio.gather(*(probe(st.client, urls[c], PROBE_TIMEOUT_SEC) for c in cands))
        live = {c: r[0] for c, r in zip(cands, results) if r is not None}
        if not live:
            return st.rng.choice(cands), f"{strategy}_probe_failed", 0.0
        if scorer == "learned":
            q = [live.get(i, 0) for i in range(n)]
            chosen = engine.learned.best_of(list(live), q)
        elif scorer == "sed":
            chosen = min(live, key=lambda c: ((live[c] + 1) / engine.nominal_rate(c), st.rng.random()))
        else:
            chosen = min(live, key=lambda c: (live[c], st.rng.random()))
        return chosen, strategy, float(live[chosen])

    local_correction = strategy.endswith("+lc")
    base = strategy[:-3] if local_correction else strategy
    if local_correction:
        sent = st.sent_since_refresh
        queue_list = [q + (sent[i] if i < len(sent) else 0) for i, q in enumerate(queue_list)]
    chosen, mode, _ = engine.select_instance(base, weights, cpu_list, queue_list)
    if local_correction and chosen < len(st.sent_since_refresh):
        st.sent_since_refresh[chosen] += 1
    return chosen, mode, float(queue_list[chosen])

def _is_valid_strategy(strategy: str) -> bool:
    base = strategy[:-3] if strategy.endswith("+lc") else strategy
    return base in VALID_STRATEGIES and not (strategy.endswith("+lc") and base in PROBING_STRATEGIES)

def _learns(strategy: str) -> bool:
    base = strategy[:-3] if strategy.endswith("+lc") else strategy
    return base in LEARNED_STRATEGIES or base in ("p2c_learned_probe", "p3c_learned_probe")

async def forward_proxy_request(
    request: Request,
    response: Response,
    priority: str = "medium",
    strategy: str = None,
    client_region: str = "us-east"
):
    """Core HTTP reverse proxy logic with Geo-Routing & QoS Traffic Shaping."""
    shared_cache = request.app.state.shared_cache
    http_client = request.app.state.client
    routing_engine = request.app.state.routing_engine
    
    active_strategy = strategy.lower() if strategy else request.app.state.current_strategy
    current_time = time.time()
    
    weights, last_update, metrics = read_cluster_state(request.app, current_time)
    cpu_list = metrics["cpu"]
    queue_list = metrics["queue"]

    # Adaptive Traffic Shaping (QoS Load Shedding under heavy cluster load)
    avg_cpu = sum(cpu_list) / max(1, len(cpu_list))
    if avg_cpu > 80.0 and priority == "low":
        qos_shed_counts["low"] += 1
        response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
        return {
            "status": "shedded",
            "priority": priority,
            "message": f"QoS Load Shedding active (Average Cluster CPU {avg_cpu:.1f}% > 80%). Low priority request shed to protect SLA."
        }

    # Safety Guardrail: Circuit Breaker for Stale Cache
    is_stale = (current_time - last_update) > STALENESS_THRESHOLD_SEC
    breaker_tripped = is_stale and active_strategy == "lin_ts"
    if breaker_tripped != request.app.state.circuit_breaker_tripped:
        # Only write on state transitions to keep Redis off the hot path.
        shared_cache.set_circuit_breaker(breaker_tripped)
        request.app.state.circuit_breaker_tripped = breaker_tripped

    if breaker_tripped:
        chosen_idx, routing_mode, effective_weights = routing_engine.select_instance("least_conn", weights, cpu_list, queue_list)
        routing_mode = "fallback_circuit_breaker_least_conn"
        feature = float(queue_list[chosen_idx])
    else:
        chosen_idx, routing_mode, feature = await choose_backend(request, active_strategy, weights, cpu_list, queue_list)

    # Estimated geo penalty (informational header only; never added to measured latency)
    node_region = get_node_region(request.app.state.registry, chosen_idx)
    geo_penalty = REGION_LATENCY_PENALTY.get((client_region.lower(), node_region))

    # Proxy request
    target_url = f"{get_node_url(request.app.state.registry, chosen_idx)}/work"
    start_time = time.time()
    
    try:
        proxy_resp = await http_client.get(target_url)
        latency_ms = (time.time() - start_time) * 1000.0
        success = (proxy_resp.status_code == status.HTTP_200_OK)

        geo_str = f"{geo_penalty:.0f}ms" if geo_penalty is not None else "unknown"
        reason = f"Strategy={routing_mode}; Target=Node-{chosen_idx + 1}; ClientRegion={client_region.upper()}; NodeRegion={node_region.upper()}; EstGeoPenalty={geo_str}; CPU={cpu_list[chosen_idx]:.1f}%"
        response.headers["X-Routed-To"] = f"Node-{chosen_idx + 1}"
        response.headers["X-Routing-Mode"] = routing_mode
        response.headers["X-Client-Region"] = client_region
        response.headers["X-Decision-Reason"] = reason
        response.headers["X-Proxy-Latency"] = f"{latency_ms:.2f} ms"
        
        response.status_code = proxy_resp.status_code
        
        breached_sla = latency_ms > SLA_LATENCY_MS
        shared_cache.record_request_outcome(chosen_idx, latency_ms, success, breached_sla)
        if _learns(active_strategy):
            routing_engine.learned.observe(chosen_idx, latency_ms / 1000.0, feature)

        if HAS_PROMETHEUS:
            try:
                PROM_REQUEST_LATENCY.labels(
                    endpoint=request.url.path,
                    priority=priority,
                    routed_to=f"Node-{chosen_idx + 1}"
                ).observe(latency_ms / 1000.0)
            except Exception:
                logger.debug("Failed to observe Prometheus latency", exc_info=True)

        res_data = proxy_resp.json()
        res_data["est_geo_penalty_ms"] = geo_penalty
        return res_data
        
    except httpx.RequestError:
        # High Availability: failover retry to another healthy node
        fallback_candidates = [i for i in range(len(cpu_list)) if i != chosen_idx and cpu_list[i] <= CPU_MASK_THRESHOLD]
        if fallback_candidates:
            alt_idx = min(fallback_candidates, key=lambda i: queue_list[i] if i < len(queue_list) else 0)
            alt_url = f"{get_node_url(request.app.state.registry, alt_idx)}/work"
            try:
                proxy_resp = await http_client.get(alt_url)
                latency_ms = (time.time() - start_time) * 1000.0
                success = (proxy_resp.status_code == status.HTTP_200_OK)
                breached_sla = latency_ms > SLA_LATENCY_MS
                shared_cache.record_request_outcome(alt_idx, latency_ms, success, breached_sla)
                response.headers["X-Routed-To"] = f"Node-{alt_idx + 1}"
                response.headers["X-Routing-Mode"] = f"{routing_mode}_failover"
                response.headers["X-Proxy-Latency"] = f"{latency_ms:.2f} ms"
                response.status_code = proxy_resp.status_code
                res_data = proxy_resp.json()
                res_data["est_geo_penalty_ms"] = geo_penalty
                return res_data
            except Exception:
                logger.warning("Failover to Node-%d also failed", alt_idx + 1, exc_info=True)

        latency_ms = (time.time() - start_time) * 1000.0
        shared_cache.record_request_outcome(chosen_idx, latency_ms, False, True)
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Connection failed proxying to Backend Node {chosen_idx + 1}"
        )

# Route Endpoints with Geo-Routing & QoS Tiers
@router.get("/auth")
@router.get("/checkout")
async def high_priority_endpoint(
    request: Request,
    response: Response,
    strategy: str = None,
    x_client_region: str = Header("us-east", alias="X-Client-Region")
):
    return await forward_proxy_request(request, response, priority="high", strategy=strategy, client_region=x_client_region)

@router.get("/play-video")
async def medium_priority_endpoint(
    request: Request,
    response: Response,
    strategy: str = None,
    x_client_region: str = Header("us-east", alias="X-Client-Region")
):
    return await forward_proxy_request(request, response, priority="medium", strategy=strategy, client_region=x_client_region)

@router.get("/analytics")
@router.get("/logs")
async def low_priority_endpoint(
    request: Request,
    response: Response,
    strategy: str = None,
    x_client_region: str = Header("us-east", alias="X-Client-Region")
):
    return await forward_proxy_request(request, response, priority="low", strategy=strategy, client_region=x_client_region)

# Service Registry Endpoints
@router.get("/registry/instances")
async def get_registered_instances(request: Request):
    """Exposes all registered healthy cluster nodes."""
    return {"active_instances": request.app.state.registry.get_active_instances()}

@router.post("/registry/register")
async def register_instance(
    request: Request,
    node_idx: int = Body(..., embed=True),
    name: str = Body(..., embed=True),
    host: str = Body("127.0.0.1", embed=True),
    port: int = Body(..., embed=True),
    region: str = Body("us-east", embed=True)
):
    """Dynamically registers a new microservice node in the cluster."""
    registry = request.app.state.registry
    info = registry.register_instance(node_idx=node_idx, name=name, host=host, port=port, region=region)
    return {"status": "registered", "instance": info}

@router.post("/registry/deregister")
async def deregister_instance(request: Request, node_idx: int = Body(..., embed=True)):
    """Deregisters a microservice node from the cluster."""
    registry = request.app.state.registry
    success = registry.deregister_instance(node_idx)
    return {"status": "deregistered" if success else "not_found", "node_idx": node_idx}

@router.get("/explain-routing")
async def explain_routing(request: Request):
    """AI & Routing Explainability Audit Endpoint."""
    shared_cache = request.app.state.shared_cache
    weights, last_update = shared_cache.get_routing_weights()
    metrics = shared_cache.get_instance_metrics()
    telemetry = shared_cache.get_telemetry()
    
    cpu_list = metrics["cpu"]
    queue_list = metrics["queue"]
    p99_list = metrics["p99"]
    
    candidates = []
    n = len(cpu_list)
    weights = [weights[i] if i < len(weights) else 0.0 for i in range(n)]
    max_weight = max(weights) if weights else 0.0

    for i in range(n):
        cpu = cpu_list[i]
        weight = weights[i]
        is_masked = cpu > CPU_MASK_THRESHOLD
        
        status_str = "MASKED_HIGH_CPU" if is_masked else ("TOP_CHOICE" if weight == max_weight else "ACTIVE")
        
        candidates.append({
            "node": f"Node-{i+1}",
            "name": BACKEND_SPECS[i]["name"] if i < len(BACKEND_SPECS) else f"Autoscaled-Node-{i+1}",
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
        "registered_instances_count": len(request.app.state.registry.get_active_instances()),
        "candidate_evaluations": candidates
    }

@router.post("/chaos/inject")
async def inject_chaos(
    request: Request,
    node_idx: int = Query(..., ge=0),
    fault_type: str = Body("cpu_spike", embed=True),
    extra_latency_ms: float = Body(300.0, embed=True)
):
    """Chaos Engineering Fault Injector Endpoint."""
    fleet_size = len(request.app.state.shared_cache.get_instance_metrics()["cpu"])
    if node_idx >= fleet_size:
        raise HTTPException(status_code=422, detail=f"node_idx must be < current fleet size ({fleet_size})")
    http_client = request.app.state.client
    target_url = f"{get_node_url(request.app.state.registry, node_idx)}/chaos/inject"
    try:
        resp = await http_client.post(target_url, json={"fault_type": fault_type, "extra_latency_ms": extra_latency_ms})
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inject chaos on Node {node_idx + 1}: {str(e)}")

@router.post("/chaos/clear")
async def clear_chaos(request: Request):
    """Clears all injected chaos faults across the cluster."""
    http_client = request.app.state.client
    fleet_size = len(request.app.state.shared_cache.get_instance_metrics()["cpu"])
    results = []
    for idx in range(fleet_size):
        url = get_node_url(request.app.state.registry, idx)
        try:
            resp = await http_client.post(f"{url}/chaos/inject", json={"fault_type": "clear"})
            results.append(resp.json())
        except Exception:
            logger.warning("Failed to clear chaos on Node-%d", idx + 1, exc_info=True)
    return {"status": "all_cleared", "nodes": results}

@router.post("/strategy")
async def set_strategy(request: Request, strategy: str = Body(..., embed=True)):
    """Dynamically switches the active load balancing strategy at runtime."""
    strat = strategy.lower()
    if not _is_valid_strategy(strat):
        raise HTTPException(status_code=400, detail=f"Invalid strategy. Choose from: {VALID_STRATEGIES}")
    
    request.app.state.current_strategy = strat
    return {"status": "updated", "current_strategy": strat}

@router.get("/metrics")
async def get_metrics(request: Request):
    """Exposes Prometheus exposition formatted metrics payload."""
    shared_cache = request.app.state.shared_cache
    telemetry = shared_cache.get_telemetry()
    content, content_type = generate_prometheus_metrics(telemetry, qos_shed_counts)
    return Response(content=content, media_type=content_type)

app = create_app()
