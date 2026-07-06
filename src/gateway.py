# src/gateway.py
"""
Data Plane API Gateway implementing fast routing, action masking, and circuit breaker.
"""
import time
import random
from fastapi import FastAPI, Request, HTTPException, status
from src.config import STALENESS_THRESHOLD_SEC

app = FastAPI(title="AI-Driven Asynchronous Load Balancer Gateway")

@app.get("/play-video")
async def play_video(request: Request):
    """
    Synchronous-like fast request router.
    Resolves a target backend instance stochastically under 1ms overhead,
    then executes the simulated request.
    """
    shared_cache = request.app.state.shared_cache
    simulator = request.app.state.simulator
    
    current_time = time.time()
    
    # 1. Fetch latest weights and metrics
    weights, last_update = shared_cache.get_routing_weights()
    metrics = shared_cache.get_instance_metrics()
    cpu_list = metrics["cpu"]
    
    # 2. Safety Guardrail: Circuit Breaker for Stale Cache
    is_stale = (current_time - last_update) > STALENESS_THRESHOLD_SEC
    
    if is_stale:
        shared_cache.set_circuit_breaker(True)
        # Fall back to Least Connections
        chosen_idx = simulator.get_least_connections_idx()
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
            # Emergency Fallback: If all backends are masked (>85% CPU)
            chosen_idx = simulator.get_least_connections_idx()
            routing_mode = "fallback_all_masked_least_conn"

    # 4. Route request to selected simulator backend instance (asynchronous sleep simulation)
    latency_ms, success = await simulator.route_to_instance(chosen_idx)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Backend Instance {chosen_idx + 1} Overloaded (503 Error)"
        )
        
    return {
        "status": "success",
        "backend_id": chosen_idx + 1,
        "latency_ms": round(latency_ms, 2),
        "routing_mode": routing_mode
    }

@app.get("/metrics")
async def get_metrics(request: Request):
    """Exposes current system telemetry metrics."""
    shared_cache = request.app.state.shared_cache
    return shared_cache.get_telemetry()
