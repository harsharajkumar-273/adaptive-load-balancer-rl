# src/config.py
"""
Configuration parameters for the distributed systems-focused Load Balancer.
"""
import os

# General System Settings
NUM_INSTANCES = 5
PORT = 8000
HOST = "0.0.0.0"  # Expose to allow external traffic
CHECKPOINT_PATH = "model_checkpoint.npz"

# Redis Shared Cache Settings
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))

# Microservice Route Mappings (Resolves to local ports or Docker container hosts)
BACKEND_URLS = [
    os.environ.get("BACKEND_URL_1", "http://127.0.0.1:8001"),
    os.environ.get("BACKEND_URL_2", "http://127.0.0.1:8002"),
    os.environ.get("BACKEND_URL_3", "http://127.0.0.1:8003"),
    os.environ.get("BACKEND_URL_4", "http://127.0.0.1:8004"),
    os.environ.get("BACKEND_URL_5", "http://127.0.0.1:8005")
]

# SLA Thresholds
SLA_LATENCY_MS = 200.0  # Hard SLA threshold in milliseconds

# Loop Intervals & Timeouts
CONTROL_PLANE_INTERVAL_SEC = 0.15  # Control plane RL loop updates every 150ms
DASHBOARD_INTERVAL_SEC = 0.20      # Telemetry dashboard prints every 200ms
STALENESS_THRESHOLD_SEC = 1.0      # Data Plane circuit breaker trips if weights are older than 1.0s

# RL Agent Hyperparameters (Linear Thompson Sampling)
RL_EXPLORATION_PARAM = 0.3         # v^2 parameter for Thompson Sampling variance
RL_TEMPERATURE = 0.2               # Softmax temperature for routing probability distribution
RL_REWARD_WEIGHTS = {
    "latency": 1.0,                # Penalty weight for P99 latency / 200
    "error": 5.0,                  # Penalty weight for fraction of error responses
    "sla": 10.0                    # Penalty weight for fraction of SLA breaches (>200ms)
}

# Backend Node Specifications (used for running microservices)
BACKEND_SPECS = [
    {
        "id": 1,
        "name": "Instance-1 (High-Compute)",
        "base_latency_ms": 10.0,
        "capacity": 150.0,           # Max request handling capacity before queue builds up
        "cpu_multiplier": 0.5,       # How fast CPU grows with queue depth
        "error_threshold_cpu": 95.0, # CPU threshold above which errors begin
        "jitter_range_ms": (0, 2)
    },
    {
        "id": 2,
        "name": "Instance-2 (Standard-Med)",
        "base_latency_ms": 25.0,
        "capacity": 80.0,
        "cpu_multiplier": 0.8,
        "error_threshold_cpu": 95.0,
        "jitter_range_ms": (0, 4)
    },
    {
        "id": 3,
        "name": "Instance-3 (Standard-Slow)",
        "base_latency_ms": 40.0,
        "capacity": 60.0,
        "cpu_multiplier": 1.1,
        "error_threshold_cpu": 95.0,
        "jitter_range_ms": (0, 6)
    },
    {
        "id": 4,
        "name": "Instance-4 (Unstable-Jittery)",
        "base_latency_ms": 20.0,
        "capacity": 50.0,
        "cpu_multiplier": 1.2,
        "error_threshold_cpu": 70.0, # Fails under moderate load!
        "jitter_range_ms": (2, 25)   # High jitter
    },
    {
        "id": 5,
        "name": "Instance-5 (Slow-Legacy)",
        "base_latency_ms": 80.0,
        "capacity": 20.0,
        "cpu_multiplier": 2.5,       # CPU grows extremely fast under load
        "error_threshold_cpu": 90.0,
        "jitter_range_ms": (5, 15)
    }
]
