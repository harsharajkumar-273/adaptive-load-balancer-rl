# src/config.py
"""
Configuration module supporting dynamic config.yaml and config.json parsing.
"""
import os
import json

def load_yaml_fallback(filepath: str) -> dict:
    """Simple parser for simple YAML key-value/list structures without external dependencies."""
    if not os.path.exists(filepath):
        return {}
    
    # Try importing PyYAML first
    try:
        import yaml
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    except ImportError:
        pass
    
    # Basic line parser fallback
    data = {}
    current_section = None
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
                
            # Section header
            if line.rstrip().endswith(":") and not line.startswith(" ") and not line.startswith("\t") and not line.startswith("-"):
                current_section = line.rstrip()[:-1].strip()
                if current_section == "backends":
                    data[current_section] = []
                else:
                    data[current_section] = {}
                continue
                
            if ":" in line:
                parts = line.split(":", 1)
                key = parts[0].strip()
                val = parts[1].strip()
                
                if "#" in val:
                    val = val.split("#")[0].strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                    
                parsed_val = val
                if val.lower() == "true":
                    parsed_val = True
                elif val.lower() == "false":
                    parsed_val = False
                elif val:
                    try:
                        if "." in val:
                            parsed_val = float(val)
                        else:
                            parsed_val = int(val)
                    except ValueError:
                        pass
                
                if current_section == "backends" and isinstance(data.get("backends"), list):
                    if key.startswith("-"):
                        key = key[1:].strip()
                        data["backends"].append({key: parsed_val})
                    elif data["backends"]:
                        data["backends"][-1][key] = parsed_val
                elif current_section and current_section in data and isinstance(data[current_section], dict):
                    data[current_section][key] = parsed_val
                else:
                    data[key] = parsed_val

    return data

# Load configuration file
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
_yaml_config = load_yaml_fallback(CONFIG_PATH)

def _get_dict(source, key):
    val = source.get(key, {}) if isinstance(source, dict) else {}
    return val if isinstance(val, dict) else {}

# Cluster Settings
_cluster = _get_dict(_yaml_config, "cluster")
NUM_INSTANCES = int(_cluster.get("num_instances", 5))
# Upper bound on fleet size (autoscaler ceiling). The live fleet size is
# discovered from node heartbeats and always lies in [NUM_INSTANCES, MAX_INSTANCES].
MAX_INSTANCES = int(os.environ.get("MAX_INSTANCES", _cluster.get("max_instances", 10)))
PORT = int(os.environ.get("PORT", _cluster.get("port", 8000)))
HOST = os.environ.get("HOST", _cluster.get("host", "0.0.0.0"))
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", _cluster.get("checkpoint_path", "model_checkpoint.npz"))
ROUTING_STRATEGY = os.environ.get("ROUTING_STRATEGY", str(_cluster.get("routing_strategy", "lin_ts"))).lower()

# Redis Shared Cache Settings
_redis = _get_dict(_yaml_config, "redis")
REDIS_HOST = os.environ.get("REDIS_HOST", _redis.get("host", "127.0.0.1"))
REDIS_PORT = int(os.environ.get("REDIS_PORT", _redis.get("port", 6379)))
REDIS_DB = int(os.environ.get("REDIS_DB", _redis.get("db", 0)))

# SLA Thresholds
_sla = _get_dict(_yaml_config, "sla")
SLA_LATENCY_MS = float(_sla.get("latency_ms", 200.0))
STALENESS_THRESHOLD_SEC = float(_sla.get("staleness_threshold_sec", 1.0))

# Agent Settings
_agent = _get_dict(_yaml_config, "agent")
CONTROL_PLANE_INTERVAL_SEC = float(_agent.get("control_plane_interval_sec", 0.15))
DASHBOARD_INTERVAL_SEC = 0.20
RL_EXPLORATION_PARAM = float(_agent.get("exploration_param", 0.3))
RL_TEMPERATURE = float(_agent.get("temperature", 0.2))

_reward_weights = _get_dict(_agent, "reward_weights")
RL_REWARD_WEIGHTS = {
    "latency": float(_reward_weights.get("latency", 1.0)),
    "error": float(_reward_weights.get("error", 5.0)),
    "sla": float(_reward_weights.get("sla", 10.0))
}

# Backend Specifications
BACKEND_SPECS = [
    {
        "id": 1,
        "name": "Instance-1 (High-Compute)",
        "base_latency_ms": 10.0,
        "capacity": 150.0,
        "cpu_multiplier": 0.5,
        "error_threshold_cpu": 95.0,
        "jitter_range_ms": (0.0, 2.0),
        "port": 8001
    },
    {
        "id": 2,
        "name": "Instance-2 (Standard-Med)",
        "base_latency_ms": 25.0,
        "capacity": 80.0,
        "cpu_multiplier": 0.8,
        "error_threshold_cpu": 95.0,
        "jitter_range_ms": (0.0, 4.0),
        "port": 8002
    },
    {
        "id": 3,
        "name": "Instance-3 (Standard-Slow)",
        "base_latency_ms": 40.0,
        "capacity": 60.0,
        "cpu_multiplier": 1.1,
        "error_threshold_cpu": 95.0,
        "jitter_range_ms": (0.0, 6.0),
        "port": 8003
    },
    {
        "id": 4,
        "name": "Instance-4 (Unstable-Jittery)",
        "base_latency_ms": 20.0,
        "capacity": 50.0,
        "cpu_multiplier": 1.2,
        "error_threshold_cpu": 70.0,
        "jitter_range_ms": (2.0, 25.0),
        "port": 8004
    },
    {
        "id": 5,
        "name": "Instance-5 (Slow-Legacy)",
        "base_latency_ms": 80.0,
        "capacity": 20.0,
        "cpu_multiplier": 2.5,
        "error_threshold_cpu": 90.0,
        "jitter_range_ms": (5.0, 15.0),
        "port": 8005
    }
]

BACKEND_URLS = [
    os.environ.get("BACKEND_URL_1", "http://127.0.0.1:8001"),
    os.environ.get("BACKEND_URL_2", "http://127.0.0.1:8002"),
    os.environ.get("BACKEND_URL_3", "http://127.0.0.1:8003"),
    os.environ.get("BACKEND_URL_4", "http://127.0.0.1:8004"),
    os.environ.get("BACKEND_URL_5", "http://127.0.0.1:8005")
]
