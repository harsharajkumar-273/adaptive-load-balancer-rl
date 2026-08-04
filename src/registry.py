# src/registry.py
"""
Dynamic Service Discovery & Registration Engine.
Allows backend microservice nodes to dynamically register on boot, heartbeat health status,
and deregister on shutdown, eliminating hardcoded node list limitations.
"""
import time
import json
import threading
from typing import Dict, List, Any, Optional

class ServiceRegistry:
    def __init__(self, shared_cache):
        self.shared_cache = shared_cache
        self._lock = threading.Lock()
        self._instances: Dict[int, Dict[str, Any]] = {}

    def register_instance(
        self,
        node_idx: int,
        name: str,
        host: str,
        port: int,
        capacity: float = 50.0,
        base_latency_ms: float = 20.0,
        region: str = "us-east"
    ) -> Dict[str, Any]:
        """Registers a new or existing backend instance in the service registry."""
        with self._lock:
            info = {
                "id": node_idx + 1,
                "node_idx": node_idx,
                "name": name,
                "host": host,
                "port": port,
                "url": f"http://{host}:{port}",
                "capacity": capacity,
                "base_latency_ms": base_latency_ms,
                "region": region,
                "registered_at": time.time(),
                "last_heartbeat": time.time(),
                "status": "HEALTHY"
            }
            self._instances[node_idx] = info
            
            # Store in Redis
            try:
                self.shared_cache.client.set(f"registry:instance:{node_idx}", json.dumps(info))
                self.shared_cache.client.set(f"registry:active_nodes_count", str(len(self._instances)))
            except Exception:
                pass
                
            return info

    def deregister_instance(self, node_idx: int) -> bool:
        """Deregisters an instance from the service registry."""
        with self._lock:
            if node_idx in self._instances:
                del self._instances[node_idx]
                try:
                    self.shared_cache.client.delete(f"registry:instance:{node_idx}")
                    self.shared_cache.client.set(f"registry:active_nodes_count", str(len(self._instances)))
                except Exception:
                    pass
                return True
            return False

    def heartbeat(self, node_idx: int) -> bool:
        """Updates last heartbeat timestamp for an instance."""
        with self._lock:
            if node_idx in self._instances:
                self._instances[node_idx]["last_heartbeat"] = time.time()
                return True
            return False

    def get_active_instances(self) -> List[Dict[str, Any]]:
        """Returns list of all active healthy instances."""
        with self._lock:
            now = time.time()
            active = []
            for idx, info in list(self._instances.items()):
                # If heartbeat is fresh (within 3 seconds)
                if now - info.get("last_heartbeat", 0) <= 3.0:
                    info["status"] = "HEALTHY"
                    active.append(info)
                else:
                    info["status"] = "UNHEALTHY"
            return active
