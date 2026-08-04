# src/chaos.py
"""
Chaos Engineering Fault Injection Module.
Injects simulated CPU spikes, artificial latency delays, and HTTP error spikes into backend nodes.
"""
from typing import Dict, Any, Optional

class ChaosEngine:
    def __init__(self):
        # Fault state per node index
        self._faults: Dict[int, Dict[str, Any]] = {}

    def inject_fault(self, node_idx: int, fault_type: str, duration_sec: float = 10.0, extra_latency_ms: float = 300.0):
        """Injects a fault into a specific node."""
        self._faults[node_idx] = {
            "type": fault_type.lower(),
            "extra_latency_ms": extra_latency_ms,
            "error_rate": 0.9 if fault_type == "error_spike" else 0.0,
            "forced_cpu": 99.0 if fault_type == "cpu_spike" else None
        }

    def clear_fault(self, node_idx: Optional[int] = None):
        """Clears chaos fault injections for a single node or all nodes."""
        if node_idx is not None:
            if node_idx in self._faults:
                del self._faults[node_idx]
        else:
            self._faults.clear()

    def get_fault(self, node_idx: int) -> Optional[Dict[str, Any]]:
        """Retrieves active fault configuration for a node."""
        return self._faults.get(node_idx)

    def list_active_faults(self) -> Dict[int, Dict[str, Any]]:
        """Returns all active chaos faults across the cluster."""
        return dict(self._faults)

# Global chaos engine instance
chaos_manager = ChaosEngine()
