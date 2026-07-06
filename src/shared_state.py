# src/shared_state.py
"""
Thread-safe shared state cache to bridge the Control Plane and Data Plane.
"""
import time
import threading
from typing import Dict, List, Any

class SharedMemoryCache:
    def __init__(self, num_instances: int):
        self._lock = threading.Lock()
        
        # Initial routing weights (equal probability)
        self._weights = [1.0 / num_instances] * num_instances
        self._last_weights_update = time.time()
        
        # Instance Metrics
        self._cpu_utilization = [0.0] * num_instances
        self._queue_depths = [0] * num_instances
        self._recent_p99_latencies = [0.0] * num_instances
        self._recent_error_rates = [0.0] * num_instances
        self._recent_sla_breach_rates = [0.0] * num_instances
        
        # Global Telemetry
        self._total_requests = 0
        self._breached_sla_requests = 0
        self._system_latencies: List[float] = []
        self._global_request_rate = 0.0
        self._traffic_spike_active = False
        self._circuit_breaker_tripped = False
        
        # Dynamic window feedback metrics (reset each control plane interval)
        self._window_requests = [0] * num_instances
        self._window_latencies = [[] for _ in range(num_instances)]
        self._window_errors = [0] * num_instances

    def get_routing_weights(self) -> tuple[List[float], float]:
        """Returns the current weights and the timestamp they were last updated."""
        with self._lock:
            return list(self._weights), self._last_weights_update

    def set_routing_weights(self, weights: List[float]):
        """Sets the new routing weights and updates the timestamp."""
        with self._lock:
            # Basic sanity check
            total = sum(weights)
            if total > 0:
                self._weights = [w / total for w in weights]
            else:
                self._weights = [1.0 / len(weights)] * len(weights)
            self._last_weights_update = time.time()

    def get_instance_metrics(self) -> Dict[str, List[Any]]:
        """Returns a copy of all current instance metrics."""
        with self._lock:
            return {
                "cpu": list(self._cpu_utilization),
                "queue": list(self._queue_depths),
                "p99": list(self._recent_p99_latencies),
                "error_rate": list(self._recent_error_rates),
                "sla_breach_rate": list(self._recent_sla_breach_rates)
            }

    def update_instance_metrics(self, instance_idx: int, cpu: float, queue: int):
        """Updates CPU and Queue Depth for a specific instance."""
        with self._lock:
            self._cpu_utilization[instance_idx] = cpu
            self._queue_depths[instance_idx] = queue

    def record_request_outcome(self, instance_idx: int, latency_ms: float, success: bool, breached_sla: bool):
        """Records the latency and success of a request at the Data Plane."""
        with self._lock:
            self._total_requests += 1
            if breached_sla:
                self._breached_sla_requests += 1
            
            # Keep system P99 latency history capped
            self._system_latencies.append(latency_ms)
            if len(self._system_latencies) > 2000:
                self._system_latencies.pop(0)

            # Record in window-level metrics for RL feedback
            self._window_requests[instance_idx] += 1
            self._window_latencies[instance_idx].append(latency_ms)
            if not success:
                self._window_errors[instance_idx] += 1

    def flush_window_feedback(self) -> Dict[str, Any]:
        """
        Calculates P99 latency, error rates, and SLA breaches for this window,
        updates the metrics, and flushes window metrics for the next RL iteration.
        """
        with self._lock:
            feedback = []
            for i in range(len(self._weights)):
                reqs = self._window_requests[i]
                lats = self._window_latencies[i]
                errs = self._window_errors[i]

                # Compute window stats
                p99 = 0.0
                sla_breach = 0.0
                err_rate = 0.0
                if reqs > 0:
                    lats.sort()
                    idx = int(len(lats) * 0.99)
                    p99 = lats[min(idx, len(lats) - 1)]
                    sla_breach = sum(1 for l in lats if l > 200.0) / reqs
                    err_rate = errs / reqs
                    
                    # Update cache values for dashboard representation
                    self._recent_p99_latencies[i] = p99
                    self._recent_error_rates[i] = err_rate
                    self._recent_sla_breach_rates[i] = sla_breach
                else:
                    # Decay latency metrics if idle
                    self._recent_p99_latencies[i] = max(0.0, self._recent_p99_latencies[i] * 0.8)
                    self._recent_error_rates[i] = max(0.0, self._recent_error_rates[i] * 0.8)
                    self._recent_sla_breach_rates[i] = max(0.0, self._recent_sla_breach_rates[i] * 0.8)

                feedback.append({
                    "instance_idx": i,
                    "requests_count": reqs,
                    "p99": p99,
                    "error_rate": err_rate,
                    "sla_breach_rate": sla_breach
                })

            # Reset window arrays
            self._window_requests = [0] * len(self._weights)
            self._window_latencies = [[] for _ in range(len(self._weights))]
            self._window_errors = [0] * len(self._weights)

            return {
                "feedback": feedback,
                "cpu": list(self._cpu_utilization),
                "queue": list(self._queue_depths)
            }

    # Dashboard Metrics Accessors
    def get_telemetry(self) -> Dict[str, Any]:
        """Fetches consolidated statistics for the telemetry dashboard."""
        with self._lock:
            # Compute system P99 latency
            sys_p99 = 0.0
            if self._system_latencies:
                sorted_lats = sorted(self._system_latencies)
                idx = int(len(sorted_lats) * 0.99)
                sys_p99 = sorted_lats[min(idx, len(sorted_lats) - 1)]

            return {
                "total_requests": self._total_requests,
                "breached_sla_requests": self._breached_sla_requests,
                "system_p99_latency": sys_p99,
                "weights": list(self._weights),
                "cpu": list(self._cpu_utilization),
                "queue": list(self._queue_depths),
                "p99_latencies": list(self._recent_p99_latencies),
                "error_rates": list(self._recent_error_rates),
                "global_request_rate": self._global_request_rate,
                "traffic_spike_active": self._traffic_spike_active,
                "circuit_breaker_tripped": self._circuit_breaker_tripped
            }

    def set_global_request_rate(self, rate: float):
        with self._lock:
            self._global_request_rate = rate

    def set_traffic_spike_active(self, active: bool):
        with self._lock:
            self._traffic_spike_active = active

    def set_circuit_breaker(self, tripped: bool):
        with self._lock:
            self._circuit_breaker_tripped = tripped
