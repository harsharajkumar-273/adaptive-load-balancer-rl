# src/shared_state.py
"""
Redis-based distributed state cache with a robust thread-safe in-memory Mock fallback.
"""
import json
import time
import threading
from typing import Dict, List, Any, Tuple, Optional
import redis
from src.config import REDIS_HOST, REDIS_PORT, REDIS_DB

class MockRedis:
    """Thread-safe in-memory mock of redis.Redis for seamless local fallback."""
    def __init__(self):
        self._db = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            val = self._db.get(key)
            if val is None:
                return None
            return str(val).encode('utf-8')

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        with self._lock:
            self._db[key] = str(value)
            return True

    def getset(self, key: str, value: Any) -> Optional[bytes]:
        with self._lock:
            old_val = self._db.get(key)
            self._db[key] = str(value)
            if old_val is None:
                return None
            return str(old_val).encode('utf-8')

    def incr(self, key: str, amount: int = 1) -> int:
        with self._lock:
            val = int(self._db.get(key, 0))
            new_val = val + amount
            self._db[key] = str(new_val)
            return new_val

    def decr(self, key: str, amount: int = 1) -> int:
        with self._lock:
            val = int(self._db.get(key, 0))
            new_val = val - amount
            self._db[key] = str(new_val)
            return new_val

    def lpush(self, key: str, *values: Any) -> int:
        with self._lock:
            if key not in self._db:
                self._db[key] = []
            elif not isinstance(self._db[key], list):
                self._db[key] = [str(self._db[key])]
            
            for val in reversed(values):
                self._db[key].insert(0, str(val))
            return len(self._db[key])

    def lrange(self, key: str, start: int, end: int) -> List[bytes]:
        with self._lock:
            lst = self._db.get(key, [])
            if not isinstance(lst, list):
                return []
            if end == -1:
                sub = lst[start:]
            else:
                sub = lst[start:end+1]
            return [str(x).encode('utf-8') for x in sub]

    def ltrim(self, key: str, start: int, end: int) -> bool:
        with self._lock:
            lst = self._db.get(key, [])
            if not isinstance(lst, list):
                return False
            if end == -1:
                self._db[key] = lst[start:]
            else:
                self._db[key] = lst[start:end+1]
            return True

    def delete(self, *keys: str) -> int:
        with self._lock:
            count = 0
            for key in keys:
                if key in self._db:
                    del self._db[key]
                    count += 1
            return count

    def ping(self) -> bool:
        return True


class DistributedStateCache:
    def __init__(self, num_instances: int):
        self.num_instances = num_instances
        
        # Connect to Redis with fallback
        try:
            self.client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                socket_connect_timeout=1.0
            )
            self.client.ping()
            self.is_mock = False
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            print(f"\033[33m[SharedState] Redis unavailable at {REDIS_HOST}:{REDIS_PORT}. Falling back to In-Memory Mock.\033[0m")
            self.client = MockRedis()
            self.is_mock = True

        # Initialize default weights if missing
        if self.client.get("rl:routing_weights") is None:
            init_weights = [1.0 / num_instances] * num_instances
            self.set_routing_weights(init_weights)

    def get_routing_weights(self) -> Tuple[List[float], float]:
        """Reads routing weights and the last update timestamp."""
        try:
            weights_raw = self.client.get("rl:routing_weights")
            last_update_raw = self.client.get("rl:last_updated")
            
            weights = json.loads(weights_raw.decode('utf-8')) if weights_raw else [1.0 / self.num_instances] * self.num_instances
            last_update = float(last_update_raw.decode('utf-8')) if last_update_raw else time.time()
            return weights, last_update
        except Exception:
            return [1.0 / self.num_instances] * self.num_instances, time.time()

    def set_routing_weights(self, weights: List[float]):
        """Sets new routing weights and updates the timestamp."""
        try:
            total = sum(weights)
            normalized = [w / total for w in weights] if total > 0 else [1.0 / len(weights)] * len(weights)
            
            self.client.set("rl:routing_weights", json.dumps(normalized))
            self.client.set("rl:last_updated", str(time.time()))
        except Exception:
            pass

    def record_node_heartbeat(self, node_idx: int, cpu: float, queue: int):
        """Called by independent backend node processes to register their load metrics."""
        try:
            self.client.set(f"node:{node_idx}:cpu", str(cpu))
            self.client.set(f"node:{node_idx}:queue", str(queue))
            self.client.set(f"node:{node_idx}:last_seen", str(time.time()))
        except Exception:
            pass

    def get_instance_metrics(self) -> Dict[str, List[Any]]:
        """Collects metrics across all nodes for gateway routing & dashboard checks."""
        cpu_list = [0.0] * self.num_instances
        queue_list = [0] * self.num_instances
        p99_list = [0.0] * self.num_instances
        error_list = [0.0] * self.num_instances
        sla_list = [0.0] * self.num_instances
        
        current_time = time.time()

        for i in range(self.num_instances):
            try:
                # Retrieve metrics. Expiry check: if node hasn't heartbeated in > 2.0s, treat it as dead.
                last_seen_raw = self.client.get(f"node:{i}:last_seen")
                last_seen = float(last_seen_raw.decode('utf-8')) if last_seen_raw else 0.0
                
                if current_time - last_seen > 2.0:
                    # Node is offline/dead: Force CPU to 100% and queue to high value to prevent routing
                    cpu_list[i] = 100.0
                    queue_list[i] = 999
                else:
                    cpu_raw = self.client.get(f"node:{i}:cpu")
                    queue_raw = self.client.get(f"node:{i}:queue")
                    cpu_list[i] = float(cpu_raw.decode('utf-8')) if cpu_raw else 5.0
                    queue_list[i] = int(queue_raw.decode('utf-8')) if queue_raw else 0

                # Fetch static historical evaluations
                p99_raw = self.client.get(f"node:{i}:recent_p99")
                err_raw = self.client.get(f"node:{i}:recent_error_rate")
                sla_raw = self.client.get(f"node:{i}:recent_sla_breach_rate")

                p99_list[i] = float(p99_raw.decode('utf-8')) if p99_raw else 0.0
                error_list[i] = float(err_raw.decode('utf-8')) if err_raw else 0.0
                sla_list[i] = float(sla_raw.decode('utf-8')) if sla_raw else 0.0
            except Exception:
                pass

        return {
            "cpu": cpu_list,
            "queue": queue_list,
            "p99": p99_list,
            "error_rate": error_list,
            "sla_breach_rate": sla_list
        }

    def record_request_outcome(self, node_idx: int, latency_ms: float, success: bool, breached_sla: bool):
        """Called by Gateway to record telemetry for each forwarded HTTP request."""
        try:
            self.client.incr("global:total_requests")
            if breached_sla:
                self.client.incr("global:breached_sla_requests")

            # Record system-wide P99 telemetry list
            self.client.lpush("global:system_latencies", str(latency_ms))
            self.client.ltrim("global:system_latencies", 0, 1999)

            # Record window-level counters for RL feedback loop
            self.client.incr(f"window:node:{node_idx}:requests")
            self.client.lpush(f"window:node:{node_idx}:latencies", str(latency_ms))
            if not success:
                self.client.incr(f"window:node:{node_idx}:errors")
            if breached_sla:
                self.client.incr(f"window:node:{node_idx}:sla_breaches")
        except Exception:
            pass

    def flush_window_feedback(self) -> Dict[str, Any]:
        """Called by RL control agent to gather the feedback of the last window and flush counters."""
        feedback = []
        cpu_list = []
        queue_list = []
        
        current_metrics = self.get_instance_metrics()
        cpu_list = current_metrics["cpu"]
        queue_list = current_metrics["queue"]

        for i in range(self.num_instances):
            try:
                # Flush window counts using getset and delete
                reqs_raw = self.client.getset(f"window:node:{i}:requests", 0)
                errs_raw = self.client.getset(f"window:node:{i}:errors", 0)
                sla_raw = self.client.getset(f"window:node:{i}:sla_breaches", 0)
                
                reqs = int(reqs_raw.decode('utf-8')) if reqs_raw else 0
                errs = int(errs_raw.decode('utf-8')) if errs_raw else 0
                sla_breaches = int(sla_raw.decode('utf-8')) if sla_raw else 0

                # Pull latencies and delete the list
                latencies_raw = self.client.lrange(f"window:node:{i}:latencies", 0, -1)
                self.client.delete(f"window:node:{i}:latencies")
                
                latencies = [float(l.decode('utf-8')) for l in latencies_raw] if latencies_raw else []

                # Calculate P99 and rates
                p99 = 0.0
                err_rate = 0.0
                sla_breach_rate = 0.0
                
                if reqs > 0:
                    if latencies:
                        latencies.sort()
                        idx = int(len(latencies) * 0.99)
                        p99 = latencies[min(idx, len(latencies) - 1)]
                    err_rate = errs / reqs
                    sla_breach_rate = sla_breaches / reqs

                    # Update keys for visualization & gateway
                    self.client.set(f"node:{i}:recent_p99", str(p99))
                    self.client.set(f"node:{i}:recent_error_rate", str(err_rate))
                    self.client.set(f"node:{i}:recent_sla_breach_rate", str(sla_breach_rate))
                else:
                    # Decay slowly
                    p99_old = float(self.client.get(f"node:{i}:recent_p99").decode('utf-8')) if self.client.get(f"node:{i}:recent_p99") else 0.0
                    err_old = float(self.client.get(f"node:{i}:recent_error_rate").decode('utf-8')) if self.client.get(f"node:{i}:recent_error_rate") else 0.0
                    sla_old = float(self.client.get(f"node:{i}:recent_sla_breach_rate").decode('utf-8')) if self.client.get(f"node:{i}:recent_sla_breach_rate") else 0.0
                    
                    self.client.set(f"node:{i}:recent_p99", str(p99_old * 0.8))
                    self.client.set(f"node:{i}:recent_error_rate", str(err_old * 0.8))
                    self.client.set(f"node:{i}:recent_sla_breach_rate", str(sla_old * 0.8))

                feedback.append({
                    "instance_idx": i,
                    "requests_count": reqs,
                    "p99": p99,
                    "error_rate": err_rate,
                    "sla_breach_rate": sla_breach_rate
                })
            except Exception:
                feedback.append({
                    "instance_idx": i,
                    "requests_count": 0,
                    "p99": 0.0,
                    "error_rate": 0.0,
                    "sla_breach_rate": 0.0
                })

        return {
            "feedback": feedback,
            "cpu": cpu_list,
            "queue": queue_list
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Reads status metrics for the telemetry dashboard."""
        try:
            total_raw = self.client.get("global:total_requests")
            breached_raw = self.client.get("global:breached_sla_requests")
            rate_raw = self.client.get("global:request_rate")
            spike_raw = self.client.get("global:traffic_spike_active")
            cb_raw = self.client.get("global:circuit_breaker_tripped")
            
            total_requests = int(total_raw.decode('utf-8')) if total_raw else 0
            breached_sla_requests = int(breached_raw.decode('utf-8')) if breached_raw else 0
            global_request_rate = float(rate_raw.decode('utf-8')) if rate_raw else 0.0
            traffic_spike_active = int(spike_raw.decode('utf-8')) == 1 if spike_raw else False
            circuit_breaker_tripped = int(cb_raw.decode('utf-8')) == 1 if cb_raw else False

            # Calculate overall system P99 from lists
            system_lats_raw = self.client.lrange("global:system_latencies", 0, -1)
            system_p99 = 0.0
            if system_lats_raw:
                lats = [float(l.decode('utf-8')) for l in system_lats_raw]
                lats.sort()
                idx = int(len(lats) * 0.99)
                system_p99 = lats[min(idx, len(lats) - 1)]

            # Backend arrays
            weights, _ = self.get_routing_weights()
            metrics = self.get_instance_metrics()

            return {
                "total_requests": total_requests,
                "breached_sla_requests": breached_sla_requests,
                "system_p99_latency": system_p99,
                "weights": weights,
                "cpu": metrics["cpu"],
                "queue": metrics["queue"],
                "p99_latencies": metrics["p99"],
                "error_rates": metrics["error_rate"],
                "global_request_rate": global_request_rate,
                "traffic_spike_active": traffic_spike_active,
                "circuit_breaker_tripped": circuit_breaker_tripped
            }
        except Exception:
            return {}

    def set_global_request_rate(self, rate: float):
        try:
            self.client.set("global:request_rate", str(rate))
        except Exception:
            pass

    def set_traffic_spike_active(self, active: bool):
        try:
            self.client.set("global:traffic_spike_active", "1" if active else "0")
        except Exception:
            pass

    def set_circuit_breaker(self, tripped: bool):
        try:
            self.client.set("global:circuit_breaker_tripped", "1" if tripped else "0")
        except Exception:
            pass
