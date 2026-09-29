# src/shared_state.py
"""
Redis-based distributed state cache with Redis Pub/Sub Event-Driven Streaming & Mock fallback.
"""
import json
import time
import logging
import threading
from typing import Dict, List, Any, Tuple, Optional
import redis
from src.config import REDIS_HOST, REDIS_PORT, REDIS_DB, MAX_INSTANCES

logger = logging.getLogger(__name__)

# A node whose heartbeat is older than this is treated as offline.
HEARTBEAT_TIMEOUT_SEC = 2.0

class MockPubSub:
    def __init__(self, channels_dict, channel_name):
        self.channels_dict = channels_dict
        self.channel_name = channel_name
        self.queue = []
        self.lock = threading.Lock()

    def subscribe(self, *args, **kwargs):
        pass

    def get_message(self, ignore_subscribe_messages=True, timeout=None):
        with self.lock:
            if self.queue:
                msg = self.queue.pop(0)
                return {"type": "message", "channel": self.channel_name, "data": str(msg).encode('utf-8')}
        return None

class MockRedis:
    """Thread-safe in-memory mock of redis.Redis with Pub/Sub support."""
    def __init__(self):
        self._db = {}
        self._pubsub_subscribers = {}
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

    def mset(self, mapping: Dict[str, Any]) -> bool:
        with self._lock:
            for k, v in mapping.items():
                self._db[k] = str(v)
            return True

    def mget(self, keys: List[str]) -> List[Optional[bytes]]:
        return [self.get(k) for k in keys]

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

    def publish(self, channel: str, message: Any) -> int:
        """Publishes message to Mock Pub/Sub subscribers."""
        with self._lock:
            subs = self._pubsub_subscribers.get(channel, [])
            for s in subs:
                with s.lock:
                    s.queue.append(message)
            return len(subs)

    def pubsub(self):
        ps = MockPubSub(self._pubsub_subscribers, "events:request_outcomes")
        with self._lock:
            if "events:request_outcomes" not in self._pubsub_subscribers:
                self._pubsub_subscribers["events:request_outcomes"] = []
            self._pubsub_subscribers["events:request_outcomes"].append(ps)
        return ps

    def ping(self) -> bool:
        return True


class DistributedStateCache:
    """
    Shared cluster state backed by Redis (or an in-process MockRedis fallback).

    `num_instances` is the live fleet size. It starts at `min_instances` and is
    re-discovered from node heartbeats on every `get_instance_metrics()` call, so
    nodes added by the autoscaler (up to `max_instances`) become routable, and
    trailing nodes that stop heartbeating are dropped again.
    """

    _NODE_FIELDS = ("last_seen", "cpu", "queue", "recent_p99", "recent_error_rate", "recent_sla_breach_rate")

    def __init__(self, num_instances: int, max_instances: int = MAX_INSTANCES):
        self.min_instances = num_instances
        self.max_instances = max(num_instances, max_instances)
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
        except Exception:
            logger.warning(
                "Redis unreachable at %s:%s; using in-process MockRedis. "
                "State will NOT be shared across processes.", REDIS_HOST, REDIS_PORT
            )
            self.client = MockRedis()
            self.is_mock = True

        # Initialize default weights if missing
        if self.client.get("rl:routing_weights") is None:
            init_weights = [1.0 / num_instances] * num_instances
            self.set_routing_weights(init_weights)

    @classmethod
    def in_memory(cls, num_instances: int, max_instances: int = MAX_INSTANCES) -> "DistributedStateCache":
        """Builds a cache on MockRedis without attempting a Redis connection (tests, simulations)."""
        cache = cls.__new__(cls)
        cache.min_instances = num_instances
        cache.max_instances = max(num_instances, max_instances)
        cache.num_instances = num_instances
        cache.client = MockRedis()
        cache.is_mock = True
        cache.set_routing_weights([1.0 / num_instances] * num_instances)
        return cache

    def get_routing_weights(self) -> Tuple[List[float], float]:
        """Reads routing weights and the last update timestamp."""
        try:
            weights_raw, last_update_raw = self.client.mget(["rl:routing_weights", "rl:last_updated"])
            weights = json.loads(weights_raw.decode('utf-8')) if weights_raw else [1.0 / self.num_instances] * self.num_instances
            last_update = float(last_update_raw.decode('utf-8')) if last_update_raw else time.time()
            return weights, last_update
        except Exception:
            logger.warning("Failed to read routing weights; using uniform weights", exc_info=True)
            return [1.0 / self.num_instances] * self.num_instances, time.time()

    def set_routing_weights(self, weights: List[float]):
        """Sets new routing weights and updates the timestamp."""
        try:
            total = sum(weights)
            normalized = [w / total for w in weights] if total > 0 else [1.0 / len(weights)] * len(weights)

            self.client.set("rl:routing_weights", json.dumps(normalized))
            self.client.set("rl:last_updated", str(time.time()))
        except Exception:
            logger.warning("Failed to write routing weights", exc_info=True)

    def record_node_heartbeat(self, node_idx: int, cpu: float, queue: int):
        """Called by independent backend node processes to register their load metrics."""
        try:
            self.client.mset({
                f"node:{node_idx}:cpu": str(cpu),
                f"node:{node_idx}:queue": str(queue),
                f"node:{node_idx}:last_seen": str(time.time()),
            })
        except Exception:
            logger.warning("Failed to record heartbeat for node %d", node_idx, exc_info=True)

    @staticmethod
    def _as_float(raw: Optional[bytes], default: float) -> float:
        return float(raw.decode('utf-8')) if raw else default

    def get_instance_metrics(self) -> Dict[str, List[Any]]:
        """
        Collects metrics for every node with a single batched MGET and refreshes
        the live fleet size from heartbeats.
        """
        n_fields = len(self._NODE_FIELDS)
        keys = [f"node:{i}:{field}" for i in range(self.max_instances) for field in self._NODE_FIELDS]
        try:
            raw = self.client.mget(keys)
        except Exception:
            logger.warning("Failed to read node metrics", exc_info=True)
            raw = [None] * len(keys)

        current_time = time.time()
        rows = [raw[i * n_fields:(i + 1) * n_fields] for i in range(self.max_instances)]
        alive = [
            current_time - self._as_float(row[0], 0.0) <= HEARTBEAT_TIMEOUT_SEC
            for row in rows
        ]

        # Fleet = the configured minimum plus any higher-indexed nodes that are alive.
        highest_alive = max((i for i, a in enumerate(alive) if a), default=-1)
        self.num_instances = max(self.min_instances, highest_alive + 1)

        metrics = {"cpu": [], "queue": [], "p99": [], "error_rate": [], "sla_breach_rate": []}
        for i in range(self.num_instances):
            _, cpu_raw, queue_raw, p99_raw, err_raw, sla_raw = rows[i]
            if alive[i]:
                metrics["cpu"].append(self._as_float(cpu_raw, 5.0))
                metrics["queue"].append(int(queue_raw.decode('utf-8')) if queue_raw else 0)
            else:
                # Offline: report saturated so every strategy masks it.
                metrics["cpu"].append(100.0)
                metrics["queue"].append(999)
            metrics["p99"].append(self._as_float(p99_raw, 0.0))
            metrics["error_rate"].append(self._as_float(err_raw, 0.0))
            metrics["sla_breach_rate"].append(self._as_float(sla_raw, 0.0))

        return metrics

    def record_request_outcome(self, node_idx: int, latency_ms: float, success: bool, breached_sla: bool):
        """Called by the Gateway to record per-request telemetry into the current feedback window."""
        try:
            pipe = self.client.pipeline() if hasattr(self.client, "pipeline") else self.client
            pipe.incr("global:total_requests")
            if breached_sla:
                pipe.incr("global:breached_sla_requests")

            pipe.lpush("global:system_latencies", str(latency_ms))
            pipe.ltrim("global:system_latencies", 0, 1999)

            pipe.incr(f"window:node:{node_idx}:requests")
            pipe.lpush(f"window:node:{node_idx}:latencies", str(latency_ms))
            if not success:
                pipe.incr(f"window:node:{node_idx}:errors")
            if breached_sla:
                pipe.incr(f"window:node:{node_idx}:sla_breaches")

            # Outcome event stream for external observers (dashboards, loggers).
            # The RL agent does NOT learn from this stream; it learns only from the
            # windowed aggregates above, so each outcome is counted exactly once.
            payload = json.dumps({
                "node_idx": node_idx,
                "latency_ms": latency_ms,
                "success": success,
                "breached_sla": breached_sla,
                "timestamp": time.time()
            })
            pipe.publish("events:request_outcomes", payload)
            if pipe is not self.client:
                pipe.execute()
        except Exception:
            logger.warning("Failed to record request outcome for node %d", node_idx, exc_info=True)

    def flush_window_feedback(self) -> Dict[str, Any]:
        """Called by RL control agent to gather window feedback and flush counters."""
        feedback = []
        current_metrics = self.get_instance_metrics()
        cpu_list = current_metrics["cpu"]
        queue_list = current_metrics["queue"]

        for i in range(self.num_instances):
            try:
                reqs_raw = self.client.getset(f"window:node:{i}:requests", 0)
                errs_raw = self.client.getset(f"window:node:{i}:errors", 0)
                sla_raw = self.client.getset(f"window:node:{i}:sla_breaches", 0)

                reqs = int(reqs_raw.decode('utf-8')) if reqs_raw else 0
                errs = int(errs_raw.decode('utf-8')) if errs_raw else 0
                sla_breaches = int(sla_raw.decode('utf-8')) if sla_raw else 0

                latencies_raw = self.client.lrange(f"window:node:{i}:latencies", 0, -1)
                self.client.delete(f"window:node:{i}:latencies")

                latencies = [float(l.decode('utf-8')) for l in latencies_raw] if latencies_raw else []

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

                    self.client.set(f"node:{i}:recent_p99", str(p99))
                    self.client.set(f"node:{i}:recent_error_rate", str(err_rate))
                    self.client.set(f"node:{i}:recent_sla_breach_rate", str(sla_breach_rate))
                else:
                    # No traffic this window: decay the stale estimates towards zero.
                    p99_old, err_old, sla_old = (
                        self._as_float(v, 0.0) for v in self.client.mget([
                            f"node:{i}:recent_p99",
                            f"node:{i}:recent_error_rate",
                            f"node:{i}:recent_sla_breach_rate",
                        ])
                    )
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
                logger.warning("Failed to flush feedback window for node %d", i, exc_info=True)
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
            total_raw, breached_raw, rate_raw, spike_raw, cb_raw = self.client.mget([
                "global:total_requests",
                "global:breached_sla_requests",
                "global:request_rate",
                "global:traffic_spike_active",
                "global:circuit_breaker_tripped",
            ])

            total_requests = int(total_raw.decode('utf-8')) if total_raw else 0
            breached_sla_requests = int(breached_raw.decode('utf-8')) if breached_raw else 0
            global_request_rate = float(rate_raw.decode('utf-8')) if rate_raw else 0.0
            traffic_spike_active = int(spike_raw.decode('utf-8')) == 1 if spike_raw else False
            circuit_breaker_tripped = int(cb_raw.decode('utf-8')) == 1 if cb_raw else False

            system_lats_raw = self.client.lrange("global:system_latencies", 0, -1)
            system_p99 = 0.0
            if system_lats_raw:
                lats = [float(l.decode('utf-8')) for l in system_lats_raw]
                lats.sort()
                idx = int(len(lats) * 0.99)
                system_p99 = lats[min(idx, len(lats) - 1)]

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
            logger.warning("Failed to read telemetry", exc_info=True)
            return {}

    def _set_flag(self, key: str, value: str):
        try:
            self.client.set(key, value)
        except Exception:
            logger.warning("Failed to set %s", key, exc_info=True)

    def set_global_request_rate(self, rate: float):
        self._set_flag("global:request_rate", str(rate))

    def set_traffic_spike_active(self, active: bool):
        self._set_flag("global:traffic_spike_active", "1" if active else "0")

    def set_circuit_breaker(self, tripped: bool):
        self._set_flag("global:circuit_breaker_tripped", "1" if tripped else "0")
