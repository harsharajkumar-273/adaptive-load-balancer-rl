# src/metrics.py
"""
Prometheus Metrics Exporter module for distributed observability.
Exposes standard Prometheus metrics (Histograms, Gauges, Counters) for P50/P95/P99 latencies,
routing weights, node CPU/queue utilization, circuit breaker status, and QoS load shedding.
"""
import time
from typing import Dict, Any

try:
    from prometheus_client import Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

if HAS_PROMETHEUS:
    # Prometheus Metrics
    PROM_ROUTING_WEIGHT = Gauge("load_balancer_routing_weight", "Active routing weight per backend node", ["node"])
    PROM_NODE_CPU = Gauge("load_balancer_node_cpu_utilization", "Real-time CPU load percentage per node", ["node"])
    PROM_NODE_QUEUE = Gauge("load_balancer_node_queue_depth", "Active connection queue depth per node", ["node"])
    PROM_CIRCUIT_BREAKER = Gauge("load_balancer_circuit_breaker_status", "Circuit breaker status (0=Closed, 1=Tripped)")
    PROM_QOS_SHED_TOTAL = Counter("load_balancer_qos_load_shed_total", "Total requests shed due to QoS traffic shaping", ["priority"])
    PROM_REQUEST_LATENCY = Histogram(
        "load_balancer_http_request_duration_seconds",
        "HTTP request duration histogram in seconds",
        ["endpoint", "priority", "routed_to"],
        buckets=(0.01, 0.025, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0)
    )
else:
    PROM_ROUTING_WEIGHT = None
    PROM_NODE_CPU = None
    PROM_NODE_QUEUE = None
    PROM_CIRCUIT_BREAKER = None
    PROM_QOS_SHED_TOTAL = None
    PROM_REQUEST_LATENCY = None

def generate_prometheus_metrics(telemetry: Dict[str, Any], qos_shed_counts: Dict[str, int]) -> tuple[bytes, str]:
    """Generates standard Prometheus exposition formatted metrics payload."""
    if HAS_PROMETHEUS:
        weights = telemetry.get("weights", [])
        cpu_list = telemetry.get("cpu", [])
        queue_list = telemetry.get("queue", [])
        cb_tripped = 1.0 if telemetry.get("circuit_breaker_tripped", False) else 0.0

        for i in range(len(weights)):
            node_label = f"Node-{i+1}"
            PROM_ROUTING_WEIGHT.labels(node=node_label).set(weights[i])
            if i < len(cpu_list):
                PROM_NODE_CPU.labels(node=node_label).set(cpu_list[i])
            if i < len(queue_list):
                PROM_NODE_QUEUE.labels(node=node_label).set(queue_list[i])

        PROM_CIRCUIT_BREAKER.set(cb_tripped)
        
        for priority, count in qos_shed_counts.items():
            current = PROM_QOS_SHED_TOTAL.labels(priority=priority)._value.get()
            if count > current:
                PROM_QOS_SHED_TOTAL.labels(priority=priority).inc(count - current)

        return generate_latest(), CONTENT_TYPE_LATEST

    else:
        # Fallback exposition formatter if library not available
        lines = []
        lines.append("# HELP load_balancer_circuit_breaker_status Circuit breaker status (0=Closed, 1=Tripped)")
        lines.append("# TYPE load_balancer_circuit_breaker_status gauge")
        cb = 1 if telemetry.get("circuit_breaker_tripped", False) else 0
        lines.append(f"load_balancer_circuit_breaker_status {cb}")

        weights = telemetry.get("weights", [])
        lines.append("# HELP load_balancer_routing_weight Active routing weight per backend node")
        lines.append("# TYPE load_balancer_routing_weight gauge")
        for i, w in enumerate(weights):
            lines.append(f'load_balancer_routing_weight{{node="Node-{i+1}"}} {w:.4f}')

        cpu_list = telemetry.get("cpu", [])
        lines.append("# HELP load_balancer_node_cpu_utilization Real-time CPU load percentage per node")
        lines.append("# TYPE load_balancer_node_cpu_utilization gauge")
        for i, c in enumerate(cpu_list):
            lines.append(f'load_balancer_node_cpu_utilization{{node="Node-{i+1}"}} {c:.2f}')

        output = "\n".join(lines) + "\n"
        return output.encode("utf-8"), "text/plain; version=0.0.4; charset=utf-8"
