# src/benchmark.py
"""
Single-gateway benchmark of the production routing strategies.

Runs a discrete-event simulation of the heterogeneous fleet in config.py and
compares Round Robin, Weighted Round Robin, Least Connections, P2C and the LinTS
agent. Every configuration is repeated over several seeds and reported as
mean +/- 95% CI, with the CPU action mask both ON (production default) and OFF
(isolates what the learning itself contributes).

For the multi-gateway herding experiments, see research/.
"""
import sys
import os
import argparse
import random
import numpy as np
from typing import Dict, List, Any

# Ensure parent directory is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.routing_strategies import RoutingEngine
from src.config import BACKEND_SPECS, NUM_INSTANCES

import heapq
import math
import time
from src.shared_state import DistributedStateCache
from src.agent import ContextualBanditAgent

class InProcessBenchmark:
    """
    Realistic in-process discrete-event benchmark suite.
    Simulates heterogeneous backend nodes with dynamic queueing, CPU saturation curves,
    and online Reinforcement Learning (Contextual Bandit) adaptation.
    """
    def __init__(self, apply_cpu_mask: bool = True):
        self.engine = RoutingEngine(num_instances=NUM_INSTANCES, apply_cpu_mask=apply_cpu_mask)
        self.specs = BACKEND_SPECS

    def _compute_node_latency(self, node_idx: int, active_conns: int, cpu: float, extra_delay: float = 0.0) -> float:
        spec = self.specs[node_idx]
        base = spec["base_latency_ms"]
        jitter = random.uniform(spec["jitter_range_ms"][0], spec["jitter_range_ms"][1])
        norm_conns = active_conns / max(1.0, spec["capacity"] / 10.0)
        queue_impact = (norm_conns ** 1.2) * 3.0
        cpu_impact = 0.0
        if cpu > 70.0:
            cpu_impact = math.exp((cpu - 70.0) / 6.0) * 6.0
        return base + jitter + queue_impact + cpu_impact + extra_delay

    def run_strategy_benchmark(
        self,
        strategy: str,
        target_rps: int,
        duration_sec: int = 3,
        chaos_node_idx: int = None,
        seed: int = 0
    ) -> Dict[str, Any]:
        """
        Executes a discrete-event traffic simulation across the target duration.
        """
        random.seed(seed)
        np.random.seed(seed)

        # Set up state cache and RL agent if strategy is lin_ts
        cache = DistributedStateCache.in_memory(num_instances=NUM_INSTANCES)
        agent = None
        if strategy == "lin_ts":
            agent = ContextualBanditAgent(num_instances=NUM_INSTANCES, shared_cache=cache, load_checkpoint=False)
            # Pre-train / warm up agent with initial uniform context
            init_contexts = [np.ones(6) for _ in range(NUM_INSTANCES)]
            agent.prev_context_vectors = init_contexts
            # Initial exploration over 50 steps
            for i in range(NUM_INSTANCES):
                cache.record_request_outcome(i, self.specs[i]["base_latency_ms"], True, False)
            fb = cache.flush_window_feedback()
            agent.update_model(fb)

        # Simulation state
        num_nodes = NUM_INSTANCES
        active_conns = [0] * num_nodes
        cpu_list = [10.0] * num_nodes
        p99_list = [self.specs[i]["base_latency_ms"] for i in range(num_nodes)]
        recent_latencies = {i: [] for i in range(num_nodes)}
        
        # Priority queue of completion events: (completion_time, node_idx, latency_ms, success, breached_sla)
        event_queue = []
        
        total_requests = target_rps * duration_sec
        inter_arrival = 1.0 / target_rps
        current_time = 0.0
        
        latencies = []
        node_counts = {i: 0 for i in range(num_nodes)}
        errors = 0
        sla_breaches = 0
        weights = [1.0 / num_nodes] * num_nodes

        last_rl_update_time = 0.0
        rl_interval = 0.15

        for req_idx in range(total_requests):
            current_time += inter_arrival
            
            # 1. Process all completed requests up to current_time
            while event_queue and event_queue[0][0] <= current_time:
                comp_time, done_node, lat_ms, success_flag, sla_flag = heapq.heappop(event_queue)
                active_conns[done_node] = max(0, active_conns[done_node] - 1)
                if agent:
                    cache.record_request_outcome(done_node, lat_ms, success_flag, sla_flag)

            # 2. Update node CPU utilization based on active queues
            for i in range(num_nodes):
                spec = self.specs[i]
                target_cpu = min(100.0, 5.0 + (active_conns[i] / spec["capacity"]) * 100.0 * spec["cpu_multiplier"])
                if chaos_node_idx == i:
                    target_cpu = 95.0  # Injected chaos fault
                cpu_list[i] = 0.80 * cpu_list[i] + 0.20 * target_cpu

            # 3. If RL LinTS, periodically update policy weights from feedback
            if strategy == "lin_ts":
                if (current_time - last_rl_update_time) >= rl_interval:
                    fb = cache.flush_window_feedback()
                    agent.update_model(fb)
                    
                    # Build real-time context vectors
                    contexts = []
                    rate_change = 0.0
                    for i in range(num_nodes):
                        contexts.append(agent._build_context_vector(cpu_list[i], active_conns[i], p99_list[i], float(target_rps), rate_change))
                    
                    weights = agent.select_action_weights(contexts)
                    cache.set_routing_weights(weights)
                    agent.prev_context_vectors = contexts
                    last_rl_update_time = current_time
            else:
                weights = [1.0 / num_nodes] * num_nodes

            # 4. Route request using RoutingEngine
            chosen_idx, mode, eff_w = self.engine.select_instance(strategy, weights, cpu_list, active_conns)
            node_counts[chosen_idx] += 1
            active_conns[chosen_idx] += 1

            # 5. Simulate execution and latency
            extra_delay = 300.0 if (chaos_node_idx == chosen_idx) else 0.0
            lat_ms = self._compute_node_latency(chosen_idx, active_conns[chosen_idx], cpu_list[chosen_idx], extra_delay)
            latencies.append(lat_ms)
            recent_latencies[chosen_idx].append(lat_ms)
            if len(recent_latencies[chosen_idx]) > 50:
                recent_latencies[chosen_idx].pop(0)
            p99_list[chosen_idx] = float(np.percentile(recent_latencies[chosen_idx], 99))

            # 6. Check SLA breach & errors
            breached_sla = lat_ms > 200.0
            if breached_sla:
                sla_breaches += 1

            # Node error threshold check
            spec = self.specs[chosen_idx]
            success = True
            if cpu_list[chosen_idx] > spec["error_threshold_cpu"]:
                excess = cpu_list[chosen_idx] - spec["error_threshold_cpu"]
                err_prob = min(0.85, excess / (100.0 - spec["error_threshold_cpu"]))
                if random.random() < err_prob:
                    errors += 1
                    success = False
            elif chaos_node_idx == chosen_idx and random.random() < 0.8:
                errors += 1
                success = False

            # Schedule completion event
            completion_time = current_time + (lat_ms / 1000.0)
            heapq.heappush(event_queue, (completion_time, chosen_idx, lat_ms, success, breached_sla))

        # Drain remaining events
        while event_queue:
            comp_time, done_node, lat_ms, success_flag, sla_flag = heapq.heappop(event_queue)
            if agent:
                cache.record_request_outcome(done_node, lat_ms, success_flag, sla_flag)

        latencies.sort()
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        return {
            "strategy": strategy,
            "target_rps": target_rps,
            "actual_rps": float(target_rps),
            "total_requests": total_requests,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "sla_violation_pct": (sla_breaches / total_requests) * 100.0,
            "error_pct": (errors / total_requests) * 100.0,
            "node_distribution": node_counts
        }

STRATEGY_LABELS = {
    "round_robin": "Round Robin",
    "weighted_round_robin": "Weighted Round Robin",
    "least_conn": "Least Connections",
    "p2c": "Power of Two Choices (P2C)",
    "lin_ts": "LinTS (contextual bandit)",
}

METRICS = ["p50_ms", "p95_ms", "p99_ms", "sla_violation_pct", "error_pct"]


def _mean_ci(values: List[float]) -> str:
    """Mean +/- 95% CI half-width (normal approximation)."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return f"{arr.mean():.1f}"
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
    return f"{arr.mean():.1f} ± {half:.1f}"


def _run_seeds(bench: InProcessBenchmark, strategy: str, seeds: int, **kwargs) -> Dict[str, List[float]]:
    results = {m: [] for m in METRICS}
    for seed in range(seeds):
        res = bench.run_strategy_benchmark(strategy, seed=seed, **kwargs)
        for m in METRICS:
            results[m].append(res[m])
    return results


def _table_header() -> str:
    return ("| Scenario | CPU mask | Strategy | P50 (ms) | P95 (ms) | P99 (ms) | SLA breach % | Error % |\n"
            "|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|\n")


def _table_row(scenario: str, mask: bool, strategy: str, r: Dict[str, List[float]]) -> str:
    cells = " | ".join(_mean_ci(r[m]) for m in METRICS)
    return f"| {scenario} | {'on' if mask else 'off'} | {STRATEGY_LABELS[strategy]} | {cells} |"


def run_all_benchmarks(seeds: int = 5, duration_sec: int = 3, output: str = "benchmark_results.md"):
    strategies = list(STRATEGY_LABELS)
    load_levels = [100, 500, 1000]
    started = time.time()

    report = [
        "# Single-Gateway Benchmark Report",
        "",
        f"Discrete-event simulation of the 5-node heterogeneous fleet in `src/config.py`. "
        f"Each cell is mean ± 95% CI over {seeds} seeds ({duration_sec}s of simulated traffic per run). "
        "Generated by `python src/benchmark.py`; do not edit by hand.",
        "",
        "The CPU mask (weight 0 for nodes above 85% CPU) is applied identically to every strategy. "
        "Rows with the mask **off** show what each routing policy achieves on its own.",
        "",
        "## Steady state",
        "",
        _table_header().rstrip("\n"),
    ]

    for mask in (True, False):
        bench = InProcessBenchmark(apply_cpu_mask=mask)
        for rps in load_levels:
            for strat in strategies:
                r = _run_seeds(bench, strat, seeds, target_rps=rps, duration_sec=duration_sec)
                line = _table_row(f"{rps} RPS", mask, strat, r)
                print(line)
                report.append(line)

    report += [
        "",
        "## Degraded node (Node 1 forced to 95% CPU, +300 ms, 80% errors) at 500 RPS",
        "",
        _table_header().rstrip("\n"),
    ]
    for mask in (True, False):
        bench = InProcessBenchmark(apply_cpu_mask=mask)
        for strat in strategies:
            r = _run_seeds(bench, strat, seeds, target_rps=500, duration_sec=duration_sec, chaos_node_idx=0)
            line = _table_row("500 RPS + fault", mask, strat, r)
            print(line)
            report.append(line)

    with open(output, "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nSaved report to '{output}' in {time.time() - started:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--duration", type=int, default=3, help="Simulated seconds per run")
    parser.add_argument("--output", default="benchmark_results.md")
    args = parser.parse_args()
    run_all_benchmarks(seeds=args.seeds, duration_sec=args.duration, output=args.output)
