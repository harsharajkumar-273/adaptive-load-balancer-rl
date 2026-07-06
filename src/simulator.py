# src/simulator.py
"""
Simulator environment modeling backend instances and traffic generator.
"""
import asyncio
import math
import random
import time
import httpx
from typing import Dict, List, Tuple
from src.config import BACKEND_SPECS, NUM_INSTANCES, SLA_LATENCY_MS

class BackendInstance:
    def __init__(self, spec: Dict):
        self.spec = spec
        self.id = spec["id"]
        self.name = spec["name"]
        self.base_latency = spec["base_latency_ms"]
        self.capacity = spec["capacity"]
        self.cpu_multiplier = spec["cpu_multiplier"]
        self.error_threshold_cpu = spec["error_threshold_cpu"]
        self.jitter_range = spec["jitter_range_ms"]
        
        # Dynamic State
        self.queue_depth = 0
        self.cpu_utilization = 5.0  # Base idle CPU
        
        # Locks
        self._lock = asyncio.Lock()

    def update_cpu(self):
        """
        Simulate dynamic CPU utilization with inertia.
        CPU is modeled based on active queue depth relative to target capacity.
        """
        # If queue depth is high, target CPU spikes.
        target_cpu = (self.queue_depth / self.capacity) * 100.0 * self.cpu_multiplier
        target_cpu = max(5.0, min(100.0, target_cpu))
        
        # Exponential moving average to simulate CPU spin-up lag
        self.cpu_utilization = 0.82 * self.cpu_utilization + 0.18 * target_cpu

    def get_cpu(self) -> float:
        self.update_cpu()
        return self.cpu_utilization

    def calculate_latency(self, cpu: float) -> float:
        """
        Computes response latency based on base latency, queue depth,
        and high CPU utilization (which triggers exponential latency growth).
        """
        jitter = random.uniform(self.jitter_range[0], self.jitter_range[1])
        
        # Linear impact of queue depth
        queue_impact = (self.queue_depth ** 1.3) * 3.5
        
        # Exponential impact of high CPU (SLA degradation threshold > 70% CPU)
        cpu_impact = 0.0
        if cpu > 70.0:
            cpu_impact = math.exp((cpu - 70.0) / 7.0) * 8.0
            
        latency = self.base_latency + queue_impact + cpu_impact + jitter
        return latency

    def check_success(self, cpu: float) -> bool:
        """
        Simulates HTTP error rates (5xx) if CPU utilization exceeds safety thresholds.
        """
        if cpu <= self.error_threshold_cpu:
            return True
        # Probability of error increases as CPU goes above threshold
        excess = cpu - self.error_threshold_cpu
        span = 100.0 - self.error_threshold_cpu
        error_probability = excess / span
        # Cap max failure rate at 85% to maintain some feedback capability
        error_probability = min(0.85, error_probability)
        return random.random() > error_probability

    async def handle_request(self, shared_cache) -> Tuple[float, bool]:
        """
        Simulates processing a request: increments queue depth, sleeps for calculated latency,
        updates cache metrics, and returns response metrics.
        """
        async with self._lock:
            self.queue_depth += 1
            
        # Update metrics in real time
        cpu = self.get_cpu()
        shared_cache.update_instance_metrics(self.id - 1, cpu, self.queue_depth)

        # Compute dynamic latency
        latency_ms = self.calculate_latency(cpu)
        success = self.check_success(cpu)

        # Simulate work execution
        await asyncio.sleep(latency_ms / 1000.0)

        async with self._lock:
            self.queue_depth = max(0, self.queue_depth - 1)
            
        # Final update after request finishes
        final_cpu = self.get_cpu()
        shared_cache.update_instance_metrics(self.id - 1, final_cpu, self.queue_depth)
        
        # Record outcome in cache
        breached_sla = latency_ms > SLA_LATENCY_MS
        shared_cache.record_request_outcome(self.id - 1, latency_ms, success, breached_sla)

        return latency_ms, success


class FleetSimulator:
    def __init__(self, shared_cache):
        self.shared_cache = shared_cache
        self.instances = [BackendInstance(spec) for spec in BACKEND_SPECS]

    async def route_to_instance(self, instance_idx: int) -> Tuple[float, bool]:
        return await self.instances[instance_idx].handle_request(self.shared_cache)

    def get_least_connections_idx(self) -> int:
        """Helper to find the backend index with the lowest queue depth (for circuit breaker fallback)."""
        min_depth = float('inf')
        best_indices = []
        for idx, inst in enumerate(self.instances):
            inst.update_cpu()
            if inst.queue_depth < min_depth:
                min_depth = inst.queue_depth
                best_indices = [idx]
            elif inst.queue_depth == min_depth:
                best_indices.append(idx)
        return random.choice(best_indices)


class TrafficGenerator:
    def __init__(self, port: int, shared_cache):
        self.port = port
        self.shared_cache = shared_cache
        self.active = False

    async def start(self):
        self.active = True
        asyncio.create_task(self._run_loop())

    async def stop(self):
        self.active = False

    async def _send_request(self, client: httpx.AsyncClient):
        """Sends a single request to the API Gateway."""
        try:
            # We target the /play-video endpoint of our gateway
            start_time = time.time()
            response = await client.get(f"http://127.0.0.1:{self.port}/play-video", timeout=3.0)
            latency_ms = (time.time() - start_time) * 1000.0
            return response.status_code, latency_ms
        except Exception:
            return 500, 3000.0  # Timeout or connection error penalty

    async def _run_loop(self):
        """
        Simulates the traffic cycle over 60 seconds:
        - 0s to 10s: Base load (15 req/s)
        - 10s to 28s: Exponential surge (up to 160 req/s) - "Stranger Things" spike
        - 28s to 45s: Recovery / Subside (declining back to 15 req/s)
        - 45s to 60s: Cooldown (15 req/s)
        Runs continuously in a repeating loop.
        """
        async with httpx.AsyncClient() as client:
            start_time = time.time()
            
            while self.active:
                elapsed = (time.time() - start_time) % 60.0
                
                # Calculate target Request Rate (RPS) based on the timeline phase
                if elapsed < 10.0:
                    # Base Load Phase
                    self.shared_cache.set_traffic_spike_active(False)
                    target_rps = 15.0
                elif elapsed < 20.0:
                    # Exponential Surge Phase
                    self.shared_cache.set_traffic_spike_active(True)
                    # Exponentiate from 15 to 160 RPS
                    progress = (elapsed - 10.0) / 10.0
                    target_rps = 15.0 + (145.0 * (progress ** 2))
                elif elapsed < 28.0:
                    # Peak Surge Phase
                    self.shared_cache.set_traffic_spike_active(True)
                    target_rps = 160.0
                elif elapsed < 45.0:
                    # Subsiding Traffic Phase
                    self.shared_cache.set_traffic_spike_active(False)
                    progress = (elapsed - 28.0) / 17.0
                    target_rps = 160.0 - (145.0 * progress)
                else:
                    # Cooldown Phase
                    self.shared_cache.set_traffic_spike_active(False)
                    target_rps = 15.0

                self.shared_cache.set_global_request_rate(target_rps)

                # Send requests concurrently for the current 100ms window
                window_duration = 0.1
                num_requests = int(target_rps * window_duration)
                
                if num_requests > 0:
                    tasks = [self._send_request(client) for _ in range(num_requests)]
                    await asyncio.gather(*tasks)

                # Sleep to align with the window pacing
                await asyncio.sleep(window_duration)
