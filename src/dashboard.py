# src/dashboard.py
"""
Real-time console telemetry dashboard for the RL load balancer.
"""
import os
import sys
import asyncio
import time
from typing import Dict, Any
from src.config import DASHBOARD_INTERVAL_SEC, SLA_LATENCY_MS

# ANSI Colors
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_RED = "\033[31m"
CLR_GREEN = "\033[32m"
CLR_YELLOW = "\033[33m"
CLR_BLUE = "\033[34m"
CLR_MAGENTA = "\033[35m"
CLR_CYAN = "\033[36m"
CLR_WHITE = "\033[37m"

# Background Colors
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_BLACK_GREY = "\033[100m"

class TelemetryDashboard:
    def __init__(self, shared_cache):
        self.shared_cache = shared_cache
        self.active = False

    async def start(self):
        self.active = True
        asyncio.create_task(self._run_loop())

    async def stop(self):
        self.active = False

    def _draw_bar(self, percentage: float, width: int = 15, color: str = CLR_GREEN) -> str:
        """Draws a beautiful progress bar."""
        filled_len = int(round(width * percentage / 100.0))
        filled_len = max(0, min(width, filled_len))
        bar = "█" * filled_len + "░" * (width - filled_len)
        return f"{color}{bar}{CLR_RESET}"

    def _render_dashboard(self, telemetry: Dict[str, Any]):
        """Renders the dashboard metrics to stdout."""
        # Clear screen and move cursor to top-left
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.flush()

        total_reqs = telemetry["total_requests"]
        breached_sla = telemetry["breached_sla_requests"]
        sys_p99 = telemetry["system_p99_latency"]
        weights = telemetry["weights"]
        cpu_list = telemetry["cpu"]
        queue_list = telemetry["queue"]
        p99_list = telemetry["p99_latencies"]
        error_rates = telemetry["error_rates"]
        global_rate = telemetry["global_request_rate"]
        is_spike = telemetry["traffic_spike_active"]
        cb_tripped = telemetry["circuit_breaker_tripped"]

        # Color-coded values
        sla_color = CLR_GREEN if breached_sla == 0 else (CLR_YELLOW if breached_sla / max(1, total_reqs) < 0.05 else CLR_RED)
        p99_color = CLR_GREEN if sys_p99 < 50.0 else (CLR_YELLOW if sys_p99 < SLA_LATENCY_MS else CLR_RED)
        
        # Traffic phase badge
        if is_spike:
            traffic_badge = f"{BG_RED}{CLR_WHITE}{CLR_BOLD}  THUNDERING HERD SPIKE ACTIVE (STRANGER THINGS RELEASE)  {CLR_RESET}"
        else:
            traffic_badge = f"{BG_GREEN}{CLR_WHITE}{CLR_BOLD}  NORMAL BASE TRAFFIC PHASE  {CLR_RESET}"

        # Circuit breaker badge
        if cb_tripped:
            cb_badge = f"{CLR_RED}{CLR_BOLD}TRIPPED (Fallback: Least Connections){CLR_RESET}"
        else:
            cb_badge = f"{CLR_GREEN}{CLR_BOLD}CLOSED (RL Contextual Bandit Routing){CLR_RESET}"

        # Header
        print("=" * 90)
        print(f" {CLR_MAGENTA}{CLR_BOLD}NETFLIX RL LOAD BALANCER SIMULATION DASHBOARD{CLR_RESET}   (Time: {time.strftime('%H:%M:%S')})")
        print("=" * 90)
        
        # Traffic State & CB
        print(f" {CLR_BOLD}Traffic Mode:{CLR_RESET}   {traffic_badge}")
        print(f" {CLR_BOLD}Circuit Breaker State:{CLR_RESET} {cb_badge}")
        print("-" * 90)

        # Global Statistics Row
        print(f" {CLR_BOLD}Total Client Requests:{CLR_RESET} {CLR_CYAN}{total_reqs}{CLR_RESET}  |"
              f"  {CLR_BOLD}Global Request Rate:{CLR_RESET} {CLR_CYAN}{global_rate:.1f} RPS{CLR_RESET}  |"
              f"  {CLR_BOLD}SLA Breaches:{CLR_RESET} {sla_color}{breached_sla} reqs{CLR_RESET} ({breached_sla / max(1, total_reqs) * 100:.1f}%)")
        print(f" {CLR_BOLD}Overall System P99 Latency:{CLR_RESET} {p99_color}{sys_p99:.2f} ms{CLR_RESET} (SLA Threshold: {SLA_LATENCY_MS} ms)")
        print("=" * 90)

        # Instance Fleet Headers
        print(f" {CLR_BOLD}{'INSTANCE ID / NAME':<30} | {'CPU UTILIZATION':<23} | {'QUEUE':<5} | {'P99 LAT':<10} | {'ERRORS':<7} | {'RL WEIGHT'}{CLR_RESET}")
        print("-" * 90)

        # Print metrics for each of the 5 instances
        for i in range(len(weights)):
            name = f"Instance {i+1}"
            cpu = cpu_list[i]
            queue = queue_list[i]
            p99 = p99_list[i]
            err = error_rates[i] * 100.0
            weight = weights[i] * 100.0

            # Color ranges
            cpu_color = CLR_GREEN if cpu < 60 else (CLR_YELLOW if cpu < 85 else CLR_RED)
            p99_color = CLR_GREEN if p99 < 80 else (CLR_YELLOW if p99 < SLA_LATENCY_MS else CLR_RED)
            err_color = CLR_GREEN if err == 0 else CLR_RED
            weight_color = CLR_CYAN if weight > 0 else CLR_YELLOW

            # CPU Bar & Weight Bar
            cpu_bar = self._draw_bar(cpu, width=12, color=cpu_color)
            weight_bar = self._draw_bar(weight, width=12, color=CLR_CYAN)

            # Format name
            if i == 0:
                full_name = f"Instance-1 (High-Compute)"
            elif i == 1:
                full_name = f"Instance-2 (Standard-Med)"
            elif i == 2:
                full_name = f"Instance-3 (Standard-Slow)"
            elif i == 3:
                full_name = f"Instance-4 (Unstable-Jittery)"
            else:
                full_name = f"Instance-5 (Slow-Legacy)"

            print(f" {full_name:<30} | {cpu_bar} {cpu_color}{cpu:>5.1f}%{CLR_RESET} | {queue:<5} | {p99_color}{p99:>6.1f} ms{CLR_RESET} | {err_color}{err:>5.1f}%{CLR_RESET} | {weight_bar} {weight_color}{weight:>5.1f}%{CLR_RESET}")

        print("=" * 90)
        print(f" {CLR_BOLD}Features Used by RL (LinTS Context):{CLR_RESET}")
        print(f"   Bias, CPU/100, QueueDepth/20, RecentP99/200, GlobalRate/150, d/dt(Rate)/50")
        print(f" {CLR_BOLD}Safety Guardrails in Action:{CLR_RESET}")
        print(f"   - Action Masking overrides weight to 0% if CPU > 85%.")
        print(f"   - Circuit Breaker trips if Control Plane updates halt for > 1.0s.")
        print("=" * 90)
        sys.stdout.flush()

    async def _run_loop(self):
        """Dashboard rendering loop."""
        while self.active:
            try:
                telemetry = self.shared_cache.get_telemetry()
                self._render_dashboard(telemetry)
            except Exception as e:
                pass
            await asyncio.sleep(DASHBOARD_INTERVAL_SEC)
