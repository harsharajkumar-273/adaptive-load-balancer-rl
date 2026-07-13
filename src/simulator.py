# src/simulator.py
"""
Traffic Generator client simulating thundering-herd spikes on the API Gateway.
"""
import asyncio
import time
import httpx
from src.config import PORT

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
            start_time = time.time()
            response = await client.get(f"http://127.0.0.1:{self.port}/play-video", timeout=4.0)
            latency_ms = (time.time() - start_time) * 1000.0
            return response.status_code, latency_ms
        except Exception:
            return 500, 4000.0  # Timeout or connection error penalty

    async def _run_loop(self):
        """
        Simulates the traffic cycle over 60 seconds:
        - 0s to 10s: Base load (15 req/s)
        - 10s to 28s: Exponential surge (up to 160 req/s) - "Stranger Things" spike
        - 28s to 45s: Recovery / Subside (declining back to 15 req/s)
        - 45s to 60s: Cooldown (15 req/s)
        Runs continuously in a repeating loop.
        """
        # Wait a moment for gateway to initialize
        await asyncio.sleep(1.0)
        
        async with httpx.AsyncClient() as client:
            start_time = time.time()
            
            while self.active:
                elapsed = (time.time() - start_time) % 60.0
                
                # Calculate target Request Rate (RPS) based on the timeline phase
                if elapsed < 10.0:
                    self.shared_cache.set_traffic_spike_active(False)
                    target_rps = 15.0
                elif elapsed < 20.0:
                    self.shared_cache.set_traffic_spike_active(True)
                    # Exponentiate from 15 to 160 RPS
                    progress = (elapsed - 10.0) / 10.0
                    target_rps = 15.0 + (145.0 * (progress ** 2))
                elif elapsed < 28.0:
                    self.shared_cache.set_traffic_spike_active(True)
                    target_rps = 160.0
                elif elapsed < 45.0:
                    self.shared_cache.set_traffic_spike_active(False)
                    progress = (elapsed - 28.0) / 17.0
                    target_rps = 160.0 - (145.0 * progress)
                else:
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
