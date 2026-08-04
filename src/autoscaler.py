# src/autoscaler.py
"""
Kubernetes Horizontal Pod Autoscaler (HPA) Simulation Engine.
Monitors real-time cluster CPU load and automatically provisions/destroys backend microservice node
subprocesses dynamically (scaling cluster capacity from 5 up to 10 nodes).
"""
import sys
import os
import asyncio
import subprocess
import time
from typing import List, Dict, Any

class ClusterAutoScaler:
    def __init__(self, shared_cache, min_nodes: int = 5, max_nodes: int = 10):
        self.shared_cache = shared_cache
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.active = False
        
        # Track dynamically spawned subprocesses
        self.dynamic_processes: Dict[int, subprocess.Popen] = {}
        self.high_cpu_duration = 0.0
        self.low_cpu_duration = 0.0

    async def start(self):
        self.active = True
        asyncio.create_task(self._monitor_loop())

    async def stop(self):
        self.active = False
        self._terminate_all_dynamic_nodes()

    def _spawn_node(self, node_idx: int):
        """Spawns a new backend node subprocess dynamically."""
        if node_idx in self.dynamic_processes:
            return
            
        port = 8001 + node_idx
        print(f"\n[HPA AutoScaler] 🚀 SCALING UP: Provisioning new Backend Node-{node_idx+1} on Port {port}...")
        
        try:
            p = subprocess.Popen([
                sys.executable, "src/backend_node.py",
                "--port", str(port),
                "--node-idx", str(node_idx)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.dynamic_processes[node_idx] = p
        except Exception as e:
            print(f"[HPA AutoScaler] Failed to spawn Node {node_idx+1}: {e}")

    def _terminate_node(self, node_idx: int):
        """Gracefully terminates a dynamic backend node subprocess."""
        if node_idx in self.dynamic_processes:
            print(f"\n[HPA AutoScaler] 📉 SCALING DOWN: Terminating Backend Node-{node_idx+1}...")
            p = self.dynamic_processes.pop(node_idx)
            try:
                p.terminate()
                p.wait(timeout=2.0)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    def _terminate_all_dynamic_nodes(self):
        for idx in list(self.dynamic_processes.keys()):
            self._terminate_node(idx)

    async def _monitor_loop(self):
        await asyncio.sleep(2.0)
        while self.active:
            try:
                metrics = self.shared_cache.get_instance_metrics()
                cpu_list = metrics.get("cpu", [])
                
                current_active_count = self.min_nodes + len(self.dynamic_processes)
                active_cpus = cpu_list[:current_active_count]
                
                if active_cpus:
                    avg_cpu = sum(active_cpus) / len(active_cpus)
                    
                    # Scale Up condition: CPU > 75% for 3 seconds
                    if avg_cpu > 75.0 and current_active_count < self.max_nodes:
                        self.high_cpu_duration += 1.0
                        if self.high_cpu_duration >= 3.0:
                            next_idx = current_active_count
                            self._spawn_node(next_idx)
                            self.high_cpu_duration = 0.0
                    else:
                        self.high_cpu_duration = max(0.0, self.high_cpu_duration - 0.5)

                    # Scale Down condition: CPU < 25% for 5 seconds
                    if avg_cpu < 25.0 and current_active_count > self.min_nodes:
                        self.low_cpu_duration += 1.0
                        if self.low_cpu_duration >= 5.0:
                            last_idx = current_active_count - 1
                            self._terminate_node(last_idx)
                            self.low_cpu_duration = 0.0
                    else:
                        self.low_cpu_duration = max(0.0, self.low_cpu_duration - 0.5)

            except Exception:
                pass
                
            await asyncio.sleep(1.0)
