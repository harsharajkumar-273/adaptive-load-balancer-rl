# src/main.py
"""
Orchestrator to launch the entire multi-process distributed cluster.
Spawns backend microservices in subprocesses and manages their lifecycles,
including Service Discovery registration and HPA Cluster AutoScaler.
"""
import sys
import os

# Ensure parent directory is on sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import subprocess
import time
import uvicorn
from src.config import PORT, HOST, BACKEND_URLS
from src.shared_state import DistributedStateCache
from src.simulator import TrafficGenerator
from src.agent import ContextualBanditAgent
from src.dashboard import TelemetryDashboard
from src.autoscaler import ClusterAutoScaler

async def run_circuit_breaker_demo(agent: ContextualBanditAgent):
    """
    Background worker that monitors elapsed time and mocks a control plane crash
    at t=30s to demonstrate the circuit breaker and Least Connections fallback.
    """
    await asyncio.sleep(30.0)
    print("\n[Orchestrator] MOCKING CONTROL PLANE CRASH (Stopping RL updates)...")
    await agent.stop()
    
    await asyncio.sleep(6.0)
    
    print("\n[Orchestrator] RECOVERING CONTROL PLANE (Resuming RL updates)...")
    await agent.start()

async def shutdown_system(tasks, node_processes, autoscaler):
    """Cleanly terminates all background tasks, autoscaled nodes, and subprocesses on exit."""
    if autoscaler:
        await autoscaler.stop()

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    
    print("[System] Terminating backend microservice subprocesses...")
    for p in node_processes:
        try:
            p.terminate()
            p.wait(timeout=2.0)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    print("[System] All services shut down successfully.")

async def main():
    cache = DistributedStateCache(num_instances=5)
    
    print("[System] Spawning initial 5 backend microservices in subprocesses...")
    node_processes = []
    
    is_docker = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"
    
    if not is_docker:
        for i in range(5):
            port = 8001 + i
            p = subprocess.Popen([
                sys.executable, "src/backend_node.py",
                "--port", str(port),
                "--node-idx", str(i)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            node_processes.append(p)
        
        print("[System] Waiting for backend microservices to initialize...")
        await asyncio.sleep(2.0)

    # Initialize Control Plane RL Agent
    agent = ContextualBanditAgent(num_instances=5, shared_cache=cache)
    
    # Initialize Telemetry Dashboard
    dashboard = TelemetryDashboard(cache)
    
    # Initialize Traffic Generator
    traffic_generator = TrafficGenerator(port=PORT, shared_cache=cache)

    # Initialize HPA Cluster AutoScaler
    autoscaler = ClusterAutoScaler(shared_cache=cache, min_nodes=5, max_nodes=10)

    # Start Control Plane, Traffic, Dashboard, and AutoScaler loops
    await agent.start()
    await dashboard.start()
    await traffic_generator.start()
    await autoscaler.start()

    cb_demo_task = asyncio.create_task(run_circuit_breaker_demo(agent))

    from src.gateway import app
    app.state.shared_cache = cache

    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
        loop="asyncio"
    )
    server = uvicorn.Server(config)
    
    bg_tasks = [cb_demo_task]

    try:
        await server.serve()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\n[System] Shutting down load balancer...")
        agent.save_checkpoint()
        await agent.stop()
        await dashboard.stop()
        await traffic_generator.stop()
        await shutdown_system(bg_tasks, node_processes, autoscaler)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[System] Execution terminated by user.")
