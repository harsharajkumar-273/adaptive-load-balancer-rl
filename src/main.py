# src/main.py
"""
Orchestrator to launch the entire multi-process distributed cluster.
Spawns backend microservices in subprocesses and manages their lifecycles.
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

async def run_circuit_breaker_demo(agent: ContextualBanditAgent):
    """
    Background worker that monitors elapsed time and mocks a control plane crash
    at t=30s to demonstrate the circuit breaker and Least Connections fallback.
    """
    await asyncio.sleep(30.0)  # Wait until t=30s (right after the peak surge)
    print("\n[Orchestrator] MOCKING CONTROL PLANE CRASH (Stopping RL updates)...")
    await agent.stop()
    
    # Keep it stopped for 6 seconds
    await asyncio.sleep(6.0)
    
    print("\n[Orchestrator] RECOVERING CONTROL PLANE (Resuming RL updates)...")
    await agent.start()

async def shutdown_system(tasks, node_processes):
    """Cleanly terminates all background tasks and backend subprocesses on exit."""
    # Cancel tasks
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Terminate backend microservices
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
    # 1. Initialize Shared State (Connect to Redis)
    cache = DistributedStateCache(num_instances=5)
    
    # 2. Spawn 5 backend microservice nodes in subprocesses
    print("[System] Spawning 5 backend microservices in subprocesses...")
    node_processes = []
    
    # Check if we are running in docker-compose.
    # In Docker compose, we don't spawn subprocesses locally; they are separate containers.
    # We detect this via an env variable.
    is_docker = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"
    
    if not is_docker:
        for i in range(5):
            port = 8001 + i
            # Execute python src/backend_node.py --port port --node-idx idx
            p = subprocess.Popen([
                sys.executable, "src/backend_node.py",
                "--port", str(port),
                "--node-idx", str(i)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            node_processes.append(p)
        
        # Give the backend nodes 2 seconds to bind to ports and initialize Redis connections
        print("[System] Waiting for backend microservices to initialize...")
        await asyncio.sleep(2.0)

    # 3. Initialize Control Plane RL Agent
    agent = ContextualBanditAgent(num_instances=5, shared_cache=cache)
    
    # 4. Initialize Telemetry Dashboard
    dashboard = TelemetryDashboard(cache)
    
    # 5. Initialize Traffic Generator
    traffic_generator = TrafficGenerator(port=PORT, shared_cache=cache)

    # 6. Start Control Plane, Traffic, and Dashboard loops
    await agent.start()
    await dashboard.start()
    await traffic_generator.start()

    # 7. Start the Circuit Breaker demo task
    cb_demo_task = asyncio.create_task(run_circuit_breaker_demo(agent))

    # 8. Import gateway app (done here to ensure app state bindings occur cleanly)
    from src.gateway import app
    app.state.shared_cache = cache

    # 9. Start the FastAPI Uvicorn Server in the event loop
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",  # Expose to external traffic (essential for Docker networks)
        port=PORT,
        log_level="warning",
        loop="asyncio"
    )
    server = uvicorn.Server(config)
    
    bg_tasks = [cb_demo_task]

    try:
        # Serve requests asynchronously
        await server.serve()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\n[System] Shutting down load balancer...")
        # Save model checkpoint
        print("[System] Saving RL agent parameters checkpoint...")
        agent.save_checkpoint()
        
        # Stop loops
        await agent.stop()
        await dashboard.stop()
        await traffic_generator.stop()
        
        # Shutdown tasks and subprocesses
        await shutdown_system(bg_tasks, node_processes)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[System] Execution terminated by user.")
