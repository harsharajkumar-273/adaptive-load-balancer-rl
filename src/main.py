import sys
import os
import asyncio

# Ensure parent directory is on sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI
from src.config import PORT, HOST
from src.shared_state import SharedMemoryCache
from src.simulator import FleetSimulator, TrafficGenerator
from src.agent import ContextualBanditAgent
from src.dashboard import TelemetryDashboard, CLR_BOLD, CLR_YELLOW, CLR_RESET
from src.gateway import app

async def run_circuit_breaker_demo(agent: ContextualBanditAgent, dashboard: TelemetryDashboard):
    """
    Background worker that monitors elapsed time and mocks a control plane crash
    at t=30s to demonstrate the circuit breaker and Least Connections fallback.
    """
    await asyncio.sleep(30.0)  # Wait until t=30s (right after the peak surge)
    
    # Stop the agent (simulate a crash/hang)
    await agent.stop()
    
    # Keep it stopped for 6 seconds
    await asyncio.sleep(6.0)
    
    # Restart the agent (simulate recovery)
    await agent.start()

async def shutdown_system(tasks):
    """Cleanly cancels all background tasks on exit."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print("\n[System] All services shut down successfully.")

async def main():
    # 1. Initialize Shared Cache
    cache = SharedMemoryCache(num_instances=5)
    
    # 2. Initialize Simulator & Fleet
    simulator = FleetSimulator(cache)
    
    # 3. Initialize Control Plane RL Agent
    agent = ContextualBanditAgent(num_instances=5, shared_cache=cache)
    
    # 4. Initialize Telemetry Dashboard
    dashboard = TelemetryDashboard(cache)
    
    # 5. Initialize Traffic Generator
    traffic_generator = TrafficGenerator(port=PORT, shared_cache=cache)

    # Attach shared resources to FastAPI app state
    app.state.shared_cache = cache
    app.state.simulator = simulator

    # 6. Start Control Plane, Traffic, and Dashboard loops
    await agent.start()
    await dashboard.start()
    await traffic_generator.start()

    # 7. Start the Circuit Breaker demo task
    cb_demo_task = asyncio.create_task(run_circuit_breaker_demo(agent, dashboard))

    # 8. Start the FastAPI Uvicorn Server in the event loop
    config = uvicorn.Config(
        app=app,
        host=HOST,
        port=PORT,
        log_level="warning",
        loop="asyncio"
    )
    server = uvicorn.Server(config)
    
    # Gather background task references for cleanup
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
        await shutdown_system(bg_tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[System] Execution terminated by user.")
