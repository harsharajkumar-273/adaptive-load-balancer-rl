# src/gateway_pool.py
"""
Runs several independent gateways in one process, one per port:

    python -m src.gateway_pool --ports 9001,9002,9003

Each gateway is built by src.gateway.create_app(), so it has its own shared-state
snapshot, learned model, probe pool and HTTP client; they only share the event
loop. This lets experiments run many gateways (e.g. 64) on a small machine.
Configuration comes from the same environment variables as src/gateway.py.
"""
import argparse
import asyncio

import uvicorn

from src.gateway import create_app


async def serve(ports):
    servers = [
        uvicorn.Server(uvicorn.Config(create_app(port_hint=port), host="127.0.0.1", port=port,
                                      log_level="warning", loop="asyncio", timeout_keep_alive=300))
        for port in ports
    ]
    await asyncio.gather(*(server.serve() for server in servers))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ports", required=True, help="comma-separated ports, one gateway each")
    args = parser.parse_args()
    asyncio.run(serve([int(p) for p in args.ports.split(",")]))


if __name__ == "__main__":
    main()
