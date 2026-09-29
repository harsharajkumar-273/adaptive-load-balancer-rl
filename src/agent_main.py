# src/agent_main.py
"""
Runs the LinTS control-plane agent (src/agent.py) on its own against Redis:

    python -m src.agent_main [--no-checkpoint]

It publishes routing weights every CONTROL_PLANE_INTERVAL_SEC for gateways
using the lin_ts strategy. src/main.py runs the same agent in-process.
"""
import argparse
import asyncio
import logging

from src.agent import ContextualBanditAgent
from src.config import NUM_INSTANCES
from src.shared_state import DistributedStateCache


async def run(load_checkpoint: bool):
    cache = DistributedStateCache(num_instances=NUM_INSTANCES)
    agent = ContextualBanditAgent(num_instances=NUM_INSTANCES, shared_cache=cache, load_checkpoint=load_checkpoint)
    await agent.start()
    while True:
        await asyncio.sleep(3600)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-checkpoint", action="store_true", help="start from an untrained model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(run(load_checkpoint=not args.no_checkpoint))


if __name__ == "__main__":
    main()
