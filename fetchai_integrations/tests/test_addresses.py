"""
tests/test_addresses.py

Print all 5 NeuralLens agent addresses.
Run with: python tests/test_addresses.py

These addresses are deterministic — same seed = same address.
Run this WITHOUT starting agents to discover addresses first,
then fill them into .env.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uagents import Agent

AGENTS = [
    (
        "ORCHESTRATOR",
        "neurallens orchestrator lahacks 2026 neuromarketing main",
        "ORCHESTRATOR_ADDRESS",
    ),
    (
        "SENSOR",
        "neurallens sensor tribe deepgaze lahacks 2026 brain",
        "SENSOR_ADDRESS",
    ),
    (
        "INTERPRETER",
        "neurallens interpreter nes roi scorer lahacks 2026",
        "INTERPRETER_ADDRESS",
    ),
    (
        "STRATEGIST",
        "neurallens strategist gemma optimizer lahacks 2026",
        "STRATEGIST_ADDRESS",
    ),
    (
        "EXECUTOR",
        "neurallens executor cloudinary image lahacks 2026",
        "EXECUTOR_ADDRESS",
    ),
]


def main():
    print("\n" + "=" * 60)
    print("NeuralLens Agent Addresses")
    print("=" * 60)
    print()
    print("Copy these into your .env file:\n")

    for name, seed, env_key in AGENTS:
        # Instantiate agent just to derive its address
        # We use a dummy port to avoid binding conflicts
        port = 9900 + AGENTS.index((name, seed, env_key))
        agent = Agent(
            name=f"neurallens-{name.lower()}",
            seed=seed,
            port=port,
            endpoint=[f"http://localhost:{port}/submit"],
        )
        print(f"{env_key}={agent.address}")

    print()
    print("=" * 60)
    print()
    print("NEXT: Register ORCHESTRATOR_ADDRESS on Agentverse")
    print("URL: https://agentverse.ai")
    print()


if __name__ == "__main__":
    main()
