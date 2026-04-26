"""
run_all_agents.py

Runs all 5 NeuralLens agents in a single process using uAgents Bureau.

WHY BUREAU instead of 5 separate terminals:
  When agents run in the same Bureau, they communicate via direct
  in-process message passing — no Almanac contract resolution needed,
  no FET token funding required, no endpoint lookup.
  This is the correct way to run a multi-agent pipeline locally.

  For Agentverse deployment each agent can still be launched separately
  (their mailbox=True flag means Agentverse queues messages for them).
  But for local development and demos, Bureau is the right tool.

Usage:
  python run_all_agents.py

The orchestrator address printed here is what you register on Agentverse
so ASI:One can discover and chat with NeuralLens.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from uagents import Bureau

# Import agents — each module prints its address on import
from agents.orchestrator import orchestrator
from agents.sensor_agent import sensor
from agents.interpreter_agent import interpreter
from agents.strategist_agent import strategist
from agents.executor_agent import executor

print("\n" + "=" * 55)
print("NeuralLens — Starting all agents via Bureau")
print("=" * 55)
print()
print("All agents are communicating in-process.")
print("No Almanac funding required for local comms.")
print()
print("Register the ORCHESTRATOR ADDRESS on Agentverse")
print("so ASI:One can discover NeuralLens.")
print()
print("Ctrl+C to stop all agents.")
print("=" * 55 + "\n")

bureau = Bureau()
bureau.add(orchestrator)
bureau.add(sensor)
bureau.add(interpreter)
bureau.add(strategist)
bureau.add(executor)

if __name__ == "__main__":
    bureau.run()
