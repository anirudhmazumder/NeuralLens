#!/usr/bin/env bash
# Run all 5 NeuralLens agents in separate terminals.
# Usage: bash run_all.sh

echo "🧠 NeuralLens — Starting all agents"
echo ""
echo "⚠️  IMPORTANT: Run this AFTER filling in .env"
echo ""
echo "Opening 5 terminal windows..."
echo ""

# macOS: open each agent in a new Terminal tab
open_tab() {
  osascript -e "tell application \"Terminal\" to do script \"cd $(pwd) && source .venv/bin/activate && $1\""
}

open_tab "python agents/executor_agent.py"
sleep 1
open_tab "python agents/strategist_agent.py"
sleep 1
open_tab "python agents/interpreter_agent.py"
sleep 1
open_tab "python agents/sensor_agent.py"
sleep 1
open_tab "python agents/orchestrator.py"

echo ""
echo "✅ All 5 agents starting in separate Terminal windows."
echo ""
echo "NEXT STEPS:"
echo "  1. Wait ~10 seconds for all agents to print their addresses"
echo "  2. Copy each address into .env:"
echo "     ORCHESTRATOR_ADDRESS=agent1q..."
echo "     SENSOR_ADDRESS=agent1q..."
echo "     INTERPRETER_ADDRESS=agent1q..."
echo "     STRATEGIST_ADDRESS=agent1q..."
echo "     EXECUTOR_ADDRESS=agent1q..."
echo "  3. Restart all agents (addresses are fixed by seed)"
echo "  4. Register ORCHESTRATOR_ADDRESS on Agentverse"
echo "  5. Test via ASI:One"
echo ""
echo "QUICK TEST (after filling .env with addresses):"
echo "  python tests/test_addresses.py"
