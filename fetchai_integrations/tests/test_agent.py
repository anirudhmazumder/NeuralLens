"""
tests/test_agent.py

Integration test for integrations/neurallens_agent.py.
Run with:  python tests/test_agent.py

IMPORTANT: The NeuralLens agent must already be running in a separate
terminal before executing this test:
    python integrations/neurallens_agent.py

The test sends a ChatMessage to the agent on localhost:8000 and
waits for a response, then prints it.
"""

# stdlib
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# third-party
from dotenv import load_dotenv
load_dotenv()

from uagents import Agent, Context, Model
from uagents.experimental.chat import ChatMessage  # type: ignore

AGENT_ENDPOINT = "http://localhost:8000/submit"
TEST_MESSAGE = "Analyse https://example.com for my SaaS product"
RESPONSE_TIMEOUT = 90  # seconds

# Caller agent — ephemeral, just for testing
caller = Agent(name="neurallens_test_caller", seed="neurallens_test_seed_42")

_received_response: list = []


@caller.on_event("startup")
async def send_test_message(ctx: Context) -> None:
    print(f"\n[test_agent] Caller address: {caller.address}")
    print(f"[test_agent] Sending test message to agent at {AGENT_ENDPOINT} ...")
    print(f"[test_agent] Message: {TEST_MESSAGE!r}\n")

    # Note: in a real test you would resolve the target agent address via
    # Agentverse lookup.  Here we send directly to localhost.
    # Replace with the actual agent address printed on startup.
    target = os.environ.get("NEURALLENS_AGENT_ADDRESS", "")
    if not target:
        print(
            "SKIP: Set NEURALLENS_AGENT_ADDRESS env var to the agent's address "
            "printed when you run:  python integrations/neurallens_agent.py"
        )
        return

    await ctx.send(target, ChatMessage(content=TEST_MESSAGE))
    print("[test_agent] Message sent, waiting for response ...")


@caller.on_message(ChatMessage)
async def receive_response(ctx: Context, sender: str, message: ChatMessage) -> None:
    print(f"\n[test_agent] Response received from {sender}:")
    print(message.content)
    _received_response.append(message.content)


def test_agent():
    print("\nPrinting agent address (requires agent to be running) ...")
    print(f"  NEURALLENS_AGENT_ADDRESS env var: {os.environ.get('NEURALLENS_AGENT_ADDRESS', 'NOT SET')}")
    print("\nStarting ephemeral caller agent ...")

    # Run for RESPONSE_TIMEOUT seconds then exit
    async def _run():
        task = asyncio.create_task(caller._run())
        await asyncio.sleep(RESPONSE_TIMEOUT)
        task.cancel()

    asyncio.run(_run())

    if _received_response:
        print("\nPASS: received response from NeuralLens agent")
    else:
        print("\nSKIP/FAIL: no response received within timeout — is the agent running?")


if __name__ == "__main__":
    test_agent()
