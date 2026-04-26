"""
tests/debug_agent_comms.py

Tests inter-agent communication directly — sends a minimal
SensorRequest to the Sensor agent and waits for FinalResult
to come back to a throwaway listener agent.

Skips the chat protocol entirely — tests the pipeline only.

Run with: python tests/debug_agent_comms.py

Requires:
  - All 5 agents running in separate terminals
  - All *_ADDRESS vars set in .env
"""

import os
import sys
import asyncio
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from uagents import Agent, Context
from agents.models import SensorRequest, FinalResult

SENSOR_ADDRESS = os.environ.get("SENSOR_ADDRESS", "")
TEST_IMAGE_URL = sys.argv[1] if len(sys.argv) > 1 else \
    "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=800"

TIMEOUT_SECONDS = 120

# Throwaway listener — receives the FinalResult that executor
# would normally send to orchestrator
listener = Agent(
    name="neurallens-debug-listener",
    seed="neurallens debug listener throwaway test only",
    port=8099,
    endpoint=["http://localhost:8099/submit"],
)

print(f"\n{'='*55}")
print(f"Debug listener address: {listener.address}")
print(f"Sending SensorRequest to sensor: {SENSOR_ADDRESS}")
print(f"Image: {TEST_IMAGE_URL}")
print(f"Timeout: {TIMEOUT_SECONDS}s")
print(f"{'='*55}\n")

_start = time.time()
_result_received = False


@listener.on_event("startup")
async def send_test_request(ctx: Context):
    if not SENSOR_ADDRESS:
        ctx.logger.error(
            "SENSOR_ADDRESS not set in .env — run test_addresses.py first"
        )
        return

    ctx.logger.info("Sending SensorRequest to sensor agent...")
    await ctx.send(
        SENSOR_ADDRESS,
        SensorRequest(
            image_url=TEST_IMAGE_URL,
            prompt="optimize this for my bakery Instagram to increase foot traffic",
            industry="bakery",
            asset_type="instagram_post",
            session_sender=listener.address,  # result returns here
        ),
    )
    ctx.logger.info("SensorRequest sent — waiting for FinalResult...")


@listener.on_message(model=FinalResult)
async def receive_final(ctx: Context, sender: str, msg: FinalResult):
    global _result_received
    _result_received = True
    elapsed = time.time() - _start

    print(f"\n{'='*55}")
    print(f"✅ FinalResult received in {elapsed:.1f}s")
    print(f"{'='*55}")
    print(f"  NES before : {msg.nes_before}")
    print(f"  NES after  : {msg.nes_after}")
    print(f"  Delta      : +{msg.delta}")
    print(f"  Profile    : {msg.profile_before} → {msg.profile_after}")
    print(f"  Optimized  : {msg.optimized_image_url}")
    print(f"  Heatmap    : {msg.heatmap_before_url}")
    print(f"\n  New copy:\n{msg.new_copy}")
    print(f"\n  Strategy: {msg.optimization_strategy}")

    try:
        changes = json.loads(msg.changes)
        print(f"\n  Changes ({len(changes)}):")
        for c in changes:
            label = c.get("reason") or c.get("description") or str(c)[:80]
            print(f"    • {label}")
    except Exception:
        pass

    try:
        issues = json.loads(msg.issues)
        if issues:
            print(f"\n  Issues ({len(issues)}):")
            for i in issues:
                print(f"    ⚠️  {i}")
    except Exception:
        pass

    print(f"\n{'='*55}\n")


@listener.on_interval(period=5.0)
async def check_timeout(ctx: Context):
    elapsed = time.time() - _start
    if not _result_received:
        ctx.logger.info(f"Still waiting... ({elapsed:.0f}s / {TIMEOUT_SECONDS}s)")
    if elapsed > TIMEOUT_SECONDS and not _result_received:
        ctx.logger.error(
            f"TIMEOUT after {TIMEOUT_SECONDS}s — no FinalResult received.\n"
            "Check that all 5 agents are running and addresses are correct in .env.\n"
            "Run debug_pipeline.py to test each stage individually."
        )


if __name__ == "__main__":
    listener.run()
