"""
NeuralLens Executor Agent.

Receives StrategistResult from Strategist Agent.
Downloads the original image (with browser-like headers).
Uploads it to Cloudinary as the base asset.
Applies visual_changes as Cloudinary URL transformations.
Estimates the after-NES from the number and type of issues resolved.
Sends FinalResult back to Orchestrator Agent.

Run standalone:   python agents/executor_agent.py
Run with Bureau:  python run_all_agents.py  (recommended)
"""

import os
import json
import time
import requests

from uagents import Agent, Context
from dotenv import load_dotenv

from agents.models import StrategistResult, FinalResult
from integrations.cloudinary_client import (
    upload_image,
    apply_visual_transformations,
)
from integrations.image_fetcher import fetch_image
from pipeline.nes_math import _get_profile

load_dotenv()

# ── AGENT SETUP ─────────────────────────────────────────────
executor = Agent(
    name="neurallens-executor",
    seed="neurallens executor cloudinary image lahacks 2026",
    port=8004,
    mailbox=True,
)

print("\n" + "=" * 55)
print(f"EXECUTOR ADDRESS: {executor.address}")
print("=" * 55 + "\n")


# ── HELPERS ─────────────────────────────────────────────────

def _download_image(url: str) -> bytes:
    """Download image bytes, retrying with multiple UA strings. See integrations/image_fetcher.py."""
    return fetch_image(url, timeout=30)


def _estimate_nes_after(nes_before: float, issues: list) -> float:
    """
    Estimate the post-optimisation NES.

    Each resolved brain-region issue contributes a fixed point lift.
    A production system would re-run TRIBE v2 on the Cloudinary-
    transformed image; this heuristic is sufficient for demo purposes.

    Args:
        nes_before: NES before optimisation.
        issues: List of issue strings from Interpreter.

    Returns:
        Estimated NES after, clamped to 0-100.
    """
    LIFTS = {
        "dlPFC":       8.0,   # reducing cognitive load is high leverage
        "striatum":    10.0,  # increasing desire is most impactful
        "amygdala":    7.0,
        "hippocampus": 5.0,
        "IPS":         6.0,
        "insula":      7.0,
    }
    total_lift = sum(
        lift for region, lift in LIFTS.items()
        if any(region.lower() in issue.lower() for issue in issues)
    )
    baseline = 5.0 if issues else 2.0
    return round(min(100.0, nes_before + baseline + total_lift), 1)


def _build_transforms(visual_changes: list) -> list:
    """
    Extract Cloudinary transformation dicts from the visual changes list.
    Falls back to a quality-improve + vibrance transform if none are found.

    Args:
        visual_changes: List of change dicts from Strategist.

    Returns:
        List of Cloudinary transformation dicts.
    """
    transforms = [
        ct for c in visual_changes
        if isinstance((ct := c.get("cloudinary_transform")), dict)
    ]
    if not transforms:
        transforms = [{"effect": "improve:40"}, {"effect": "vibrance:30"}]
    return transforms


# ── MESSAGE HANDLER ─────────────────────────────────────────

@executor.on_message(model=StrategistResult)
async def handle_strategist_result(ctx: Context, sender: str, msg: StrategistResult):
    """
    Apply Cloudinary transforms, estimate after-NES, deliver FinalResult
    to the Orchestrator Agent so it can be sent to the original user.
    """
    ctx.logger.info(f"📥 StrategistResult: {msg.image_url}")

    orchestrator_addr = os.environ.get("ORCHESTRATOR_ADDRESS", "")
    if not orchestrator_addr:
        ctx.logger.error("ORCHESTRATOR_ADDRESS not set in .env")
        return

    ts = int(time.time())

    try:
        # 1. Parse changes from Strategist
        try:
            visual_changes = json.loads(msg.visual_changes)
        except Exception:
            visual_changes = []
        try:
            text_changes = json.loads(msg.text_changes)
        except Exception:
            text_changes = []
        try:
            issues = json.loads(msg.issues)
        except Exception:
            issues = []

        all_changes = visual_changes + text_changes
        ctx.logger.info(
            f"{len(visual_changes)} visual + {len(text_changes)} text changes"
        )

        # 2. Download original image
        ctx.logger.info("Downloading original image for Cloudinary...")
        image_bytes = _download_image(msg.image_url)

        # 3. Upload original to Cloudinary
        original_id = f"neurallens_{ts}_original"
        ctx.logger.info(f"Uploading original as {original_id}...")
        upload_image(image_bytes, original_id)

        # 4. Apply Cloudinary transformations
        transforms = _build_transforms(visual_changes)
        ctx.logger.info(f"Applying {len(transforms)} Cloudinary transform(s)...")
        optimized_url = apply_visual_transformations(original_id, transforms)
        ctx.logger.info(f"Optimized URL: {optimized_url}")

        # 5. Estimate after-NES
        nes_after = _estimate_nes_after(msg.nes_total, issues)
        delta = round(nes_after - msg.nes_total, 1)
        ctx.logger.info(f"NES: {msg.nes_total} → {nes_after} (Δ{delta:+.1f})")

        # 6. Derive after-profile heuristically
        if delta >= 20:
            profile_after = "high desire and emotion — strong engagement"
        elif delta >= 10:
            profile_after = "moderate engagement — measurable improvement"
        else:
            profile_after = msg.profile  # unchanged if small delta

        # 7. Send FinalResult to Orchestrator
        await ctx.send(
            orchestrator_addr,
            FinalResult(
                nes_before=msg.nes_total,
                nes_after=nes_after,
                delta=delta,
                profile_before=msg.profile,
                profile_after=profile_after,
                issues=msg.issues,
                changes=json.dumps(all_changes),
                optimized_image_url=optimized_url,
                heatmap_before_url=msg.heatmap_url,
                heatmap_after_url=optimized_url,
                new_copy=msg.new_copy,
                optimization_strategy=msg.optimization_strategy,
                session_sender=msg.session_sender,
            ),
        )
        ctx.logger.info(f"✅ FinalResult sent to orchestrator {orchestrator_addr}")

    except Exception as e:
        ctx.logger.error(f"❌ Executor failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    executor.run()
