"""
NeuralLens Interpreter Agent.

Receives SensorResult from Sensor Agent.
Decompresses the zlib-encoded activation array.
Runs: extract_roi_values → compute_nes → analyze_intersection.
Forwards InterpreterResult to Strategist Agent.

Run standalone:   python agents/interpreter_agent.py
Run with Bureau:  python run_all_agents.py  (recommended)
"""

import os
import json
import zlib
import base64
import numpy as np

from uagents import Agent, Context
from dotenv import load_dotenv

from agents.models import SensorResult, InterpreterResult
from pipeline.nes_math import extract_roi_values, compute_nes, analyze_intersection

load_dotenv()

# ── AGENT SETUP ─────────────────────────────────────────────
interpreter = Agent(
    name="neurallens-interpreter",
    seed="neurallens interpreter nes roi scorer lahacks 2026",
    port=8002,
    mailbox=True,
)

print("\n" + "=" * 55)
print(f"INTERPRETER ADDRESS: {interpreter.address}")
print("=" * 55 + "\n")


# ── MESSAGE HANDLER ─────────────────────────────────────────

@interpreter.on_message(model=SensorResult)
async def handle_sensor_result(ctx: Context, sender: str, msg: SensorResult):
    """
    Decompress activations, run NES math, forward to Strategist.
    """
    ctx.logger.info("📥 SensorResult received")

    strategist_addr = os.environ.get("STRATEGIST_ADDRESS", "")
    if not strategist_addr:
        ctx.logger.error("STRATEGIST_ADDRESS not set in .env")
        return

    try:
        # 1. Decompress activations: base64 → zlib → float32 numpy array
        raw_bytes = zlib.decompress(base64.b64decode(msg.tribe_activations))
        activations = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        ctx.logger.info(f"Decompressed: {len(activations):,} activations")

        # 2. Deserialise DeepGaze saliency
        saliency = json.loads(msg.deepgaze_heatmap)
        ctx.logger.info(f"Saliency: {len(saliency):,} values")

        # 3. Extract 7 ROI means, normalised 0-100
        roi_values = extract_roi_values(activations)
        ctx.logger.info(f"ROI values: {roi_values}")

        # 4. Compute NES, valence, arousal, profile, issues
        nes = compute_nes(roi_values)
        ctx.logger.info(
            f"NES: {nes['nes_total']} | {nes['profile']}"
        )

        # 5. 9-zone TRIBE × DeepGaze intersection
        insights = analyze_intersection(activations, saliency)
        ctx.logger.info(f"Zones: {len(insights)}")

        # 6. Forward to Strategist
        await ctx.send(
            strategist_addr,
            InterpreterResult(
                image_url=msg.image_url,
                nes_total=nes["nes_total"],
                roi_values=json.dumps(roi_values),
                profile=nes["profile"],
                issues=json.dumps(nes["issues"]),
                insights=json.dumps(insights),
                valence=nes["valence"],
                arousal=nes["arousal"],
                heatmap_url=msg.heatmap_url,
                prompt=msg.prompt,
                industry=msg.industry,
                asset_type=msg.asset_type,
                session_sender=msg.session_sender,
            ),
        )
        ctx.logger.info(f"✅ InterpreterResult forwarded to {strategist_addr}")

    except Exception as e:
        ctx.logger.error(f"❌ Interpreter failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    interpreter.run()
