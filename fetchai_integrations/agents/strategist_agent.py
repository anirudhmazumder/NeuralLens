"""
NeuralLens Strategist Agent.

Receives InterpreterResult from Interpreter Agent.
Calls ASI:One LLM (Fetch.ai's native model, OpenAI-compatible API)
to generate 3 visual changes, 2 text changes, new copy, and a strategy.
Falls back to sensible defaults if the API is unavailable.
Forwards StrategistResult to Executor Agent.

Using ASI:One (not Gemma) because:
  - ASI:One API key already works (confirmed)
  - Scores points on the Fetch.ai hackathon track
  - No separate Google Cloud project setup required

Run standalone:   python agents/strategist_agent.py
Run with Bureau:  python run_all_agents.py  (recommended)
"""

import os
import json
import re

from openai import OpenAI
from uagents import Agent, Context
from dotenv import load_dotenv

from agents.models import InterpreterResult, StrategistResult

load_dotenv()

# ── ASI:ONE CLIENT ───────────────────────────────────────────
_asi = OpenAI(
    base_url="https://api.asi1.ai/v1",
    api_key=os.environ.get("ASI1_API_KEY", ""),
)

# ── AGENT SETUP ─────────────────────────────────────────────
strategist = Agent(
    name="neurallens-strategist",
    seed="neurallens strategist gemma optimizer lahacks 2026",
    port=8003,
    mailbox=True,
)

print("\n" + "=" * 55)
print(f"STRATEGIST ADDRESS: {strategist.address}")
print("=" * 55 + "\n")

# ── SYSTEM PROMPT ────────────────────────────────────────────
_SYSTEM = """You are a neuromarketing optimization expert for visual content.

You receive:
- Asset type (Instagram post, menu, flyer, logo, etc.)
- Business industry
- User's optimization goal
- NES score breakdown across 7 brain regions (0-100 each)
- Zone-level analysis (9 zones: power_zone, attention_trap, hidden_value, dead_zone)
- Specific brain region issues

Return ONLY valid JSON. No markdown. No backticks. No explanation. No preamble.

{
  "visual_changes": [
    {
      "zone": "which zone (top-left, center, etc.)",
      "change_type": "color|contrast|size|position|crop|filter",
      "description": "specific visual change to make",
      "cloudinary_transform": {"effect": "brightness:30"},
      "target_region": "brain region targeted",
      "reason": "one-sentence neuroscience explanation"
    }
  ],
  "text_changes": [
    {
      "element": "headline|cta|body|tagline",
      "old_text": "current text or description",
      "new_text": "exact replacement text",
      "target_region": "brain region targeted",
      "reason": "one-sentence neuroscience explanation"
    }
  ],
  "new_copy": "complete ready-to-paste caption with emojis and hashtags",
  "optimization_strategy": "one sentence overall strategy"
}

RULES:
- visual_changes: EXACTLY 3 items
- text_changes: EXACTLY 2 items
- new_copy: 3-5 sentences, include emojis and relevant hashtags
- cloudinary_transform: must be a valid Cloudinary transformation dict
- Write SPECIFIC values, not vague descriptions
- Prioritize highest-impact brain region issues first
- For dead zones: suggest removal or replacement content
- For attention traps: suggest emotional enrichment
- For hidden value zones: suggest making that zone dominant"""


def _build_user_prompt(msg: InterpreterResult) -> str:
    """Assemble the user-turn prompt from InterpreterResult fields."""
    try:
        roi = json.loads(msg.roi_values)
    except Exception:
        roi = {}
    try:
        issues = json.loads(msg.issues)
    except Exception:
        issues = []
    try:
        insights = json.loads(msg.insights)
    except Exception:
        insights = []

    issues_text = "\n".join(f"- {i}" for i in issues) or "(none)"
    zones_text = "\n".join(
        f"  {z['zone']} [{z['type']}]: {z['action']}"
        for z in insights
        if z.get("type") in ("attention_trap", "hidden_value", "dead_zone")
    ) or "  (no critical zones)"

    return (
        f"Asset: {msg.asset_type.replace('_', ' ')}\n"
        f"Industry: {msg.industry}\n"
        f"Goal: {msg.prompt}\n\n"
        f"NES Total: {msg.nes_total} / 100\n"
        f"Profile: {msg.profile}\n"
        f"Valence: {msg.valence}  Arousal: {msg.arousal}\n\n"
        f"Brain region issues:\n{issues_text}\n\n"
        f"ROI values (0-100):\n{json.dumps(roi, indent=2)}\n\n"
        f"Critical zones:\n{zones_text}"
    )


def _parse_json_response(raw: str) -> dict:
    """
    Parse a JSON dict from a raw LLM response.
    Strips markdown fences and uses regex fallback.
    Raises ValueError if nothing parseable is found.
    """
    # Strip markdown fences
    raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^```\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Regex fallback: grab first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot parse response as JSON:\n{raw[:400]}")


def _call_asi_one(msg: InterpreterResult) -> dict:
    """
    Call ASI:One with neuromarketing context.
    Returns parsed strategy dict.
    Raises ValueError if response cannot be parsed.
    """
    response = _asi.chat.completions.create(
        model="asi1",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _build_user_prompt(msg)},
        ],
        max_tokens=2048,
    )
    raw = response.choices[0].message.content.strip()
    return _parse_json_response(raw)


def _fallback_strategy(msg: InterpreterResult) -> dict:
    """
    Minimal valid strategy when ASI:One is unavailable.
    Targets the most commonly flagged brain regions.
    """
    try:
        issues = json.loads(msg.issues)
        top_region = "striatum"
        for issue in issues:
            for r in ("dlPFC", "striatum", "amygdala", "IPS", "insula"):
                if r.lower() in issue.lower():
                    top_region = r
                    break
    except Exception:
        top_region = "striatum"

    return {
        "visual_changes": [
            {
                "zone": "center",
                "change_type": "contrast",
                "description": "Boost contrast by 25% to strengthen focal point",
                "cloudinary_transform": {"effect": "improve:50"},
                "target_region": "IPS",
                "reason": "Higher contrast boosts IPS activation and makes the focal element unmissable",
            },
            {
                "zone": "top-center",
                "change_type": "filter",
                "description": "Warm up color temperature to evoke emotion",
                "cloudinary_transform": {"effect": "vibrance:40"},
                "target_region": "amygdala",
                "reason": "Warm, saturated colors consistently elevate amygdala emotional arousal",
            },
            {
                "zone": "bottom-center",
                "change_type": "brightness",
                "description": "Auto-enhance overall image quality",
                "cloudinary_transform": {"effect": "auto_brightness"},
                "target_region": top_region,
                "reason": "Quality enhancement raises desire signal and reduces cognitive friction",
            },
        ],
        "text_changes": [
            {
                "element": "headline",
                "old_text": "(current headline)",
                "new_text": "Made for people who want more — not less.",
                "target_region": "mPFC",
                "reason": "Self-relevant framing directly activates medial PFC and raises self-relevance score",
            },
            {
                "element": "cta",
                "old_text": "(current CTA)",
                "new_text": "Get started free →",
                "target_region": "striatum",
                "reason": "Low-friction, benefit-first CTA maximises striatum desire and reduces dlPFC load",
            },
        ],
        "new_copy": (
            "✨ What if your results could double this month?\n\n"
            "We've helped hundreds of businesses like yours grow faster "
            "without the guesswork.\n\n"
            "🔗 Try it free today — no credit card needed.\n\n"
            f"#{msg.industry.replace(' ', '')} #SmallBusiness #Growth #Marketing"
        ),
        "optimization_strategy": (
            "Strengthen contrast and warm color temperature to boost amygdala "
            "and IPS activation, while simplifying copy to cut dlPFC cognitive "
            "load and raise the striatum desire signal."
        ),
    }


# ── MESSAGE HANDLER ─────────────────────────────────────────

@strategist.on_message(model=InterpreterResult)
async def handle_interpreter_result(ctx: Context, sender: str, msg: InterpreterResult):
    """
    Call ASI:One for optimization strategy, forward to Executor.
    """
    ctx.logger.info(
        f"📥 InterpreterResult: NES={msg.nes_total} | {msg.profile}"
    )

    executor_addr = os.environ.get("EXECUTOR_ADDRESS", "")
    if not executor_addr:
        ctx.logger.error("EXECUTOR_ADDRESS not set in .env")
        return

    try:
        ctx.logger.info("Calling ASI:One for neuromarketing strategy...")
        strategy = _call_asi_one(msg)
        ctx.logger.info("✅ ASI:One strategy parsed successfully")
    except Exception as e:
        ctx.logger.error(f"ASI:One call failed ({e}) — using fallback strategy")
        strategy = _fallback_strategy(msg)

    try:
        await ctx.send(
            executor_addr,
            StrategistResult(
                image_url=msg.image_url,
                visual_changes=json.dumps(strategy.get("visual_changes", [])),
                text_changes=json.dumps(strategy.get("text_changes", [])),
                new_copy=strategy.get("new_copy", ""),
                optimization_strategy=strategy.get("optimization_strategy", ""),
                nes_total=msg.nes_total,
                profile=msg.profile,
                issues=msg.issues,
                heatmap_url=msg.heatmap_url,
                session_sender=msg.session_sender,
            ),
        )
        ctx.logger.info(f"✅ StrategistResult forwarded to {executor_addr}")
    except Exception as e:
        ctx.logger.error(f"❌ Strategist forward failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    strategist.run()
