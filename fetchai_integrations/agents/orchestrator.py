"""
NeuralLens Orchestrator Agent.

The ONLY agent the user interacts with directly via ASI:One.
Implements Fetch.ai Chat Protocol for ASI:One discoverability.
Uses ASI:One LLM to parse user intent from natural language.
Coordinates the full 4-agent pipeline.
Receives FinalResult and formats it for the user.

Run standalone:   python agents/orchestrator.py
Run with Bureau:  python run_all_agents.py  (recommended)
"""

from datetime import datetime
from uuid import uuid4
import os
import re
import json

from openai import OpenAI
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)
from dotenv import load_dotenv

from agents.models import SensorRequest, FinalResult

load_dotenv()

AGENTVERSE_KEY = os.environ.get("AGENTVERSE_KEY", "")

# ── AGENT SETUP ─────────────────────────────────────────────
# No endpoint/port needed — Agentverse mailbox handles routing.
orchestrator = Agent(
    name="neurallens-orchestrator",
    seed="neurallens orchestrator lahacks 2026 neuromarketing main",
    port=8000,
    mailbox=True,
)

print("\n" + "=" * 55)
print(f"ORCHESTRATOR ADDRESS: {orchestrator.address}")
print("=" * 55 + "\n")

# ── ASI:ONE LLM CLIENT ──────────────────────────────────────
asi_client = OpenAI(
    base_url="https://api.asi1.ai/v1",
    api_key=os.environ.get("ASI1_API_KEY", ""),
)

# ── SESSION STORAGE ─────────────────────────────────────────
sessions: dict = {}

# ── CHAT PROTOCOL ───────────────────────────────────────────
chat_proto = Protocol(spec=chat_protocol_spec)

# ── SAFE IMAGE FETCH HEADERS ────────────────────────────────
FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}


def create_text_message(text: str) -> ChatMessage:
    """Wrap plain text into a ChatMessage."""
    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)],
    )


def parse_user_intent(user_text: str) -> dict:
    """
    Use ASI:One LLM to extract image_url, industry, asset_type, goal.
    Falls back to regex on failure.
    """
    system_prompt = """You are a precise JSON parser.
Extract information from the user's message and return
ONLY a valid JSON object with no markdown, no backticks, no explanation.

JSON format:
{
  "image_url": "full URL to image or empty string",
  "industry": "one of: bakery, restaurant, retail, saas, beauty, fitness, general",
  "asset_type": "one of: instagram_post, menu, flyer, logo, poster, advertisement, image",
  "goal": "the user's optimization goal in one sentence"
}

Rules:
- image_url: extract any URL ending in .jpg .jpeg .png .webp .gif OR from image CDNs
- industry: infer from context (bakery, cafe, salon, shop, etc.)
- asset_type: infer from words like instagram, post, menu, flyer, logo
- goal: rephrase the user's intent as one clear sentence
- Default industry: "general", default asset_type: "image" """

    try:
        response = asi_client.chat.completions.create(
            model="asi1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        url_match = re.search(r"https?://\S+", user_text)
        return {
            "image_url": url_match.group().rstrip(".,)") if url_match else "",
            "industry": "general",
            "asset_type": "image",
            "goal": user_text.strip(),
        }


def format_final_response(msg: FinalResult) -> str:
    """Format FinalResult into readable ASI:One chat text."""
    try:
        issues = json.loads(msg.issues)
    except Exception:
        issues = []

    try:
        changes = json.loads(msg.changes)
    except Exception:
        changes = []

    delta_emoji = "🟢" if msg.delta > 20 else "🟡" if msg.delta > 5 else "🔴"

    issues_text = "\n".join(
        f"  ⚠️  {issue}" for issue in issues[:4]
    ) or "  No critical issues detected"

    changes_text = "\n".join(
        f"  ✏️  {c.get('reason', c.get('description', str(c)[:80]))}"
        for c in changes[:4]
    ) or "  Visual and copy optimizations applied"

    return (
        f"✅ NeuralLens Analysis Complete\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 NEURAL ENGAGEMENT SCORE\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Before: {msg.nes_before:.1f} / 100\n"
        f"  After:  {msg.nes_after:.1f} / 100\n"
        f"  Change: {delta_emoji} +{msg.delta:.1f} points\n\n"
        f"  Profile before: {msg.profile_before}\n"
        f"  Profile after:  {msg.profile_after}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️  ISSUES DETECTED\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{issues_text}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✏️  CHANGES APPLIED\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{changes_text}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🖼️  YOUR OPTIMIZED IMAGE\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  {msg.optimized_image_url}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📝  NEW COPY (paste ready)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{msg.new_copy}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Strategy: {msg.optimization_strategy}\n\n"
        f"Powered by TRIBE v2 + DeepGaze + Gemma 3 + Cloudinary AI\n"
        f"NeuralLens — neuromarketing for every business\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


# ── CHAT PROTOCOL HANDLERS ──────────────────────────────────

@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle all incoming ASI:One chat messages."""
    ctx.logger.info(f"Chat message from {sender}")

    # Always acknowledge first (mandatory for Chat Protocol)
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.utcnow(),
        acknowledged_msg_id=msg.msg_id,
    ))

    for item in msg.content:
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"New session: {sender}")
            sessions[sender] = {"status": "ready"}
            await ctx.send(sender, create_text_message(
                "🧠 Welcome to NeuralLens!\n\n"
                "I simulate how a real human brain responds to your visual "
                "content and automatically optimize both the image and copy.\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "HOW TO USE\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "Send me a direct image URL + your goal:\n\n"
                "https://yoursite.com/post.jpg\n"
                "optimize this Instagram post for my bakery\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "WHAT YOU GET BACK\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "✅ Optimized image URL\n"
                "✅ NES score before and after\n"
                "✅ Brain region analysis\n"
                "✅ 3 specific changes with neuroscience reasons\n"
                "✅ Ready-to-paste caption\n"
            ))

        elif isinstance(item, TextContent):
            ctx.logger.info(f"Text from {sender}: {item.text[:80]}")
            parsed = parse_user_intent(item.text)
            ctx.logger.info(f"Parsed: {parsed}")

            if not parsed.get("image_url"):
                await ctx.send(sender, create_text_message(
                    "I need a direct image URL to analyze. 🖼️\n\n"
                    "Please include a link ending in .jpg, .png, or .webp\n"
                    "along with your optimization goal."
                ))
                continue

            sessions[sender] = {
                "status": "processing",
                **parsed,
            }

            await ctx.send(sender, create_text_message(
                f"🔬 Analyzing your "
                f"{parsed['asset_type'].replace('_', ' ')} "
                f"for {parsed['industry']}...\n\n"
                f"Running TRIBE v2 brain simulation + DeepGaze eye tracking.\n"
                f"⏳ This takes 30-60 seconds. Stand by."
            ))

            sensor_addr = os.environ.get("SENSOR_ADDRESS", "")
            if not sensor_addr:
                await ctx.send(sender, create_text_message(
                    "⚠️ Configuration error: SENSOR_ADDRESS not set.\n"
                    "Run: python run_all_agents.py"
                ))
                continue

            try:
                await ctx.send(sensor_addr, SensorRequest(
                    image_url=parsed["image_url"],
                    prompt=parsed["goal"],
                    industry=parsed["industry"],
                    asset_type=parsed["asset_type"],
                    session_sender=sender,
                ))
                ctx.logger.info(f"SensorRequest sent to {sensor_addr}")
            except Exception as e:
                ctx.logger.error(f"Failed to reach sensor: {e}")
                await ctx.send(sender, create_text_message(
                    f"⚠️ Pipeline error: {e}\n"
                    "Make sure all agents are running via: python run_all_agents.py"
                ))

        elif isinstance(item, EndSessionContent):
            ctx.logger.info(f"Session ended: {sender}")
            sessions.pop(sender, None)


@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Ack from {sender}")


# ── FINAL RESULT HANDLER ────────────────────────────────────

@orchestrator.on_message(model=FinalResult)
async def handle_final_result(ctx: Context, sender: str, msg: FinalResult):
    """Receive completed result from Executor, deliver to original user."""
    ctx.logger.info(
        f"✅ FinalResult: NES {msg.nes_before} → {msg.nes_after}"
    )

    original_sender = msg.session_sender
    if not original_sender:
        ctx.logger.error("No session_sender in FinalResult — cannot deliver")
        return

    if original_sender in sessions:
        sessions[original_sender]["status"] = "complete"

    try:
        await ctx.send(original_sender, create_text_message(
            format_final_response(msg)
        ))
        ctx.logger.info(f"Result delivered to {original_sender}")
    except Exception as e:
        ctx.logger.error(f"Failed to deliver to {original_sender}: {e}")


# ── PROTOCOL + RUN ──────────────────────────────────────────
orchestrator.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    orchestrator.run()
