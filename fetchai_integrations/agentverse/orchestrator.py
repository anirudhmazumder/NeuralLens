import os, re, json, io, time, requests, subprocess

def _free_port(port: int):
    r = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
    pids = r.stdout.strip()
    if pids:
        subprocess.run(["kill", "-9"] + pids.split(), capture_output=True)

_free_port(18000)
os.environ.setdefault("SSL_CERT_FILE", "/Users/peemmac/Library/Python/3.13/lib/python/site-packages/certifi/cacert.pem")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/Users/peemmac/Library/Python/3.13/lib/python/site-packages/certifi/cacert.pem")

import asyncio, stripe
from datetime import datetime
from uuid import uuid4
from openai import OpenAI
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement, ChatMessage, EndSessionContent,
    StartSessionContent, TextContent, chat_protocol_spec,
)
from uagents_core.contrib.protocols.payment import (
    CommitPayment, RejectPayment, CompletePayment,
    RequestPayment, Funds, payment_protocol_spec,
)
from dotenv import load_dotenv
load_dotenv()
import subprocess; subprocess.run(["lsof", "-ti", f"tcp:18000"], capture_output=True, text=True).stdout.strip() and subprocess.run(["kill", "-9"] + subprocess.run(["lsof", "-ti", f"tcp:18000"], capture_output=True, text=True).stdout.strip().split(), capture_output=True)

_asi = OpenAI(base_url="https://api.asi1.ai/v1", api_key=os.environ.get("ASI1_API_KEY", ""))
_openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")

def _create_checkout(description: str) -> dict:
    session = stripe.checkout.Session.create(
        mode="payment",
        payment_method_types=["card"],
        success_url="https://agentverse.ai?payment=success&session_id={CHECKOUT_SESSION_ID}",
        cancel_url="https://agentverse.ai?payment=cancelled",
        line_items=[{"price_data": {"currency": "usd", "product_data": {"name": "NeuralLens Optimization", "description": description}, "unit_amount": 100}, "quantity": 1}],
    )
    return {"checkout_session_id": session.id, "url": session.url}

def _verify_payment(checkout_session_id: str) -> bool:
    session = stripe.checkout.Session.retrieve(checkout_session_id)
    return getattr(session, "payment_status", None) == "paid"
import cloudinary, cloudinary.uploader
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", ""),
    api_key=os.environ.get("CLOUDINARY_API_KEY", ""),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", ""),
    secure=True,
)

# ── STEP 1: ASI:One analyzes image and generates an optimized prompt ──
def _analyze_image(image_url: str, goal: str) -> dict:
    """Ask ASI:One to analyze the image and return what to improve + a generation prompt."""
    system = (
        "You are a neuromarketing expert and creative director. "
        "Analyze the marketing image and return ONLY valid JSON:\n"
        '{"issues": ["issue1", "issue2"], "improvements": ["improvement1", "improvement2"], '
        '"generation_prompt": "detailed prompt for generating an improved version of this image"}\n'
        "IMPORTANT: Each issue MUST reference a specific brain region and explain why it is triggered. "
        "Use these regions naturally: amygdala (emotion/fear/desire), striatum (reward/dopamine), "
        "dlPFC (cognitive load/confusion), hippocampus (memory encoding), IPS (visual attention/saliency), "
        "insula (distrust/discomfort), mPFC (self-relevance). Example issue: "
        "'Low contrast suppresses IPS activation — the brain's attention system never locks onto the focal point.' "
        "The generation_prompt must be an edit instruction for the EXISTING image — keep ALL text, "
        "menu items, logos, layout, and content exactly as-is. Only describe visual improvements: "
        "richer colors, higher contrast, better lighting, sharper typography, more vibrant background. "
        "Start with: 'Keep all text and content identical. Enhance this image by:'"
    )
    # Try vision first
    try:
        resp = _asi.chat.completions.create(
            model="asi1",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": f"Goal: {goal}\nAnalyze this image and return the JSON."},
                ]},
            ],
            max_tokens=600,
        )
    except Exception:
        # Fall back to text-only if vision not supported
        resp = _asi.chat.completions.create(
            model="asi1",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Image URL: {image_url}\nGoal: {goal}\nAnalyze and return the JSON."},
            ],
            max_tokens=600,
        )

    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^```\s*|```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    # Fallback
    return {
        "issues": ["Low visual contrast", "Weak emotional impact"],
        "improvements": ["Boost saturation and contrast", "Add dramatic lighting"],
        "generation_prompt": f"High-quality marketing photo, vibrant colors, dramatic lighting, professional composition, emotionally compelling, {goal}, ultra-detailed, 4k",
    }

# ── STEP 2: OpenAI gpt-image-1 edit — iterates on original image ─
def _generate_image(prompt: str, image_url: str) -> str:
    """Download original image, send to OpenAI image edit, upload result to Cloudinary."""
    import base64
    from PIL import Image as PILImage

    # Download original
    img_bytes = requests.get(image_url, timeout=30).content

    # Convert to RGBA PNG and resize to exact OpenAI supported size
    buf = io.BytesIO(img_bytes)
    pil_img = PILImage.open(buf).convert("RGBA")
    w, h = pil_img.size
    ratio = h / w
    if ratio > 1.2:
        target = (1024, 1536)
    elif ratio < 0.8:
        target = (1536, 1024)
    else:
        target = (1024, 1024)
    size = f"{target[0]}x{target[1]}"
    pil_img = pil_img.resize(target, PILImage.LANCZOS)
    out_buf = io.BytesIO()
    pil_img.save(out_buf, format="PNG")
    out_buf.seek(0)

    # Call OpenAI image edit
    response = _openai.images.edit(
        model="gpt-image-1",
        image=("image.png", out_buf, "image/png"),
        prompt=prompt,
        size=size,
    )

    # Decode base64 result and upload to Cloudinary for a permanent URL
    image_data = base64.b64decode(response.data[0].b64_json)
    ts = int(time.time())
    result = cloudinary.uploader.upload(
        image_data, public_id=f"neurallens_{ts}_optimized", overwrite=True, resource_type="image"
    )
    return result["secure_url"]

# ── FULL PIPELINE ─────────────────────────────────────────────
def _run_pipeline_sync(image_url: str, goal: str) -> str:
    print(f"[pipeline] Step 1 — ASI:One analyzing image")
    analysis = _analyze_image(image_url, goal)
    print(f"[pipeline]   Issues: {analysis.get('issues')}")
    print(f"[pipeline]   Prompt: {analysis.get('generation_prompt', '')[:80]}...")

    print(f"[pipeline] Step 2 — Replicate img2img generating optimized version")
    new_image_url = _generate_image(analysis["generation_prompt"], image_url)
    print(f"[pipeline]   Generated: {new_image_url}")

    issues_text = "\n".join(f"  ⚠️  {i}" for i in analysis.get("issues", [])[:4]) or "  No critical issues"
    improvements_text = "\n".join(f"  ✅  {i}" for i in analysis.get("improvements", [])[:4]) or "  Applied general optimizations"

    return (
        f"✅ NeuralLens Optimization Complete\n\n"
        f"🔍 ISSUES FOUND IN ORIGINAL\n{issues_text}\n\n"
        f"🧠 NEURO-OPTIMIZATIONS APPLIED\n{improvements_text}\n\n"
        f"🖼️  NEW OPTIMIZED IMAGE\n  {new_image_url}\n\n"
        f"📝  GENERATION PROMPT USED\n  {analysis.get('generation_prompt', '')[:200]}\n\n"
        f"Powered by ASI:One + Replicate Flux — NeuralLens"
    )

# ── INTENT PARSER ─────────────────────────────────────────────
def _parse_intent(text: str) -> dict:
    try:
        resp = _asi.chat.completions.create(
            model="asi1",
            messages=[
                {"role": "system", "content": 'Extract and return ONLY valid JSON: {"image_url": "full URL or empty", "goal": "one sentence goal"}'},
                {"role": "user", "content": text},
            ],
            max_tokens=150,
        )
        raw = re.sub(r"```json|```", "", resp.choices[0].message.content.strip()).strip()
        return json.loads(raw)
    except Exception:
        url = re.search(r"https?://\S+", text)
        return {"image_url": url.group().rstrip(".,)") if url else "", "goal": text.strip()}

# ── INTER-AGENT NOTIFICATION MODEL ───────────────────────────
from uagents import Model

class PipelineNotification(Model):
    step: str
    data: str

class PipelineAck(Model):
    step: str
    status: str

# ── AGENT ─────────────────────────────────────────────────────
orchestrator = Agent(
    name="neurallens-orchestrator",
    seed="neurallens orchestrator lahacks 2026 neuromarketing main",
    port=18000,
    agentverse=os.environ.get("AGENTVERSE_KEY", ""),
    mailbox=True,
)
print(f"ORCHESTRATOR ADDRESS: {orchestrator.address}")

sessions: dict = {}
chat_proto = Protocol(spec=chat_protocol_spec)

def _msg(text: str) -> ChatMessage:
    return ChatMessage(timestamp=datetime.utcnow(), msg_id=uuid4(), content=[TextContent(type="text", text=text)])

@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(sender, ChatAcknowledgement(timestamp=datetime.utcnow(), acknowledged_msg_id=msg.msg_id))
    for item in msg.content:
        if isinstance(item, StartSessionContent):
            sessions[sender] = {}
            await ctx.send(sender, _msg(
                "🧠 Welcome to NeuralLens!\n\n"
                "Send me an image URL and what you want to optimize it for:\n\n"
                "  https://example.com/ad.jpg optimize for instagram engagement\n\n"
                "I'll analyze it with AI and generate a new, better version."
            ))
        elif isinstance(item, TextContent):
            # Check if user confirming payment
            if item.text.strip().upper() == "PAID" and sender in sessions and sessions[sender].get("checkout_session_id"):
                session_data = sessions[sender]
                if _verify_payment(session_data["checkout_session_id"]):
                    await ctx.send(sender, _msg("✅ Payment confirmed! Running brain analysis + image optimization... ⏳ ~30 seconds"))
                    for env_key, step in [("SENSOR_ADDRESS","image_fetch"),("INTERPRETER_ADDRESS","nes_analysis"),("STRATEGIST_ADDRESS","strategy"),("EXECUTOR_ADDRESS","execution")]:
                        addr = os.environ.get(env_key, "")
                        if addr:
                            await ctx.send(addr, PipelineNotification(step=step, data=session_data["image_url"]))
                    try:
                        result = await asyncio.wait_for(asyncio.to_thread(_run_pipeline_sync, session_data["image_url"], session_data["goal"]), timeout=240)
                        await ctx.send(sender, _msg(result))
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        await ctx.send(sender, _msg(f"⚠️ Failed: {e}"))
                else:
                    await ctx.send(sender, _msg("⚠️ Payment not found yet. Please complete checkout and try again."))
                continue
            parsed = _parse_intent(item.text)
            if not parsed.get("image_url"):
                await ctx.send(sender, _msg("Please include a direct image URL + your goal."))
                continue
            sessions[sender] = {"image_url": parsed["image_url"], "goal": parsed["goal"]}
            try:
                checkout = _create_checkout(f"Neuro-optimize: {parsed['goal'][:80]}")
                sessions[sender]["checkout_session_id"] = checkout["checkout_session_id"]
                await ctx.send(sender, _msg(
                    f"💳 Pay $1 to receive your AI brain-optimized image:\n\n"
                    f"👉 {checkout['url']}\n\n"
                    f"After paying, reply: PAID"
                ))
            except Exception as e:
                await ctx.send(sender, _msg(f"⚠️ Payment setup failed: {e}"))
        elif isinstance(item, EndSessionContent):
            sessions.pop(sender, None)

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement): pass

# ── PAYMENT HANDLER ───────────────────────────────────────────
payment_proto_seller = Protocol(spec=payment_protocol_spec, role="seller")

@payment_proto_seller.on_message(CommitPayment)
async def on_commit(ctx: Context, sender: str, msg: CommitPayment):
    if msg.funds.payment_method != "stripe" or not msg.transaction_id:
        await ctx.send(sender, RejectPayment(reason="Expected stripe payment."))
        return
    if not _verify_payment(msg.transaction_id):
        await ctx.send(sender, RejectPayment(reason="Payment not completed yet."))
        return
    await ctx.send(sender, CompletePayment(transaction_id=msg.transaction_id))
    await ctx.send(sender, _msg("✅ Payment confirmed! Running brain analysis + image optimization... ⏳ ~30 seconds"))
    session = sessions.get(sender, {})
    image_url = session.get("image_url", "")
    goal = session.get("goal", "optimize for engagement")
    if not image_url:
        await ctx.send(sender, _msg("⚠️ Session expired. Please send your image URL again."))
        return
    # Notify sub-agents
    for env_key, step in [("SENSOR_ADDRESS","image_fetch"),("INTERPRETER_ADDRESS","nes_analysis"),("STRATEGIST_ADDRESS","strategy"),("EXECUTOR_ADDRESS","execution")]:
        addr = os.environ.get(env_key, "")
        if addr:
            await ctx.send(addr, PipelineNotification(step=step, data=image_url))
    try:
        result = await asyncio.wait_for(asyncio.to_thread(_run_pipeline_sync, image_url, goal), timeout=240)
        await ctx.send(sender, _msg(result))
    except asyncio.TimeoutError:
        await ctx.send(sender, _msg("⚠️ Timed out. Try again."))
    except Exception as e:
        import traceback; traceback.print_exc()
        await ctx.send(sender, _msg(f"⚠️ Failed: {e}"))

@payment_proto_seller.on_message(RejectPayment)
async def on_reject(ctx: Context, sender: str, msg: RejectPayment):
    await ctx.send(sender, _msg(f"❌ Payment cancelled."))

# Handle acks from sub-agents
pipeline_proto = Protocol(name="NeuralLensPipelineProtocol", version="1.0.0")

@pipeline_proto.on_message(model=PipelineAck)
async def handle_pipeline_ack(ctx: Context, sender: str, msg: PipelineAck):
    ctx.logger.info(f"✅ Agent ack: {msg.step} — {msg.status}")

orchestrator.include(chat_proto, publish_manifest=True)
orchestrator.include(payment_proto_seller, publish_manifest=True)
orchestrator.include(pipeline_proto, publish_manifest=True)

if __name__ == "__main__":
    orchestrator.run()
