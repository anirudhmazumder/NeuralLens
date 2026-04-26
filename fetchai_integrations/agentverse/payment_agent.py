import os, re, json, io, time, requests, subprocess

def _free_port(port: int):
    r = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
    pids = r.stdout.strip()
    if pids:
        subprocess.run(["kill", "-9"] + pids.split(), capture_output=True)

_free_port(18005)
os.environ.setdefault("SSL_CERT_FILE", "/Users/peemmac/Library/Python/3.13/lib/python/site-packages/certifi/cacert.pem")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/Users/peemmac/Library/Python/3.13/lib/python/site-packages/certifi/cacert.pem")

import asyncio, stripe, urllib3
import cloudinary, cloudinary.uploader
from PIL import Image as PILImage
from datetime import datetime
from uuid import uuid4
from openai import OpenAI
from uagents import Agent, Context, Protocol, Model
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
import subprocess; subprocess.run(["lsof", "-ti", f"tcp:18005"], capture_output=True, text=True).stdout.strip() and subprocess.run(["kill", "-9"] + subprocess.run(["lsof", "-ti", f"tcp:18005"], capture_output=True, text=True).stdout.strip().split(), capture_output=True)

# ── CLIENTS ───────────────────────────────────────────────────
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
_asi   = OpenAI(base_url="https://api.asi1.ai/v1", api_key=os.environ.get("ASI1_API_KEY", ""))
_openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", ""),
    api_key=os.environ.get("CLOUDINARY_API_KEY", ""),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", ""),
    secure=True,
)

# ── STRIPE HELPERS ────────────────────────────────────────────
def _create_checkout(description: str) -> dict:
    session = stripe.checkout.Session.create(
        ui_mode="embedded",
        redirect_on_completion="if_required",
        mode="payment",
        payment_method_types=["card"],
        return_url="https://agentverse.ai?session_id={CHECKOUT_SESSION_ID}",
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {"name": "NeuralLens Image Optimization", "description": description},
                "unit_amount": 100,
            },
            "quantity": 1,
        }],
    )
    return {
        "client_secret": session.client_secret,
        "checkout_session_id": session.id,
        "publishable_key": STRIPE_PUBLISHABLE_KEY,
        "currency": "usd",
        "amount_cents": 100,
        "ui_mode": "embedded",
    }

def _verify_payment(checkout_session_id: str) -> bool:
    session = stripe.checkout.Session.retrieve(checkout_session_id)
    return getattr(session, "payment_status", None) == "paid"

# ── IMAGE PIPELINE (self-contained) ───────────────────────────
_UAS = ["Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"]

def _fetch_image(url: str) -> bytes:
    try:
        r = requests.get(url, headers={"User-Agent": _UAS[0], "Accept": "image/*,*/*"}, timeout=30, allow_redirects=True)
        r.raise_for_status()
        return r.content
    except Exception:
        urllib3.disable_warnings()
        r = requests.get(url, headers={"User-Agent": _UAS[0]}, timeout=30, allow_redirects=True, verify=False)
        r.raise_for_status()
        return r.content

def _analyze_image(image_url: str, goal: str) -> dict:
    system = (
        "You are a neuromarketing expert and creative director. "
        "Analyze the marketing image and return ONLY valid JSON:\n"
        '{"issues": ["issue1"], "improvements": ["improvement1"], "generation_prompt": "..."}\n'
        "Each issue MUST reference a brain region: amygdala (emotion), striatum (reward), "
        "dlPFC (cognitive load), hippocampus (memory), IPS (visual attention), insula (distrust), mPFC (self-relevance). "
        "The generation_prompt must say 'Keep all text and content identical. Enhance this image by:' "
        "then describe ONLY visual improvements (colors, contrast, lighting, vibrancy)."
    )
    try:
        resp = _asi.chat.completions.create(
            model="asi1",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": f"Goal: {goal}. Analyze and return JSON."},
                ]},
            ],
            max_tokens=600,
        )
    except Exception:
        resp = _asi.chat.completions.create(
            model="asi1",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Image: {image_url}\nGoal: {goal}"},
            ],
            max_tokens=600,
        )
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", resp.choices[0].message.content.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    return {
        "issues": ["Low amygdala activation — emotional pull is weak"],
        "improvements": ["Boost contrast and warmth"],
        "generation_prompt": "Keep all text and content identical. Enhance this image by: boosting saturation, adding dramatic warm lighting, increasing contrast for visual impact.",
    }

def _generate_image(prompt: str, image_url: str) -> str:
    img_bytes = _fetch_image(image_url)
    buf = io.BytesIO(img_bytes)
    pil_img = PILImage.open(buf).convert("RGBA")
    w, h = pil_img.size
    ratio = h / w
    target = (1024, 1536) if ratio > 1.2 else (1536, 1024) if ratio < 0.8 else (1024, 1024)
    size = f"{target[0]}x{target[1]}"
    pil_img = pil_img.resize(target, PILImage.LANCZOS)
    out_buf = io.BytesIO()
    pil_img.save(out_buf, format="PNG")
    out_buf.seek(0)
    response = _openai.images.edit(
        model="gpt-image-1",
        image=("image.png", out_buf, "image/png"),
        prompt=prompt,
        size=size,
    )
    import base64
    image_data = base64.b64decode(response.data[0].b64_json)
    ts = int(time.time())
    result = cloudinary.uploader.upload(image_data, public_id=f"neurallens_pay_{ts}", overwrite=True, resource_type="image")
    return result["secure_url"]

def _run_pipeline_sync(image_url: str, goal: str) -> str:
    print(f"[payment-pipeline] Step 1 — ASI:One analyzing image")
    analysis = _analyze_image(image_url, goal)
    print(f"[payment-pipeline] Step 2 — OpenAI generating optimized image")
    new_url = _generate_image(analysis["generation_prompt"], image_url)
    issues_text = "\n".join(f"  ⚠️  {i}" for i in analysis.get("issues", [])[:4])
    improvements_text = "\n".join(f"  ✅  {i}" for i in analysis.get("improvements", [])[:4])
    return (
        f"✅ NeuralLens Optimization Complete\n\n"
        f"🔍 BRAIN ISSUES DETECTED\n{issues_text}\n\n"
        f"🧠 NEURO-OPTIMIZATIONS APPLIED\n{improvements_text}\n\n"
        f"🖼️  OPTIMIZED IMAGE\n  {new_url}\n\n"
        f"Powered by ASI:One + GPT-Image-1 — NeuralLens"
    )

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

# ── AGENT ─────────────────────────────────────────────────────
payment_agent = Agent(
    name="neurallens-payment",
    seed="neurallens payment stripe agent lahacks 2026",
    port=18005,
    agentverse=os.environ.get("AGENTVERSE_KEY", ""),
    mailbox=True,
)
print(f"PAYMENT AGENT ADDRESS: {payment_agent.address}")

sessions: dict = {}  # sender → {image_url, goal, checkout_session_id}

chat_proto    = Protocol(spec=chat_protocol_spec)
payment_proto = Protocol(spec=payment_protocol_spec, role="seller")

def _msg(text: str) -> ChatMessage:
    return ChatMessage(timestamp=datetime.utcnow(), msg_id=uuid4(), content=[TextContent(type="text", text=text)])

# ── CHAT HANDLER ──────────────────────────────────────────────
@chat_proto.on_message(ChatMessage)
async def handle_chat(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(sender, ChatAcknowledgement(timestamp=datetime.utcnow(), acknowledged_msg_id=msg.msg_id))
    for item in msg.content:
        if isinstance(item, StartSessionContent):
            sessions[sender] = {}
            await ctx.send(sender, _msg(
                "🧠 NeuralLens Pro — AI-Powered Image Optimization\n\n"
                "Send me an image URL and your goal, and I'll generate a neuro-optimized version for just $1.\n\n"
                "Example:\n"
                "  https://example.com/flyer.jpg optimize for instagram engagement\n\n"
                "Powered by ASI:One brain analysis + GPT-Image-1 generation."
            ))
        elif isinstance(item, TextContent):
            parsed = _parse_intent(item.text)
            if not parsed.get("image_url"):
                await ctx.send(sender, _msg("Please include a direct image URL + your goal."))
                continue

            sessions[sender] = {"image_url": parsed["image_url"], "goal": parsed["goal"]}
            description = f"Neuro-optimize: {parsed['goal'][:80]}"

            try:
                checkout = _create_checkout(description)
                sessions[sender]["checkout_session_id"] = checkout["checkout_session_id"]
                req = RequestPayment(
                    accepted_funds=[Funds(currency="USD", amount="1.00", payment_method="stripe")],
                    recipient=str(ctx.agent.address),
                    description="Pay $1 to receive your AI-optimized image with brain science analysis.",
                    metadata={"stripe": checkout, "service": "neurallens_optimization"},
                )
                await ctx.send(sender, req)
                await ctx.send(sender, _msg("💳 Please complete the $1 payment above to receive your optimized image."))
            except Exception as e:
                await ctx.send(sender, _msg(f"⚠️ Payment setup failed: {e}"))
        elif isinstance(item, EndSessionContent):
            sessions.pop(sender, None)

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement): pass

# ── PAYMENT HANDLER ───────────────────────────────────────────
@payment_proto.on_message(CommitPayment)
async def on_commit(ctx: Context, sender: str, msg: CommitPayment):
    if msg.funds.payment_method != "stripe" or not msg.transaction_id:
        await ctx.send(sender, RejectPayment(reason="Expected stripe payment method."))
        return

    paid = _verify_payment(msg.transaction_id)
    if not paid:
        await ctx.send(sender, RejectPayment(reason="Payment not completed yet. Please finish checkout."))
        return

    await ctx.send(sender, CompletePayment(transaction_id=msg.transaction_id))
    await ctx.send(sender, _msg("✅ Payment confirmed! Running brain analysis + image optimization... ⏳ ~30 seconds"))

    session = sessions.get(sender, {})
    image_url = session.get("image_url", "")
    goal      = session.get("goal", "optimize for engagement")

    if not image_url:
        await ctx.send(sender, _msg("⚠️ Session expired. Please send your image URL again."))
        return

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_run_pipeline_sync, image_url, goal),
            timeout=240,
        )
        await ctx.send(sender, _msg(result))
    except asyncio.TimeoutError:
        await ctx.send(sender, _msg("⚠️ Optimization timed out. Contact support for a refund."))
    except Exception as e:
        import traceback; traceback.print_exc()
        await ctx.send(sender, _msg(f"⚠️ Optimization failed: {e}"))

@payment_proto.on_message(RejectPayment)
async def on_reject(ctx: Context, sender: str, msg: RejectPayment):
    await ctx.send(sender, _msg(f"❌ Payment rejected: {msg.reason}"))

payment_agent.include(chat_proto, publish_manifest=True)
payment_agent.include(payment_proto, publish_manifest=True)

if __name__ == "__main__":
    payment_agent.run()
