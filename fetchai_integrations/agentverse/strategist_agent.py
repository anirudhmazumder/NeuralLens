import os, json, re, subprocess

def _free_port(port: int):
    r = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
    pids = r.stdout.strip()
    if pids:
        subprocess.run(["kill", "-9"] + pids.split(), capture_output=True)

_free_port(18003)
os.environ.setdefault("SSL_CERT_FILE", "/Users/peemmac/Library/Python/3.13/lib/python/site-packages/certifi/cacert.pem")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/Users/peemmac/Library/Python/3.13/lib/python/site-packages/certifi/cacert.pem")
from openai import OpenAI
from uagents import Agent, Context, Model, Protocol
from dotenv import load_dotenv
load_dotenv()
import subprocess; subprocess.run(["lsof", "-ti", f"tcp:18003"], capture_output=True, text=True).stdout.strip() and subprocess.run(["kill", "-9"] + subprocess.run(["lsof", "-ti", f"tcp:18003"], capture_output=True, text=True).stdout.strip().split(), capture_output=True)

# ── MODELS ───────────────────────────────────────────────────
class InterpreterResult(Model):
    image_url: str
    nes_total: float
    roi_values: str
    profile: str
    issues: str
    insights: str
    valence: float
    arousal: float
    heatmap_url: str
    prompt: str
    industry: str
    asset_type: str
    session_sender: str

class StrategistResult(Model):
    image_url: str
    visual_changes: str
    text_changes: str
    new_copy: str
    optimization_strategy: str
    nes_total: float
    profile: str
    issues: str
    heatmap_url: str
    session_sender: str

# ── ASI:ONE ──────────────────────────────────────────────────
_asi = OpenAI(base_url="https://api.asi1.ai/v1", api_key=os.environ.get("ASI1_API_KEY", ""))

_SYSTEM = """You are a neuromarketing optimization expert. Return ONLY valid JSON, no markdown, no backticks.
{
  "visual_changes": [{"zone":"string","change_type":"color|contrast|size|position|crop|filter","description":"string","cloudinary_transform":{"effect":"string"},"target_region":"string","reason":"string"}],
  "text_changes": [{"element":"headline|cta|body|tagline","old_text":"string","new_text":"string","target_region":"string","reason":"string"}],
  "new_copy": "string",
  "optimization_strategy": "string"
}
RULES: visual_changes EXACTLY 3 items, text_changes EXACTLY 2 items, new_copy 3-5 sentences with emojis and hashtags."""

def _build_prompt(msg: InterpreterResult) -> str:
    try: roi = json.loads(msg.roi_values)
    except: roi = {}
    try: issues = json.loads(msg.issues)
    except: issues = []
    try: insights = json.loads(msg.insights)
    except: insights = []
    issues_text = "\n".join(f"- {i}" for i in issues) or "(none)"
    zones_text = "\n".join(f"  {z['zone']} [{z['type']}]: {z['action']}" for z in insights if z.get("type") in ("attention_trap","hidden_value","dead_zone")) or "  (no critical zones)"
    return (f"Asset: {msg.asset_type.replace('_',' ')}\nIndustry: {msg.industry}\nGoal: {msg.prompt}\n\n"
            f"NES Total: {msg.nes_total}/100\nProfile: {msg.profile}\nValence: {msg.valence}  Arousal: {msg.arousal}\n\n"
            f"Issues:\n{issues_text}\n\nROI:\n{json.dumps(roi,indent=2)}\n\nCritical zones:\n{zones_text}")

def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^```\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw).strip()
    try: return json.loads(raw)
    except: pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    raise ValueError(f"Cannot parse JSON: {raw[:200]}")

def _call_asi(msg: InterpreterResult) -> dict:
    resp = _asi.chat.completions.create(
        model="asi1",
        messages=[{"role":"system","content":_SYSTEM},{"role":"user","content":_build_prompt(msg)}],
        max_tokens=2048,
    )
    return _parse_json(resp.choices[0].message.content.strip())

def _fallback(msg: InterpreterResult) -> dict:
    try: issues = json.loads(msg.issues); top = next((r for i in issues for r in ("dlPFC","striatum","amygdala","IPS","insula") if r.lower() in i.lower()), "striatum")
    except: top = "striatum"
    return {
        "visual_changes": [
            {"zone":"center","change_type":"contrast","description":"Boost contrast 25%","cloudinary_transform":{"effect":"improve:50"},"target_region":"IPS","reason":"Higher contrast boosts IPS activation"},
            {"zone":"top-center","change_type":"filter","description":"Warm color temperature","cloudinary_transform":{"effect":"vibrance:40"},"target_region":"amygdala","reason":"Warm colors elevate amygdala arousal"},
            {"zone":"bottom-center","change_type":"brightness","description":"Auto-enhance quality","cloudinary_transform":{"effect":"auto_brightness"},"target_region":top,"reason":"Quality enhancement reduces cognitive friction"},
        ],
        "text_changes": [
            {"element":"headline","old_text":"(current)","new_text":"Made for people who want more — not less.","target_region":"mPFC","reason":"Self-relevant framing activates medial PFC"},
            {"element":"cta","old_text":"(current)","new_text":"Get started free →","target_region":"striatum","reason":"Low-friction CTA maximises striatum desire"},
        ],
        "new_copy": f"✨ What if your results could double?\n\nWe help businesses like yours grow faster without guesswork.\n\n🔗 Try free today — no credit card needed.\n\n#{msg.industry.replace(' ','')} #Growth #Marketing",
        "optimization_strategy": "Strengthen contrast and warm colors to boost amygdala and IPS, simplify copy to cut dlPFC load.",
    }

# ── AGENT ────────────────────────────────────────────────────
strategist = Agent(
    name="neurallens-strategist",
    seed="neurallens strategist gemma optimizer lahacks 2026",
    port=18003,
    agentverse=os.environ.get("AGENTVERSE_KEY", ""),
    mailbox=True,
)
print(f"STRATEGIST ADDRESS: {strategist.address}")

strategist_proto = Protocol(name="NeuralLensStrategistProtocol", version="1.0.0")

@strategist_proto.on_message(model=InterpreterResult, replies=StrategistResult)
async def handle_interpreter_result(ctx: Context, sender: str, msg: InterpreterResult):
    ctx.logger.info(f"📥 InterpreterResult: NES={msg.nes_total}")
    executor_addr = os.environ.get("EXECUTOR_ADDRESS", "")
    if not executor_addr:
        ctx.logger.error("EXECUTOR_ADDRESS not set")
        return
    try:
        strategy = _call_asi(msg)
        ctx.logger.info("✅ ASI:One strategy parsed")
    except Exception as e:
        ctx.logger.error(f"ASI:One failed ({e}) — using fallback")
        strategy = _fallback(msg)
    try:
        await ctx.send(executor_addr, StrategistResult(
            image_url=msg.image_url, visual_changes=json.dumps(strategy.get("visual_changes",[])),
            text_changes=json.dumps(strategy.get("text_changes",[])), new_copy=strategy.get("new_copy",""),
            optimization_strategy=strategy.get("optimization_strategy",""), nes_total=msg.nes_total,
            profile=msg.profile, issues=msg.issues, heatmap_url=msg.heatmap_url,
            session_sender=msg.session_sender,
        ))
        ctx.logger.info(f"✅ StrategistResult → {executor_addr}")
    except Exception as e:
        ctx.logger.error(f"❌ Forward failed: {e}")

class PipelineNotification(Model):
    step: str
    data: str

class PipelineAck(Model):
    step: str
    status: str

pipeline_proto = Protocol(name="NeuralLensPipelineProtocol", version="1.0.0")

@pipeline_proto.on_message(model=PipelineNotification, replies=PipelineAck)
async def handle_notification(ctx: Context, sender: str, msg: PipelineNotification):
    ctx.logger.info(f"📡 [{msg.step}] {msg.data[:80]}")
    await ctx.send(sender, PipelineAck(step=msg.step, status="received"))

strategist.include(strategist_proto, publish_manifest=True)
strategist.include(pipeline_proto, publish_manifest=True)

if __name__ == "__main__":
    strategist.run()
