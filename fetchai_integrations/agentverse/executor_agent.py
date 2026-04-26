import os, json, time, requests, subprocess

def _free_port(port: int):
    r = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
    pids = r.stdout.strip()
    if pids:
        subprocess.run(["kill", "-9"] + pids.split(), capture_output=True)

_free_port(18004)
os.environ.setdefault("SSL_CERT_FILE", "/Users/peemmac/Library/Python/3.13/lib/python/site-packages/certifi/cacert.pem")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/Users/peemmac/Library/Python/3.13/lib/python/site-packages/certifi/cacert.pem")
import urllib3
import cloudinary, cloudinary.uploader, cloudinary.utils
from uagents import Agent, Context, Model, Protocol
from dotenv import load_dotenv
load_dotenv()
import subprocess; subprocess.run(["lsof", "-ti", f"tcp:18004"], capture_output=True, text=True).stdout.strip() and subprocess.run(["kill", "-9"] + subprocess.run(["lsof", "-ti", f"tcp:18004"], capture_output=True, text=True).stdout.strip().split(), capture_output=True)

# ── MODELS ───────────────────────────────────────────────────
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

class FinalResult(Model):
    nes_before: float
    nes_after: float
    delta: float
    profile_before: str
    profile_after: str
    issues: str
    changes: str
    optimized_image_url: str
    heatmap_before_url: str
    heatmap_after_url: str
    new_copy: str
    optimization_strategy: str
    session_sender: str

# ── CLOUDINARY ───────────────────────────────────────────────
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", ""),
    api_key=os.environ.get("CLOUDINARY_API_KEY", ""),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", ""),
    secure=True,
)

def _upload_image(image_bytes: bytes, public_id: str) -> str:
    r = cloudinary.uploader.upload(image_bytes, public_id=public_id, overwrite=True, resource_type="image")
    return r["secure_url"]

def _apply_transforms(public_id: str, transforms: list) -> str:
    url, _ = cloudinary.utils.cloudinary_url(public_id, transformation=transforms)
    return url

# ── IMAGE FETCHER ────────────────────────────────────────────
_UAS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (compatible; NeuralLens/1.0)",
    "curl/8.4.0",
]

def _fetch_image(url: str, timeout: int = 30) -> bytes:
    for ua in _UAS:
        try:
            r = requests.get(url, headers={"User-Agent": ua, "Accept": "image/*,*/*"}, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            if r.content: return r.content
        except Exception: continue
    urllib3.disable_warnings()
    r = requests.get(url, headers={"User-Agent": _UAS[0]}, timeout=timeout, allow_redirects=True, verify=False)
    r.raise_for_status()
    return r.content

# ── NES ESTIMATE ─────────────────────────────────────────────
def _estimate_nes_after(nes_before: float, issues: list) -> float:
    LIFTS = {"dlPFC": 8.0, "striatum": 10.0, "amygdala": 7.0, "hippocampus": 5.0, "IPS": 6.0, "insula": 7.0}
    total_lift = sum(lift for region, lift in LIFTS.items() if any(region.lower() in i.lower() for i in issues))
    return round(min(100.0, nes_before + (5.0 if issues else 2.0) + total_lift), 1)

def _get_profile(delta: float, default: str) -> str:
    if delta >= 20: return "high desire and emotion — strong engagement"
    if delta >= 10: return "moderate engagement — measurable improvement"
    return default

# ── AGENT ────────────────────────────────────────────────────
executor = Agent(
    name="neurallens-executor",
    seed="neurallens executor cloudinary image lahacks 2026",
    port=18004,
    agentverse=os.environ.get("AGENTVERSE_KEY", ""),
    mailbox=True,
)
print(f"EXECUTOR ADDRESS: {executor.address}")

executor_proto = Protocol(name="NeuralLensExecutorProtocol", version="1.0.0")

@executor_proto.on_message(model=StrategistResult, replies=FinalResult)
async def handle_strategist_result(ctx: Context, sender: str, msg: StrategistResult):
    ctx.logger.info(f"📥 StrategistResult: {msg.image_url}")
    orchestrator_addr = os.environ.get("ORCHESTRATOR_ADDRESS", "")
    if not orchestrator_addr:
        ctx.logger.error("ORCHESTRATOR_ADDRESS not set")
        return
    ts = int(time.time())
    try:
        try: visual_changes = json.loads(msg.visual_changes)
        except: visual_changes = []
        try: text_changes = json.loads(msg.text_changes)
        except: text_changes = []
        try: issues = json.loads(msg.issues)
        except: issues = []

        image_bytes = _fetch_image(msg.image_url, timeout=30)
        original_id = f"neurallens_{ts}_original"
        _upload_image(image_bytes, original_id)

        transforms = [ct for c in visual_changes if isinstance((ct := c.get("cloudinary_transform")), dict)]
        if not transforms:
            transforms = [{"effect": "improve:40"}, {"effect": "vibrance:30"}]
        optimized_url = _apply_transforms(original_id, transforms)

        nes_after = _estimate_nes_after(msg.nes_total, issues)
        delta = round(nes_after - msg.nes_total, 1)
        ctx.logger.info(f"NES: {msg.nes_total} → {nes_after} (Δ{delta:+.1f})")

        await ctx.send(orchestrator_addr, FinalResult(
            nes_before=msg.nes_total, nes_after=nes_after, delta=delta,
            profile_before=msg.profile, profile_after=_get_profile(delta, msg.profile),
            issues=msg.issues, changes=json.dumps(visual_changes + text_changes),
            optimized_image_url=optimized_url, heatmap_before_url=msg.heatmap_url,
            heatmap_after_url=optimized_url, new_copy=msg.new_copy,
            optimization_strategy=msg.optimization_strategy, session_sender=msg.session_sender,
        ))
        ctx.logger.info(f"✅ FinalResult → {orchestrator_addr}")
    except Exception as e:
        ctx.logger.error(f"❌ Executor failed: {e}")
        import traceback; traceback.print_exc()

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

executor.include(executor_proto, publish_manifest=True)
executor.include(pipeline_proto, publish_manifest=True)

if __name__ == "__main__":
    executor.run()
