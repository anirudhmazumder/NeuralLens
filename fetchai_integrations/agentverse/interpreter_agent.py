import os, json, zlib, base64, subprocess

def _free_port(port: int):
    r = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
    pids = r.stdout.strip()
    if pids:
        subprocess.run(["kill", "-9"] + pids.split(), capture_output=True)

_free_port(18002)
os.environ.setdefault("SSL_CERT_FILE", "/Users/peemmac/Library/Python/3.13/lib/python/site-packages/certifi/cacert.pem")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/Users/peemmac/Library/Python/3.13/lib/python/site-packages/certifi/cacert.pem")
import numpy as np
from typing import Dict, List, Tuple
from uagents import Agent, Context, Model, Protocol
from dotenv import load_dotenv
load_dotenv()
import subprocess; subprocess.run(["lsof", "-ti", f"tcp:18002"], capture_output=True, text=True).stdout.strip() and subprocess.run(["kill", "-9"] + subprocess.run(["lsof", "-ti", f"tcp:18002"], capture_output=True, text=True).stdout.strip().split(), capture_output=True)

# ── MODELS ───────────────────────────────────────────────────
class SensorResult(Model):
    image_url: str
    tribe_activations: str
    deepgaze_heatmap: str
    heatmap_url: str
    prompt: str
    industry: str
    asset_type: str
    session_sender: str

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

# ── NES MATH ─────────────────────────────────────────────────
ROI_INDICES: Dict[str, Tuple[int, int]] = {
    "amygdala":    (1200, 1450),
    "striatum":    (2100, 2380),
    "hippocampus": (3400, 3700),
    "dlPFC":       (8200, 8600),
    "IPS":         (5100, 5400),
    "mPFC":        (7800, 8100),
    "insula":      (4200, 4500),
}
NES_WEIGHTS: Dict[str, float] = {
    "amygdala": +0.20, "striatum": +0.25, "hippocampus": +0.15,
    "dlPFC": -0.20, "IPS": +0.10, "mPFC": +0.10, "insula": +0.00,
}

def extract_roi_values(activations: np.ndarray) -> Dict[str, float]:
    raw = {}
    for region, (s, e) in ROI_INDICES.items():
        if e <= len(activations):
            raw[region] = float(np.mean(activations[s:e]))
        else:
            avail = activations[s:] if s < len(activations) else np.array([0.0])
            raw[region] = float(np.mean(avail))
    vals = list(raw.values())
    mn, mx = min(vals), max(vals)
    r = mx - mn
    return {k: round(((v - mn) / r) * 100, 1) if r > 1e-8 else 50.0 for k, v in raw.items()}

def compute_nes(roi: Dict[str, float]) -> Dict:
    raw = sum(roi.get(r, 50.0) * w for r, w in NES_WEIGHTS.items())
    nes_total = round(max(0.0, min(100.0, raw)), 1)
    valence = round(max(-1.0, min(1.0, ((roi.get("striatum",50)-50)*0.4 + (roi.get("amygdala",50)-50)*0.3 + (roi.get("insula",50)-50)*-0.3) / 100)), 3)
    arousal = round(max(-1.0, min(1.0, ((roi.get("amygdala",50)-50)*0.4 + (roi.get("dlPFC",50)-50)*0.3 + (roi.get("IPS",50)-50)*0.3) / 100)), 3)
    profile = _get_profile(roi)
    issues = _get_issues(roi)
    return {"nes_total": nes_total, "valence": valence, "arousal": arousal, "profile": profile, "issues": issues, "roi_values": roi}

def _get_profile(roi: Dict) -> str:
    if roi.get("dlPFC", 50) > 65: return "high cognitive load — page is confusing"
    if roi.get("striatum", 50) < 35 and roi.get("amygdala", 50) < 35: return "low desire and emotion — flat and forgettable"
    if roi.get("insula", 50) > 65 and roi.get("dlPFC", 50) > 60: return "distrust and confusion — something feels off"
    if roi.get("striatum", 50) > 65 and roi.get("amygdala", 50) > 60: return "high desire and emotion — strong engagement"
    if roi.get("hippocampus", 50) < 35: return "low memory encoding — will be forgotten"
    return "moderate engagement — room for improvement"

def _get_issues(roi: Dict) -> List[str]:
    issues = []
    if roi.get("dlPFC", 50) > 65: issues.append(f"dlPFC {roi['dlPFC']}/100 — HIGH cognitive load, simplify the layout")
    if roi.get("striatum", 50) < 40: issues.append(f"striatum {roi['striatum']}/100 — LOW desire, strengthen value proposition")
    if roi.get("amygdala", 50) < 40: issues.append(f"amygdala {roi['amygdala']}/100 — LOW emotional pull, copy is flat")
    if roi.get("hippocampus", 50) < 40: issues.append(f"hippocampus {roi['hippocampus']}/100 — LOW memory encoding, add narrative")
    if roi.get("IPS", 50) < 35: issues.append(f"IPS {roi['IPS']}/100 — CTA not visually salient, reposition or recolor")
    if roi.get("insula", 50) > 65: issues.append(f"insula {roi['insula']}/100 — HIGH distrust signal, add social proof")
    return issues

def analyze_intersection(activations: np.ndarray, deepgaze: list) -> List[Dict]:
    zones = ["top-left","top-center","top-right","middle-left","center","middle-right","bottom-left","bottom-center","bottom-right"]
    t = np.array(activations[:min(900, len(activations))])
    g = np.array(deepgaze[:min(900, len(deepgaze))]) if deepgaze else np.zeros(900)
    if len(t) < 900: t = np.pad(t, (0, 900 - len(t)))
    if len(g) < 900: g = np.pad(g, (0, 900 - len(g)))
    def n100(a):
        mn, mx = a.min(), a.max()
        return ((a - mn) / (mx - mn)) * 100 if mx - mn > 1e-8 else np.full_like(a, 50.0)
    tn, gn = n100(t), n100(g)
    insights = []
    for i, zone in enumerate(zones):
        ts = float(np.mean(tn[i*100:(i+1)*100]))
        gs = float(np.mean(gn[i*100:(i+1)*100]))
        if gs > 60 and ts < 40:
            zt, m, a = "attention_trap", "Eyes land here but brain feels nothing.", "enrich emotionally or remove"
        elif ts > 60 and gs < 40:
            zt, m, a = "hidden_value", "Strong brain response but eyes never land.", "make this zone visually dominant"
        elif ts > 60 and gs > 60:
            zt, m, a = "power_zone", "Eyes go here AND brain responds strongly.", "protect and amplify"
        else:
            zt, m, a = "dead_zone", "Nobody looks here and brain feels nothing.", "remove or replace"
        insights.append({"zone": zone, "type": zt, "meaning": m, "action": a, "tribe_score": round(ts,1), "gaze_score": round(gs,1)})
    return insights

# ── AGENT ────────────────────────────────────────────────────
interpreter = Agent(
    name="neurallens-interpreter",
    seed="neurallens interpreter nes roi scorer lahacks 2026",
    port=18002,
    agentverse=os.environ.get("AGENTVERSE_KEY", ""),
    mailbox=True,
)
print(f"INTERPRETER ADDRESS: {interpreter.address}")

interpreter_proto = Protocol(name="NeuralLensInterpreterProtocol", version="1.0.0")

@interpreter_proto.on_message(model=SensorResult, replies=InterpreterResult)
async def handle_sensor_result(ctx: Context, sender: str, msg: SensorResult):
    ctx.logger.info("📥 SensorResult received")
    strategist_addr = os.environ.get("STRATEGIST_ADDRESS", "")
    if not strategist_addr:
        ctx.logger.error("STRATEGIST_ADDRESS not set")
        return
    try:
        raw_bytes = zlib.decompress(base64.b64decode(msg.tribe_activations))
        activations = np.frombuffer(raw_bytes, dtype=np.float32)
        ctx.logger.info(f"Activations: {len(activations):,}")
        try: deepgaze = json.loads(msg.deepgaze_heatmap)
        except: deepgaze = []
        roi = extract_roi_values(activations)
        nes = compute_nes(roi)
        insights = analyze_intersection(activations, deepgaze)
        ctx.logger.info(f"NES: {nes['nes_total']} | {nes['profile']}")
        await ctx.send(strategist_addr, InterpreterResult(
            image_url=msg.image_url, nes_total=nes["nes_total"],
            roi_values=json.dumps(nes["roi_values"]), profile=nes["profile"],
            issues=json.dumps(nes["issues"]), insights=json.dumps(insights),
            valence=nes["valence"], arousal=nes["arousal"],
            heatmap_url=msg.heatmap_url, prompt=msg.prompt,
            industry=msg.industry, asset_type=msg.asset_type,
            session_sender=msg.session_sender,
        ))
        ctx.logger.info(f"✅ InterpreterResult → {strategist_addr}")
    except Exception as e:
        ctx.logger.error(f"❌ Interpreter failed: {e}")
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

interpreter.include(interpreter_proto, publish_manifest=True)
interpreter.include(pipeline_proto, publish_manifest=True)

if __name__ == "__main__":
    interpreter.run()
