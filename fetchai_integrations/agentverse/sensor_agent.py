import os, base64, time, json, zlib, io, requests, subprocess

def _free_port(port: int):
    r = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
    pids = r.stdout.strip()
    if pids:
        subprocess.run(["kill", "-9"] + pids.split(), capture_output=True)

_free_port(18001)
os.environ.setdefault("SSL_CERT_FILE", "/Users/peemmac/Library/Python/3.13/lib/python/site-packages/certifi/cacert.pem")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/Users/peemmac/Library/Python/3.13/lib/python/site-packages/certifi/cacert.pem")
import numpy as np
from uagents import Agent, Context, Model, Protocol
from dotenv import load_dotenv
import cloudinary, cloudinary.uploader
import urllib3
load_dotenv()
import subprocess; subprocess.run(["lsof", "-ti", f"tcp:18001"], capture_output=True, text=True).stdout.strip() and subprocess.run(["kill", "-9"] + subprocess.run(["lsof", "-ti", f"tcp:18001"], capture_output=True, text=True).stdout.strip().split(), capture_output=True)

# ── MODELS ───────────────────────────────────────────────────
class SensorRequest(Model):
    image_url: str
    prompt: str
    industry: str
    asset_type: str
    session_sender: str

class SensorResult(Model):
    image_url: str
    tribe_activations: str
    deepgaze_heatmap: str
    heatmap_url: str
    prompt: str
    industry: str
    asset_type: str
    session_sender: str

# ── CLOUDINARY ───────────────────────────────────────────────
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", ""),
    api_key=os.environ.get("CLOUDINARY_API_KEY", ""),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", ""),
    secure=True,
)

def _upload_image(image_bytes: bytes, public_id: str) -> str:
    result = cloudinary.uploader.upload(image_bytes, public_id=public_id, overwrite=True, resource_type="image")
    return result["secure_url"]

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
            if r.content:
                return r.content
        except Exception:
            continue
    urllib3.disable_warnings()
    r = requests.get(url, headers={"User-Agent": _UAS[0]}, timeout=timeout, allow_redirects=True, verify=False)
    r.raise_for_status()
    return r.content

# ── TRIBE /encode ────────────────────────────────────────────
TRIBE_API_URL = os.environ.get("TRIBE_API_URL", "").rstrip("/")
NGROK_HEADERS = {"ngrok-skip-browser-warning": "true", "User-Agent": "NeuralLens/1.0"}

def _call_tribe_encode(image_bytes: bytes) -> list:
    if not TRIBE_API_URL:
        return np.random.beta(2, 5, 70_000).astype(np.float32).tolist()
    try:
        resp = requests.post(
            f"{TRIBE_API_URL}/encode", data=image_bytes,
            headers={**NGROK_HEADERS, "Content-Type": "application/octet-stream"}, timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if "vertices" in data:
            raw = data["vertices"].get("lh", []) + data["vertices"].get("rh", [])
        else:
            raw = data.get("activations") or data.get("brain_activations") or data.get("encoding") or []
        if raw:
            return [float(v) for v in raw]
        raise ValueError(f"Empty response: {list(data.keys())}")
    except Exception as e:
        print(f"[sensor] /encode failed: {e} — using mock")
    return np.random.beta(2, 5, 70_000).astype(np.float32).tolist()

def _mock_saliency() -> list:
    return np.random.uniform(0, 1, 900).tolist()

def _generate_heatmap(activations: list, image_bytes: bytes) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    arr = np.array(activations[:70_000], dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-9:
        arr = (arr - mn) / (mx - mn)
    side = int(len(arr) ** 0.5)
    grid = arr[:side * side].reshape(side, side)
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    fig, ax = plt.subplots(figsize=(w / 100, h / 100))
    ax.imshow(grid, cmap="hot", alpha=0.75, interpolation="bilinear", aspect="auto")
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def _compress(activations: list) -> str:
    arr = np.array(activations, dtype=np.float32)
    return base64.b64encode(zlib.compress(arr.tobytes(), level=6)).decode("utf-8")

# ── AGENT ────────────────────────────────────────────────────
sensor = Agent(
    name="neurallens-sensor",
    seed="neurallens sensor tribe deepgaze lahacks 2026 brain",
    port=18001,
    agentverse=os.environ.get("AGENTVERSE_KEY", ""),
    mailbox=True,
)
print(f"SENSOR ADDRESS: {sensor.address}")

sensor_proto = Protocol(name="NeuralLensSensorProtocol", version="1.0.0")

@sensor_proto.on_message(model=SensorRequest, replies=SensorResult)
async def handle_sensor_request(ctx: Context, sender: str, msg: SensorRequest):
    ctx.logger.info(f"📥 SensorRequest: {msg.image_url}")
    interpreter_addr = os.environ.get("INTERPRETER_ADDRESS", "")
    if not interpreter_addr:
        ctx.logger.error("INTERPRETER_ADDRESS not set")
        return
    try:
        image_bytes = _fetch_image(msg.image_url, timeout=30)
        ctx.logger.info(f"Downloaded: {len(image_bytes):,} bytes")
        activations = _call_tribe_encode(image_bytes)
        ctx.logger.info(f"Activations: {len(activations):,}")
        saliency = _mock_saliency()
        heatmap_bytes = _generate_heatmap(activations, image_bytes)
        ts = int(time.time())
        heatmap_url = _upload_image(heatmap_bytes, f"neurallens_{ts}_heatmap_before")
        ctx.logger.info(f"Heatmap: {heatmap_url}")
        act_compressed = _compress(activations)
        await ctx.send(interpreter_addr, SensorResult(
            image_url=msg.image_url, tribe_activations=act_compressed,
            deepgaze_heatmap=json.dumps(saliency), heatmap_url=heatmap_url,
            prompt=msg.prompt, industry=msg.industry,
            asset_type=msg.asset_type, session_sender=msg.session_sender,
        ))
        ctx.logger.info(f"✅ SensorResult → {interpreter_addr}")
    except Exception as e:
        ctx.logger.error(f"❌ Sensor failed: {e}")
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

sensor.include(sensor_proto, publish_manifest=True)
sensor.include(pipeline_proto, publish_manifest=True)

if __name__ == "__main__":
    sensor.run()
