"""
NeuralLens Sensor Agent.

Receives SensorRequest from Orchestrator.
Downloads the image (with browser-like headers to avoid 403s).
Calls TRIBE v2 /encode on the Colab ngrok server.
Falls back to beta(2,5) mock activations if /encode is unavailable.
Always generates heatmap locally with matplotlib (no remote heatmap endpoint).
Always uses mock uniform saliency for DeepGaze (no remote endpoint).
Compresses the 70k-float activation array before sending
  (raw = ~280KB, compressed = ~65KB — stays within uAgents limits).
Uploads the heatmap to Cloudinary.
Forwards SensorResult to Interpreter Agent.

Run standalone:   python agents/sensor_agent.py
Run with Bureau:  python run_all_agents.py  (recommended)
"""

import os
import base64
import time
import json
import zlib
import requests
import numpy as np

from uagents import Agent, Context
from dotenv import load_dotenv

from agents.models import SensorRequest, SensorResult
from integrations.cloudinary_client import upload_image
from integrations.image_fetcher import fetch_image

load_dotenv()

TRIBE_API_URL = os.environ.get("TRIBE_API_URL", "").rstrip("/")

# Headers to skip ngrok browser warning page
NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "NeuralLens/1.0",
}

# ── AGENT SETUP ─────────────────────────────────────────────
sensor = Agent(
    name="neurallens-sensor",
    seed="neurallens sensor tribe deepgaze lahacks 2026 brain",
    port=8001,
    mailbox=True,
)

print("\n" + "=" * 55)
print(f"SENSOR ADDRESS: {sensor.address}")
print("=" * 55 + "\n")


# ── IMAGE HELPERS ────────────────────────────────────────────

def _download_image(url: str) -> bytes:
    """Download image bytes, retrying with multiple UA strings. See integrations/image_fetcher.py."""
    return fetch_image(url, timeout=30)


def _to_base64(image_bytes: bytes) -> str:
    """Encode bytes as base64 string for JSON API payloads."""
    return base64.b64encode(image_bytes).decode("utf-8")


# ── TRIBE /encode ────────────────────────────────────────────

def _call_tribe_encode(image_bytes: bytes) -> list:
    """
    POST /encode to the Colab TRIBE v2 endpoint.
    Sends raw image bytes (what the server expects).
    Returns flat list of 20484 floats (lh + rh vertices).
    Falls back to beta(2,5) mock on any failure.
    """
    if not TRIBE_API_URL:
        print("[sensor] TRIBE_API_URL not set — using mock activations")
        return np.random.beta(2, 5, 70_000).astype(np.float32).tolist()

    try:
        resp = requests.post(
            f"{TRIBE_API_URL}/encode",
            data=image_bytes,
            headers={**NGROK_HEADERS, "Content-Type": "application/octet-stream"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        # Server returns {"vertices": {"lh": [...], "rh": [...]}, ...}
        if "vertices" in data:
            lh = data["vertices"].get("lh", [])
            rh = data["vertices"].get("rh", [])
            raw = lh + rh
        else:
            raw = (
                data.get("activations")
                or data.get("brain_activations")
                or data.get("encoding")
                or data.get("features")
                or []
            )
        if raw:
            return [float(v) for v in raw]
        raise ValueError(f"Empty activations in /encode response: {list(data.keys())}")
    except Exception as e:
        print(f"[sensor] TRIBE /encode failed: {e} — using mock activations")

    return np.random.beta(2, 5, 70_000).astype(np.float32).tolist()


# ── SALIENCY (always mock) ───────────────────────────────────

def _mock_deepgaze_saliency() -> list:
    """Return uniform-random mock saliency (900 values for a 30×30 grid)."""
    return np.random.uniform(0, 1, 900).tolist()


# ── LOCAL HEATMAP ────────────────────────────────────────────

def _generate_local_heatmap(activations: list, image_bytes: bytes) -> bytes:
    """
    Generate a matplotlib hot-colormap heatmap overlaid on the original image.
    Returns PNG bytes sized to the original image dimensions.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    arr = np.array(activations[:70_000], dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-9:
        arr = (arr - mn) / (mx - mn)

    side = int(len(arr) ** 0.5)
    grid = arr[: side * side].reshape(side, side)

    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size

    fig, ax = plt.subplots(figsize=(w / 100, h / 100))
    ax.imshow(grid, cmap="hot", alpha=0.75,
              interpolation="bilinear", aspect="auto")
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── COMPRESSION ──────────────────────────────────────────────

def _compress_activations(activations: list) -> str:
    """
    Compress 70k float32 activations for safe uAgents transport.
    Raw list JSON ≈ 280KB → zlib+base64 ≈ 65KB.

    Returns base64 string of zlib-compressed float32 bytes.
    """
    arr = np.array(activations, dtype=np.float32)
    compressed = zlib.compress(arr.tobytes(), level=6)
    return base64.b64encode(compressed).decode("utf-8")


# ── MESSAGE HANDLER ─────────────────────────────────────────

@sensor.on_message(model=SensorRequest)
async def handle_sensor_request(ctx: Context, sender: str, msg: SensorRequest):
    """
    Full sensor pipeline: download → TRIBE /encode → local heatmap →
    Cloudinary upload → compress → forward to Interpreter.
    """
    ctx.logger.info(f"📥 SensorRequest received: {msg.image_url}")

    interpreter_addr = os.environ.get("INTERPRETER_ADDRESS", "")
    if not interpreter_addr:
        ctx.logger.error("INTERPRETER_ADDRESS not set in .env")
        return

    try:
        # 1. Download image
        ctx.logger.info("Downloading image...")
        image_bytes = _download_image(msg.image_url)
        image_b64 = _to_base64(image_bytes)
        ctx.logger.info(f"Downloaded: {len(image_bytes):,} bytes")

        # 2. TRIBE v2 brain activations via /encode
        ctx.logger.info("Calling TRIBE v2 /encode...")
        activations = _call_tribe_encode(image_bytes)
        ctx.logger.info(f"Activations: {len(activations):,} values")

        # 3. DeepGaze saliency (mock — no live endpoint)
        saliency = _mock_deepgaze_saliency()
        ctx.logger.info(f"Saliency (mock): {len(saliency):,} values")

        # 4. Local heatmap (always — no remote heatmap endpoint)
        ctx.logger.info("Generating local heatmap...")
        heatmap_bytes = _generate_local_heatmap(activations, image_bytes)
        ctx.logger.info(f"Heatmap: {len(heatmap_bytes):,} bytes")

        # 5. Upload heatmap to Cloudinary
        ctx.logger.info("Uploading heatmap to Cloudinary...")
        ts = int(time.time())
        heatmap_url = upload_image(
            heatmap_bytes,
            public_id=f"neurallens_{ts}_heatmap_before",
        )
        ctx.logger.info(f"Heatmap URL: {heatmap_url}")

        # 6. Compress activations (critical — raw 280KB would stall uAgents)
        ctx.logger.info("Compressing activations...")
        act_compressed = _compress_activations(activations)
        gaze_str = json.dumps(saliency)
        ctx.logger.info(
            f"Compressed: {len(np.array(activations, np.float32).tobytes())//1024}KB "
            f"→ {len(act_compressed)//1024}KB"
        )

        # 7. Forward to Interpreter
        await ctx.send(
            interpreter_addr,
            SensorResult(
                image_url=msg.image_url,
                tribe_activations=act_compressed,
                deepgaze_heatmap=gaze_str,
                heatmap_url=heatmap_url,
                prompt=msg.prompt,
                industry=msg.industry,
                asset_type=msg.asset_type,
                session_sender=msg.session_sender,
            ),
        )
        ctx.logger.info(f"✅ SensorResult forwarded to {interpreter_addr}")

    except Exception as e:
        ctx.logger.error(f"❌ Sensor failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    sensor.run()
