"""
tests/debug_pipeline.py

Step-by-step pipeline debugger — runs each stage locally (no agents),
prints exactly where it breaks and how long each step takes.

Run with: python tests/debug_pipeline.py <image_url>

Example:
  python tests/debug_pipeline.py https://picsum.photos/seed/neurallens/800/600
"""

import os
import sys
import time
import json
import base64
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests
import numpy as np
from integrations.image_fetcher import fetch_image as _download_image_robust

# Reliable fallback — picsum.photos serves clean JPEGs with no auth
DEFAULT_IMAGE_URL = "https://picsum.photos/seed/neurallens/800/600"
TEST_IMAGE_URL = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_URL

TRIBE_API_URL = os.environ.get("TRIBE_API_URL", "").rstrip("/")
HEADERS = {"ngrok-skip-browser-warning": "true", "User-Agent": "NeuralLens/1.0"}


def step(n: int, label: str):
    print(f"\n{'─'*55}")
    print(f"STEP {n}: {label}")
    print(f"{'─'*55}")


def ok(msg: str):
    print(f"  ✅ {msg}")


def fail(msg: str):
    print(f"  ❌ {msg}")


def timing(start: float) -> str:
    return f"{time.time() - start:.2f}s"


# ── STEP 1: Download image ───────────────────────────────────

step(1, f"Download image\n  URL: {TEST_IMAGE_URL}")
t = time.time()
try:
    image_bytes = _download_image_robust(TEST_IMAGE_URL, timeout=20)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    ok(f"{len(image_bytes):,} bytes downloaded ({timing(t)})")
    ok(f"Base64 length: {len(image_b64):,} chars")
    ok(f"Magic bytes: {image_bytes[:4].hex()} "
       f"({'JPEG' if image_bytes[:3]==b'\\xff\\xd8\\xff' else 'PNG' if image_bytes[:4]==b'\\x89PNG' else 'other'})")
except Exception as e:
    fail(f"Download failed: {e}")
    print(f"\n  Tip: use a direct image URL like:")
    print(f"    https://picsum.photos/seed/test/800/600")
    sys.exit(1)


# ── STEP 2: TRIBE /encode ────────────────────────────────────

step(2, f"TRIBE v2 /encode  →  {TRIBE_API_URL}/encode")
t = time.time()
activations = None
try:
    r = requests.post(
        f"{TRIBE_API_URL}/encode",
        data=image_bytes,
        headers={**HEADERS, "Content-Type": "application/octet-stream"},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    # Server returns {"vertices": {"lh": [...], "rh": [...]}, ...}
    if "vertices" in data:
        raw = data["vertices"].get("lh", []) + data["vertices"].get("rh", [])
    else:
        raw = (
            data.get("activations")
            or data.get("brain_activations")
            or data.get("encoding")
            or data.get("features")
            or []
        )
    if not raw:
        raise ValueError(f"Empty activations in response: {list(data.keys())}")
    activations = [float(v) for v in raw]
    ok(f"{len(activations):,} activations returned ({timing(t)})")
    arr = np.array(activations)
    ok(f"Range: [{arr.min():.4f}, {arr.max():.4f}]  mean={arr.mean():.4f}")
except Exception as e:
    fail(f"/encode failed: {e}")
    print("  → Falling back to mock activations (beta(2,5))")
    activations = np.random.beta(2, 5, 70_000).astype(np.float32).tolist()
    ok(f"Mock: {len(activations):,} values")


# ── STEP 3: Local heatmap ────────────────────────────────────

step(3, "Local heatmap (matplotlib hot-colormap)")
t = time.time()
heatmap_bytes = None
try:
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
    grid = arr[:side * side].reshape(side, side)

    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size

    fig, ax = plt.subplots(figsize=(w / 100, h / 100))
    ax.imshow(grid, cmap="hot", alpha=0.75,
              interpolation="bilinear", aspect="auto")
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    heatmap_bytes = buf.read()
    ok(f"Heatmap PNG: {len(heatmap_bytes):,} bytes ({timing(t)})")
except Exception as e:
    fail(f"Local heatmap failed: {e}")
    traceback.print_exc()
    sys.exit(1)


# ── STEP 4: Mock DeepGaze saliency ──────────────────────────

step(4, "DeepGaze saliency (mock — no live endpoint)")
saliency = np.random.uniform(0, 1, 900).tolist()
ok(f"Mock: {len(saliency):,} values (uniform 0–1)")


# ── STEP 5: Cloudinary upload ────────────────────────────────

step(5, "Cloudinary upload")
t = time.time()
heatmap_url = None
try:
    from integrations.cloudinary_client import upload_image
    ts = int(time.time())
    heatmap_url = upload_image(heatmap_bytes, f"neurallens_debug_{ts}_heatmap")
    ok(f"Uploaded: {heatmap_url} ({timing(t)})")
except Exception as e:
    fail(f"Cloudinary upload failed: {e}")
    heatmap_url = "https://placeholder.cloudinary.com/heatmap.png"
    print(f"  → Using placeholder URL: {heatmap_url}")


# ── STEP 6: NES math ─────────────────────────────────────────

step(6, "NES math (extract_roi + compute_nes + analyze_intersection)")
t = time.time()
try:
    from pipeline.nes_math import extract_roi_values, compute_nes, analyze_intersection
    arr = np.array(activations, dtype=np.float32)
    roi = extract_roi_values(arr)
    nes = compute_nes(roi)
    insights = analyze_intersection(arr, saliency)
    ok(f"NES total: {nes['nes_total']} ({timing(t)})")
    ok(f"Profile: {nes['profile']}")
    ok(f"Issues: {len(nes['issues'])}")
    ok(f"Zones: {len(insights)}")
    print(f"\n  ROI values:")
    for region, val in roi.items():
        bar = "█" * int(val / 10)
        print(f"    {region:12s} {val:5.1f}  {bar}")
    if nes["issues"]:
        print(f"\n  Issues:")
        for issue in nes["issues"]:
            print(f"    ⚠️  {issue}")
    zone_types = {}
    for z in insights:
        zone_types[z["type"]] = zone_types.get(z["type"], 0) + 1
    print(f"\n  Zone breakdown: {zone_types}")
except Exception as e:
    fail(f"NES math failed: {e}")
    traceback.print_exc()
    sys.exit(1)


# ── STEP 7: Payload size check ───────────────────────────────

step(7, "SensorResult payload size check (as actually sent by sensor_agent)")
import zlib as _zlib

# Show raw list size (what the old broken code sent)
raw_payload_kb = len(json.dumps(activations).encode()) / 1024

# Show compressed size (what sensor_agent.py actually sends)
act_arr = np.array(activations, dtype=np.float32)
compressed_b64 = base64.b64encode(_zlib.compress(act_arr.tobytes(), level=6)).decode()
gaze_str = json.dumps(saliency)

compressed_result = {
    "image_url": TEST_IMAGE_URL,
    "tribe_activations": compressed_b64,   # ← this is the actual field sent
    "deepgaze_heatmap": gaze_str,
    "heatmap_url": heatmap_url or "",
    "prompt": "test prompt",
    "industry": "general",
    "asset_type": "image",
    "session_sender": "agent1qtest",
}
payload_kb = len(json.dumps(compressed_result).encode()) / 1024

print(f"  Raw activation list    : {raw_payload_kb:.0f} KB  (NOT what agent sends)")
print(f"  Compressed + base64    : {payload_kb:.1f} KB  (what sensor_agent.py sends)")

if payload_kb < 200:
    ok(f"Payload safe ({payload_kb:.1f} KB) — compression is working ✓")
elif payload_kb < 500:
    print(f"  ⚠️  Payload {payload_kb:.0f} KB — large but within uAgents limits")
else:
    fail(f"Compressed payload still too large ({payload_kb:.0f} KB)")


# ── STEP 8: ASI:One strategy call ───────────────────────────

step(8, "ASI:One strategy call (used by strategist_agent)")
t = time.time()
try:
    from openai import OpenAI
    import re as _re
    asi = OpenAI(
        base_url="https://api.asi1.ai/v1",
        api_key=os.environ.get("ASI1_API_KEY", ""),
    )
    resp = asi.chat.completions.create(
        model="asi1",
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": 'Return {"ok": true}'},
        ],
        max_tokens=32,
    )
    raw = resp.choices[0].message.content.strip()
    ok(f"ASI:One responded in {timing(t)}: {raw[:80]}")
except Exception as e:
    fail(f"ASI:One failed: {e}")
    print("  → Check ASI1_API_KEY in .env")


# ── SUMMARY ──────────────────────────────────────────────────

print(f"\n{'='*55}")
print("PIPELINE DEBUG SUMMARY")
print(f"{'='*55}")
print(f"  Image URL   : {TEST_IMAGE_URL}")
print(f"  Activations : {len(activations):,} values")
print(f"  Saliency    : {len(saliency):,} values")
print(f"  Heatmap     : {len(heatmap_bytes):,} bytes")
print(f"  Heatmap URL : {heatmap_url}")
print(f"  NES         : {nes['nes_total']}")
print(f"  Payload (compressed): {payload_kb:.1f} KB")
print(f"{'='*55}\n")
