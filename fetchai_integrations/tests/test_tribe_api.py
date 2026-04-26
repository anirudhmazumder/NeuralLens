"""
tests/test_tribe_api.py

Tests the TRIBE v2 Colab endpoint via ngrok.
Run with: python tests/test_tribe_api.py

Requires TRIBE_API_URL in .env.
Only /encode is a known working endpoint.
"""

import os
import sys
import base64
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv

load_dotenv()

TRIBE_API_URL = os.environ.get("TRIBE_API_URL", "").rstrip("/")
TEST_IMAGE_URL = "https://picsum.photos/seed/tribe_test/400/300"

DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}
HEADERS = {"ngrok-skip-browser-warning": "true", "User-Agent": "NeuralLens/1.0"}


def _download_test_image() -> str:
    """Download test image and return as base64 string."""
    print(f"  Downloading test image from {TEST_IMAGE_URL}...")
    resp = requests.get(TEST_IMAGE_URL, headers=DOWNLOAD_HEADERS, timeout=15)
    resp.raise_for_status()
    b64 = base64.b64encode(resp.content).decode("utf-8")
    print(f"  Image: {len(resp.content):,} bytes → {len(b64)} chars base64")
    return b64


def test_encode():
    """Test POST /encode endpoint — the only active TRIBE v2 endpoint."""
    print(f"\n[1/1] Testing POST {TRIBE_API_URL}/encode ...")
    try:
        image_b64 = _download_test_image()
        resp = requests.post(
            f"{TRIBE_API_URL}/encode",
            json={"image": image_b64},
            headers=HEADERS,
            timeout=120,
        )
        print(f"  HTTP status: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        print(f"  Response keys: {list(data.keys())}")
        activations = (
            data.get("activations")
            or data.get("brain_activations")
            or data.get("encoding")
            or data.get("features")
            or []
        )
        print(f"  Activations: {len(activations):,} values")
        if activations:
            import numpy as np
            arr = np.array(activations[:1000])
            print(f"  Range: [{arr.min():.4f}, {arr.max():.4f}]")
        print("  PASS: /encode")
        return True
    except Exception as e:
        print(f"  FAIL: /encode — {e}")
        return False


def main():
    print("\n" + "=" * 55)
    print(f"Testing TRIBE v2 API at: {TRIBE_API_URL}")
    print("=" * 55)

    if not TRIBE_API_URL:
        print("ERROR: TRIBE_API_URL not set in .env")
        sys.exit(1)

    passed = test_encode()

    print(f"\n{'='*55}")
    if passed:
        print("✅ /encode OK — TRIBE v2 is ready")
    else:
        print("⚠️  /encode failed — sensor will use mock activations (beta(2,5))")
        print("   This is expected if Colab is not running or is overloaded.")
    print("=" * 55)


if __name__ == "__main__":
    main()
