"""
tests/test_tribe.py

Standalone test for pipeline/tribe_inference.py.
Run with:  python tests/test_tribe.py

If USE_MOCK_TRIBE is True the test uses mock inference (expected during
development before TRIBE v2 weights are downloaded).
"""

# stdlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# third-party
from dotenv import load_dotenv

load_dotenv()

# local
import pipeline.tribe_inference as tribe_module
from pipeline.tribe_inference import (
    USE_MOCK_TRIBE,
    run_tribe_inference,
    activations_to_heatmap,
    mock_tribe_inference,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs")
SCREENSHOT_PATH = os.path.join(OUTPUT_DIR, "screenshot.png")


def _load_or_create_screenshot() -> bytes:
    if os.path.exists(SCREENSHOT_PATH):
        with open(SCREENSHOT_PATH, "rb") as f:
            return f.read()
    # Fallback: tiny 1×1 white PNG
    import struct, zlib
    def make_png():
        sig = b"\x89PNG\r\n\x1a\n"
        def chunk(tag, data):
            crc = zlib.crc32(tag + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        idat = chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        iend = chunk(b"IEND", b"")
        return sig + ihdr + idat + iend
    return make_png()


def test_tribe_inference():
    print(f"\nUSE_MOCK_TRIBE = {USE_MOCK_TRIBE}")
    screenshot = _load_or_create_screenshot()
    text = "Example domain — This domain is for use in illustrative examples."

    print("Testing run_tribe_inference ...")
    try:
        activations = run_tribe_inference(screenshot, text)
        print(f"  activation shape : {activations.shape}")
        print(f"  activation range : [{activations.min():.4f}, {activations.max():.4f}]")
        print(f"  activation mean  : {activations.mean():.4f}")
        assert activations.ndim == 1, "activations must be 1-D"
        assert len(activations) > 0, "activations must not be empty"
        print("PASS: run_tribe_inference")
    except Exception as exc:
        print(f"FAIL: run_tribe_inference — {exc}")
        raise

    print("\nTesting activations_to_heatmap ...")
    try:
        heatmap_bytes = activations_to_heatmap(activations, screenshot)
        assert isinstance(heatmap_bytes, bytes), "heatmap must be bytes"
        assert heatmap_bytes[:4] == b"\x89PNG", "heatmap must be PNG"
        print(f"  heatmap size: {len(heatmap_bytes)} bytes")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "heatmap.png")
        with open(out_path, "wb") as f:
            f.write(heatmap_bytes)
        print(f"  saved to: {out_path}")
        print("PASS: activations_to_heatmap")
    except Exception as exc:
        print(f"FAIL: activations_to_heatmap — {exc}")
        raise


if __name__ == "__main__":
    test_tribe_inference()
