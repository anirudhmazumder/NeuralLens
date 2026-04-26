"""
tests/test_cloudinary.py

Tests Cloudinary upload and transformation functions.
Run with: python tests/test_cloudinary.py

Requires CLOUDINARY_* vars in .env.
Downloads a small test image and uploads it twice to test
upload + overlay transformation.
"""

import os
import sys
import time
import struct
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from integrations.cloudinary_client import (
    upload_image,
    create_heatmap_overlay,
    apply_visual_transformations,
)


def _make_test_png(color: tuple = (255, 100, 50)) -> bytes:
    """Create a minimal valid 100x100 single-color PNG."""
    width, height = 100, 100
    r, g, b = color

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = chunk(b"IHDR", ihdr_data)

    raw_row = b"\x00" + bytes([r, g, b]) * width
    raw_data = raw_row * height
    idat = chunk(b"IDAT", zlib.compress(raw_data))
    iend = chunk(b"IEND", b"")

    return sig + ihdr + idat + iend


def test_upload():
    print("\n[1/3] Testing upload_image...")
    ts = int(time.time())
    png = _make_test_png((255, 100, 50))  # orange

    public_id = f"neurallens_test_{ts}_base"
    url = upload_image(png, public_id)
    print(f"  public_id : {public_id}")
    print(f"  URL       : {url}")
    assert url.startswith("https://"), "URL must be HTTPS"
    assert "cloudinary.com" in url, "URL must be from Cloudinary"
    print("  PASS: upload_image")
    return public_id, url


def test_overlay(base_id: str):
    print("\n[2/3] Testing create_heatmap_overlay...")
    ts = int(time.time())
    heatmap_png = _make_test_png((200, 0, 0))  # red heatmap

    overlay_id = f"neurallens_test_{ts}_heatmap"
    upload_image(heatmap_png, overlay_id)

    overlay_url = create_heatmap_overlay(base_id, overlay_id)
    print(f"  Overlay URL: {overlay_url}")
    assert overlay_url.startswith("https://") or overlay_url.startswith("http://"), \
        "Overlay URL should be a URL"
    print("  PASS: create_heatmap_overlay")
    return overlay_url


def test_transformations(base_id: str):
    print("\n[3/3] Testing apply_visual_transformations...")
    transforms = [
        {"effect": "improve:40"},
        {"effect": "vibrance:30"},
        {"effect": "brightness:15"},
    ]
    url = apply_visual_transformations(base_id, transforms)
    print(f"  Transform URL: {url}")
    assert url, "URL should not be empty"
    print("  PASS: apply_visual_transformations")


def main():
    print("=" * 55)
    print("NeuralLens Cloudinary Tests")
    print("=" * 55)

    missing = [
        k for k in ["CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY",
                     "CLOUDINARY_API_SECRET"]
        if not os.environ.get(k)
    ]
    if missing:
        print(f"ERROR: Missing env vars: {missing}")
        sys.exit(1)

    base_id, _ = test_upload()
    test_overlay(base_id)
    test_transformations(base_id)

    print("\n✅ All Cloudinary tests passed!")


if __name__ == "__main__":
    main()
