"""
integrations/image_fetcher.py

Robust image downloader shared by sensor_agent and executor_agent.
Tries multiple User-Agent strings and falls back to SSL-unverified
as a last resort so CDN/wiki/Unsplash URLs all work.
"""

import requests
import urllib3

# Ordered list — most CDN-friendly first
_USER_AGENTS = [
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Mozilla/5.0 (compatible; NeuralLens/1.0; +https://neurallens.ai)",
    "curl/8.4.0",
    "python-requests/2.31.0",
]

_IMAGE_MAGIC = {
    b"\xff\xd8\xff": "JPEG",
    b"\x89PN": "PNG",
    b"GIF": "GIF",
    b"RIFF": "WEBP",
}


def _looks_like_image(data: bytes, content_type: str) -> bool:
    for magic in _IMAGE_MAGIC:
        if data[:len(magic)] == magic:
            return True
    return "image" in content_type.lower()


def fetch_image(url: str, timeout: int = 30) -> bytes:
    """
    Download raw image bytes from any public URL.

    Tries multiple User-Agent strings in order.
    Falls back to SSL verify=False as a last resort.

    Args:
        url: Direct image URL (jpg, png, webp, gif, etc.).
        timeout: Per-attempt timeout in seconds.

    Returns:
        Raw image bytes.

    Raises:
        RuntimeError: If all strategies fail, with a user-friendly message.
    """
    last_err = None

    # Strategy 1: each User-Agent, SSL verified
    for ua in _USER_AGENTS:
        try:
            resp = requests.get(
                url,
                headers={
                    "User-Agent": ua,
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                    "Referer": "https://www.google.com/",
                },
                timeout=timeout,
                allow_redirects=True,
            )
            resp.raise_for_status()
            if resp.content and _looks_like_image(resp.content, resp.headers.get("Content-Type", "")):
                return resp.content
            last_err = ValueError(
                f"Response body doesn't look like an image "
                f"(content-type={resp.headers.get('Content-Type')!r}, "
                f"magic={resp.content[:4].hex()!r})"
            )
        except Exception as e:
            last_err = e
            continue

    # Strategy 2: SSL verify=False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _USER_AGENTS[0]},
            timeout=timeout,
            allow_redirects=True,
            verify=False,
        )
        resp.raise_for_status()
        if resp.content:
            return resp.content
    except Exception as e:
        last_err = e

    raise RuntimeError(
        f"Could not download image from {url!r}.\n"
        f"Last error: {last_err}\n"
        f"Tips:\n"
        f"  • Paste the URL in a browser — does it load directly?\n"
        f"  • Use a direct image link ending in .jpg/.png/.webp\n"
        f"  • Good test URLs: https://picsum.photos/seed/test/800/600\n"
        f"                    https://i.imgur.com/XXXXXX.jpg"
    )
