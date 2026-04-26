"""Human viewing simulator using Playwright.

Simulates a person loading a webpage, scrolling through it, and experiencing
its content. Outputs a PageContent bundle with screenshot, scroll video,
readable text, optional audio, and metadata.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    url: str
    text: str
    html: str
    screenshot_path: str
    video_path: str
    audio_path: Optional[str]
    metadata: dict = field(default_factory=dict)


async def scrape(url: str, out_dir: str) -> PageContent:
    """Simulate a human viewing the URL and return all captured content."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    screenshot_path = str(out_path / "screenshot.png")
    video_path = str(out_path / "scroll_video.mp4")

    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--no-sandbox"])
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        await page.goto(url, wait_until="load", timeout=45_000)

        # Full-page screenshot
        await page.screenshot(path=screenshot_path, full_page=True)

        # Metadata
        title = await page.title()
        description = await page.evaluate(
            "document.querySelector('meta[name=\"description\"]')?.content || ''"
        )
        headings = await page.evaluate("""
            () => Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6'))
                .map(h => ({tag: h.tagName.toLowerCase(), text: h.innerText.trim()}))
                .filter(h => h.text.length > 0)
        """)
        cta_texts = await page.evaluate("""
            () => Array.from(document.querySelectorAll(
                'button,[role="button"],a.btn,.cta,input[type="submit"],input[type="button"]'
            ))
            .map(el => (el.innerText || el.value || '').trim())
            .filter(t => t && t.length < 100)
            .slice(0, 10)
        """)

        # Visible text in reading order via TreeWalker
        text = await page.evaluate("""
            () => {
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    {
                        acceptNode(node) {
                            const p = node.parentElement;
                            if (!p) return NodeFilter.FILTER_REJECT;
                            const s = window.getComputedStyle(p);
                            if (s.display==='none'||s.visibility==='hidden'||s.opacity==='0')
                                return NodeFilter.FILTER_REJECT;
                            if (['SCRIPT','STYLE','NOSCRIPT','HEAD'].includes(p.tagName))
                                return NodeFilter.FILTER_REJECT;
                            const t = node.textContent.trim();
                            return t ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
                        }
                    }
                );
                const parts = [];
                let node;
                while ((node = walker.nextNode())) {
                    const t = node.textContent.trim();
                    if (t) parts.push(t);
                }
                return parts.join('\\n');
            }
        """)

        # Cleaned body HTML (scripts/styles/iframes removed) for agent context
        html = await page.evaluate("""
            () => {
                const clone = document.body.cloneNode(true);
                for (const el of clone.querySelectorAll(
                    'script,style,noscript,iframe,svg,canvas,video,audio'
                )) el.remove();
                return clone.innerHTML.substring(0, 8000);
            }
        """)

        # Scroll recording: 20 frames over ~10 seconds at 2 fps
        page_height = await page.evaluate("document.documentElement.scrollHeight")
        viewport_h = 800
        scroll_steps = 20
        frames: list[Image.Image] = []

        for i in range(scroll_steps):
            frac = i / max(scroll_steps - 1, 1)
            pos = int(frac * max(0, page_height - viewport_h))
            await page.evaluate(f"window.scrollTo(0, {pos})")
            await asyncio.sleep(0.05)
            raw = await page.screenshot()
            frames.append(Image.open(io.BytesIO(raw)).convert("RGB"))

        # Check for audio/video src
        audio_src = await page.evaluate("""
            () => {
                const a = document.querySelector('audio[src],audio source');
                const v = document.querySelector('video[src],video source');
                return (a && (a.src || a.getAttribute('src')))
                    || (v && (v.src || v.getAttribute('src')))
                    || null;
            }
        """)

        await context.close()
        await browser.close()

    # Stitch frames into MP4
    video_path = _write_video(frames, video_path)

    # Download audio if present
    audio_path: Optional[str] = None
    if audio_src:
        audio_path = await _download_audio(audio_src, str(out_path / "audio.mp3"))

    return PageContent(
        url=url,
        text=text,
        html=html,
        screenshot_path=screenshot_path,
        video_path=video_path,
        audio_path=audio_path,
        metadata={
            "title": title,
            "description": description,
            "headings": headings,
            "cta_texts": cta_texts,
        },
    )


def _write_video(frames: list[Image.Image], video_path: str) -> str:
    if not frames:
        return ""
    arr_frames = [np.array(f) for f in frames]
    h, w = arr_frames[0].shape[:2]

    # Try imageio first (requires imageio-ffmpeg)
    try:
        import imageio.v2 as iio  # type: ignore

        writer = iio.get_writer(video_path, fps=2, codec="libx264", quality=5)
        for arr in arr_frames:
            writer.append_data(arr)
        writer.close()
        return video_path
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("imageio video writer failed; falling back to cv2: %s", exc)

    # Fallback to cv2
    try:
        import cv2  # type: ignore

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 2, (w, h))
        if not out.isOpened():
            logger.error("cv2 VideoWriter could not open output path: %s", video_path)
            return ""
        for arr in arr_frames:
            out.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        out.release()
        return video_path
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.error("cv2 video writer failed; no scroll video generated: %s", exc)

    logger.error("video generation failed with all available backends: %s", video_path)
    return ""


async def _download_audio(src: str, dest: str) -> Optional[str]:
    try:
        import httpx  # type: ignore

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(src)
            if resp.status_code == 200:
                Path(dest).write_bytes(resp.content)
                return dest
            logger.warning(
                "audio download returned non-200 status (%s) from %s",
                resp.status_code,
                src,
            )
    except ImportError as exc:
        logger.warning("httpx unavailable; skipping audio download from %s: %s", src, exc)
    except (OSError, ValueError) as exc:
        logger.error("failed writing downloaded audio to %s: %s", dest, exc)
    except Exception as exc:
        logger.error("audio download failed from %s: %s", src, exc)
    return None
