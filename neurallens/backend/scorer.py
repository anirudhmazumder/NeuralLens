"""TRIBE v2 multimodal brain-encoding scorer.

Two modes:
  stub (default) — calls fake_tribe_score, no API keys needed.
  live  (TRIBE_LIVE=true) — POSTs to a real TRIBE v2 endpoint; if the endpoint
        returns full voxel/region data the atlas_regions key is populated.

Gaze weighting (GAZE_LIVE=true):
  When screenshot_path is provided and GAZE_LIVE=true, the overall_score is
  reweighted by saliency-weighted text quality using the gaze predictor.
  Falls back to the base score silently if gaze analysis fails.

Brain region scoring:
  score_brain_regions(screenshot_path) returns all 9 HCP-MMP1 region activations
  (FFA, V4, MT+, Hippocampus, PFC, ACC, Amygdala, Insula, NAcc) using the local
  StubEncoder. This runs in parallel to the main TRIBE score.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from stubs import fake_tribe_score


class TribeScorer:
    def __init__(self) -> None:
        self.live = os.getenv("TRIBE_LIVE", "false").lower() == "true"
        self.endpoint = os.getenv("TRIBE_ENDPOINT", "http://localhost:9090")

    async def score(
        self,
        video_path: str,
        text: str,
        audio_path: Optional[str] = None,
        *,
        screenshot_path: Optional[str] = None,
    ) -> dict:
        # ── Base TRIBE score ───────────────────────────────────────────────────
        if not self.live:
            base = await fake_tribe_score(video_path, text, audio_path)
        else:
            import httpx  # type: ignore
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.endpoint}/score",
                    json={"video_path": video_path, "text": text, "audio_path": audio_path},
                )
                resp.raise_for_status()
                base = resp.json()

        # ── Gaze-weighted adjustment ───────────────────────────────────────────
        if (
            os.getenv("GAZE_LIVE", "false").lower() == "true"
            and screenshot_path
            and Path(screenshot_path).exists()
        ):
            try:
                from gaze_predictor import get_gaze_predictor, gaze_weighted_score
                gaze_data = await get_gaze_predictor().analyze(screenshot_path)
                if gaze_data["regions"]:
                    base = gaze_weighted_score(base, text, gaze_data["regions"])
            except Exception:
                pass  # gaze failure is non-fatal; return base score

        return base

    async def score_brain_regions(
        self,
        screenshot_path: Optional[str] = None,
        *,
        text: str = "",
    ) -> dict[str, float]:
        """Return all 9 HCP-MMP1 region activations.

        If TRIBE_LIVE=true and the endpoint returns region-level data, use that.
        Otherwise use the local StubEncoder on the screenshot (fast, no GPU).
        Falls back to text-derived heuristic if no screenshot is available.
        """
        if self.live and screenshot_path and Path(screenshot_path).exists():
            try:
                import httpx  # type: ignore
                async with httpx.AsyncClient(timeout=30.0) as client:
                    with open(screenshot_path, "rb") as fh:
                        import base64 as _b64
                        img_b64 = _b64.b64encode(fh.read()).decode()
                    resp = await client.post(
                        f"{self.endpoint}/brain-regions",
                        json={"image_base64": img_b64, "text": text},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if "regions" in data:
                            return data["regions"]
            except Exception:
                pass  # fall through to local stub

        if screenshot_path and Path(screenshot_path).exists():
            from brain_regions import score_screenshot
            return score_screenshot(screenshot_path)

        # Text-only fallback: derive approximate region scores from text length/complexity
        return _text_region_heuristic(text)


def _text_region_heuristic(text: str) -> dict[str, float]:
    """Cheap text-only approximation when no screenshot is available."""
    import math

    words = text.split()
    wc = len(words)
    # Longer, richer text → higher language/hippocampus engagement
    length_factor = min(1.0, math.log1p(wc) / math.log1p(500))
    # Short punchy text → higher NAcc (reward signal)
    punch = max(0.0, 1.0 - wc / 200.0)

    return {
        "FFA": 0.4,
        "V4": 0.4,
        "MT+": 0.35,
        "Hippocampus": round(0.3 + 0.5 * length_factor, 4),
        "PFC": round(0.45 + 0.2 * length_factor, 4),
        "ACC": round(0.35 + 0.1 * (1.0 - length_factor), 4),
        "Amygdala": 0.3,
        "Insula": 0.3,
        "NAcc": round(0.35 + 0.3 * punch, 4),
    }
