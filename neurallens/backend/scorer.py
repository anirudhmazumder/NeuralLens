"""TRIBE v2 multimodal brain-encoding scorer.

Two modes:
  stub (default) — calls fake_tribe_score, no API keys needed.
  live  (TRIBE_LIVE=true) — POSTs to a real TRIBE v2 endpoint.

Gaze weighting (GAZE_LIVE=true):
  When screenshot_path is provided and GAZE_LIVE=true, the overall_score is
  reweighted by saliency-weighted text quality using the gaze predictor.
  Falls back to the base score silently if gaze analysis fails.
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
