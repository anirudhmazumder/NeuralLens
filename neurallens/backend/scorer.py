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

import base64
import os
from pathlib import Path
from typing import Optional

from stubs import fake_tribe_score


class TribeScorer:
    def __init__(self) -> None:
        self.live = os.getenv("TRIBE_LIVE", "false").lower() == "true"
        self.endpoint = os.getenv("TRIBE_ENDPOINT", "http://localhost:9090/encode")
        token = os.getenv("TRIBE_TOKEN", "").strip()
        self.token = token or None

    def _encode_url(self) -> str:
        ep = self.endpoint.rstrip("/")
        if ep.endswith("/encode"):
            return ep
        return f"{ep}/encode"

    @staticmethod
    def _extract_regions(payload: dict) -> dict[str, float]:
        # Accept a few schema variants from live TRIBE services.
        if isinstance(payload.get("region_scores"), dict):
            base = payload["region_scores"]
        elif isinstance(payload.get("scores"), dict):
            base = payload["scores"]
        elif isinstance(payload.get("regions"), dict):
            base = payload["regions"]
        else:
            base = payload

        regions: dict[str, float] = {}
        for key in ("FFA", "V4", "MT+", "PFC", "ACC", "Insula"):
            if key in base:
                regions[key] = float(base[key])

        sub = payload.get("subcortical_estimates")
        sub_vals = sub.get("values") if isinstance(sub, dict) and isinstance(sub.get("values"), dict) else {}
        for key in ("Hippocampus", "Amygdala", "NAcc"):
            if key in base:
                regions[key] = float(base[key])
            elif key in sub_vals:
                regions[key] = float(sub_vals[key])

        required = {"FFA", "V4", "MT+", "Hippocampus", "PFC", "ACC", "Amygdala", "Insula", "NAcc"}
        missing = sorted(required - set(regions.keys()))
        if missing:
            raise RuntimeError(
                "TRIBE live response missing required regions: "
                + ", ".join(missing)
            )
        return regions

    async def _call_live_encode(self, screenshot_path: str, text: str) -> dict[str, float]:
        import httpx  # type: ignore

        url = self._encode_url()
        with open(screenshot_path, "rb") as fh:
            png = fh.read()

        headers = {"Content-Type": "image/png"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            # First try raw PNG body (Flask request.data style).
            r = await client.post(url, content=png, headers=headers)
            if r.status_code >= 400:
                # Fallback to JSON base64 (common FastAPI shape).
                payload = {"image_base64": base64.b64encode(png).decode(), "text": text}
                json_headers = {"Content-Type": "application/json"}
                if self.token:
                    json_headers["Authorization"] = f"Bearer {self.token}"
                r = await client.post(url, json=payload, headers=json_headers)
            r.raise_for_status()
            data = r.json()
        return self._extract_regions(data)

    @staticmethod
    def _regions_to_multimodal_score(regions: dict[str, float], *, audio_path: Optional[str], text: str) -> dict:
        # Keep output contract expected by optimizer/frontend.
        ffa = regions["FFA"]
        v4 = regions["V4"]
        mt = regions["MT+"]
        hip = regions["Hippocampus"]
        pfc = regions["PFC"]
        acc = regions["ACC"]
        amyg = regions["Amygdala"]
        ins = regions["Insula"]
        nacc = regions["NAcc"]

        visual_roi = max(0.0, min(1.0, 0.45 * v4 + 0.35 * ffa + 0.20 * mt))
        attention_roi = max(0.0, min(1.0, 0.30 * ffa + 0.25 * v4 + 0.15 * mt + 0.20 * nacc + 0.10 * pfc))
        language_roi = max(0.0, min(1.0, 0.55 * pfc + 0.30 * hip + 0.15 * ffa))
        penalty = 0.55 * amyg + 0.30 * ins + 0.15 * acc
        overall = max(0.0, min(1.0, 0.36 * attention_roi + 0.34 * language_roi + 0.30 * visual_roi - 0.35 * penalty))
        # Use lightweight text/audio modulation, but never stubbed random values.
        text_bonus = min(0.05, max(0.0, len(text.split()) / 2000.0))
        audio_score = 0.55 if audio_path else 0.30
        overall = max(0.0, min(1.0, overall + text_bonus))

        return {
            "overall_score": round(overall, 4),
            "visual_score": round(visual_roi, 4),
            "text_score": round(language_roi, 4),
            "audio_score": round(audio_score, 4),
            "language_roi": round(language_roi, 4),
            "attention_roi": round(attention_roi, 4),
            "visual_roi": round(visual_roi, 4),
            "atlas_regions": {k: round(float(v), 4) for k, v in regions.items()},
        }

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
            if not screenshot_path or not Path(screenshot_path).exists():
                raise RuntimeError(
                    "TRIBE_LIVE=true requires a valid screenshot_path for /encode scoring."
                )
            regions = await self._call_live_encode(screenshot_path, text)
            base = self._regions_to_multimodal_score(regions, audio_path=audio_path, text=text)

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
        if self.live:
            if not screenshot_path or not Path(screenshot_path).exists():
                raise RuntimeError(
                    "TRIBE_LIVE=true requires a valid screenshot_path for /encode region scoring."
                )
            return await self._call_live_encode(screenshot_path, text)

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
