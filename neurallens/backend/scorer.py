"""TRIBE v2 multimodal brain-encoding scorer.

Always calls the live TRIBE endpoint (TRIBE_ENDPOINT env var).
POSTs a screenshot PNG + text to /encode and receives 9 HCP-MMP1 region scores.

Gaze weighting (GAZE_LIVE=true):
  When screenshot_path is provided and GAZE_LIVE=true, overall_score is
  reweighted by saliency-weighted text quality. Fails silently.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

class TribeScorer:
    def __init__(self) -> None:
        self.endpoint = os.getenv("TRIBE_ENDPOINT", "http://localhost:9090/encode")
        # Free tunnels / cold model starts can exceed 30s, so make timeout configurable.
        self.timeout_seconds = float(os.getenv("TRIBE_TIMEOUT_SECONDS", "90"))

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
        """POST to TRIBE endpoint. Falls back to local image encoder on connection failure."""
        import httpx  # type: ignore

        url = self._encode_url()
        with open(screenshot_path, "rb") as fh:
            png = fh.read()

        try:
            timeout = httpx.Timeout(
                connect=min(15.0, self.timeout_seconds),
                read=self.timeout_seconds,
                write=min(30.0, self.timeout_seconds),
                pool=min(15.0, self.timeout_seconds),
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Try raw PNG body first (Flask request.data style).
                r = await client.post(url, content=png, headers={"Content-Type": "image/png"})
                if r.status_code >= 400:
                    # Fallback to JSON base64 (common FastAPI shape).
                    payload = {"image_base64": base64.b64encode(png).decode(), "text": text}
                    r = await client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
                # Strip large vertex arrays — we only need region_scores + subcortical
                data.pop("vertices", None)
                return self._extract_regions(data)

        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            print(
                f"[TRIBE] endpoint unreachable ({exc.__class__.__name__}) "
                f"after timeout={self.timeout_seconds:.1f}s — using local encoder"
            )
            from brain_regions import score_screenshot
            return score_screenshot(screenshot_path)

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
        # ── Live TRIBE score (always) ──────────────────────────────────────────
        if not screenshot_path or not Path(screenshot_path).exists():
            raise RuntimeError(
                "TRIBE endpoint requires a valid screenshot_path. "
                "Ensure TRIBE_ENDPOINT is reachable and a screenshot was taken."
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
        """Return all 9 HCP-MMP1 region activations from the live TRIBE endpoint.

        Falls back to the local image-feature encoder (brain_regions.score_screenshot)
        if screenshot_path is missing — this keeps the RL loop running when a
        resample screenshot hasn't arrived yet.
        """
        if screenshot_path and Path(screenshot_path).exists():
            return await self._call_live_encode(screenshot_path, text)
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
