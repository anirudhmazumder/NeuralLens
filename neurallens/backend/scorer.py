"""TRIBE v2 multimodal brain-encoding scorer.

Always calls the configured live TRIBE endpoint (TRIBE_ENDPOINT env var).
No fake-score fallback is used in live mode: endpoint failures are surfaced.

In-memory response caching is keyed by (screenshot path + mtime, text hash).
This avoids re-paying TRIBE latency when the optimizer scores the same
screenshot twice in one run (e.g., baseline vs. iteration 1 with no edit
applied yet, or repeat hits during retry).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional


_TRIBE_CACHE_SIZE = int(os.getenv("TRIBE_RESPONSE_CACHE_SIZE", "64"))
_shared_cache_lock = threading.Lock()
_shared_cache: "OrderedDict[tuple[str, int, str], dict[str, float]]" = OrderedDict()


class TribeScorer:
    def __init__(self) -> None:
        self.endpoint = os.getenv("TRIBE_ENDPOINT", "http://localhost:9090/encode")
        # Allow slower remote TRIBE backends (cold starts / queued workers).
        self.timeout_seconds = float(os.getenv("TRIBE_TIMEOUT_SECONDS", "90"))
        self.retry_attempts = int(os.getenv("TRIBE_RETRY_ATTEMPTS", "3"))
        self.retry_backoff_seconds = float(os.getenv("TRIBE_RETRY_BACKOFF_SECONDS", "1.5"))
        # Process-wide cache shared across instances; bounded LRU.
        self._cache_lock = _shared_cache_lock
        self._cache: "OrderedDict[tuple[str, int, str], dict[str, float]]" = _shared_cache

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

    @staticmethod
    def _cache_key(screenshot_path: str, text: str) -> Optional[tuple[str, int, str]]:
        try:
            p = Path(screenshot_path).resolve()
            mtime_ns = p.stat().st_mtime_ns
        except OSError:
            return None
        text_hash = hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]
        return (str(p), mtime_ns, text_hash)

    def _cache_get(self, key: Optional[tuple]) -> Optional[dict[str, float]]:
        if key is None:
            return None
        with self._cache_lock:
            value = self._cache.get(key)
            if value is None:
                return None
            self._cache.move_to_end(key)
            return dict(value)

    def _cache_put(self, key: Optional[tuple], value: dict[str, float]) -> None:
        if key is None:
            return
        with self._cache_lock:
            self._cache[key] = dict(value)
            self._cache.move_to_end(key)
            while len(self._cache) > _TRIBE_CACHE_SIZE:
                self._cache.popitem(last=False)

    async def _call_live_encode(self, screenshot_path: str, text: str) -> dict[str, float]:
        """POST screenshot data to live TRIBE endpoint (with response cache)."""
        import httpx  # type: ignore

        cache_key = self._cache_key(screenshot_path, text)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        url = self._encode_url()
        with open(screenshot_path, "rb") as fh:
            png = fh.read()

        timeout = httpx.Timeout(
            connect=min(12.0, self.timeout_seconds),
            read=self.timeout_seconds,
            write=min(20.0, self.timeout_seconds),
            pool=min(12.0, self.timeout_seconds),
        )
        last_timeout_exc: Optional[Exception] = None
        last_status_exc: Optional[Exception] = None
        last_status_code: Optional[int] = None
        last_error_excerpt = ""

        async with httpx.AsyncClient(timeout=timeout) as client:
            payload_json = {"image_base64": base64.b64encode(png).decode(), "text": text}
            request_variants = (
                {"json": payload_json},
                {"content": png, "headers": {"Content-Type": "image/png"}},
            )

            for attempt in range(1, max(1, self.retry_attempts) + 1):
                for request_kwargs in request_variants:
                    try:
                        r = await client.post(url, **request_kwargs)
                        r.raise_for_status()
                        data = r.json()
                        # Strip large vertex arrays — we only need region_scores + subcortical
                        data.pop("vertices", None)
                        regions = self._extract_regions(data)
                        self._cache_put(cache_key, regions)
                        return regions
                    except httpx.TimeoutException as exc:
                        last_timeout_exc = exc
                    except httpx.HTTPStatusError as exc:
                        last_status_exc = exc
                        if exc.response is not None:
                            last_status_code = exc.response.status_code
                            body = exc.response.text or ""
                            last_error_excerpt = body.strip().replace("\n", " ")[:280]

                if attempt < max(1, self.retry_attempts):
                    await asyncio.sleep(self.retry_backoff_seconds * attempt)

        if last_timeout_exc is not None and last_status_exc is None:
            raise RuntimeError(
                f"TRIBE /encode timed out after {self.timeout_seconds:.0f}s at {url} "
                f"(retried {max(1, self.retry_attempts)}x). "
                "Your remote TRIBE endpoint is reachable but not responding in time."
            ) from last_timeout_exc

        if last_status_exc is not None:
            detail = f" Last response body: {last_error_excerpt}" if last_error_excerpt else ""
            raise RuntimeError(
                f"TRIBE /encode returned HTTP {last_status_code or 'unknown'} at {url} "
                f"(retried {max(1, self.retry_attempts)}x)."
                f"{detail} Check Colab server logs for endpoint-side exceptions."
            ) from last_status_exc

        raise RuntimeError(f"TRIBE /encode failed at {url} for an unknown reason.")

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

        Hits the in-memory cache first (so when the optimizer has already called
        ``score()`` for the same screenshot+text, the brain-region lookup is
        essentially free). Falls back to a text-only heuristic if there's no
        screenshot to send.
        """
        if screenshot_path and Path(screenshot_path).exists():
            cache_key = self._cache_key(screenshot_path, text)
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached
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
