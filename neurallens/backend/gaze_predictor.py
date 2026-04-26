"""Gaze prediction layer — DeepGaze IIE (live) or F-pattern stub.

GAZE_LIVE=false  → F-pattern heuristic saliency, no GPU, no model download
GAZE_LIVE=true   → DeepGaze IIE with MIT1003 centerbias

The stub produces scientifically plausible fixation distributions:
- Strong horizontal scan band at the top (first headline / hero area)
- Second horizontal band ~25% down (sub-headline / key value prop)
- Left-edge fixation bias throughout (beginning of each scan line)
- Gaussian noise to simulate natural variance

Singleton via get_gaze_predictor() — model loaded once, reused across requests.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_CENTERBIAS_PATH = os.getenv("CENTERBIAS_PATH", "/workspace/centerbias_mit1003.npy")
_GAZE_DEVICE = os.getenv("GAZE_DEVICE", "cpu")
_MODEL_CACHE_PATH = Path(os.getenv(
    "DEEPGAZE_CACHE_PATH",
    os.path.expanduser("~/.cache/neurallens/deepgaze_iie.pth"),
))


@dataclass
class GazeRegion:
    rank: int
    bbox: list[int]          # [x1, y1, x2, y2] in screenshot pixels
    saliency_score: float
    peak_coords: list[int]   # [x, y]


# ── Stub ──────────────────────────────────────────────────────────────────────

def _stub_saliency(h: int, w: int) -> np.ndarray:
    """F-pattern saliency: bright top band, second band ~25% down, left bias."""
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    y_norm = y_idx / h
    x_norm = x_idx / w

    row1 = np.exp(-y_norm * 8) * 0.55           # strong top scan
    row2 = np.exp(-np.abs(y_norm - 0.25) * 12) * 0.30  # second fixation band
    left = np.exp(-x_norm * 2.5) * 0.15         # left-edge fixation

    sal = row1 + row2 + left
    rng = np.random.default_rng(seed=42)
    sal += rng.normal(0, 0.015, sal.shape)
    sal = np.clip(sal, 0, None)
    sal /= sal.max() + 1e-9
    return sal.astype(np.float32)


def _suppress_and_find_peaks(
    sal_norm: np.ndarray,
    top_k: int,
    win_h: int,
    win_w: int,
) -> list[GazeRegion]:
    h, w = sal_norm.shape
    regions: list[GazeRegion] = []
    sal_copy = sal_norm.copy()

    for rank in range(1, top_k + 1):
        py, px = np.unravel_index(sal_copy.argmax(), sal_copy.shape)
        y1 = max(0, int(py) - win_h // 2)
        y2 = min(h, int(py) + win_h // 2)
        x1 = max(0, int(px) - win_w // 2)
        x2 = min(w, int(px) + win_w // 2)
        score = float(sal_norm[y1:y2, x1:x2].mean())
        regions.append(GazeRegion(
            rank=rank,
            bbox=[x1, y1, x2, y2],
            saliency_score=round(score, 4),
            peak_coords=[int(px), int(py)],
        ))
        sal_copy[y1:y2, x1:x2] = 0.0

    return regions


# ── GazePredictor ─────────────────────────────────────────────────────────────

class GazePredictor:
    def __init__(self) -> None:
        self.live = os.getenv("GAZE_LIVE", "false").lower() == "true"
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from deepgaze_pytorch import DeepGazeIIE  # type: ignore[import-untyped]

            if _MODEL_CACHE_PATH.exists():
                # Load from local cache — no internet call on subsequent starts
                model = DeepGazeIIE(pretrained=False)
                model.load_state_dict(
                    torch.load(str(_MODEL_CACHE_PATH), map_location=_GAZE_DEVICE)
                )
                print(f"[GazePredictor] DeepGaze IIE loaded from cache: {_MODEL_CACHE_PATH}")
            else:
                # First run: download once, then save for future starts
                model = DeepGazeIIE(pretrained=True)
                _MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(_MODEL_CACHE_PATH))
                print(f"[GazePredictor] DeepGaze IIE downloaded and cached to {_MODEL_CACHE_PATH}")

            self._model = model.to(_GAZE_DEVICE)
            self._model.eval()
        except (RuntimeError, OSError, Exception) as exc:
            # Corrupted checkpoint, missing file, or other load failure — fall back to stub
            print(f"[GazePredictor] model load failed ({exc.__class__.__name__}: {exc}) — using F-pattern stub")
            self.live = False
            self._model = None

    # ── Saliency ───────────────────────────────────────────────────────────────

    def predict_saliency(self, screenshot_path: str) -> tuple[np.ndarray, int, int]:
        """Return (saliency_map, height, width)."""
        try:
            from PIL import Image
            img = Image.open(screenshot_path).convert("RGB")
            w, h = img.size
        except Exception:
            h, w = 800, 1280

        if not self.live:
            return _stub_saliency(h, w), h, w

        # ── Live: DeepGaze IIE ─────────────────────────────────────────────
        import torch
        from scipy.ndimage import zoom
        from scipy.special import logsumexp
        from PIL import Image as _PILImage

        self._load_model()
        # _load_model may have flipped live=False on failure
        if not self.live:
            return _stub_saliency(h, w), h, w

        image = np.array(_PILImage.open(screenshot_path).convert("RGB"))
        h_img, w_img = image.shape[:2]

        centerbias_template = np.load(_CENTERBIAS_PATH)
        cb = zoom(
            centerbias_template,
            (h_img / centerbias_template.shape[0], w_img / centerbias_template.shape[1]),
            order=0, mode="nearest",
        )
        cb -= logsumexp(cb)

        img_t = torch.tensor([image.transpose(2, 0, 1)], dtype=torch.float32).to(_GAZE_DEVICE)
        cb_t  = torch.tensor([cb], dtype=torch.float32).to(_GAZE_DEVICE)

        with torch.no_grad():
            log_density = self._model(img_t, cb_t)

        sal = log_density.exp().cpu().numpy()[0, 0].astype(np.float32)
        return sal, h_img, w_img

    # ── Regions ────────────────────────────────────────────────────────────────

    def get_salient_regions(self, screenshot_path: str, top_k: int = 5) -> list[GazeRegion]:
        sal, h, w = self.predict_saliency(screenshot_path)

        mn, mx = sal.min(), sal.max()
        sal_norm = (sal - mn) / (mx - mn + 1e-9)

        win_h = max(h // 5, 60)
        win_w = max(w // 4, 80)
        return _suppress_and_find_peaks(sal_norm, top_k, win_h, win_w)

    # ── Heatmap overlay ────────────────────────────────────────────────────────

    def generate_heatmap_overlay(self, screenshot_path: str, output_path: str) -> str:
        import cv2

        image = cv2.imread(screenshot_path)
        if image is None:
            return ""

        sal, h, w = self.predict_saliency(screenshot_path)
        sal_resized = cv2.resize(sal, (image.shape[1], image.shape[0]))
        mn, mx = sal_resized.min(), sal_resized.max()
        if mx > mn:
            sal_u8 = ((sal_resized - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            sal_u8 = np.zeros(sal_resized.shape, dtype=np.uint8)

        heatmap = cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)
        cv2.imwrite(output_path, overlay)
        return output_path

    # ── Async entry point ──────────────────────────────────────────────────────

    async def analyze(self, screenshot_path: str, top_k: int = 5) -> dict:
        """Run in executor so CPU-bound work doesn't block the event loop."""
        if not screenshot_path or not Path(screenshot_path).exists():
            return {"regions": [], "gaze_live": self.live}

        loop = asyncio.get_event_loop()
        regions = await loop.run_in_executor(
            None, self.get_salient_regions, screenshot_path, top_k
        )
        return {
            "regions": [
                {
                    "rank": r.rank,
                    "bbox": r.bbox,
                    "saliency_score": r.saliency_score,
                    "peak_coords": r.peak_coords,
                }
                for r in regions
            ],
            "gaze_live": self.live,
        }


# ── Module singleton ──────────────────────────────────────────────────────────

_predictor: Optional[GazePredictor] = None


def get_gaze_predictor() -> GazePredictor:
    global _predictor
    if _predictor is None:
        _predictor = GazePredictor()
    return _predictor


# ── Gaze-weighted score helper (used by scorer.py) ────────────────────────────

def gaze_weighted_score(base_score: dict, text: str, regions: list[dict]) -> dict:
    """Reweight overall_score by saliency-weighted text quality per region.

    Splits text into N equal chunks (top→bottom), assigns each chunk the
    saliency weight of the corresponding gaze region, then blends the
    saliency-weighted quality average with the base overall_score (50/50).
    No async calls — runs instantly.
    """
    from stubs import _text_quality_score  # type: ignore

    words = text.split()
    n = len(regions)
    if not words or n == 0:
        return base_score

    chunk_size = max(1, len(words) // n)
    chunks = [" ".join(words[i * chunk_size:(i + 1) * chunk_size]) for i in range(n)]
    qualities = [_text_quality_score(c) for c in chunks]
    saliencies = [r.get("saliency_score", 1.0 / n) for r in regions]
    total_sal = sum(saliencies) or 1.0

    weighted_quality = sum(q * s / total_sal for q, s in zip(qualities, saliencies))
    gaze_overall = round(0.5 * base_score["overall_score"] + 0.5 * weighted_quality, 4)

    result = dict(base_score)
    result["overall_score"] = gaze_overall
    result["gaze_weighted"] = True
    result["salient_regions"] = [
        {**r, "text_quality": round(q, 4)}
        for r, q in zip(regions, qualities)
    ]
    return result
