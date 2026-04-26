"""Page annotator — draws component overlays, gaze heatmap, scan-path on screenshots.

Generates base64 JPEG annotations for the Agent Vision live feed.
All drawing is synchronous (cv2); callers should run in a thread executor.
"""
from __future__ import annotations

import base64
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Color helpers ──────────────────────────────────────────────────────────────

def _score_bgr(score: float) -> tuple[int, int, int]:
    """BGR: red (low < 0.35) → yellow (mid) → green (high > 0.6)."""
    if score < 0.35:
        return (60, 60, 210)
    elif score < 0.6:
        return (40, 200, 210)
    return (60, 200, 60)


def _gaussian_heatmap(image: np.ndarray, salient_regions: list[dict]) -> np.ndarray:
    """Blend a gaussian-blob gaze heatmap approximation onto the image."""
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    sigma = min(h, w) // 7

    for r in salient_regions:
        px, py = r.get("peak_coords", [w // 2, h // 4])
        sal = r.get("saliency_score", 0.5)
        y_idx, x_idx = np.mgrid[0:h, 0:w]
        blob = sal * np.exp(-((x_idx - px) ** 2 + (y_idx - py) ** 2) / (2 * sigma ** 2))
        heatmap = np.maximum(heatmap, blob.astype(np.float32))

    mx = heatmap.max()
    heatmap_u8 = ((heatmap / mx) * 255).astype(np.uint8) if mx > 0 else np.zeros((h, w), dtype=np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.60, heatmap_bgr, 0.40, 0)


def _dashed_line(image: np.ndarray, pt1: tuple, pt2: tuple, color, thickness: int, dash: int = 20) -> None:
    """Draw a dashed line between two points."""
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    steps = max(1, int(dist / dash))
    for s in range(steps):
        if s % 2 == 0:
            sx = int(pt1[0] + dx * s / steps)
            sy = int(pt1[1] + dy * s / steps)
            ex = int(pt1[0] + dx * (s + 1) / steps)
            ey = int(pt1[1] + dy * (s + 1) / steps)
            cv2.line(image, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)


# ── PageVisualizer ─────────────────────────────────────────────────────────────

class PageVisualizer:

    def draw_annotated_screenshot(
        self,
        screenshot_path: str,
        viz_components: list[dict],
        salient_regions: list[dict],
        target_id: str = "",
        accepted_ids: Optional[list[str]] = None,
        rejected_ids: Optional[list[str]] = None,
    ) -> str:
        """Return base64 JPEG string of annotated screenshot, or "" on failure."""
        accepted_ids = accepted_ids or []
        rejected_ids = rejected_ids or []

        try:
            if screenshot_path and Path(screenshot_path).exists():
                image = cv2.imread(screenshot_path)
            else:
                image = None

            if image is None:
                # Stub gray canvas
                image = np.full((800, 1280, 3), 38, dtype=np.uint8)
                cv2.putText(image, "NeuralLens — Agent Vision (stub)",
                            (320, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

            h, w = image.shape[:2]

            # ── 1. Gaze heatmap ───────────────────────────────────────────
            if salient_regions:
                image = _gaussian_heatmap(image, salient_regions)

            # ── 2. Component bounding boxes ───────────────────────────────
            if viz_components:
                n = len(viz_components)
                per_h = max(1, h // n)
                overlay = image.copy()

                for i, comp in enumerate(viz_components):
                    y1 = i * per_h
                    y2 = min(h - 1, (i + 1) * per_h)
                    cid = comp.get("id", "")
                    score = comp.get("neural_score", 0.5)
                    color = _score_bgr(score)

                    is_target   = cid == target_id
                    is_accepted = cid in accepted_ids
                    is_rejected = cid in rejected_ids

                    # Semi-transparent fill for special states
                    fill = None
                    if is_target:
                        fill = (120, 120, 255)
                    elif is_accepted:
                        fill = (40, 200, 40)
                    elif is_rejected:
                        fill = (40, 40, 200)
                    if fill:
                        cv2.rectangle(overlay, (0, y1), (w, y2), fill, -1)

                    # Border
                    thickness = 4 if is_target else 2
                    border = (255, 255, 255) if is_target else color
                    cv2.rectangle(image, (1, y1 + 1), (w - 1, y2 - 1), border, thickness)

                    # Score badge
                    badge = f"{comp.get('type', 'blk')[:4].upper()}  {score:.2f}"
                    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                    cv2.rectangle(image, (4, y1 + 3), (4 + bw + 6, y1 + bh + 9), (15, 15, 15), -1)
                    cv2.putText(image, badge, (7, y1 + bh + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

                    # "🎯 TARGETING" label on right side
                    if is_target:
                        lbl = "TARGETING"
                        (lw2, lh2), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
                        lx = w - lw2 - 14
                        cv2.rectangle(image, (lx - 4, y1 + 4), (w - 4, y1 + lh2 + 12), (60, 40, 160), -1)
                        cv2.putText(image, lbl, (lx, y1 + lh2 + 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

                image = cv2.addWeighted(overlay, 0.12, image, 0.88, 0)

            # ── 3. Gaze scan path + numbered circles ──────────────────────
            if salient_regions:
                peaks = [r.get("peak_coords", [w // 2, h // 4]) for r in salient_regions]

                # Dashed lines connecting fixation points
                for i in range(len(peaks) - 1):
                    _dashed_line(image, tuple(peaks[i]), tuple(peaks[i + 1]),
                                 (220, 220, 220), 2, dash=18)

                # Numbered circles
                for r in salient_regions:
                    px, py = r.get("peak_coords", [w // 2, h // 4])
                    rank = r.get("rank", 1)
                    cv2.circle(image, (px, py), 22, (20, 20, 20), -1)
                    cv2.circle(image, (px, py), 22, (124, 58, 237), 3, cv2.LINE_AA)
                    cv2.putText(image, str(rank), (px - 7 if rank < 10 else px - 10, py + 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # ── Encode ────────────────────────────────────────────────────
            _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
            return base64.b64encode(buf).decode("utf-8")

        except Exception as exc:
            print(f"[PageVisualizer] annotation failed: {exc}")
            return ""
