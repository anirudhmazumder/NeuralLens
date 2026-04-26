"""Local brain-region activation estimator — fallback when TRIBE API is unreachable.

Only a small subset of the 9 regions can be reliably estimated from raw screenshot
pixels. The rest return 0.5 (neutral) so the optimizer keeps running without
fabricating scores.

  Estimable from pixels:
    V4    — colour saturation (Conway & Tsao 2006 — well supported)
    PFC   — whitespace + luminance contrast proxy (directionally OK)
    ACC   — visual busyness / low whitespace proxy (directionally OK)
    NAcc  — saturation × contrast (rough; kept as fallback only)

  NOT estimable from pixels (return 0.5 neutral):
    FFA   — face-selective; HSV skin-colour detection fires on any warm-toned
             surface. Requires face detection or live TRIBE.
    MT+   — motion complex; diagonal edges in static images are not a reliable
             MT+ proxy (Kourtzi & Kanwisher 2000 — effect is weak and stimulus-
             specific). Requires motion stimulus or live TRIBE.
    Insula — colour-clash correlation with visceral unease is unproven.
             Requires live TRIBE.

  Subcortical (Harvard-Oxford atlas — estimated by the Colab API):
    Hippocampus — novelty/memory encoding
    Amygdala    — threat/anxiety signal
    NAcc        — reward anticipation
    These are NOT in the HCP-MMP1 cortical atlas and must come from the API.
    The pixel-estimated NAcc fallback is retained only so the optimizer never
    receives a missing key; it should be ignored when the API is live.

When connected to the live TRIBE v2 API the Colab returns all 9 regions,
including subcortical scores computed from Harvard-Oxford atlas masks
(labels 6/13=Hippocampus, 7/14=Amygdala, 8/15=NAcc in the thr25-2mm volume).
"""
from __future__ import annotations

import math
from typing import Optional

from PIL import Image, ImageStat

# ── Image feature extractors ──────────────────────────────────────────────────

def _sigmoid(x: float, center: float = 0.5, k: float = 6.0) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - center)))


def _saturation(img: Image.Image) -> float:
    hsv = img.convert("HSV")
    s_band = hsv.split()[1]
    return ImageStat.Stat(s_band).mean[0] / 255.0


def _luma_contrast(img: Image.Image) -> float:
    g = img.convert("L")
    return min(1.0, ImageStat.Stat(g).stddev[0] / 96.0)


def _whitespace_fraction(img: Image.Image) -> float:
    g = img.convert("L").resize((64, 64))
    px = g.load()
    light = sum(1 for y in range(64) for x in range(64) if px[x, y] > 230)
    return light / (64 * 64)


# ── Encoder ───────────────────────────────────────────────────────────────────

def score_screenshot(path: str) -> dict[str, float]:
    """Return all 9 HCP-MMP1 region activations for the screenshot at *path*.

    Only V4, PFC, ACC, and NAcc are estimated from image features.
    All other regions return 0.5 (neutral) — they require the live TRIBE API.
    Subcortical regions (Hippocampus, Amygdala, NAcc) are estimated by the API
    via Harvard-Oxford atlas; the NAcc pixel proxy here is a last-resort fallback.

    Falls back gracefully — if the image can't be loaded, returns 0.5 for all
    regions so downstream code never crashes.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return {r: 0.5 for r in _ALL_REGIONS}

    sat      = _saturation(img)
    contrast = _luma_contrast(img)
    white    = _whitespace_fraction(img)

    return {
        # ── Cortical engagement ───────────────────────────────────────────────
        "FFA":  0.5,  # face-selective — not estimable from pixel colour
        "V4":   round(_sigmoid(sat, center=0.4, k=8.0), 4),  # colour saturation ✓
        "MT+":  0.5,  # motion complex — static images are not a reliable proxy
        # ── Subcortical (Harvard-Oxford atlas via API) ────────────────────────
        "Hippocampus": 0.5,  # returned by API; not estimable from pixels
        # ── Trust ─────────────────────────────────────────────────────────────
        "PFC":  round(_sigmoid(0.6 * white + 0.4 * contrast, center=0.3), 4),
        # ── Penalty ───────────────────────────────────────────────────────────
        "ACC":     round(_sigmoid(1.0 - white, center=0.55, k=7.0), 4),  # busyness proxy
        "Amygdala": 0.5,  # subcortical — returned by API
        "Insula":   0.5,  # colour-clash correlation with Insula is unproven
        # ── Dual ──────────────────────────────────────────────────────────────
        "NAcc": round(_sigmoid(sat * contrast, center=0.35, k=8.0), 4),  # rough fallback
    }


_ALL_REGIONS = ("FFA", "V4", "MT+", "Hippocampus", "PFC", "ACC", "Amygdala", "Insula", "NAcc")

REGION_CATEGORIES: dict[str, str] = {
    "FFA":         "engagement",
    "V4":          "engagement",
    "MT+":         "engagement",
    "Hippocampus": "engagement",
    "PFC":         "trust",
    "ACC":         "penalty",
    "Amygdala":    "penalty",
    "Insula":      "penalty",
    "NAcc":        "dual",
}

# Subcortical regions — NOT in the HCP-MMP1 cortical atlas.
# Estimated by the Colab API using Harvard-Oxford subcortical atlas
# (fetch_atlas_harvard_oxford 'sub-maxprob-thr25-2mm').
SUBCORTICAL_REGIONS: frozenset[str] = frozenset({"Hippocampus", "Amygdala", "NAcc"})


# ── Ethical guardrails ────────────────────────────────────────────────────────

DARK_PATTERN_AMYG   = 0.60
DARK_PATTERN_NACC   = 0.70
AROUSAL_CEILING     = 0.85
NACC_ALLOWED_INTENTS = {"gamification"}

# Intent-aware target weights.
# Kept in sync with src/neurolens/reward.py::TARGET_WEIGHTS.
# NAcc is a target ONLY for gamification; it is penalised for all other intents.
INTENT_WEIGHTS: dict[str, dict[str, float]] = {
    "engage":        {"FFA": 1.0, "V4": 0.8, "MT+": 0.7, "Hippocampus": 0.6, "PFC": 0.3},
    "trust":         {"PFC": 1.4, "FFA": 0.4, "V4": 0.2, "Hippocampus": 0.3},
    "convert":       {"FFA": 0.9, "PFC": 0.7, "V4": 0.5, "MT+": 0.4, "Hippocampus": 0.3},
    "accessibility": {"PFC": 1.5, "V4": 0.2},
    "gamification":  {"FFA": 0.5, "V4": 0.7, "MT+": 0.7, "Hippocampus": 0.6, "NAcc": 1.0},
}


def evaluate_ethics(
    scores: dict[str, float],
    intent: str,
    prev_scores: Optional[dict[str, float]] = None,
) -> list[dict]:
    """Return a list of ethics flag dicts — empty means all clear."""
    flags: list[dict] = []

    amyg = scores.get("Amygdala", 0.0)
    nacc = scores.get("NAcc", 0.0)
    pfc  = scores.get("PFC", 0.0)
    ffa  = scores.get("FFA", 0.0)

    if amyg > DARK_PATTERN_AMYG:
        flags.append({
            "code":     "dark_pattern_amygdala",
            "severity": "block",
            "message":  f"Amygdala {amyg:.2f} > {DARK_PATTERN_AMYG} — likely anxiety/urgency dark pattern.",
        })

    if nacc > DARK_PATTERN_NACC and intent not in NACC_ALLOWED_INTENTS:
        flags.append({
            "code":     "dark_pattern_nacc",
            "severity": "block",
            "message":  f"NAcc {nacc:.2f} > {DARK_PATTERN_NACC} — possible compulsive-design trigger.",
        })

    arousal  = max(amyg, scores.get("Insula", 0.0), scores.get("ACC", 0.0))
    positive = (pfc + ffa) / 2.0
    if arousal > 0.7 and positive < 0.3:
        flags.append({
            "code":     "valence_negative",
            "severity": "warn",
            "message":  f"High-arousal-negative state (arousal={arousal:.2f}, positive={positive:.2f}).",
        })

    over = [r for r, s in scores.items() if s > AROUSAL_CEILING]
    if over:
        flags.append({
            "code":     "yerkes_ceiling",
            "severity": "warn",
            "message":  f"Region(s) near overstimulation ceiling: {', '.join(over)}.",
        })

    if prev_scores is not None:
        d_amyg = amyg - prev_scores.get("Amygdala", amyg)
        d_nacc = nacc - prev_scores.get("NAcc", nacc)
        if d_amyg > 0.05:
            flags.append({
                "code":     "amygdala_regression",
                "severity": "warn",
                "message":  f"Amygdala rose {d_amyg:+.2f} — last edit may have added anxiety cues.",
            })
        if d_nacc > 0.05 and intent not in NACC_ALLOWED_INTENTS:
            flags.append({
                "code":     "nacc_regression",
                "severity": "warn",
                "message":  f"NAcc rose {d_nacc:+.2f} outside gamification intent.",
            })

    return flags


def compute_intent_reward(
    scores: dict[str, float],
    prev_scores: dict[str, float],
    intent: str,
) -> float:
    """Return shaped reward incorporating intent weights and penalty regions.

    Aligned with src/neurolens/reward.py::compute():
      - Target regions weighted by INTENT_WEIGHTS (deltas)
      - Amygdala, Insula, ACC always penalised
      - NAcc penalised when intent is not gamification
    """
    weights = INTENT_WEIGHTS.get(intent, INTENT_WEIGHTS["engage"])

    raw = sum(
        w * (scores.get(r, 0.5) - prev_scores.get(r, 0.5))
        for r, w in weights.items()
    )

    # Penalty regions always subtracted
    penalty = (
        1.5 * (scores.get("Amygdala", 0.0) - prev_scores.get("Amygdala", 0.0))
        + 0.8 * (scores.get("Insula",   0.0) - prev_scores.get("Insula",   0.0))
        + 0.5 * (scores.get("ACC",      0.0) - prev_scores.get("ACC",      0.0))
    )

    # NAcc: target only for gamification; penalise for all other intents
    nacc_penalty = 0.0
    if intent not in NACC_ALLOWED_INTENTS:
        nacc_penalty = 1.0 * (scores.get("NAcc", 0.0) - prev_scores.get("NAcc", 0.0))

    return round(raw - penalty - nacc_penalty, 4)
