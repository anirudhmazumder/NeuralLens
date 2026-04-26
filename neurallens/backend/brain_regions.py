"""Local brain-region activation estimator.

Ports the StubEncoder from src/neurolens/tribe.py into the backend so we can
score all 9 HCP-MMP1 regions directly from a screenshot — without the full
neurolens library import or nilearn dependency.

Also ports the ethical-guardrail evaluator from src/neurolens/ethics.py.

Outputs are on [0, 1]. Same image → same scores (deterministic).

Regions and their neuroscience categories:
  engagement  — FFA, V4, MT+, Hippocampus
  trust       — PFC
  penalty     — ACC, Amygdala, Insula
  dual        — NAcc  (reward; acceptable in gamification intent)

Key trio for the UI highlight:
  Amygdala  — dark-pattern threat detector (anxiety / urgency triggers)
  Hippocampus — novelty / distinctiveness encoding
  NAcc      — reward anticipation signal
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFilter, ImageStat

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


def _skin_fraction(img: Image.Image) -> float:
    small = img.convert("HSV").resize((64, 64))
    pixels = small.load()
    hits = 0
    for y in range(64):
        for x in range(64):
            h, s, v = pixels[x, y]
            if 0 <= h <= 30 and 50 <= s <= 200 and 80 <= v <= 230:
                hits += 1
    return hits / (64 * 64)


def _edge_diagonality(img: Image.Image) -> float:
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES).resize((64, 64))
    px = edges.load()
    diag = axis = 0
    for y in range(1, 63):
        for x in range(1, 63):
            c = px[x, y]
            if c < 40:
                continue
            if abs(px[x, y - 1] - px[x, y + 1]) > abs(px[x - 1, y - 1] - px[x + 1, y + 1]):
                axis += 1
            else:
                diag += 1
    if diag + axis == 0:
        return 0.0
    return diag / (diag + axis)


def _whitespace_fraction(img: Image.Image) -> float:
    g = img.convert("L").resize((64, 64))
    px = g.load()
    light = sum(1 for y in range(64) for x in range(64) if px[x, y] > 230)
    return light / (64 * 64)


def _palette_clash(img: Image.Image) -> float:
    hsv = img.convert("HSV").resize((64, 64))
    h_band, s_band, _ = hsv.split()
    s_mean = ImageStat.Stat(s_band).mean[0] / 255.0
    hue_var = ImageStat.Stat(h_band).stddev[0] / 128.0
    return min(1.0, s_mean * hue_var * 1.6)


def _novelty_hash(img: Image.Image) -> float:
    small = img.convert("RGB").resize((16, 16))
    h = hashlib.sha256(small.tobytes()).digest()
    n = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
    return 0.3 + 0.6 * n


def _red_dominance(img: Image.Image) -> float:
    rgb = img.convert("RGB").resize((64, 64))
    px = rgb.load()
    hits = 0
    for y in range(64):
        for x in range(64):
            r, g, b = px[x, y]
            if r > 160 and r > g + 40 and r > b + 40:
                hits += 1
    return hits / (64 * 64)


# ── Encoder ───────────────────────────────────────────────────────────────────

def score_screenshot(path: str) -> dict[str, float]:
    """Return all 9 HCP-MMP1 region activations for the screenshot at *path*.

    Falls back gracefully — if the image can't be loaded, returns 0.5 for all
    regions so downstream code never crashes.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return {r: 0.5 for r in _ALL_REGIONS}

    sat = _saturation(img)
    contrast = _luma_contrast(img)
    skin = _skin_fraction(img)
    diag = _edge_diagonality(img)
    white = _whitespace_fraction(img)
    clash = _palette_clash(img)
    novelty = _novelty_hash(img)
    red = _red_dominance(img)

    return {
        # engagement
        "FFA": round(_sigmoid(skin * 6.0, center=0.5), 4),
        "V4": round(_sigmoid(sat, center=0.4, k=8.0), 4),
        "MT+": round(_sigmoid(diag, center=0.45, k=8.0), 4),
        "Hippocampus": round(novelty, 4),
        # trust
        "PFC": round(_sigmoid(0.6 * white + 0.4 * contrast - 0.5 * clash, center=0.3), 4),
        # penalty
        "ACC": round(_sigmoid(diag * (1.0 - white), center=0.45, k=7.0), 4),
        "Amygdala": round(_sigmoid(0.6 * red + 0.4 * (1.0 - white), center=0.55, k=8.0), 4),
        "Insula": round(_sigmoid(clash, center=0.4, k=8.0), 4),
        # dual
        "NAcc": round(_sigmoid(sat * contrast, center=0.35, k=8.0), 4),
    }


_ALL_REGIONS = ("FFA", "V4", "MT+", "Hippocampus", "PFC", "ACC", "Amygdala", "Insula", "NAcc")

REGION_CATEGORIES: dict[str, str] = {
    "FFA": "engagement",
    "V4": "engagement",
    "MT+": "engagement",
    "Hippocampus": "engagement",
    "PFC": "trust",
    "ACC": "penalty",
    "Amygdala": "penalty",
    "Insula": "penalty",
    "NAcc": "dual",
}

# ── Ethical guardrails ────────────────────────────────────────────────────────

DARK_PATTERN_AMYG = 0.60
DARK_PATTERN_NACC = 0.70
AROUSAL_CEILING = 0.85
NACC_ALLOWED_INTENTS = {"gamification"}

# Intent-aware reward weights: (engagement, trust, penalty_scale)
INTENT_WEIGHTS: dict[str, dict[str, float]] = {
    "engage":          {"FFA": 1.0, "V4": 0.8, "MT+": 0.7, "Hippocampus": 1.0, "PFC": 0.5, "NAcc": 0.6},
    "trust":           {"PFC": 1.5, "FFA": 0.5, "Hippocampus": 0.7, "NAcc": 0.3},
    "convert":         {"NAcc": 1.2, "FFA": 0.8, "V4": 0.6, "PFC": 0.6},
    "accessibility":   {"PFC": 1.2, "ACC": -1.5, "Insula": -1.5},
    "gamification":    {"NAcc": 1.5, "MT+": 0.8, "FFA": 0.7},
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
    pfc = scores.get("PFC", 0.0)
    ffa = scores.get("FFA", 0.0)

    if amyg > DARK_PATTERN_AMYG:
        flags.append({
            "code": "dark_pattern_amygdala",
            "severity": "block",
            "message": f"Amygdala {amyg:.2f} > {DARK_PATTERN_AMYG} — likely anxiety/urgency dark pattern.",
        })

    if nacc > DARK_PATTERN_NACC and intent not in NACC_ALLOWED_INTENTS:
        flags.append({
            "code": "dark_pattern_nacc",
            "severity": "block",
            "message": f"NAcc {nacc:.2f} > {DARK_PATTERN_NACC} — possible compulsive-design trigger.",
        })

    arousal = max(amyg, scores.get("Insula", 0.0), scores.get("ACC", 0.0))
    positive = (pfc + ffa) / 2.0
    if arousal > 0.7 and positive < 0.3:
        flags.append({
            "code": "valence_negative",
            "severity": "warn",
            "message": f"High-arousal-negative state (arousal={arousal:.2f}, positive={positive:.2f}).",
        })

    over = [r for r, s in scores.items() if s > AROUSAL_CEILING]
    if over:
        flags.append({
            "code": "yerkes_ceiling",
            "severity": "warn",
            "message": f"Region(s) near overstimulation ceiling: {', '.join(over)}.",
        })

    if prev_scores is not None:
        d_amyg = amyg - prev_scores.get("Amygdala", amyg)
        d_nacc = nacc - prev_scores.get("NAcc", nacc)
        if d_amyg > 0.05:
            flags.append({
                "code": "amygdala_regression",
                "severity": "warn",
                "message": f"Amygdala rose {d_amyg:+.2f} — last edit may have added anxiety cues.",
            })
        if d_nacc > 0.05 and intent not in NACC_ALLOWED_INTENTS:
            flags.append({
                "code": "nacc_regression",
                "severity": "warn",
                "message": f"NAcc rose {d_nacc:+.2f} outside gamification intent.",
            })

    return flags


def compute_intent_reward(
    scores: dict[str, float],
    prev_scores: dict[str, float],
    intent: str,
) -> float:
    """Return shaped reward incorporating intent weights and penalty regions."""
    weights = INTENT_WEIGHTS.get(intent, INTENT_WEIGHTS["engage"])

    raw = sum(
        w * (scores.get(r, 0.5) - prev_scores.get(r, 0.5))
        for r, w in weights.items()
    )
    # Penalty regions always subtracted
    penalty = (
        1.5 * (scores.get("Amygdala", 0) - prev_scores.get("Amygdala", 0))
        + 0.8 * (scores.get("Insula", 0) - prev_scores.get("Insula", 0))
        + 0.5 * (scores.get("ACC", 0) - prev_scores.get("ACC", 0))
    )
    return round(raw - penalty, 4)
