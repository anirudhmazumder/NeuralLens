"""Fake implementations for testing without API keys.

TRIBE stub: simulates multimodal brain scoring with realistic variance.
LLM stub: cycles through plausible text edit suggestions.
"""
from __future__ import annotations

import asyncio
import hashlib
import random
from typing import Optional

ACTION_TYPES = [
    "rewrite_headline",
    "rewrite_cta",
    "rewrite_body_paragraph",
    "adjust_meta_description",
    "change_visual_hierarchy",
    "adjust_color_contrast",
    "reorder_sections",
    "simplify_language",
    "add_social_proof",
    "strengthen_value_prop",
]

_FAKE_EDITS = [
    {
        "action_type": "rewrite_headline",
        "target": "main H1 heading",
        "original": "Welcome to Our Platform",
        "replacement": "Turn Visitors into Customers — Starting Today",
        "reasoning": "Direct value proposition activates language ROI; active voice reduces cognitive load.",
        "expected_roi": "language_roi",
    },
    {
        "action_type": "rewrite_cta",
        "target": "primary call-to-action button",
        "original": "Submit",
        "replacement": "Get Started Free →",
        "reasoning": "Specific action language with directional cue fires attention ROI circuits.",
        "expected_roi": "attention_roi",
    },
    {
        "action_type": "strengthen_value_prop",
        "target": "hero subheadline",
        "original": "We help businesses grow online.",
        "replacement": "Join 12,000+ teams who doubled conversions in 30 days.",
        "reasoning": "Social proof + specific outcome engages language and attention regions simultaneously.",
        "expected_roi": "language_roi",
    },
    {
        "action_type": "simplify_language",
        "target": "product description paragraph",
        "original": "Our comprehensive solution facilitates the optimization of your digital presence through synergistic methodologies.",
        "replacement": "We make your website convert more visitors into paying customers.",
        "reasoning": "Reduced cognitive load on language processing ROI improves neural engagement scores.",
        "expected_roi": "language_roi",
    },
    {
        "action_type": "add_social_proof",
        "target": "testimonials section",
        "original": "Trusted by businesses worldwide.",
        "replacement": "Trusted by 50,000+ businesses • 4.9/5 stars • Forbes #1 Growth Tool 2024",
        "reasoning": "Concrete social proof triggers trust encoding in attention ROI circuits.",
        "expected_roi": "attention_roi",
    },
    {
        "action_type": "rewrite_body_paragraph",
        "target": "features description",
        "original": "Our platform offers many features for your needs.",
        "replacement": "Launch in 5 minutes. No code required. Cancel any time.",
        "reasoning": "Scannable benefit-forward copy reduces visual ROI processing load.",
        "expected_roi": "visual_roi",
    },
    {
        "action_type": "adjust_meta_description",
        "target": "page meta description",
        "original": "Company offering digital services.",
        "replacement": "Boost revenue 40% with AI-powered marketing — free 14-day trial, no credit card.",
        "reasoning": "Benefit-led copy with urgency-free offer maximizes language ROI activation.",
        "expected_roi": "language_roi",
    },
    {
        "action_type": "reorder_sections",
        "target": "page content flow",
        "original": "Features → Pricing → Testimonials → Hero",
        "replacement": "Hero → Social Proof → Features → Pricing",
        "reasoning": "Trust-first flow matches natural attention ROI top-down scanning pattern.",
        "expected_roi": "attention_roi",
    },
    {
        "action_type": "change_visual_hierarchy",
        "target": "navigation and hero layout",
        "original": "Dense navigation with 8 items, small hero text",
        "replacement": "Minimal 4-item nav, large bold headline above the fold",
        "reasoning": "Cleaner visual hierarchy reduces visual ROI processing cost.",
        "expected_roi": "visual_roi",
    },
    {
        "action_type": "adjust_color_contrast",
        "target": "CTA button and background",
        "original": "Light gray button on white background",
        "replacement": "High-contrast coral button (#FF6B6B) on white, 4.5:1 contrast ratio",
        "reasoning": "WCAG-compliant contrast with warm hue maximizes visual ROI salience.",
        "expected_roi": "visual_roi",
    },
]


def _text_quality_score(text: str) -> float:
    """Cheap deterministic readability proxy used to bias stub scores."""
    if not text:
        return 0.5
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return 0.5
    # Reward concise text near 150-300 words
    optimal = 200
    verbosity_penalty = min(0.15, abs(word_count - optimal) / optimal * 0.15)
    # Reward shorter average word length (simpler vocabulary)
    avg_word_len = sum(len(w.strip(".,!?;:")) for w in words) / word_count
    complexity_penalty = max(0.0, (avg_word_len - 5.0) * 0.02)
    return max(0.0, min(1.0, 0.7 - verbosity_penalty - complexity_penalty))


async def fake_tribe_score(
    video_path: str,
    text: str,
    audio_path: Optional[str] = None,
) -> dict:
    """Simulate 2-second TRIBE v2 inference with realistic score distribution."""
    await asyncio.sleep(2)

    # Deterministic seed from text content so the same text always scores the same
    text_seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
    rng = random.Random(text_seed)

    base = rng.uniform(0.3, 0.7)
    if video_path:
        base += 0.1
    if audio_path:
        base += 0.05

    # Add small gaussian noise from system random (non-deterministic)
    overall = min(1.0, max(0.0, base + random.gauss(0, 0.02)))

    tq = _text_quality_score(text)

    def _roi(center: float) -> float:
        return round(min(1.0, max(0.0, center + random.gauss(0, 0.04))), 4)

    return {
        "overall_score": round(overall, 4),
        "visual_score": _roi(overall * 0.98),
        "text_score": _roi(tq * 0.6 + overall * 0.4),
        "audio_score": _roi((0.55 if audio_path else 0.30)),
        "language_roi": _roi(tq * 0.55 + overall * 0.45),
        "attention_roi": _roi(overall * 1.05),
        "visual_roi": _roi(overall * 0.93),
    }


async def fake_llm_edit(
    screenshot_base64: str,
    page_content: str,
    score_history: list,
    action_type: str,
) -> dict:
    """Return a realistic edit suggestion after a simulated 1-second delay."""
    await asyncio.sleep(1)
    for edit in _FAKE_EDITS:
        if edit["action_type"] == action_type:
            return dict(edit)
    idx = len(score_history) % len(_FAKE_EDITS)
    return dict(_FAKE_EDITS[idx])
