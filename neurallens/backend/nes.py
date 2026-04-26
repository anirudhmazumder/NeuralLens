"""Neurological Evaluation Scale (NES) loss for TRIBE-guided optimization.

NES measures neurological soft signs (NSS) — subtle, non-localizable indicators
of dysfunction in the cerebello-thalamo-prefrontal cortical circuit.

A HIGH NES score = MORE dysfunction (more soft signs).
We MINIMIZE the NES score each optimization iteration.

NSS indicators mapped to HCP-MMP1 regions:

  Hyper-activation (region exceeds threshold → dysfunction):
    ACC       — motor-sequencing conflicts, task-switching rigidity   threshold 0.40
    Amygdala  — emotional dysregulation, disinhibition               threshold 0.40
    Insula    — sensory integration overload, interoceptive noise     threshold 0.40

  Hypo-activation (region falls below threshold → circuit failure):
    PFC         — prefrontal hypofunctionality, executive circuit     threshold 0.50
    Hippocampus — memory-encoding failure, novelty-gating deficit     threshold 0.40

Content that scores low on NES:
  - Doesn't trigger fear/urgency responses (Amygdala stays quiet)
  - Doesn't overload sensory channels (Insula stays quiet)
  - Doesn't create cognitive conflict (ACC stays quiet)
  - Engages executive attention (PFC elevated)
  - Encodes novelty into memory (Hippocampus elevated)
"""
from __future__ import annotations

# Each entry: region → {direction, threshold, weight}
# direction "above" = dysfunction when region EXCEEDS threshold
# direction "below" = dysfunction when region FALLS BELOW threshold
NSS_INDICATORS: dict[str, dict] = {
    "ACC":         {"direction": "above", "threshold": 0.40, "weight": 2.0},
    "Amygdala":    {"direction": "above", "threshold": 0.40, "weight": 2.0},
    "Insula":      {"direction": "above", "threshold": 0.40, "weight": 1.5},
    "PFC":         {"direction": "below", "threshold": 0.50, "weight": 1.5},
    "Hippocampus": {"direction": "below", "threshold": 0.40, "weight": 1.0},
}

# Engagement regions — tracked alongside NES for context, not part of the loss
ENGAGEMENT_REGIONS = ("FFA", "V4", "MT+", "NAcc")


def nes_score(regions: dict[str, float]) -> float:
    """Compute NES dysfunction score (minimize this; lower = fewer soft signs)."""
    total = 0.0
    for region, cfg in NSS_INDICATORS.items():
        val = regions.get(region, 0.5)
        if cfg["direction"] == "above":
            total += cfg["weight"] * max(0.0, val - cfg["threshold"])
        else:
            total += cfg["weight"] * max(0.0, cfg["threshold"] - val)
    return round(total, 4)


def nes_reward(
    before: dict[str, float],
    after: dict[str, float],
    intent: str = "engage",  # kept for API compat; NES is intent-independent
) -> float:
    """Reduction in NES dysfunction score (positive = improvement, fewer soft signs)."""
    return round(nes_score(before) - nes_score(after), 4)


def nes_summary(regions: dict[str, float], intent: str = "engage") -> dict:
    """Return human-readable NES breakdown for SSE events and agent context."""
    score = nes_score(regions)

    # Per-indicator contributions
    violations: dict[str, float] = {}
    for region, cfg in NSS_INDICATORS.items():
        val = regions.get(region, 0.5)
        if cfg["direction"] == "above":
            delta = max(0.0, val - cfg["threshold"])
        else:
            delta = max(0.0, cfg["threshold"] - val)
        if delta > 0.0:
            violations[region] = round(delta * cfg["weight"], 4)

    engagement = round(
        sum(max(0.0, regions.get(r, 0.0) - 0.3) for r in ENGAGEMENT_REGIONS), 4
    )
    penalty = round(
        sum(max(0.0, regions.get(r, 0.0) - 0.4) for r in ("Amygdala", "ACC", "Insula")), 4
    )

    dominant_penalty = (
        max(violations, key=violations.get) if violations else "none"
    )
    dominant_positive = max(
        ENGAGEMENT_REGIONS,
        key=lambda r: regions.get(r, 0.0),
        default="FFA",
    )

    return {
        "nes_loss":          score,   # key kept as nes_loss for frontend compat
        "nes_score":         score,
        "engagement_score":  engagement,
        "penalty_score":     penalty,
        "intent":            intent,
        "dominant_penalty":  dominant_penalty,
        "dominant_positive": dominant_positive,
        "violations":        violations,
    }
