"""Reward function: region scores -> single scalar.

Intent-aware. The designer declares an intent at run start; that intent picks
which regions are targets, which are penalties, and what weights to use.
NAcc is dual-use — target only under `gamification`, penalty otherwise.

Rules baked in (from the project brief):

    total -= 1.5 * Amygdala
    total -= 0.8 * Insula
    total -= 0.5 * ACC
    total -= 1.0 * NAcc        (when intent is not gamification)
    total -=   penalty if any region exceeds Yerkes-Dodson ceiling (0.85)

Targets contribute with intent-specific weights. Trust mode prizes PFC over
raw engagement; engage mode does the opposite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Intent = Literal["engage", "trust", "convert", "accessibility", "gamification"]

YERKES_CEILING = 0.85
YERKES_PENALTY = 0.4  # multiplier for graded over-stimulation penalty

PENALTY_WEIGHTS: dict[str, float] = {
    "Amygdala": 1.5,
    "Insula": 0.8,
    "ACC": 0.5,
}

# Per-intent target weights. Anything not listed contributes zero on the
# target side. NAcc handled separately because it's dual-use.
TARGET_WEIGHTS: dict[Intent, dict[str, float]] = {
    "engage":        {"FFA": 1.0, "V4": 0.8, "MT+": 0.7, "Hippocampus": 0.6, "PFC": 0.3},
    "trust":         {"PFC": 1.4, "FFA": 0.4, "V4": 0.2, "Hippocampus": 0.3},
    "convert":       {"FFA": 0.9, "PFC": 0.7, "V4": 0.5, "MT+": 0.4, "Hippocampus": 0.3},
    "accessibility": {"PFC": 1.5, "V4": 0.2},  # readability + low arousal
    "gamification":  {"FFA": 0.5, "V4": 0.7, "MT+": 0.7, "Hippocampus": 0.6, "NAcc": 1.0},
}


@dataclass
class RewardBreakdown:
    intent: Intent
    total: float
    targets: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    yerkes_violations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"intent={self.intent}  total={self.total:+.3f}"]
        if self.targets:
            t = ", ".join(f"{k}={v:+.2f}" for k, v in self.targets.items())
            lines.append(f"  targets:   {t}")
        if self.penalties:
            p = ", ".join(f"{k}={v:+.2f}" for k, v in self.penalties.items())
            lines.append(f"  penalties: {p}")
        if self.yerkes_violations:
            lines.append(f"  yerkes:    over-stimulation in {', '.join(self.yerkes_violations)}")
        return "\n".join(lines)


def compute(scores: dict[str, float], intent: Intent) -> RewardBreakdown:
    """Compute the reward breakdown for one set of region scores."""
    if intent not in TARGET_WEIGHTS:
        raise ValueError(f"unknown intent: {intent!r}")

    targets: dict[str, float] = {}
    for region, weight in TARGET_WEIGHTS[intent].items():
        s = scores.get(region, 0.0)
        targets[region] = weight * s

    penalties: dict[str, float] = {}
    for region, weight in PENALTY_WEIGHTS.items():
        s = scores.get(region, 0.0)
        penalties[region] = -weight * s

    # NAcc dual-signal: only penalty when not gamification (already a target above)
    if intent != "gamification":
        penalties["NAcc"] = -1.0 * scores.get("NAcc", 0.0)

    # Graded Yerkes-Dodson penalty: grows with distance above the ceiling,
    # not a flat step. A region barely over (0.86) costs much less than one at
    # 0.99. Formula: -YERKES_PENALTY * (excess)^1.5 per region.
    yerkes: list[str] = [r for r, s in scores.items() if s > YERKES_CEILING]
    yerkes_pen = -sum(
        YERKES_PENALTY * ((scores[r] - YERKES_CEILING) ** 1.5)
        for r in yerkes
    )

    total = sum(targets.values()) + sum(penalties.values()) + yerkes_pen

    return RewardBreakdown(
        intent=intent,
        total=total,
        targets=targets,
        penalties=penalties,
        yerkes_violations=yerkes,
    )
