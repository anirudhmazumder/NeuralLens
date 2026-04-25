"""Brain region registry.

Each region carries the metadata needed by the reward function, the agent
prompt, and the transparency report. Keeping this in one place means
adding/tuning a region is a single-file edit.

HCP indices are placeholders — the real HCP-MMP1 parcellation has 360 regions
indexed 1..360 (left + right hemisphere). When the real atlas loads, fill these
in from Glasser et al. (2016) supplement Table S2. Until then, the stub encoder
generates synthetic activations keyed by region name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Category = Literal["engagement", "trust", "penalty", "dual", "language"]


@dataclass(frozen=True)
class Region:
    name: str
    category: Category
    hcp_indices: tuple[int, ...]
    drives: str
    agent_hint: str
    full_name: str = ""


REGIONS: dict[str, Region] = {
    "FFA": Region(
        name="FFA",
        full_name="Fusiform Face Area",
        category="engagement",
        hcp_indices=(18, 22, 198, 202),
        drives="Human faces, eye contact, emotional expressions",
        agent_hint=(
            "FFA score is {score:.2f}. To increase it, consider adding a human face, "
            "enlarging an existing face, or ensuring faces make direct eye contact "
            "with the viewer."
        ),
    ),
    "V4": Region(
        name="V4",
        full_name="Color-Selective Cortex (V4)",
        category="engagement",
        hcp_indices=(6, 7, 186, 187),
        drives="Color saturation, hue contrast, bold complementary palettes",
        agent_hint=(
            "V4 score is {score:.2f}. To increase it, consider raising saturation of "
            "primary UI colors, introducing a bold accent color, or adding a "
            "colorful hero illustration."
        ),
    ),
    "MT+": Region(
        name="MT+",
        full_name="Motion Area (MT+/V5)",
        category="engagement",
        hcp_indices=(2, 23, 182, 203),
        drives="Implied motion: arrows, diagonal layouts, blur trails, action poses",
        agent_hint=(
            "MT+ score is {score:.2f}. To increase it, add directional cues "
            "(arrows, diagonal lines), use action-oriented photography, or apply "
            "a subtle blur trail to a CTA element."
        ),
    ),
    "Hippocampus": Region(
        name="Hippocampus",
        full_name="Hippocampus",
        category="engagement",
        hcp_indices=(120, 300),  # subcortical; placeholder — real HCP focuses on cortex
        drives="Novelty, distinctiveness, memorable / unexpected design choices",
        agent_hint=(
            "Hippocampus score is {score:.2f}. Suggest one distinctive element that "
            "differentiates this UI from generic templates — a unique illustration, "
            "an unexpected layout choice, or a memorable brand color application."
        ),
    ),
    "PFC": Region(
        name="PFC",
        full_name="Dorsolateral Prefrontal Cortex (DLPFC)",
        category="trust",
        hcp_indices=(70, 71, 250, 251),
        drives="Clear hierarchy, whitespace, readable typography, calm layout",
        agent_hint=(
            "PFC score is {score:.2f}. To increase it, reduce visual clutter, improve "
            "typography readability, or clarify the hierarchy so the most important "
            "action is immediately obvious."
        ),
    ),
    "ACC": Region(
        name="ACC",
        full_name="Anterior Cingulate Cortex",
        category="penalty",
        hcp_indices=(60, 240),
        drives="Confusion: too many CTAs, ambiguous labels, contradictory signals",
        agent_hint=(
            "ACC score is {score:.2f} (high = bad). Reduce it by clarifying the "
            "primary action, removing competing CTAs, or improving label clarity."
        ),
    ),
    "Amygdala": Region(
        name="Amygdala",
        full_name="Amygdala",
        category="penalty",
        hcp_indices=(150, 330),  # subcortical placeholder
        drives="Threat/anxiety: red urgency, countdowns, scarcity, FOMO copy",
        agent_hint=(
            "Amygdala score is {score:.2f} (high = bad / dark-pattern risk). "
            "Reduce urgency-coded colors, remove countdowns, soften scarcity copy."
        ),
    ),
    "Insula": Region(
        name="Insula",
        full_name="Insular Cortex",
        category="penalty",
        hcp_indices=(105, 285),
        drives="Visceral unease: clashing colors, low contrast, visual clutter",
        agent_hint=(
            "Insula score is {score:.2f} (high = bad). Improve color harmony, raise "
            "text contrast, add whitespace, or replace low-quality images."
        ),
    ),
    "NAcc": Region(
        name="NAcc",
        full_name="Nucleus Accumbens",
        category="dual",
        hcp_indices=(140, 320),
        drives="Reward anticipation: variable rewards, badges, streaks, near-completion",
        agent_hint=(
            "NAcc score is {score:.2f}. Treated as a target for gamification intent, "
            "as a penalty otherwise (compulsion risk)."
        ),
    ),
}


def by_category(cat: Category) -> list[Region]:
    return [r for r in REGIONS.values() if r.category == cat]


def all_names() -> list[str]:
    return list(REGIONS.keys())
