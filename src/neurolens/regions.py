"""Brain region registry.

Each region carries the metadata needed by the reward function, the agent
prompt, and the transparency report. Keeping this in one place means
adding/tuning a region is a single-file edit.

The `glasser_names` field lists the HCP-MMP1 (Glasser 2016) region names that
make up each ROI, expressed without hemisphere prefix. Both hemispheres are
included automatically - for `FFC`, the atlas contains `L_FFC_ROI` and
`R_FFC_ROI` and we average across both.

HCP-MMP1 is cortical-only. Regions marked `cortical=False` (Hippocampus,
Amygdala, NAcc) are subcortical and require a separate atlas (Harvard-Oxford
subcortical or AAL). Until that is wired up, those regions remain
stub-encoder-only and should be flagged in reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Category = Literal["engagement", "trust", "penalty", "dual", "language"]


@dataclass(frozen=True)
class Region:
    name: str
    category: Category
    glasser_names: tuple[str, ...]  # HCP-MMP1 names, hemisphere-agnostic
    drives: str
    agent_hint: str
    full_name: str = ""
    cortical: bool = True  # False = needs subcortical atlas (not in HCP-MMP1)


REGIONS: dict[str, Region] = {
    "FFA": Region(
        name="FFA",
        full_name="Fusiform Face Area (HCP: FFC)",
        category="engagement",
        glasser_names=("FFC",),
        drives="Human faces, eye contact, emotional expressions",
        agent_hint=(
            "FFA score is {score:.2f}. To increase it, consider adding a human face, "
            "enlarging an existing face, or ensuring faces make direct eye contact "
            "with the viewer."
        ),
    ),
    "V4": Region(
        name="V4",
        full_name="Color-Selective Cortex (HCP: V4)",
        category="engagement",
        glasser_names=("V4",),
        drives="Color saturation, hue contrast, bold complementary palettes",
        agent_hint=(
            "V4 score is {score:.2f}. To increase it, consider raising saturation of "
            "primary UI colors, introducing a bold accent color, or adding a "
            "colorful hero illustration."
        ),
    ),
    "MT+": Region(
        name="MT+",
        full_name="Motion Complex (HCP: MT, MST, V4t, FST)",
        category="engagement",
        glasser_names=("MT", "MST", "V4t", "FST"),
        drives="Implied motion: arrows, diagonal layouts, blur trails, action poses",
        agent_hint=(
            "MT+ score is {score:.2f}. To increase it, add directional cues "
            "(arrows, diagonal lines), use action-oriented photography, or apply "
            "a subtle blur trail to a CTA element."
        ),
    ),
    "Hippocampus": Region(
        name="Hippocampus",
        full_name="Hippocampus (subcortical - separate atlas needed)",
        category="engagement",
        glasser_names=(),
        drives="Novelty, distinctiveness, memorable / unexpected design choices",
        agent_hint=(
            "Hippocampus score is {score:.2f}. Suggest one distinctive element that "
            "differentiates this UI from generic templates."
        ),
        cortical=False,
    ),
    "PFC": Region(
        name="PFC",
        full_name="Dorsolateral Prefrontal Cortex (HCP: 46, 9-46d, p9-46v, a9-46v)",
        category="trust",
        glasser_names=("46", "9-46d", "p9-46v", "a9-46v"),
        drives="Clear hierarchy, whitespace, readable typography, calm layout",
        agent_hint=(
            "PFC score is {score:.2f}. To increase it, reduce visual clutter, improve "
            "typography readability, or clarify the hierarchy so the most important "
            "action is immediately obvious."
        ),
    ),
    "ACC": Region(
        name="ACC",
        full_name="Anterior Cingulate (HCP: a24, p24, p32, s32, 24dd, 24dv)",
        category="penalty",
        glasser_names=("a24", "p24", "p32", "s32", "24dd", "24dv"),
        drives="Confusion: too many CTAs, ambiguous labels, contradictory signals",
        agent_hint=(
            "ACC score is {score:.2f} (high = bad). Reduce it by clarifying the "
            "primary action, removing competing CTAs, or improving label clarity."
        ),
    ),
    "Amygdala": Region(
        name="Amygdala",
        full_name="Amygdala (subcortical - separate atlas needed)",
        category="penalty",
        glasser_names=(),
        drives="Threat/anxiety: red urgency, countdowns, scarcity, FOMO copy",
        agent_hint=(
            "Amygdala score is {score:.2f} (high = bad / dark-pattern risk). "
            "Reduce urgency-coded colors, remove countdowns, soften scarcity copy."
        ),
        cortical=False,
    ),
    "Insula": Region(
        name="Insula",
        full_name="Insular Cortex (HCP: AAIC, AVI, MI, PI)",
        category="penalty",
        glasser_names=("AAIC", "AVI", "MI", "PI"),
        drives="Visceral unease: clashing colors, low contrast, visual clutter",
        agent_hint=(
            "Insula score is {score:.2f} (high = bad). Improve color harmony, raise "
            "text contrast, add whitespace, or replace low-quality images."
        ),
    ),
    "NAcc": Region(
        name="NAcc",
        full_name="Nucleus Accumbens (subcortical - separate atlas needed)",
        category="dual",
        glasser_names=(),
        drives="Reward anticipation: variable rewards, badges, streaks, near-completion",
        agent_hint=(
            "NAcc score is {score:.2f}. Treated as a target for gamification intent, "
            "as a penalty otherwise (compulsion risk)."
        ),
        cortical=False,
    ),
}


def by_category(cat: Category) -> list[Region]:
    return [r for r in REGIONS.values() if r.category == cat]


def all_names() -> list[str]:
    return list(REGIONS.keys())


def cortical_names() -> list[str]:
    return [n for n, r in REGIONS.items() if r.cortical]


def subcortical_names() -> list[str]:
    return [n for n, r in REGIONS.items() if not r.cortical]
