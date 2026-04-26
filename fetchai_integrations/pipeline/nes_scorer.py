"""
pipeline/nes_scorer.py

Computes the Neural Engagement Score (NES) from ROI activation values.
The NES is a weighted composite that rewards desire/reward circuitry and
penalises high cognitive load.  Valence and arousal dimensions are also
computed to place the page in Russell's circumplex emotion model.
"""

# stdlib
from typing import Dict, List

# third-party  (none required here)


NES_WEIGHTS: Dict[str, float] = {
    "amygdala":    +0.20,   # emotional pull — want HIGH
    "striatum":    +0.25,   # desire/reward  — want HIGH (most important)
    "hippocampus": +0.15,   # memory encoding — want HIGH
    "dlPFC":       -0.20,   # cognitive load  — want LOW (penalise)
    "IPS":         +0.10,   # attention salience — want HIGH
    "mPFC":        +0.10,   # self-relevance  — want HIGH
    "insula":      +0.00,   # context-dependent — neutral
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _get_profile(roi: dict, valence: float, arousal: float) -> str:
    """
    Map ROI activation pattern to a human-readable neuromarketing profile.

    Args:
        roi: Normalised ROI values (0-100 scale).
        valence: Computed valence score (-1 to 1).
        arousal: Computed arousal score (-1 to 1).

    Returns:
        A single descriptive string label.
    """
    if roi["dlPFC"] > 65:
        return "high cognitive load — page is confusing"
    if roi["striatum"] < 35 and roi["amygdala"] < 35:
        return "low desire and emotion — flat and forgettable"
    if roi["insula"] > 65 and roi["dlPFC"] > 60:
        return "distrust and confusion — something feels off"
    if roi["striatum"] > 65 and roi["amygdala"] > 60:
        return "high desire and emotion — strong engagement"
    if roi["hippocampus"] < 35:
        return "low memory encoding — page will be forgotten"
    return "moderate engagement — room for improvement"


def _get_issues(roi: dict) -> List[str]:
    """
    Identify brain-region-specific issues outside healthy engagement ranges.

    Args:
        roi: Normalised ROI values (0-100 scale).

    Returns:
        List of issue strings.  Empty list means no actionable issues found.
    """
    issues: List[str] = []

    if roi["dlPFC"] > 65:
        issues.append(
            f"dlPFC: {roi['dlPFC']}/100 — HIGH cognitive load, simplify the page"
        )
    if roi["striatum"] < 40:
        issues.append(
            f"striatum: {roi['striatum']}/100 — LOW desire, strengthen value proposition"
        )
    if roi["amygdala"] < 40:
        issues.append(
            f"amygdala: {roi['amygdala']}/100 — LOW emotional pull, copy is flat"
        )
    if roi["hippocampus"] < 40:
        issues.append(
            f"hippocampus: {roi['hippocampus']}/100 — LOW memory encoding, add narrative"
        )
    if roi["IPS"] < 35:
        issues.append(
            f"IPS: {roi['IPS']}/100 — CTA not visually salient, reposition it"
        )
    if roi["insula"] > 65:
        issues.append(
            f"insula: {roi['insula']}/100 — HIGH distrust signal, add social proof"
        )

    return issues


def compute_nes(roi_values: dict) -> dict:
    """
    Compute the Neural Engagement Score and associated affective dimensions.

    Args:
        roi_values: Dict returned by extract_roi_values().  May contain the
                    private "_peaks" key — it is stripped internally.

    Returns:
        dict with keys:
            nes_total (float): Composite score 0-100.
            valence (float): Affective valence, -1 to 1.
            arousal (float): Affective arousal, -1 to 1.
            profile (str): Human-readable neuromarketing profile label.
            issues (list[str]): Actionable issue strings.
            roi_values (dict): The input ROI values, minus "_peaks".
    """
    # Strip internal key before scoring
    roi = {k: v for k, v in roi_values.items() if k != "_peaks"}

    # Weighted sum
    raw_nes = sum(roi[region] * weight for region, weight in NES_WEIGHTS.items())
    nes_total = round(_clamp(raw_nes, 0.0, 100.0), 1)

    # Valence: positive emotion minus visceral discomfort
    valence_raw = (
        (roi["striatum"] - 50) * 0.4
        + (roi["amygdala"] - 50) * 0.3
        + (roi["insula"] - 50) * -0.3
    ) / 100
    valence = round(_clamp(valence_raw, -1.0, 1.0), 3)

    # Arousal: emotional intensity modulated by attention and cognitive effort
    arousal_raw = (
        (roi["amygdala"] - 50) * 0.4
        + (roi["dlPFC"] - 50) * 0.3
        + (roi["IPS"] - 50) * 0.3
    ) / 100
    arousal = round(_clamp(arousal_raw, -1.0, 1.0), 3)

    profile = _get_profile(roi, valence, arousal)
    issues = _get_issues(roi)

    return {
        "nes_total": nes_total,
        "valence": valence,
        "arousal": arousal,
        "profile": profile,
        "issues": issues,
        "roi_values": roi,
    }
