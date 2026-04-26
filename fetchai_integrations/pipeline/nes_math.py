"""
Pure Python NES (Neural Engagement Score) computation.
No agents, no APIs. Just math.
Imported by interpreter_agent.py.
"""

import numpy as np
from typing import Dict, List, Tuple


# Approximate voxel index ranges for each brain region
# in TRIBE v2's output space.
# IMPORTANT: Verify these against the TRIBE v2 paper
# appendix / HuggingFace model card before hackathon.
# These are best estimates based on standard brain atlases.
ROI_INDICES: Dict[str, Tuple[int, int]] = {
    "amygdala":    (1200, 1450),   # bilateral amygdala
                                    # emotional salience
    "striatum":    (2100, 2380),   # ventral striatum
                                    # reward / desire
    "hippocampus": (3400, 3700),   # bilateral hippocampus
                                    # memory encoding
    "dlPFC":       (8200, 8600),   # dorsolateral PFC
                                    # cognitive load (want LOW)
    "IPS":         (5100, 5400),   # intraparietal sulcus
                                    # visual attention salience
    "mPFC":        (7800, 8100),   # medial PFC
                                    # self-relevance / social
    "insula":      (4200, 4500),   # bilateral insula
                                    # trust / gut feeling
}

# Weights for NES formula
# Positive = want high, Negative = want low (penalize)
NES_WEIGHTS: Dict[str, float] = {
    "amygdala":    +0.20,   # emotional pull — want HIGH
    "striatum":    +0.25,   # desire/reward — want HIGH
                             # most important predictor
    "hippocampus": +0.15,   # memory encoding — want HIGH
    "dlPFC":       -0.20,   # cognitive load — want LOW
    "IPS":         +0.10,   # attention salience — want HIGH
    "mPFC":        +0.10,   # self-relevance — want HIGH
    "insula":      +0.00,   # trust — context dependent
}


def extract_roi_values(activations: np.ndarray) -> Dict[str, float]:
    """
    Takes full 70k activation array from TRIBE v2.
    Returns mean activation for each of 7 key brain regions.
    All values normalized to 0-100 scale.

    Args:
        activations: numpy array of shape (N,) from TRIBE v2
    Returns:
        dict mapping region name to normalized float 0-100
    """
    raw_values = {}

    for region, (start, end) in ROI_INDICES.items():
        if end <= len(activations):
            region_slice = activations[start:end]
            raw_values[region] = float(np.mean(region_slice))
        else:
            # fallback if activation array shorter than expected
            # use available data or zero
            available = activations[start:] if start < len(activations) else np.array([0.0])
            raw_values[region] = float(np.mean(available))

    # normalize all 7 values to 0-100 scale
    all_vals = list(raw_values.values())
    min_val = min(all_vals)
    max_val = max(all_vals)
    val_range = max_val - min_val

    normalized = {}
    for region, raw in raw_values.items():
        if val_range > 1e-8:
            normalized[region] = round(
                ((raw - min_val) / val_range) * 100, 1
            )
        else:
            normalized[region] = 50.0  # all equal → midpoint

    return normalized


def compute_nes(roi_values: Dict[str, float]) -> Dict:
    """
    Takes 7 normalized ROI values (0-100 each).
    Returns NES total, valence, arousal, profile, issues.

    Args:
        roi_values: dict from extract_roi_values()
    Returns:
        dict with nes_total, valence, arousal, profile, issues
    """
    # weighted sum
    raw_score = sum(
        roi_values.get(region, 50.0) * weight
        for region, weight in NES_WEIGHTS.items()
    )

    # clamp to 0-100
    nes_total = round(max(0.0, min(100.0, raw_score)), 1)

    # valence: how positive/negative the emotional response is
    # range: -1.0 (very negative) to +1.0 (very positive)
    valence = (
        (roi_values.get("striatum", 50) - 50) * 0.4
        + (roi_values.get("amygdala", 50) - 50) * 0.3
        + (roi_values.get("insula", 50) - 50) * -0.3
    ) / 100
    valence = round(max(-1.0, min(1.0, valence)), 3)

    # arousal: how activating/calm the response is
    # range: -1.0 (calm) to +1.0 (highly aroused)
    arousal = (
        (roi_values.get("amygdala", 50) - 50) * 0.4
        + (roi_values.get("dlPFC", 50) - 50) * 0.3
        + (roi_values.get("IPS", 50) - 50) * 0.3
    ) / 100
    arousal = round(max(-1.0, min(1.0, arousal)), 3)

    profile = _get_profile(roi_values, valence, arousal)
    issues = _get_issues(roi_values)

    return {
        "nes_total": nes_total,
        "valence": valence,
        "arousal": arousal,
        "profile": profile,
        "issues": issues,
        "roi_values": roi_values,
    }


def _get_profile(roi: Dict, valence: float = 0.0, arousal: float = 0.0) -> str:
    """Returns human-readable emotion profile string."""
    dlPFC = roi.get("dlPFC", 50)
    striatum = roi.get("striatum", 50)
    amygdala = roi.get("amygdala", 50)
    insula = roi.get("insula", 50)
    hippocampus = roi.get("hippocampus", 50)

    if dlPFC > 65:
        return "high cognitive load — page is confusing"
    if striatum < 35 and amygdala < 35:
        return "low desire and emotion — flat and forgettable"
    if insula > 65 and dlPFC > 60:
        return "distrust and confusion — something feels off"
    if striatum > 65 and amygdala > 60:
        return "high desire and emotion — strong engagement"
    if hippocampus < 35:
        return "low memory encoding — will be forgotten"
    return "moderate engagement — room for improvement"


def _get_issues(roi: Dict) -> List[str]:
    """Returns list of specific issue strings for flagged regions."""
    issues = []

    if roi.get("dlPFC", 50) > 65:
        issues.append(
            f"dlPFC {roi['dlPFC']}/100 — HIGH cognitive load, "
            f"simplify the layout"
        )
    if roi.get("striatum", 50) < 40:
        issues.append(
            f"striatum {roi['striatum']}/100 — LOW desire, "
            f"strengthen value proposition"
        )
    if roi.get("amygdala", 50) < 40:
        issues.append(
            f"amygdala {roi['amygdala']}/100 — LOW emotional pull, "
            f"copy is flat"
        )
    if roi.get("hippocampus", 50) < 40:
        issues.append(
            f"hippocampus {roi['hippocampus']}/100 — LOW memory "
            f"encoding, add narrative or specificity"
        )
    if roi.get("IPS", 50) < 35:
        issues.append(
            f"IPS {roi['IPS']}/100 — CTA not visually salient, "
            f"reposition or recolor"
        )
    if roi.get("insula", 50) > 65:
        issues.append(
            f"insula {roi['insula']}/100 — HIGH distrust signal, "
            f"add social proof or testimonials"
        )

    return issues


def analyze_intersection(
    activations: np.ndarray,
    deepgaze: list,
) -> List[Dict]:
    """
    Intersects TRIBE v2 brain activation with DeepGaze saliency.
    Divides into 9 zones (3x3 grid).
    Classifies each zone as:
      power_zone:      high gaze + high brain → protect this
      attention_trap:  high gaze + low brain  → fix this
      hidden_value:    low gaze  + high brain → elevate this
      dead_zone:       low gaze  + low brain  → remove this

    Args:
        activations: TRIBE v2 activation array
        deepgaze: DeepGaze saliency map as flat list
    Returns:
        list of dicts with zone analysis
    """
    insights = []

    zones = [
        "top-left",    "top-center",    "top-right",
        "middle-left", "center",        "middle-right",
        "bottom-left", "bottom-center", "bottom-right",
    ]

    # work with what we have — normalize both arrays
    tribe_arr = np.array(activations[: min(900, len(activations))])
    gaze_arr = (
        np.array(deepgaze[: min(900, len(deepgaze))])
        if deepgaze
        else np.zeros(900)
    )

    # pad to 900 if shorter
    if len(tribe_arr) < 900:
        tribe_arr = np.pad(tribe_arr, (0, 900 - len(tribe_arr)))
    if len(gaze_arr) < 900:
        gaze_arr = np.pad(gaze_arr, (0, 900 - len(gaze_arr)))

    def norm100(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.full_like(arr, 50.0)
        return ((arr - mn) / (mx - mn)) * 100

    t_norm = norm100(tribe_arr)
    g_norm = norm100(gaze_arr)

    chunk = 100  # 900 / 9 zones

    for i, zone in enumerate(zones):
        t_score = float(np.mean(t_norm[i * chunk : (i + 1) * chunk]))
        g_score = float(np.mean(g_norm[i * chunk : (i + 1) * chunk]))

        if g_score > 60 and t_score < 40:
            zone_type = "attention_trap"
            meaning = (
                "Eyes land here but brain feels nothing. "
                "Stealing attention without creating desire."
            )
            action = "enrich emotionally or remove this element"

        elif t_score > 60 and g_score < 40:
            zone_type = "hidden_value"
            meaning = (
                "Strong brain response here but eyes never land. "
                "Your best asset is invisible."
            )
            action = "make this zone visually dominant"

        elif t_score > 60 and g_score > 60:
            zone_type = "power_zone"
            meaning = (
                "Eyes go here AND brain responds strongly. "
                "This is your strongest asset."
            )
            action = "protect and amplify this element"

        else:
            zone_type = "dead_zone"
            meaning = (
                "Nobody looks here and brain feels nothing. "
                "Wasted space."
            )
            action = "remove or replace with stronger content"

        insights.append({
            "zone":        zone,
            "type":        zone_type,
            "meaning":     meaning,
            "action":      action,
            "tribe_score": round(t_score, 1),
            "gaze_score":  round(g_score, 1),
        })

    return insights
