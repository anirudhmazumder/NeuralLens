"""
pipeline/roi_extractor.py

Maps raw TRIBE v2 voxel activations onto 7 neurologically meaningful
Regions of Interest (ROIs) relevant to consumer decision-making.

NOTE: Voxel index ranges below are approximate and should be verified
against the TRIBE v2 paper appendix and the MNI brain atlas it references.
"""

# stdlib
from typing import Dict, Tuple

# third-party
import numpy as np


# Approximate voxel index ranges for each ROI.
# These MUST be verified against the TRIBE v2 paper appendix / brain atlas.
ROI_INDICES: Dict[str, Tuple[int, int]] = {
    "amygdala":    (1200, 1450),   # bilateral amygdala
    "striatum":    (2100, 2380),   # ventral striatum / nucleus accumbens
    "hippocampus": (3400, 3700),   # bilateral hippocampus
    "dlPFC":       (8200, 8600),   # dorsolateral prefrontal cortex
    "IPS":         (5100, 5400),   # intraparietal sulcus
    "mPFC":        (7800, 8100),   # medial prefrontal cortex
    "insula":      (4200, 4500),   # bilateral insula
}


def extract_roi_values(activations: np.ndarray) -> dict:
    """
    Slice the full voxel activation array into 7 ROI mean values,
    then normalise each to a 0-100 scale.

    Args:
        activations: 1-D numpy array of voxel activations from TRIBE v2,
                     expected length >= 8600 (highest ROI upper bound).

    Returns:
        dict with keys for each ROI (float, 0-100 scale),
        plus a private "_peaks" key holding the within-ROI argmax indices
        for optional spatial visualisation.

    Side effects:
        None — pure function.
    """
    raw_means: Dict[str, float] = {}
    roi_peaks: Dict[str, int] = {}

    n = len(activations)

    for region, (start, end) in ROI_INDICES.items():
        # Clamp indices to actual array length to avoid IndexError with mock data
        clamped_start = min(start, n)
        clamped_end = min(end, n)

        if clamped_start >= clamped_end:
            # Array too short — use zero as a safe fallback
            raw_means[region] = 0.0
            roi_peaks[region] = clamped_start
            continue

        slice_ = activations[clamped_start:clamped_end]
        raw_means[region] = float(np.mean(slice_))
        roi_peaks[region] = int(np.argmax(slice_)) + clamped_start

    # Min-max normalise all 7 means to [0, 100]
    values = list(raw_means.values())
    v_min, v_max = min(values), max(values)
    span = v_max - v_min

    normalised: Dict[str, float] = {}
    for region, raw in raw_means.items():
        if span < 1e-9:
            normalised[region] = 50.0
        else:
            normalised[region] = round((raw - v_min) / span * 100, 1)

    result = dict(normalised)
    result["_peaks"] = roi_peaks
    return result
