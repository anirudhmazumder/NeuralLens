"""
pipeline/__init__.py

Pure math functions for NeuralLens — no agents, no APIs.
"""

from pipeline.nes_math import (
    extract_roi_values,
    compute_nes,
    analyze_intersection,
    ROI_INDICES,
    NES_WEIGHTS,
)

__all__ = [
    "extract_roi_values",
    "compute_nes",
    "analyze_intersection",
    "ROI_INDICES",
    "NES_WEIGHTS",
]
