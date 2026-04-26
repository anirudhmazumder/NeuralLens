"""
tests/test_nes_math.py

Tests for pipeline/nes_math.py pure functions.
Run with: python tests/test_nes_math.py

No API keys required — pure math only.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pipeline.nes_math import (
    extract_roi_values,
    compute_nes,
    analyze_intersection,
)


EDGE_CASES = [
    ("all zeros",   np.zeros(70_000)),
    ("all ones",    np.ones(70_000)),
    ("beta(2,5)",   np.random.beta(2, 5, 70_000).astype(np.float32)),
    ("short array", np.random.rand(1000).astype(np.float32)),
]


def test_extract_roi_values():
    print("\n[1/3] test_extract_roi_values")
    for label, arr in EDGE_CASES:
        roi = extract_roi_values(arr)
        assert len(roi) == 7, f"{label}: expected 7 ROIs, got {len(roi)}"
        for region, val in roi.items():
            assert 0.0 <= val <= 100.0, (
                f"{label}/{region}: value {val} out of [0,100]"
            )
        print(f"  PASS [{label}]: {json.dumps({k: v for k, v in list(roi.items())[:3]}, indent=0)} ...")
    print("PASS: extract_roi_values")


def test_compute_nes():
    print("\n[2/3] test_compute_nes")

    known_roi = {
        "amygdala":    55.0,
        "striatum":    70.0,
        "hippocampus": 45.0,
        "dlPFC":       40.0,
        "IPS":         60.0,
        "mPFC":        65.0,
        "insula":      30.0,
    }
    result = compute_nes(known_roi)
    print(f"  NES total   : {result['nes_total']}")
    print(f"  Valence     : {result['valence']}")
    print(f"  Arousal     : {result['arousal']}")
    print(f"  Profile     : {result['profile']}")
    print(f"  Issues      : {result['issues']}")

    assert 0 <= result["nes_total"] <= 100
    assert -1 <= result["valence"] <= 1
    assert -1 <= result["arousal"] <= 1
    assert isinstance(result["profile"], str)
    assert isinstance(result["issues"], list)

    # Edge: all zeros → NES should be 0
    zero_roi = {r: 0.0 for r in known_roi}
    res_zero = compute_nes(zero_roi)
    print(f"  All-zero NES: {res_zero['nes_total']} (expect 0)")
    assert res_zero["nes_total"] == 0.0

    # Edge: all 100s → NES bounded at 100
    full_roi = {r: 100.0 for r in known_roi}
    res_full = compute_nes(full_roi)
    print(f"  All-100 NES : {res_full['nes_total']} (expect ≤100)")
    assert 0 <= res_full["nes_total"] <= 100

    print("PASS: compute_nes")


def test_analyze_intersection():
    print("\n[3/3] test_analyze_intersection")
    activations = np.random.beta(2, 5, 70_000).astype(np.float32)
    deepgaze = np.random.uniform(0, 1, 900).tolist()

    insights = analyze_intersection(activations, deepgaze)
    assert len(insights) == 9, f"Expected 9 zones, got {len(insights)}"

    valid_types = {"power_zone", "attention_trap", "hidden_value", "dead_zone"}
    for zone in insights:
        assert zone["type"] in valid_types, (
            f"Unknown zone type: {zone['type']}"
        )
        assert 0 <= zone["tribe_score"] <= 100
        assert 0 <= zone["gaze_score"] <= 100

    type_counts = {}
    for z in insights:
        type_counts[z["type"]] = type_counts.get(z["type"], 0) + 1
    print(f"  Zone breakdown: {type_counts}")

    print("PASS: analyze_intersection")


def main():
    print("=" * 55)
    print("NeuralLens NES Math Tests")
    print("=" * 55)
    test_extract_roi_values()
    test_compute_nes()
    test_analyze_intersection()
    print("\n✅ All NES math tests passed!")


if __name__ == "__main__":
    main()
