"""
tests/test_nes.py

Standalone test for pipeline/nes_scorer.py.
Run with:  python tests/test_nes.py
"""

# stdlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# local
from pipeline.nes_scorer import compute_nes


KNOWN_ROI = {
    "amygdala":    55.0,
    "striatum":    70.0,
    "hippocampus": 45.0,
    "dlPFC":       40.0,
    "IPS":         60.0,
    "mPFC":        65.0,
    "insula":      30.0,
}

EDGE_CASES = [
    ("all zeros",  {r: 0.0  for r in KNOWN_ROI}),
    ("all 100s",   {r: 100.0 for r in KNOWN_ROI}),
    ("mixed",      {"amygdala": 80, "striatum": 20, "hippocampus": 50,
                    "dlPFC": 90, "IPS": 10, "mPFC": 60, "insula": 70}),
]


def run_case(label: str, roi: dict):
    print(f"\n--- {label} ---")
    try:
        result = compute_nes(roi)
        print(json.dumps(result, indent=2))
        assert 0 <= result["nes_total"] <= 100, "NES must be in [0, 100]"
        assert -1 <= result["valence"] <= 1, "valence must be in [-1, 1]"
        assert -1 <= result["arousal"] <= 1, "arousal must be in [-1, 1]"
        assert isinstance(result["profile"], str) and result["profile"]
        assert isinstance(result["issues"], list)
        print(f"PASS: {label}")
    except Exception as exc:
        print(f"FAIL: {label} — {exc}")
        raise


def test_nes():
    print("\nTesting compute_nes with known ROI values ...")
    run_case("known ROI", KNOWN_ROI)

    print("\nTesting edge cases ...")
    for label, roi in EDGE_CASES:
        run_case(label, roi)


if __name__ == "__main__":
    test_nes()
