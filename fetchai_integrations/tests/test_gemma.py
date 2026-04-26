"""
tests/test_gemma.py

Tests the Gemma 3 27B strategist prompt + JSON parsing.
Run with: python tests/test_gemma.py

Requires GOOGLE_API_KEY in .env.
"""

import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
_gemma = genai.GenerativeModel("gemma-3-27b-it")

SYSTEM_PROMPT = """You are a neuromarketing optimization expert for visual content.
Return ONLY valid JSON. No markdown. No backticks. No explanation.

{
  "visual_changes": [3 items],
  "text_changes": [2 items],
  "new_copy": "string",
  "optimization_strategy": "string"
}"""

SAMPLE_INPUT = """Asset type: instagram post
Industry: bakery
User goal: increase engagement and foot traffic

NES Total: 38.5 / 100
Emotional profile: low desire and emotion — flat and forgettable
Valence: -0.12
Arousal: 0.05

Brain region issues:
- striatum 32.0/100 — LOW desire, strengthen value proposition
- amygdala 28.0/100 — LOW emotional pull, copy is flat

ROI values (0-100):
{"amygdala": 28.0, "striatum": 32.0, "hippocampus": 55.0, "dlPFC": 45.0, "IPS": 60.0, "mPFC": 50.0, "insula": 35.0}

Critical zone issues:
  center [attention_trap]: enrich emotionally or remove this element
  bottom-center [dead_zone]: remove or replace with stronger content"""

REQUIRED_VISUAL_FIELDS = {"zone", "change_type", "description",
                           "cloudinary_transform", "target_region", "reason"}
REQUIRED_TEXT_FIELDS = {"element", "old_text", "new_text",
                         "target_region", "reason"}


def test_gemma_strategy():
    print("\n[1/1] Testing Gemma 3 strategy generation...")
    print(f"  Model: gemma-3-27b-it")
    print(f"  API key: {'set' if os.environ.get('GOOGLE_API_KEY') else 'MISSING'}")

    full_prompt = (
        f"<start_of_turn>system\n{SYSTEM_PROMPT}\n<end_of_turn>\n"
        f"<start_of_turn>user\n{SAMPLE_INPUT}\n<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    try:
        response = _gemma.generate_content(full_prompt)
        raw = response.text.strip()
        print(f"\n  Raw response ({len(raw)} chars):\n  {raw[:200]}...")

        # Strip fences
        raw = re.sub(r"```json|```", "", raw).strip()

        # Parse
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            result = json.loads(match.group()) if match else None

        assert result is not None, "Could not parse JSON"
        print(f"\n  Parsed keys: {list(result.keys())}")

        # Validate visual_changes
        vc = result.get("visual_changes", [])
        print(f"  visual_changes count: {len(vc)} (expect 3)")
        for i, c in enumerate(vc):
            missing = REQUIRED_VISUAL_FIELDS - set(c.keys())
            if missing:
                print(f"  ⚠️  visual_changes[{i}] missing: {missing}")

        # Validate text_changes
        tc = result.get("text_changes", [])
        print(f"  text_changes count: {len(tc)} (expect 2)")
        for i, c in enumerate(tc):
            missing = REQUIRED_TEXT_FIELDS - set(c.keys())
            if missing:
                print(f"  ⚠️  text_changes[{i}] missing: {missing}")

        print(f"  new_copy length: {len(result.get('new_copy', ''))} chars")
        print(f"  strategy: {result.get('optimization_strategy', '')[:80]}...")

        print("\nPASS: Gemma strategy generation")
        print("\nFull result:")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"\nFAIL: {e}")
        raise


if __name__ == "__main__":
    test_gemma_strategy()
