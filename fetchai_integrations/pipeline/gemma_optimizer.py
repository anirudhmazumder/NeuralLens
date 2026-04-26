"""
pipeline/gemma_optimizer.py

Calls Google Gemma 3 (27B instruct) via the google-generativeai SDK to
produce three targeted webpage optimisations based on the NES breakdown.
Returns a validated JSON dict ready for Playwright to apply.
"""

# stdlib
import json
import os
import re

# third-party
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Module-level initialisation — runs once at import time
# ---------------------------------------------------------------------------

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
_gemma = genai.GenerativeModel("gemma-3-27b-it")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a neuromarketing optimization expert with deep knowledge
of how brain regions respond to web design and copy.

You receive a webpage's HTML, Neural Engagement Score breakdown
across 7 brain regions, and the business industry.

You return ONLY a valid JSON object.
No markdown. No explanation. No backticks. Just raw JSON.

JSON format:
{
  "changes": [
    {
      "selector": "CSS selector of element to change",
      "property": "innerHTML OR style.backgroundColor OR style.color OR style.fontSize OR style.display OR style.padding",
      "old_value": "current value as it appears on the page",
      "new_value": "the optimized replacement value",
      "target_region": "which brain region this targets",
      "reason": "one sentence neuroscience explanation"
    }
  ],
  "summary": "one sentence describing the overall optimization strategy"
}

STRICT RULES:
- Return EXACTLY 3 changes, no more, no less
- Only modify elements that EXIST in the provided HTML
- Only use these properties: innerHTML, style.backgroundColor,
  style.color, style.fontSize, style.display, style.padding
- Use valid CSS hex colors only (e.g. #E8450A not 'orange')
- Selectors must be valid CSS and must match the HTML
- Be SPECIFIC — write the actual new text, not a description of it
- Prioritize the highest-impact brain region issues first
"""


def get_suggestions(html: str, nes_result: dict, industry: str) -> dict:
    """
    Ask Gemma 3 for three concrete webpage changes to improve the NES.

    Args:
        html: Raw page HTML (will be truncated to first 3000 chars).
        nes_result: Output dict from compute_nes().
        industry: Business vertical string (e.g. "e-commerce", "SaaS").

    Returns:
        Parsed JSON dict with keys "changes" (list of 3 dicts) and "summary".

    Raises:
        ValueError: If the model response cannot be parsed as valid JSON
                    after fallback regex extraction.
    """
    roi_json = json.dumps(nes_result.get("roi_values", {}), indent=2)
    issues_text = "\n".join(f"- {i}" for i in nes_result.get("issues", []))

    user_prompt = f"""Industry: {industry}

NES Total: {nes_result.get('nes_total', 'N/A')} / 100
Profile: {nes_result.get('profile', 'N/A')}
Valence: {nes_result.get('valence', 'N/A')}  (−1 = negative, +1 = positive)
Arousal: {nes_result.get('arousal', 'N/A')}  (−1 = calm, +1 = excited)

Brain region issues detected:
{issues_text or '(none detected)'}

ROI values (0-100 scale):
{roi_json}

HTML (first 3000 characters):
{html[:3000]}
"""

    full_prompt = (
        f"<start_of_turn>system\n{SYSTEM_PROMPT}\n<end_of_turn>\n"
        f"<start_of_turn>user\n{user_prompt}\n<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    response = _gemma.generate_content(full_prompt)
    raw = response.text.strip()

    # Strip any accidental markdown fences
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    # Primary parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Gemma returned a response that could not be parsed as JSON.\n"
        f"Raw response:\n{raw}"
    )
