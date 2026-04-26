"""NeuralLens agent skills — edit capabilities, HTML targets, and brain region focus.

Used to:
  1. Build the detailed OpenAI system prompt
  2. Map action_type → CSS selectors the agent should target
  3. Document which brain regions each skill raises or avoids

ACTION_TYPES is the canonical list shared with optimizer and stubs.
"""
from __future__ import annotations

SKILLS: dict[str, dict] = {
    "rewrite_headline": {
        "desc": "Rewrite primary headings to be direct, benefit-led, and emotionally resonant.",
        "selectors": ["h1", "h2", ".hero-title", ".headline", "[data-component='headline']"],
        "targets": ["FFA", "Hippocampus"],
        "avoids": ["Amygdala"],
        "technique": (
            "Active voice. Specific outcome ('3× more leads'). Sensory language. "
            "No puns, no vague promises. Front-load the benefit."
        ),
    },
    "rewrite_cta": {
        "desc": "Rewrite CTA buttons/links to be action-oriented and friction-free.",
        "selectors": ["button", "a.btn", ".cta", "input[type=submit]", "[role=button]"],
        "targets": ["NAcc", "PFC"],
        "avoids": ["ACC", "Amygdala"],
        "technique": (
            "Action verb + benefit + optional directional cue (→). "
            "Replace 'Submit', 'Click here', 'Learn more'. "
            "Eliminate risk language. Max 5 words."
        ),
    },
    "rewrite_body_paragraph": {
        "desc": "Rewrite body paragraphs for clarity, scanability, and benefit focus.",
        "selectors": ["p", ".body-text", "article p", ".description", "li"],
        "targets": ["PFC", "Hippocampus"],
        "avoids": ["ACC", "Insula"],
        "technique": (
            "Max 20 words per sentence. Active voice. Concrete specifics, not abstractions. "
            "One idea per sentence. Remove all filler words and jargon."
        ),
    },
    "rewrite_subheadline": {
        "desc": "Rewrite subheadlines or hero taglines to bridge the headline with specifics.",
        "selectors": ["h3", "h4", ".subheadline", ".tagline", ".hero-subtitle", "h2 + p"],
        "targets": ["FFA", "PFC"],
        "avoids": ["ACC"],
        "technique": (
            "Adds specificity or social proof beneath the main headline. "
            "Outcome-first, then mechanism. Under 15 words."
        ),
    },
    "add_social_proof": {
        "desc": "Add or strengthen social proof (testimonials, stats, logos, reviews).",
        "selectors": [".testimonial", ".review", ".social-proof", "blockquote", ".trust-badge"],
        "targets": ["PFC", "Hippocampus"],
        "avoids": ["Amygdala"],
        "technique": (
            "Specific numbers, named sources, credible authority signals. "
            "No vague superlatives. Format: [number] + [unit] + [timeframe]."
        ),
    },
    "strengthen_value_prop": {
        "desc": "Improve the core value proposition to be outcome-specific and outcome-led.",
        "selectors": [".value-prop", ".hero p", ".tagline", "h1 + p", ".benefit", ".intro"],
        "targets": ["FFA", "Hippocampus", "NAcc"],
        "avoids": ["Amygdala", "ACC"],
        "technique": (
            "Jobs-to-be-done framing. Before/after contrast. "
            "Quantified outcome. Problem → solution arc. No buzzwords."
        ),
    },
    "simplify_language": {
        "desc": "Reduce cognitive load by rewriting complex or jargon-heavy text.",
        "selectors": ["p", ".description", ".features-text", "li", ".body-text", "section p"],
        "targets": ["PFC"],
        "avoids": ["ACC", "Insula"],
        "technique": (
            "Grade 8 reading level (Flesch-Kincaid). Concrete nouns over abstractions. "
            "Remove: 'leverage', 'synergy', 'comprehensive', 'facilitate'. "
            "Replace with plain equivalents."
        ),
    },
    "reorder_sections": {
        "desc": "Change content ordering to improve attention flow.",
        "selectors": ["section", ".section", "article > div", "[data-section]"],
        "targets": ["FFA", "PFC"],
        "avoids": ["ACC"],
        "technique": (
            "Trust-first: social proof before features. "
            "Outcome before process. Hero → proof → features → pricing → CTA."
        ),
    },
    "change_visual_hierarchy": {
        "desc": "Propose text-based changes to visual hierarchy (bold phrases, structure).",
        "selectors": ["h1", "h2", ".hero", "strong", "em", ".highlight", "nav"],
        "targets": ["V4", "FFA"],
        "avoids": ["Insula"],
        "technique": (
            "Bold key benefit phrases. Reduce nav items. "
            "Align to F-pattern: most important info top-left. "
            "Reduce visual noise — remove or simplify secondary copy."
        ),
    },
    "adjust_meta_description": {
        "desc": "Improve the page meta description for engagement and click-through rate.",
        "selectors": ["meta[name='description']", "title"],
        "targets": ["PFC", "NAcc"],
        "avoids": ["ACC"],
        "technique": (
            "Benefit-led opener, specific outcome, zero friction. "
            "Under 155 characters. Ends with implied or explicit action."
        ),
    },
}

ACTION_TYPES: list[str] = list(SKILLS.keys())


def skills_prompt_block() -> str:
    """Return a formatted Markdown block for the agent system prompt."""
    lines = []
    for name, s in SKILLS.items():
        targets = ", ".join(s["targets"])
        avoids = ", ".join(s["avoids"]) if s["avoids"] else "none"
        sel = ", ".join(s["selectors"][:3])
        lines.append(
            f"**{name}**\n"
            f"  Purpose: {s['desc']}\n"
            f"  Brain: raise {targets} | keep low {avoids}\n"
            f"  HTML selectors: {sel}\n"
            f"  Technique: {s['technique']}"
        )
    return "\n\n".join(lines)
