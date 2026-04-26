"""OpenAI-powered NeuralLens optimization agent.

Uses GPT-4.1-nano with vision. Full context per call:
  - Screenshot (base64 PNG, high detail)
  - Cleaned body HTML excerpt (6000 chars — primary source for exact text)
  - All 9 HCP-MMP1 TRIBE region activations
  - NES dysfunction score + per-violation breakdown
  - Gaze fixation map (DeepGaze IIE or F-pattern)
  - Cross-session memory stats (what has worked / failed)
  - Ethics flags
  - Score trajectory

Agent has full autonomy over action_type selection — it should choose based
on the current brain state, not follow a fixed rotation.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from skills import ACTION_TYPES, SKILLS, skills_prompt_block

if TYPE_CHECKING:
    from memory import ExperienceBuffer

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """\
You are NeuralLens, an expert neuromarketing AI that edits web pages to minimize \
neurological soft signs (NSS) as measured by TRIBE v2, a multimodal fMRI-trained \
brain encoder. You are a specialist — you think in brain regions before you think \
in words.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — THE NES OBJECTIVE (what you minimize)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NES = Neurological Evaluation Scale. Measures dysfunction in the \
cerebello-thalamo-prefrontal cortical circuit. LOW NES = healthy neural engagement.

  HYPER-ACTIVATION (above threshold → circuit overload, raises NES):
    Amygdala  > 0.40   weight 2.0   → fear, urgency, anxiety, distrust
    ACC       > 0.40   weight 2.0   → cognitive conflict, decision paralysis
    Insula    > 0.40   weight 1.5   → disgust, sensory overload, discomfort

  HYPO-ACTIVATION (below threshold → circuit failure, raises NES):
    PFC       < 0.50   weight 1.5   → executive dropout, trust failure
    Hippocampus < 0.40 weight 1.0   → memory encoding failure, nothing sticks

  ENGAGEMENT (never penalized, higher = better):
    FFA    face/identity recognition — rises with social content, faces, named people
    V4     color & form perception   — rises with vivid descriptive language, visual structure
    MT+    motion & visual flow      — rises with action verbs, dynamic phrasing
    NAcc   reward anticipation       — rises with clear benefit, specificity, novelty
           ⚠ HARD LIMIT: NAcc ≤ 0.70 | Amygdala ≤ 0.60 (dark-pattern guardrails)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — REGION PLAYBOOK (what moves each region)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PFC (prefrontal — trust, executive attention)
  RAISES: clear logical structure, explicit credentials/authority, numbered steps,
          concrete specifics ("3 steps" vs "a few steps"), transparent pricing,
          calm confident tone, reduced choice complexity, summary sentences
  LOWERS: vague promises, inconsistent claims, hidden costs, too many options,
          jargon the reader can't evaluate, passive voice

Hippocampus (memory encoding — novelty gating, retention)
  RAISES: surprising facts, counterintuitive claims, specific stories/examples,
          concrete vivid details ("52,000 users" not "many users"),
          emotional anchors, before/after contrasts, named protagonists
  LOWERS: generic copy that matches prior expectations, clichés, bullet-point lists
          with no narrative, abstract statistics without context

FFA (face/identity — recognition, social trust)
  RAISES: first-person testimonials with names, "you" direct address,
          founder/team mentions, specific named customers, personalization,
          human faces referenced in text ("our team of 40 engineers"),
          conversational register
  LOWERS: corporate third-person, passive voice, unnamed "users" or "clients"

NAcc (reward anticipation — motivation, desire)
  RAISES: clear outcome statements, specific quantified benefits,
          scarcity or time-framing (ethical, not manipulative),
          curiosity gaps, achievement framing ("you will...")
  LOWERS: vague outcomes, no clear next step, benefit buried after features

V4 (color/form — visual engagement)
  RAISES: vivid sensory adjectives, color words, spatial description,
          visual metaphors, short punchy sentences that create white space mentally
  LOWERS: dense paragraphs, abstract nouns, gray corporate language

MT+ (motion/flow — dynamic processing)
  RAISES: action verbs at sentence start, present tense, motion metaphors
          ("launch", "drive", "accelerate"), progressive verbs, demos described
  LOWERS: past tense, passive constructions, noun-heavy sentences

Amygdala (fear/anxiety — KEEP LOW)
  RAISED BY: urgency pressure ("expires tonight!"), loss framing ("don't miss"),
             threat language, fear-of-missing-out without benefit, risk emphasis,
             aggressive pop-up style copy, all-caps, excessive exclamation points
  LOWER IT: reframe to gain ("join" not "don't miss"), remove urgency copy,
            soften tone, add reassurance ("cancel any time", "no risk"),
            replace negatives with positive equivalents

ACC (cognitive conflict — KEEP LOW)
  RAISED BY: contradictory claims, too many choices, unclear CTAs,
             mixed messages about audience, unclear page hierarchy,
             sentences requiring re-reading, ambiguous pronoun references
  LOWER IT: single clear CTA, consolidate contradictory sections,
            simplify navigation language, eliminate hedging phrases,
            choose one value prop and commit to it

Insula (sensory overload — KEEP LOW)
  RAISED BY: dense text blocks, complex nested sentences, information overload,
             multiple competing visual focal points described in text,
             excessive jargon, run-on paragraphs, too many numbers at once
  LOWER IT: break up text, shorter sentences, white space signals,
            remove redundant information, prioritize top 3 facts only

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — EDIT SKILLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{SKILLS}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — HOW TO REASON (follow this chain every call)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — DIAGNOSE: Which region is the PRIMARY NES driver right now?
  Check the violations list. The highest-weight violation is your target.
  If multiple violations, address the highest-weight one first.
  If no violations, maximize the lowest engagement region.

Step 2 — LOCALIZE: Where on the page is the problem?
  Use the gaze map — the rank-1 salient region gets the most neural processing.
  Use the HTML excerpt — find the EXACT element causing the problem.
  Match element type to skill selectors.

Step 3 — SELECT: Which skill addresses your target region best?
  Don't cycle mechanically. Choose freely based on the playbook above.
  If memory stats show an action_type has negative avg_reward, avoid it.
  If one has strongly positive avg_reward, prefer it.

Step 4 — CRAFT: Write the replacement with surgical precision.
  The "original" must be verbatim text from the HTML/page text.
  The "replacement" must be a single drop-in improvement.
  Measure against the playbook: does it raise the target region?
  Does it keep penalty regions from rising?

Step 5 — ESTIMATE: What delta do you expect on NES?
  Each 0.1 improvement in a penalized region reduces NES by weight × 0.1.
  Be honest about your estimate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — "original" FIELD RULES (critical)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The "original" value MUST:
  ✓ Appear VERBATIM in the HTML excerpt or page text you received
  ✓ Be copy-pasteable — no paraphrasing, no approximation
  ✓ Contain only text content — no HTML tags, no attributes
  ✓ Be long enough to be unique on the page (5+ words preferred)

If no suitable text exists for the target, set "original": "" and use
"html_selector" to describe the structural target.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — valid JSON only, no markdown fences
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
  "action_type": "<skill name — YOUR choice based on diagnosis>",
  "target": "<element description: 'primary H1', 'hero subheadline', etc.>",
  "html_selector": "<CSS selector: 'h1.hero-title', 'p.tagline', etc.>",
  "original": "<verbatim text from page>",
  "replacement": "<improved text>",
  "reasoning": "<Step 1-5 chain: which region, why this edit, expected mechanism>",
  "expected_roi_impact": {{"language": 0.0, "attention": 0.0, "visual": 0.0}},
  "nes_loss_delta_estimate": 0.0
}}
""".format(SKILLS=skills_prompt_block())


def _build_memory_block(
    memory: Optional["ExperienceBuffer"],
    action_type: str,
) -> str:
    if memory is None:
        return ""

    stats = memory.get_action_stats()
    failures = memory.get_negative_experiences(action_type)

    lines = []
    if stats:
        lines.append("Cross-session performance (sorted by avg_reward):")
        for atype, s in sorted(stats.items(), key=lambda x: -x[1]["avg_reward"]):
            mark = "✓" if s["avg_reward"] > 0 else "✗"
            lines.append(
                f"  {mark} {atype}: avg_reward={s['avg_reward']:+.4f}  "
                f"success={s['success_rate']:.0%}  n={s['count']}"
            )

    if failures:
        lines.append(f"\nRecent failures for '{action_type}' — avoid repeating:")
        for f in failures[:4]:
            detail = json.loads(f.get("action_detail", "{}"))
            lines.append(
                f"  - target='{detail.get('target', '?')}' "
                f"reward={f['reward']:+.4f}"
            )

    return "\n".join(lines)


def _build_diagnosis_block(brain_scores: dict, intent: str) -> str:
    """Pre-compute a human-readable diagnosis to guide the agent's reasoning."""
    from nes import NSS_INDICATORS, ENGAGEMENT_REGIONS

    violations = []
    for region, cfg in NSS_INDICATORS.items():
        val = brain_scores.get(region, 0.5)
        if cfg["direction"] == "above":
            delta = val - cfg["threshold"]
        else:
            delta = cfg["threshold"] - val
        if delta > 0:
            violations.append((region, delta, cfg["weight"], delta * cfg["weight"]))

    violations.sort(key=lambda x: -x[3])

    lines = []
    if violations:
        lines.append("NSS VIOLATIONS (primary NES drivers — address these first):")
        for region, delta, weight, contribution in violations:
            cfg = NSS_INDICATORS[region]
            direction = "ABOVE" if cfg["direction"] == "above" else "BELOW"
            lines.append(
                f"  ► {region} = {brain_scores[region]:.3f} "
                f"({direction} threshold {cfg['threshold']}) "
                f"→ NES contribution {contribution:.3f}  "
                f"[playbook: {'LOWER this region' if cfg['direction'] == 'above' else 'RAISE this region'}]"
            )
    else:
        lines.append("No NSS violations active — focus on raising engagement regions.")

    low_engagement = [
        (r, brain_scores.get(r, 0.0))
        for r in ENGAGEMENT_REGIONS
        if brain_scores.get(r, 0.0) < 0.5
    ]
    if low_engagement:
        lines.append("LOW ENGAGEMENT (secondary targets):")
        for r, v in sorted(low_engagement, key=lambda x: x[1]):
            lines.append(f"  ○ {r} = {v:.3f}  [raise this — see playbook]")

    return "\n".join(lines)


# ── Agent ──────────────────────────────────────────────────────────────────────

class OptimizationAgent:
    def __init__(self, memory: Optional["ExperienceBuffer"] = None) -> None:
        self._live = os.getenv("OPENAI_LIVE", "false").lower() == "true"
        self._action_index = 0
        self._memory = memory

    def _next_action_type(self) -> str:
        action = ACTION_TYPES[self._action_index % len(ACTION_TYPES)]
        self._action_index += 1
        return action

    async def select_action(
        self,
        page_content,
        score_history: list,
        iteration: int,
        screenshot_path: Optional[str] = None,
        gaze_regions: Optional[list] = None,
        brain_scores: Optional[dict] = None,
        ethics_flags: Optional[list] = None,
        intent: str = "engage",
    ) -> dict:
        if not self._live:
            raise RuntimeError(
                "OPENAI_LIVE is not set to true. "
                "Set OPENAI_LIVE=true and OPENAI_API_KEY in .env to run the optimizer."
            )

        # Suggest a starting action_type but let the model override freely
        suggested_action = self._next_action_type()

        screenshot_b64 = ""
        if screenshot_path and Path(screenshot_path).exists():
            with open(screenshot_path, "rb") as fh:
                screenshot_b64 = base64.b64encode(fh.read()).decode()

        return await self._call_openai(
            screenshot_b64, page_content, score_history, suggested_action,
            gaze_regions=gaze_regions,
            brain_scores=brain_scores,
            ethics_flags=ethics_flags,
            intent=intent,
        )

    async def _call_openai(
        self,
        screenshot_b64: str,
        page_content,
        score_history: list,
        suggested_action: str,
        gaze_regions: Optional[list] = None,
        brain_scores: Optional[dict] = None,
        ethics_flags: Optional[list] = None,
        intent: str = "engage",
    ) -> dict:
        import openai  # type: ignore

        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # ── Score context ──────────────────────────────────────────────────────

        current = score_history[-1] if score_history else {}
        score_block = "\n".join(
            f"  {k}: {v:.4f}"
            for k, v in current.items()
            if k.endswith("_roi") or k == "overall_score"
        )
        history_block = "\n".join(
            f"  iter {i}: overall={s.get('overall_score',0):.4f}  "
            f"L={s.get('language_roi',0):.3f}  "
            f"A={s.get('attention_roi',0):.3f}  "
            f"V={s.get('visual_roi',0):.3f}"
            for i, s in enumerate(score_history[-8:])
        )

        # ── NES block ──────────────────────────────────────────────────────────
        nes_block = ""
        diagnosis_block = ""
        if brain_scores:
            from nes import nes_summary
            ns = nes_summary(brain_scores, intent)
            violations_str = "  ".join(
                f"{r} +{v:.3f}" for r, v in ns.get("violations", {}).items()
            ) or "none"
            nes_block = (
                f"\nNES dysfunction score: {ns['nes_score']:.4f}  "
                f"(0 = no soft signs, higher = more dysfunction)\n"
                f"Active violations: {violations_str}\n"
                f"Dominant soft-sign: {ns['dominant_penalty']}  |  "
                f"Strongest engagement: {ns['dominant_positive']}\n"
                f"Engagement score: {ns['engagement_score']:.4f}  "
                f"Penalty score: {ns['penalty_score']:.4f}"
            )
            diagnosis_block = "\n\n--- Diagnosis (follow this for Step 1-2) ---\n" + \
                _build_diagnosis_block(brain_scores, intent)

        # ── Brain region scores ────────────────────────────────────────────────
        brain_block = ""
        if brain_scores:
            region_lines = "\n".join(
                f"  {r}: {v:.4f}" for r, v in sorted(brain_scores.items(), key=lambda x: -x[1])
            )
            brain_block = f"\n--- All TRIBE region activations ---\n{region_lines}"

        # ── Gaze block ─────────────────────────────────────────────────────────
        gaze_block = ""
        if gaze_regions:
            top = gaze_regions[0]
            gaze_lines = "\n".join(
                f"  #{r['rank']} saliency={r['saliency_score']:.2f}  "
                f"bbox={r['bbox']}  peak={r['peak_coords']}"
                for r in gaze_regions[:5]
            )
            gaze_block = (
                f"\n--- Gaze fixation map ---\n{gaze_lines}\n"
                f"Humans look at rank-1 first (peak {top['peak_coords']}). "
                f"Edits in this zone have the highest neural impact."
            )

        # ── Ethics block ───────────────────────────────────────────────────────
        ethics_block = ""
        if ethics_flags:
            flag_lines = "\n".join(
                f"  [{f['severity'].upper()}] {f['code']}: {f['message']}"
                for f in ethics_flags
            )
            ethics_block = f"\n--- Ethics flags (MANDATORY: do not worsen these) ---\n{flag_lines}"

        # ── Memory block ───────────────────────────────────────────────────────
        memory_block = _build_memory_block(self._memory, suggested_action)

        # ── Page content ───────────────────────────────────────────────────────
        html_excerpt = ""
        if hasattr(page_content, "html") and page_content.html:
            html_excerpt = page_content.html[:6000]
        text_excerpt = (
            page_content.text if hasattr(page_content, "text") else str(page_content)
        )[:3000]

        # ── Build user message ─────────────────────────────────────────────────

        user_parts: list = []
        if screenshot_b64:
            user_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_b64}",
                    "detail": "high",
                },
            })

        user_text = (
            f"URL: {getattr(page_content, 'url', '')}\n"
            f"Intent: {intent}  |  Iteration context: suggested_action={suggested_action}\n"
            f"\n--- Current TRIBE scores ---\n{score_block}\n"
            f"\n--- Score history ---\n{history_block}\n"
            f"{nes_block}"
            f"{diagnosis_block}"
            f"{brain_block}"
            f"{gaze_block}"
            f"{ethics_block}"
            + (f"\n\n--- Memory ---\n{memory_block}" if memory_block else "")
            + f"\n\n--- Page HTML (copy verbatim text from here for 'original') ---\n{html_excerpt}"
            + f"\n\n--- Page text ---\n{text_excerpt}"
            + f"\n\n"
            f"TASK: Follow the 5-step reasoning chain in Part 4 of your instructions.\n"
            f"Suggested action_type: '{suggested_action}' — but choose freely based on diagnosis.\n"
            f"If memory shows suggested type has negative avg_reward AND another has strongly positive, switch.\n"
            f"Output ONE JSON edit. No markdown fences."
        )

        user_parts.append({"type": "text", "text": user_text})

        nes_val = nes_block.split()[3] if nes_block and len(nes_block.split()) > 3 else "?"
        print(
            f"[NeuralLens] → OpenAI | suggested={suggested_action} | intent={intent} "
            f"| nes={nes_val} | gaze={bool(gaze_regions)} | html={len(html_excerpt)}chars"
        )

        resp = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
            max_tokens=1500,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_parts},
            ],
        )

        usage = resp.usage
        print(
            f"[NeuralLens] ← OpenAI | "
            f"in={usage.prompt_tokens} out={usage.completion_tokens} tokens"
        )

        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(raw[start:end])
                if "expected_roi_impact" not in result:
                    result["expected_roi_impact"] = {"language": 0.0, "attention": 0.0, "visual": 0.0}
                if "expected_roi" not in result:
                    impact = result["expected_roi_impact"]
                    best = max(impact, key=impact.get)
                    result["expected_roi"] = f"{best}_roi"
                return result
            except json.JSONDecodeError:
                pass

        print("[NeuralLens] ✗ JSON parse failed — returning no-op edit")
        return {
            "action_type": suggested_action,
            "target": "",
            "html_selector": "",
            "original": "",
            "replacement": "",
            "reasoning": "JSON parse failed",
            "expected_roi_impact": {"language": 0.0, "attention": 0.0, "visual": 0.0},
            "expected_roi": "language_roi",
        }
