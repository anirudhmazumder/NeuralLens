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
on the current brain state while keeping iteration strategy diverse.
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
You are NeuralLens, an expert neuromarketing AI that edits web pages to MAXIMIZE \
the TRIBE v2 multimodal `overall_score` — a 0..1 engagement quality predicted \
from a fMRI-trained brain encoder. You are a specialist: you think in brain \
regions before you think in words.

OPERATIVE METRIC (what the optimizer actually compares against):
  - `overall_score` = 0.36·attention_roi + 0.34·language_roi + 0.30·visual_roi − 0.35·penalty
  - Higher overall_score = better. Each iteration is ACCEPTED only if the new
    overall_score exceeds the current overall_score.
  - You will see the running history (baseline + every prior step). Use the
    DIRECTION of the last delta to decide whether to push further on the same
    lever or pivot to a different one.
  - The Neural Engagement Score (NES) below is a secondary diagnostic lens —
    LOWERING NES generally raises overall_score because penalty regions drop
    and engagement regions rise. Treat NES as your "why this edit", but treat
    overall_score as your "did it work".

CRITICAL TASK-ALIGNMENT RULES:
  - Edit the user's existing page copy; do not invent a new campaign concept.
  - Never fabricate claims, numbers, timelines, prices, discounts, or trial periods.
  - Do not inject generic ad slogans or hype copy that is not grounded in existing text.
  - Keep edits semantically close to the original intent of the target element.
  - Prefer concrete, factual clarity over dramatic persuasion language.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — THE NES DIAGNOSTIC (secondary lens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NES = Neural Engagement Score. Measures over-activation of stress/overload \
regions and under-activation of engagement/clarity regions. LOW NES = better \
neural engagement (and, almost always, higher overall_score).

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
            soften tone, add reassurance (without adding invented offers),
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

Step 0 — REVIEW LAST STEP: Did your previous edit move overall_score up or down?
  If your last attempt was REJECTED (delta ≤ 0): change strategy. Don't repeat
  the same family of edit. Look at the ROI deltas — which sub-score regressed?
  If your last attempt was ACCEPTED: you are on a good trajectory. Decide
  whether to keep pushing the same lever (same target region) or to pivot to
  a different lever that's now the top remaining gap.

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
  Keep strategy diverse across attempts: rotate between text, visual hierarchy/color,
  social proof/trust, and structural ordering when possible.
  Avoid repeating the same action_type in back-to-back iterations unless it is
  clearly the only high-impact option.

Step 4 — CRAFT: Write the replacement with surgical precision.
  The "original" must be verbatim text from the HTML/page text.
  The "replacement" must be a single drop-in improvement.
  Measure against the playbook: does it raise the target region?
  Does it keep penalty regions from rising?

Step 5 — ESTIMATE: What delta do you expect on overall_score?
  Each 0.1 improvement in a penalized region typically lifts overall_score by
  roughly 0.03–0.05 (penalty weight × 0.35). Each 0.1 raise of an engagement
  region lifts the matched ROI by roughly its weighting. Be honest — small,
  realistic estimates beat over-promising.

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
        lines.append("NES VIOLATIONS (primary score drivers — address these first):")
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
        lines.append("No NES violations active — focus on raising engagement regions.")

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
        recent_actions: Optional[list[str]] = None,
        last_outcome: Optional[dict] = None,
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
            recent_actions=recent_actions,
            last_outcome=last_outcome,
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
        recent_actions: Optional[list[str]] = None,
        last_outcome: Optional[dict] = None,
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

        # ── Last-step outcome (drives Step 0 of the reasoning chain) ───────────
        outcome_block = ""
        if last_outcome:
            verdict = "ACCEPTED ✓" if last_outcome.get("accepted") else "REJECTED ✗"
            roi_d = last_outcome.get("roi_deltas") or {}
            roi_str = "  ".join(f"Δ{k}={float(v):+.4f}" for k, v in roi_d.items())
            outcome_block = (
                "\n\n--- Last step outcome (use this for Step 0) ---\n"
                f"action_type   = {last_outcome.get('action_type', '?')}\n"
                f"verdict       = {verdict}\n"
                f"score_delta   = {float(last_outcome.get('score_delta', 0)):+.4f}\n"
                f"roi deltas    = {roi_str or 'n/a'}\n"
                f"target        = {last_outcome.get('target', '')}\n"
                f"original  → \"{(last_outcome.get('original') or '')[:120]}\"\n"
                f"replacement → \"{(last_outcome.get('replacement') or '')[:120]}\"\n"
                "If REJECTED, do NOT repeat this action_type or this lever — pivot.\n"
                "If ACCEPTED, you may keep pushing the same lever, OR switch to the\n"
                "next-largest gap if returns are diminishing."
            )

        user_text = (
            f"URL: {getattr(page_content, 'url', '')}\n"
            f"Intent: {intent}  |  Iteration context: suggested_action={suggested_action}\n"
            f"OBJECTIVE: maximize TRIBE overall_score. Your edit will be ACCEPTED only "
            f"if it lifts overall_score above the current value.\n"
            f"\n--- Current TRIBE scores ---\n{score_block}\n"
            f"\n--- Score history ---\n{history_block}\n"
            f"{nes_block}"
            f"{diagnosis_block}"
            f"{brain_block}"
            f"{gaze_block}"
            f"{ethics_block}"
            f"{outcome_block}"
            + (f"\n\n--- Memory ---\n{memory_block}" if memory_block else "")
            + (
                f"\n\n--- Recent actions (avoid repeating unless clearly best) ---\n"
                f"{', '.join(recent_actions[-6:])}"
                if recent_actions else ""
            )
            + f"\n\n--- Page HTML (copy verbatim text from here for 'original') ---\n{html_excerpt}"
            + f"\n\n--- Page text ---\n{text_excerpt}"
            + f"\n\n"
            f"TASK: Follow the 6-step reasoning chain (Step 0-5) in Part 4.\n"
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

    async def select_html_action(
        self,
        html_content: str,
        score_history: list,
        iteration: int,
        screenshot_path: Optional[str] = None,
        gaze_regions: Optional[list] = None,
        brain_scores: Optional[dict] = None,
    ) -> dict:
        """Propose an HTML/CSS design edit to improve neural engagement.

        Unlike select_action (which targets copywriting), this method asks the
        model to make visual design changes: colors, typography, spacing, CTA
        prominence, layout — returning html_original / html_replacement pairs
        that are applied as string replacements directly in the HTML source.
        """
        if not self._live:
            return _fake_html_edit(iteration, html_content)

        screenshot_b64 = ""
        if screenshot_path and Path(screenshot_path).exists():
            with open(screenshot_path, "rb") as fh:
                screenshot_b64 = base64.b64encode(fh.read()).decode()

        return await self._call_openai_html(
            html_content, screenshot_b64, score_history, iteration,
            gaze_regions=gaze_regions,
            brain_scores=brain_scores,
        )

    async def _call_openai_html(
        self,
        html_content: str,
        screenshot_b64: str,
        score_history: list,
        iteration: int,
        gaze_regions: Optional[list] = None,
        brain_scores: Optional[dict] = None,
    ) -> dict:
        import openai  # type: ignore

        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        current = score_history[-1] if score_history else {}
        score_block = "\n".join(
            f"  {k}: {v:.4f}"
            for k, v in current.items()
            if k.endswith("_roi") or k == "overall_score"
        )

        nes_block = ""
        brain_block = ""
        if brain_scores:
            from nes import nes_summary
            ns = nes_summary(brain_scores, "engage")
            violations_str = "  ".join(
                f"{r} +{v:.3f}" for r, v in ns.get("violations", {}).items()
            ) or "none"
            nes_block = (
                f"\nNES score: {ns['nes_score']:.4f}  Active violations: {violations_str}"
            )
            region_lines = "\n".join(
                f"  {r}: {v:.4f}" for r, v in sorted(brain_scores.items(), key=lambda x: -x[1])
            )
            brain_block = f"\nAll TRIBE region activations:\n{region_lines}"

        gaze_block = ""
        if gaze_regions:
            top = gaze_regions[0]
            gaze_lines = "\n".join(
                f"  #{r['rank']} saliency={r['saliency_score']:.2f}  bbox={r['bbox']}"
                for r in gaze_regions[:5]
            )
            gaze_block = (
                f"\nGaze map (rank-1 = most viewed):\n{gaze_lines}\n"
                f"Focus visual design changes near peak {top['peak_coords']}."
            )

        html_excerpt = html_content[:8000]
        action_idx = (iteration - 1) % len(HTML_ACTION_TYPES)
        suggested_action = HTML_ACTION_TYPES[action_idx]

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
            f"Iteration: {iteration}\n"
            f"Suggested action_type: '{suggested_action}' — choose freely if another is more impactful.\n"
            f"\n--- Current TRIBE scores ---\n{score_block}"
            f"{nes_block}"
            f"{brain_block}"
            f"{gaze_block}"
            f"\n\n--- Full HTML source (make your edit here) ---\n{html_excerpt}"
            "\n\nTASK: Propose ONE visual design edit to improve neural engagement.\n"
            "Rules:\n"
            "  1. html_original must be an EXACT substring of the HTML above (copy-paste it).\n"
            "  2. html_replacement is your improved version of that exact substring.\n"
            "  3. Only change CSS properties or HTML structure — do NOT alter visible text content.\n"
            "  4. Focus on: button colors, font sizes, contrast, whitespace, CTA prominence, "
            "background colors, border-radius, box-shadow, font-weight, padding, layout order.\n"
            "  5. Keep changes minimal and targeted — one element at a time.\n"
            "Output ONE JSON object. No markdown fences:\n"
            "{\n"
            '  "action_type": "<from the HTML_ACTION_TYPES list>",\n'
            '  "target": "<element description: \'primary CTA button\', \'hero section\', etc.>",\n'
            '  "html_original": "<exact substring from the HTML above>",\n'
            '  "html_replacement": "<your improved version>",\n'
            '  "reasoning": "<which brain region this targets, how the change improves engagement>",\n'
            '  "expected_roi_impact": {"language": 0.0, "attention": 0.0, "visual": 0.0}\n'
            "}"
        )

        user_parts.append({"type": "text", "text": user_text})

        print(
            f"[NeuralLens HTML] → OpenAI | iter={iteration} | suggested={suggested_action} "
            f"| html={len(html_excerpt)}chars | gaze={bool(gaze_regions)}"
        )

        resp = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
            max_tokens=1200,
            messages=[
                {"role": "system", "content": _HTML_SYSTEM},
                {"role": "user",   "content": user_parts},
            ],
        )

        usage = resp.usage
        print(
            f"[NeuralLens HTML] ← OpenAI | "
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
                return result
            except json.JSONDecodeError:
                pass

        print("[NeuralLens HTML] ✗ JSON parse failed — returning no-op")
        return _fake_html_edit(iteration, html_content)


# ── HTML system prompt ─────────────────────────────────────────────────────────

_HTML_SYSTEM = """\
You are NeuralLens HTML Designer, an expert in neuromarketing and visual design.
You analyze uploaded HTML pages and propose precise CSS/HTML edits to maximize
predicted neural engagement as measured by TRIBE v2 brain region activations.

You target the VISUAL brain regions primarily:
  V4  (color & form) — improved by: vivid colors, strong contrast, clear visual hierarchy,
       deliberate whitespace, consistent color palette with one dominant accent color
  FFA (face/identity) — improved by: prominent human imagery cues, personalized sections
  NAcc (reward anticipation) — improved by: prominent CTAs, clear benefit visibility,
       high-contrast buttons, visual emphasis on the offer

Design principles you follow:
  • CTAs: high-contrast background (#7c3aed violet or bold accent), adequate padding,
    border-radius 6-10px, font-weight 600+, box-shadow for depth
  • Typography: body 16-18px, headings 2x+ body size, sufficient line-height (1.6+),
    high contrast text (#111 on white or #f8f8f8 on dark)
  • Whitespace: generous padding (2-4rem sections), breathable layout reduces Insula load
  • Color: one dominant accent, neutral background, avoid red/orange urgency cues
    (raises Amygdala)
  • Visual hierarchy: most important element (headline or CTA) should be visually
    dominant — larger, bolder, or higher contrast than everything else

RULES:
  1. html_original must be a verbatim exact copy from the provided HTML.
  2. html_replacement is your modified version — minimal targeted change.
  3. Do NOT change visible text content — only CSS/HTML attributes/structure.
  4. Make one focused change per iteration.
  5. Output valid JSON only, no markdown fences.
"""


# ── HTML_ACTION_TYPES list (used by HtmlOptimizationLoop) ─────────────────────

HTML_ACTION_TYPES = [
    "change_button_color",
    "increase_cta_size",
    "improve_font_contrast",
    "add_whitespace",
    "change_background_color",
    "increase_heading_size",
    "change_font_family",
    "improve_color_scheme",
    "highlight_cta_section",
    "restructure_hero_layout",
]


# ── Stub for HTML mode (OPENAI_LIVE=false) ────────────────────────────────────

def _fake_html_edit(iteration: int, html_content: str) -> dict:
    """Return a no-op stub edit when running without OpenAI credentials."""
    action = HTML_ACTION_TYPES[iteration % len(HTML_ACTION_TYPES)]
    return {
        "action_type": action,
        "target": "stub",
        "html_original": "",
        "html_replacement": "",
        "reasoning": f"Stub mode — set OPENAI_LIVE=true to enable real HTML edits ({action})",
        "expected_roi_impact": {"language": 0.0, "attention": 0.0, "visual": 0.0},
    }
