"""OpenAI-powered NeuralLens optimization agent.

Uses GPT-4.1-nano with vision. Receives:
  - Screenshot (base64 PNG)
  - Cleaned body HTML excerpt (structural context for finding exact elements)
  - Current TRIBE brain region activations (9 HCP-MMP1 regions)
  - NES loss value + summary (the objective we're minimizing)
  - Gaze analysis (ranked salient regions from DeepGaze IIE)
  - Memory stats (cross-session learning — what has worked before)
  - Ethics flags (regions approaching dark-pattern thresholds)
  - Score history (trajectory over the episode)

Proposes ONE edit per call as structured JSON.
If OPENAI_LIVE=false, raises RuntimeError — stub mode has been removed.
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
You are NeuralLens, an AI agent that optimizes web pages for neural engagement \
as measured by TRIBE v2 — a multimodal brain encoder trained on fMRI data that predicts \
HCP-MMP1 cortical activation from page screenshots and text content.

## Your Objective: Minimize the NES Score

NES = Neurological Evaluation Scale. It measures neurological soft signs (NSS) — \
subtle, non-localizable indicators of dysfunction in the cerebello-thalamo-prefrontal \
cortical circuit. A HIGH NES score means more dysfunction. You MINIMIZE it each iteration.

NSS are triggered by two failure modes:

  HYPER-ACTIVATION (region exceeds threshold → circuit overload):
    ACC       > 0.40 → motor-sequencing conflict, task-switching rigidity     weight 2.0
    Amygdala  > 0.40 → emotional dysregulation, disinhibition                 weight 2.0
    Insula    > 0.40 → sensory integration overload, interoceptive noise       weight 1.5

  HYPO-ACTIVATION (region below threshold → circuit failure):
    PFC         < 0.50 → prefrontal hypofunctionality, executive dropout       weight 1.5
    Hippocampus < 0.40 → memory-encoding failure, novelty-gating deficit       weight 1.0

Content with LOW NES score: does not trigger fear/urgency, does not overload senses, \
does not create cognitive conflict, actively engages prefrontal executive attention, \
and encodes novelty into memory.

## The 9 Brain Regions (HCP-MMP1)

| Region      | Circuit role                              | NSS signal when | Hard limit |
|-------------|-------------------------------------------|-----------------|------------|
| FFA         | Face/identity recognition                 | — (engagement)  | —          |
| V4          | Color & form perception                   | — (engagement)  | —          |
| MT+         | Motion & visual flow                      | — (engagement)  | —          |
| Hippocampus | Memory encoding & novelty gating          | < 0.40 (low)    | —          |
| PFC         | Executive attention & trust               | < 0.50 (low)    | —          |
| NAcc        | Reward anticipation                       | — (engagement)  | ≤ 0.70     |
| Amygdala    | Fear, anxiety, urgency                    | > 0.40 (high)   | ≤ 0.60     |
| Insula      | Sensory overload, discomfort              | > 0.40 (high)   | ≤ 0.55     |
| ACC         | Cognitive conflict, sequencing rigidity   | > 0.40 (high)   | ≤ 0.55     |

**HARD GUARDRAIL**: Never propose an edit that would raise Amygdala above 0.60 \
or NAcc above 0.70 (outside gamification intent). These are dark-pattern thresholds.

## Your Edit Skills

{SKILLS}

## How to Use the Context You Receive

1. **Screenshot** — visual context. Identify the dominant element. Note clutter, color balance,
   hierarchy issues.

2. **HTML excerpt** — structural context. This is the page's body innerHTML with scripts removed.
   Use it to find the EXACT text to replace (copy it verbatim into the "original" field).
   Target elements by their CSS classes, IDs, and tag types shown here.

3. **TRIBE region scores** — current brain state. Focus on:
   - Which positive regions are lowest (most room to improve)?
   - Which penalty regions are highest (most urgency to fix)?

4. **NES loss** — single number. Your job is to reduce it. A −0.02 improvement per step
   compounds significantly over 10 iterations.

5. **Gaze analysis** — where humans look first. The rank-1 region gets the most attention.
   Prioritize editing content in that area unless ethics flags demand otherwise.

6. **Memory stats** — what has worked before across all sessions. If an action_type has
   negative avg_reward, avoid it unless you have a compelling reason. If one has high
   avg_reward, prefer it.

7. **Ethics flags** — if any flag has severity "block", DO NOT propose an edit that would
   worsen that region. Address the flagged region first.

## How to Write the "original" Field

This is critical. The "original" value MUST:
- Appear VERBATIM in the HTML or page text you received (copy-paste it)
- Be the exact string that will be search-replaced in the DOM
- Not include HTML tags (plain text content only)
- Be unique enough to identify the right element

If you cannot find exact text to replace, set "original" to "" and use "target" to
describe what should be changed (the system will handle it structurally).

## Output Format

Respond ONLY with valid JSON. No markdown fences, no explanation outside the JSON.

{{
  "action_type": "<skill_name from the list above>",
  "target": "<human-readable description of the element, e.g. 'primary H1 heading'>",
  "html_selector": "<CSS selector for the target, e.g. 'h1.hero-title'>",
  "original": "<exact verbatim text from the page to replace>",
  "replacement": "<improved text>",
  "reasoning": "<which brain regions this targets, why it reduces NES loss, citing current scores>",
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
        lines.append("Cross-session action performance (use to guide your choice):")
        for atype, s in sorted(stats.items(), key=lambda x: -x[1]["avg_reward"]):
            mark = "✓" if s["avg_reward"] > 0 else "✗"
            lines.append(
                f"  {mark} {atype}: avg_reward={s['avg_reward']:+.4f}  "
                f"success={s['success_rate']:.0%}  n={s['count']}"
            )

    if failures:
        lines.append(f"\nRecent failures for '{action_type}' — do not repeat these patterns:")
        for f in failures[:4]:
            detail = json.loads(f.get("action_detail", "{}"))
            lines.append(
                f"  - target='{detail.get('target', '?')}' "
                f"reward={f['reward']:+.4f}"
            )

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

        action_type = self._next_action_type()

        screenshot_b64 = ""
        if screenshot_path and Path(screenshot_path).exists():
            with open(screenshot_path, "rb") as fh:
                screenshot_b64 = base64.b64encode(fh.read()).decode()

        return await self._call_openai(
            screenshot_b64, page_content, score_history, action_type,
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
        action_type: str,
        gaze_regions: Optional[list] = None,
        brain_scores: Optional[dict] = None,
        ethics_flags: Optional[list] = None,
        intent: str = "engage",
    ) -> dict:
        import openai  # type: ignore

        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # ── Assemble context blocks ────────────────────────────────────────────

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
            for i, s in enumerate(score_history[-6:])
        )

        # NES soft-sign block
        nes_block = ""
        if brain_scores:
            from nes import nes_summary
            ns = nes_summary(brain_scores, intent)
            violations_str = "  ".join(
                f"{r}+{v:.3f}" for r, v in ns.get("violations", {}).items()
            ) or "none"
            nes_block = (
                f"\nNES dysfunction score: {ns['nes_score']:.4f}  "
                f"(lower = fewer neurological soft signs)\n"
                f"Active NSS violations: {violations_str}\n"
                f"Dominant soft-sign driver: {ns['dominant_penalty']}  |  "
                f"Strongest engagement region: {ns['dominant_positive']}"
            )

        # Brain region scores
        brain_block = ""
        if brain_scores:
            region_lines = "\n".join(
                f"  {r}: {v:.4f}" for r, v in sorted(brain_scores.items(), key=lambda x: -x[1])
            )
            brain_block = f"\nCurrent TRIBE region activations:\n{region_lines}"

        # Gaze analysis
        gaze_block = ""
        if gaze_regions:
            top = gaze_regions[0]
            gaze_lines = "\n".join(
                f"  #{r['rank']} saliency={r['saliency_score']:.2f}  bbox={r['bbox']}  peak={r['peak_coords']}"
                for r in gaze_regions[:4]
            )
            gaze_block = (
                f"\nGaze fixation map (rank 1 = most attended by human viewers):\n{gaze_lines}\n"
                f"PRIORITY: focus your edit on the content near rank-1 gaze point {top['peak_coords']}."
            )

        # Ethics flags
        ethics_block = ""
        if ethics_flags:
            flag_lines = "\n".join(
                f"  [{f['severity'].upper()}] {f['code']}: {f['message']}"
                for f in ethics_flags
            )
            ethics_block = f"\n⚠ Ethics flags (do NOT worsen these):\n{flag_lines}"

        # Memory
        memory_block = _build_memory_block(self._memory, action_type)

        # HTML excerpt — main source for finding exact text
        html_excerpt = ""
        if hasattr(page_content, "html") and page_content.html:
            html_excerpt = page_content.html[:4000]
        text_excerpt = (
            page_content.text if hasattr(page_content, "text") else str(page_content)
        )[:2000]

        # ── Build user message ────────────────────────────────────────────────

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
            f"Intent: {intent}\n"
            f"\n--- Current TRIBE scores ---\n{score_block}\n"
            f"\n--- Score history (last 6 iters) ---\n{history_block}\n"
            f"{nes_block}"
            f"\n--- Brain region activations ---\n"
            + (
                "\n".join(f"  {r}: {v:.4f}" for r, v in sorted(brain_scores.items(), key=lambda x: -x[1]))
                if brain_scores else "  (not available)"
            )
            + f"\n{gaze_block}"
            + f"\n{ethics_block}"
            + (f"\n\n--- Memory stats ---\n{memory_block}" if memory_block else "")
            + f"\n\n--- Page HTML (find exact text here) ---\n{html_excerpt}"
            + f"\n\n--- Page text (plain) ---\n{text_excerpt}"
            + f"\n\nPropose ONE edit of action_type '{action_type}'. "
            "Override the action_type only if memory stats show it consistently negative "
            "AND another type has significantly higher avg_reward. "
            "Respond ONLY with valid JSON, no markdown fences."
        )

        user_parts.append({"type": "text", "text": user_text})

        nes_val = nes_block.split()[3] if nes_block and len(nes_block.split()) > 3 else "?"
        print(
            f"[NeuralLens] → OpenAI | action={action_type} | intent={intent} "
            f"| nes={nes_val} "
            f"| gaze={bool(gaze_regions)}"
        )

        resp = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
            max_tokens=600,
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

        # Fallback: return a minimal no-op so the loop doesn't crash
        print("[NeuralLens] ✗ JSON parse failed — returning no-op edit")
        return {
            "action_type": action_type,
            "target": "",
            "html_selector": "",
            "original": "",
            "replacement": "",
            "reasoning": "JSON parse failed",
            "expected_roi_impact": {"language": 0.0, "attention": 0.0, "visual": 0.0},
            "expected_roi": "language_roi",
        }
