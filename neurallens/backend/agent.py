"""Vision-aware Claude optimization agent.

Two modes:
  stub  (default, CLAUDE_LIVE=false) — calls fake_llm_edit, cycles through
        action types systematically.
  live  (CLAUDE_LIVE=true) — uses Claude (claude-sonnet-4-6) with vision + memory.

Memory context is injected every call so the model learns across sessions:
  - action_stats: per-type avg_reward, success_rate, count from SQLite
  - recent_failures: last N failures for the current action_type

Gaze context (when provided):
  - Ranked salient regions from GazePredictor
  - Agent is instructed to prioritize edits in highest-saliency regions

Brain region context (when provided):
  - Full 9-region activation scores from brain_regions.score_screenshot()
  - Ethics flags surfaced to the agent so it can self-correct

select_action() returns a structured edit dict every call.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from stubs import ACTION_TYPES, fake_llm_edit

if TYPE_CHECKING:
    from memory import ExperienceBuffer

_BASE_SYSTEM = """\
You are NeuralLens, a neural engagement optimizer. You analyze webpages and \
propose precise text and copy edits to maximize predicted brain activation as \
measured by TRIBE v2, a multimodal brain encoding model. You reason about which \
brain regions (FFA, V4, MT+, Hippocampus, PFC, ACC, Amygdala, Insula, NAcc) each \
edit targets. You learn from past actions — if an action hurt performance, avoid \
similar actions. You respect ethical guardrails: never propose edits that would \
raise Amygdala above 0.60 or NAcc above 0.70 outside gamification intent. \
Propose ONE specific edit as JSON.\
"""


def _build_system_prompt(
    memory: Optional["ExperienceBuffer"],
    action_type: str,
    gaze_regions: Optional[list] = None,
    brain_scores: Optional[dict] = None,
    ethics_flags: Optional[list] = None,
    intent: str = "engage",
) -> str:
    extra = f"\n\nOptimization intent: {intent}"

    if memory is not None:
        stats = memory.get_action_stats()
        failures = memory.get_negative_experiences(action_type)

        stats_lines = []
        for atype, s in stats.items():
            marker = "✓" if s["avg_reward"] > 0 else "✗"
            stats_lines.append(
                f"  {marker} {atype}: avg_reward={s['avg_reward']:+.4f}, "
                f"success_rate={s['success_rate']:.0%}, n={s['count']}"
            )

        failure_lines = []
        for f in failures[:5]:
            detail = json.loads(f.get("action_detail", "{}"))
            failure_lines.append(
                f"  - target='{detail.get('target', '?')}' "
                f"reward={f['reward']:+.4f} url={f['url']}"
            )

        if stats_lines:
            extra += "\n\nPast performance stats (use this to guide your choice):\n"
            extra += "\n".join(stats_lines)
        if failure_lines:
            extra += f"\n\nRecent failures for '{action_type}' to avoid repeating:\n"
            extra += "\n".join(failure_lines)

    if brain_scores:
        region_lines = "\n".join(
            f"  {r}: {s:.3f}" for r, s in sorted(brain_scores.items(), key=lambda x: -x[1])
        )
        extra += "\n\nBrain region activations (9 HCP-MMP1 regions, stub encoder):\n" + region_lines

    if ethics_flags:
        flag_lines = "\n".join(
            f"  [{f['severity'].upper()}] {f['code']}: {f['message']}"
            for f in ethics_flags
        )
        extra += "\n\n⚠ Ethics flags — do NOT propose edits that worsen these:\n" + flag_lines

    if gaze_regions:
        gaze_lines = "\n".join(
            f"  #{r['rank']} saliency={r['saliency_score']:.2f}  "
            f"region={r['bbox']}  peak={r['peak_coords']}"
            for r in gaze_regions[:5]
        )
        extra += (
            "\n\nGaze analysis — predicted human fixation map (rank 1 = most attended):\n"
            + gaze_lines
            + "\n\nPRIORITY: target the content in the rank-1 salient region first."
        )

    return _BASE_SYSTEM + extra


class OptimizationAgent:
    def __init__(self, memory: Optional["ExperienceBuffer"] = None) -> None:
        self.live = os.getenv("CLAUDE_LIVE", "false").lower() == "true"
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
        action_type = self._next_action_type()

        screenshot_b64 = ""
        if screenshot_path and Path(screenshot_path).exists():
            with open(screenshot_path, "rb") as fh:
                screenshot_b64 = base64.b64encode(fh.read()).decode()

        if not self.live:
            text = page_content.text if hasattr(page_content, "text") else str(page_content)
            return await fake_llm_edit(screenshot_b64, text, score_history, action_type)

        return await self._live_select(
            screenshot_b64, page_content, score_history, action_type,
            gaze_regions=gaze_regions,
            brain_scores=brain_scores,
            ethics_flags=ethics_flags,
            intent=intent,
        )

    async def _live_select(
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
        import anthropic  # type: ignore

        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        system = _build_system_prompt(
            self._memory, action_type, gaze_regions, brain_scores, ethics_flags, intent
        )

        current = score_history[-1] if score_history else {}
        score_lines = "\n".join(
            f"  {k}: {v:.4f}"
            for k, v in current.items()
            if k.endswith("_roi") or k == "overall_score"
        )
        history_lines = "\n".join(
            f"  iter {i}: overall={s.get('overall_score', 0):.4f}  "
            f"lang={s.get('language_roi', 0):.4f}  "
            f"attn={s.get('attention_roi', 0):.4f}  "
            f"vis={s.get('visual_roi', 0):.4f}"
            for i, s in enumerate(score_history[-8:])
        )

        text_excerpt = (
            page_content.text if hasattr(page_content, "text") else str(page_content)
        )[:2500]

        gaze_hint = ""
        if gaze_regions:
            top = gaze_regions[0]
            gaze_hint = (
                f"\n\nGaze priority: Region #1 (saliency={top['saliency_score']:.2f}) is "
                f"at bounding-box {top['bbox']}. Target content in this area first."
            )

        user_parts: list = []
        if screenshot_b64:
            user_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_b64,
                },
            })

        user_parts.append({
            "type": "text",
            "text": (
                f"URL: {getattr(page_content, 'url', '')}\n\n"
                f"Current neural scores:\n{score_lines}\n\n"
                f"Full score history (most recent last):\n{history_lines}"
                f"{gaze_hint}\n\n"
                f"Page text excerpt:\n{text_excerpt}\n\n"
                f"Propose ONE edit of action_type '{action_type}'. "
                "You may override the action_type if memory stats show it performs poorly "
                "and another type would be more impactful. "
                "Respond ONLY with valid JSON, no markdown fences:\n"
                "{\n"
                '  "action_type": "<type>",\n'
                '  "target": "<element or section to change>",\n'
                '  "original": "<exact original text verbatim>",\n'
                '  "replacement": "<improved text>",\n'
                '  "reasoning": "<which brain region this targets and why>",\n'
                '  "expected_roi_impact": {"language": 0.0, "attention": 0.0, "visual": 0.0}\n'
                "}"
            ),
        })

        print(f"[NeuralLens] → Claude call | action={action_type} | intent={intent} | gaze={bool(gaze_regions)}")
        resp = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user_parts}],
        )

        usage = resp.usage
        print(f"[NeuralLens] ← Claude reply | input={usage.input_tokens} output={usage.output_tokens} tokens")
        raw = resp.content[0].text if resp.content else ""
        raw = raw.strip()
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

        text_excerpt_fallback = (
            page_content.text if hasattr(page_content, "text") else str(page_content)
        )[:2500]
        return await fake_llm_edit(screenshot_b64, text_excerpt_fallback, score_history, action_type)
