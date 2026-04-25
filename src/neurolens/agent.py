"""Claude-powered edit agent.

Inputs per iteration:
  * the current screenshot (sent as a vision input)
  * region scores
  * intent
  * ethics flags from the previous iteration
  * the catalog of available programmatic edits

Output: one structured edit choice — `{"edit": <name>, "params": {...}, "rationale": "..."}` —
selected via Claude's tool-use API so we get a typed result back instead of
parsing free text.

Uses Claude Opus 4.7 (`claude-opus-4-7`). Expects ANTHROPIC_API_KEY in env.
A `MockAgent` is provided for tests / offline runs.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from .edit import CATALOG, catalog_for_prompt
from .ethics import EthicsReport
from .regions import REGIONS
from .reward import Intent, RewardBreakdown


@dataclass
class EditChoice:
    edit: str
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


SYSTEM_PROMPT = """You are NeuroUI, an AI design partner that improves UI screenshots by optimizing
for predicted human brain activation. Each iteration you receive:

  * the current screenshot
  * the designer's declared intent (engage / trust / convert / accessibility / gamification)
  * predicted activation scores for 8 brain regions
  * any ethical flags from the prior iteration
  * a catalog of programmatic edits you can request

Your job: pick exactly ONE edit per turn and explain it briefly. Choose the
edit whose target regions, given the current scores, will most improve the
reward for the declared intent — without violating any blocking ethical flags.

Hard rules:
  * NEVER pick an edit that would push amygdala or NAcc up if the run is
    already flagged for those (the supervisor will reject it).
  * Avoid pushing any single region above 0.85 (Yerkes-Dodson ceiling).
  * Prefer reversible, conservative parameter values.

Available edits:
{catalog}

Available regions: {regions}

Respond ONLY by calling the `choose_edit` tool.
"""


CHOOSE_EDIT_TOOL = {
    "name": "choose_edit",
    "description": "Select one edit to apply this iteration.",
    "input_schema": {
        "type": "object",
        "properties": {
            "edit": {
                "type": "string",
                "enum": list(CATALOG.keys()),
                "description": "Name of the edit from the catalog.",
            },
            "params": {
                "type": "object",
                "description": "Edit parameters (typically {factor: number} or {pad_pct: number}). May be empty.",
            },
            "rationale": {
                "type": "string",
                "description": "One short sentence explaining which region the edit targets and why.",
            },
        },
        "required": ["edit", "rationale"],
    },
}


class Agent(Protocol):
    def choose(
        self,
        image: Image.Image,
        scores: dict[str, float],
        reward: RewardBreakdown,
        intent: Intent,
        ethics: EthicsReport,
    ) -> EditChoice: ...


def _image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _format_user_message(
    scores: dict[str, float],
    reward: RewardBreakdown,
    intent: Intent,
    ethics: EthicsReport,
) -> str:
    lines = [f"Intent: {intent}", "", "Current region scores:"]
    for r in REGIONS:
        lines.append(f"  {r:14s} {scores.get(r, 0.0):+.3f}")
    lines += ["", "Reward breakdown:", reward.summary()]
    if ethics.flags:
        lines += ["", "Ethics flags from prior iteration:", ethics.summary()]
    lines += ["", "Choose one edit via the choose_edit tool."]
    return "\n".join(lines)


@dataclass
class ClaudeAgent:
    """Live agent backed by the Anthropic API."""

    model: str = "claude-opus-4-7"
    max_tokens: int = 1024
    api_key: str | None = None

    def __post_init__(self) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._system = SYSTEM_PROMPT.format(
            catalog=catalog_for_prompt(),
            regions=", ".join(REGIONS.keys()),
        )

    def choose(
        self,
        image: Image.Image,
        scores: dict[str, float],
        reward: RewardBreakdown,
        intent: Intent,
        ethics: EthicsReport,
    ) -> EditChoice:
        user_text = _format_user_message(scores, reward, intent, ethics)
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system,
            tools=[CHOOSE_EDIT_TOOL],
            tool_choice={"type": "tool", "name": "choose_edit"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _image_to_b64(image),
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        )
        for block in msg.content:
            if block.type == "tool_use" and block.name == "choose_edit":
                inp = block.input or {}
                return EditChoice(
                    edit=inp["edit"],
                    params=inp.get("params") or {},
                    rationale=inp.get("rationale", ""),
                )
        raise RuntimeError(f"agent did not call choose_edit; got: {msg.content!r}")


@dataclass
class MockAgent:
    """Deterministic offline agent for tests and demos without an API key.

    Picks the edit whose target regions are currently weakest (for non-penalty
    regions) or strongest (for penalty regions). Simple, predictable, useful
    for verifying the loop end-to-end.
    """

    def choose(
        self,
        image: Image.Image,
        scores: dict[str, float],
        reward: RewardBreakdown,
        intent: Intent,
        ethics: EthicsReport,
    ) -> EditChoice:
        # If ethics flagged amygdala/NAcc, prioritize calming edits
        if any(f.code.startswith("dark_pattern") for f in ethics.flags):
            if "Amygdala" in {r for f in ethics.flags for r in (f.code,) if "amyg" in f.code}:
                return EditChoice(
                    edit="desaturate_reds",
                    params={"factor": 0.65},
                    rationale="Ethics flagged amygdala; reducing red dominance.",
                )
            return EditChoice(
                edit="add_whitespace",
                params={"pad_pct": 6},
                rationale="Ethics flagged compulsion risk; adding breathing room.",
            )

        # Pick the most under-performing target for the intent
        from .reward import TARGET_WEIGHTS

        weights = TARGET_WEIGHTS.get(intent, {})
        if not weights:
            return EditChoice(edit="add_whitespace", rationale="default fallback")
        weakest = min(weights, key=lambda r: scores.get(r, 0.0))
        # Map region -> sensible default edit
        region_to_edit = {
            "FFA": "increase_saturation",   # no face-insertion primitive yet
            "V4": "increase_saturation",
            "MT+": "motion_blur",
            "Hippocampus": "increase_contrast",
            "PFC": "add_whitespace",
            "NAcc": "increase_saturation",
        }
        edit = region_to_edit.get(weakest, "add_whitespace")
        return EditChoice(
            edit=edit,
            rationale=f"{weakest} is the weakest target ({scores.get(weakest, 0):.2f}); "
            f"{edit} is its best-matching primitive.",
        )


def load_agent(kind: str = "claude") -> Agent:
    if kind == "claude":
        return ClaudeAgent()
    if kind == "mock":
        return MockAgent()
    raise ValueError(f"unknown agent: {kind!r}")
