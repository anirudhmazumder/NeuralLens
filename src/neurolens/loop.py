"""End-to-end optimization loop.

For N iterations:
    1. encode current image -> region scores
    2. compute reward
    3. evaluate ethics (using prev scores for trend checks)
    4. ask agent for one edit
    5. veto if the edit targets a region the ethics layer blocked
    6. apply edit -> new image
    7. record everything to the run directory

The loop accepts the encoder/agent as parameters so tests can swap in
deterministic implementations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .agent import Agent, EditChoice, MockAgent
from .edit import CATALOG, apply_edit
from .ethics import EthicsReport, evaluate as ethics_evaluate
from .reward import Intent, compute as reward_compute
from .tribe import Encoder, StubEncoder, load_image
from .transparency import append_jsonl, iteration_record, render_run


@dataclass
class LoopConfig:
    iterations: int = 5
    intent: Intent = "engage"
    out_dir: Path = Path("runs")
    save_every: bool = True


def _veto_if_unsafe(choice: EditChoice, ethics: EthicsReport) -> EditChoice | None:
    """Reject edits that would push a blocked region in the wrong direction."""
    if not ethics.blocked:
        return choice
    spec = CATALOG.get(choice.edit)
    if spec is None:
        return None  # unknown edit -> reject
    blocked_codes = {f.code for f in ethics.flags if f.severity == "block"}
    # Block any edit that *raises* amygdala or NAcc when those are flagged
    if "dark_pattern_amygdala" in blocked_codes and "Amygdala" in spec.targets_regions:
        return None
    if "dark_pattern_nacc" in blocked_codes and "NAcc" in spec.targets_regions:
        return None
    # Conservative: block any saturation/contrast/sharpen boost while flagged
    if choice.edit in {"increase_saturation", "increase_contrast", "sharpen"} and blocked_codes:
        return None
    return choice


def run(
    image_path: str | Path,
    config: LoopConfig,
    encoder: Encoder | None = None,
    agent: Agent | None = None,
) -> dict[str, Any]:
    encoder = encoder or StubEncoder()
    agent = agent or MockAgent()

    image_path = Path(image_path)
    run_dir = config.out_dir / f"{image_path.stem}_{config.intent}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "iterations.jsonl"
    if log_path.exists():
        log_path.unlink()

    img = load_image(image_path)
    if config.save_every:
        img.save(run_dir / "iter_00_baseline.png")

    records: list[dict[str, Any]] = []
    prev_scores: dict[str, float] | None = None
    can_render_heatmap = False
    heatmap_error_printed = False

    atlas = getattr(encoder, "atlas", None)
    masks = getattr(encoder, "masks", None)
    if atlas is not None and masks is not None:
        can_render_heatmap = True

    def _maybe_render_heatmap(iteration: int, scores: dict[str, float]) -> None:
        nonlocal can_render_heatmap, heatmap_error_printed
        if not can_render_heatmap:
            return
        try:
            from .rois import render_heatmap_png

            render_heatmap_png(
                scores,
                atlas=atlas,
                masks=masks,
                out_path=run_dir / f"fmri_{iteration:02d}.png",
            )
        except ImportError as e:
            can_render_heatmap = False
            if not heatmap_error_printed:
                print(f"heatmap rendering disabled: {e}")
                heatmap_error_printed = True

    for i in range(config.iterations + 1):
        scores = encoder.encode(img)
        _maybe_render_heatmap(i, scores)
        reward = reward_compute(scores, config.intent)
        ethics = ethics_evaluate(scores, config.intent, prev_scores)

        if i == config.iterations:
            rec = iteration_record(i, config.intent, scores, prev_scores, reward, None, ethics)
            append_jsonl(log_path, rec)
            records.append(rec)
            break

        choice = agent.choose(img, scores, reward, config.intent, ethics)
        safe_choice = _veto_if_unsafe(choice, ethics)
        rec = iteration_record(
            i, config.intent, scores, prev_scores, reward, safe_choice or choice, ethics
        )
        if safe_choice is None:
            rec["edit_vetoed"] = True
            rec["edit"]["rationale"] += "  [VETOED by ethics supervisor]"
        append_jsonl(log_path, rec)
        records.append(rec)

        if safe_choice is not None:
            img = apply_edit(img, safe_choice.edit, safe_choice.params)
            if config.save_every:
                img.save(run_dir / f"iter_{i+1:02d}_{safe_choice.edit}.png")

        prev_scores = scores

    img.save(run_dir / "final.png")
    render_run(records, run_dir / "report.md")
    (run_dir / "summary.json").write_text(json.dumps(records, indent=2))
    return {"run_dir": str(run_dir), "records": records}
