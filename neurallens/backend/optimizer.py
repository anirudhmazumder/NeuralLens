"""RL-style neural optimizer.

Each run is an "episode":
  - Epsilon-greedy action selection (explore early, exploit learned patterns later)
  - Shaped reward: score_delta + language_bonus + novelty_bonus
  - Experiences stored in both memory.py (simple) and pattern_learner.py (feature-based)
  - Pattern discovery triggered every 5 experiences
  - OptimizationLoop aliased to NeuralOptimizer for backward compatibility
"""
from __future__ import annotations

import asyncio
import os
import random
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright  # type: ignore[import-untyped]

import re as _re

from agent import OptimizationAgent
from memory import get_buffer
from pattern_learner import get_library
from scraper import PageContent, scrape
from scorer import TribeScorer
from stubs import ACTION_TYPES
from visualizer import PageVisualizer


@dataclass
class JobState:
    job_id: str
    url: str
    status: str = "pending"
    max_iterations: int = 10
    current_iteration: int = 0
    intent: str = "engage"
    result: Optional[dict] = None
    error: Optional[str] = None
    events: list[dict] = field(default_factory=list)
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    gaze_regions: list = field(default_factory=list)


jobs: dict[str, JobState] = {}


# ── Reward shaping ────────────────────────────────────────────────────────────

def _shaped_reward(score_delta: float, roi_deltas: dict, novelty: bool) -> float:
    base = score_delta * 10
    lang_bonus = roi_deltas.get("language", 0) * 3
    novelty_bonus = 0.05 if novelty else 0.0
    return round(base + lang_bonus + novelty_bonus, 4)


def _exploitative_action() -> str:
    """Pick action type with best historical avg_reward, or random if no history."""
    stats = get_buffer().get_action_stats()
    if not stats:
        return random.choice(ACTION_TYPES)
    return max(stats, key=lambda k: stats[k]["avg_reward"])


def _exploratory_action(tried: list[str]) -> str:
    untried = [a for a in ACTION_TYPES if a not in tried]
    return random.choice(untried) if untried else random.choice(ACTION_TYPES)


# ── Viz-component helpers ─────────────────────────────────────────────────────

def _text_to_viz_components(text: str) -> list[dict]:
    """Split page text into lightweight component dicts for annotation."""
    import uuid as _uuid
    blocks = _re.split(r"\n{2,}", text.strip())
    comps: list[dict] = []
    for block in blocks:
        block = block.strip()
        if not block or len(block) < 10:
            continue
        words = block.split()
        wc = len(words)
        comp_type = "headline" if wc <= 10 and not block.endswith(".") else "body"
        comps.append({
            "id": _uuid.uuid4().hex[:8],
            "type": comp_type,
            "content": block[:200],
            "neural_score": 0.5,
            "is_target": False,
        })
        if len(comps) >= 15:
            break
    return comps


def _find_target_comp(viz_components: list[dict], original: str) -> str:
    """Return the id of the component most likely containing the edit target."""
    if not original or not viz_components:
        return viz_components[0]["id"] if viz_components else ""
    for comp in viz_components:
        if original[:80] in comp["content"] or comp["content"][:80] in original:
            return comp["id"]
    return viz_components[0]["id"]


# ── Main optimizer ────────────────────────────────────────────────────────────

@dataclass
class NeuralOptimizer:
    scorer: TribeScorer = field(default_factory=TribeScorer)
    agent: OptimizationAgent = field(
        default_factory=lambda: OptimizationAgent(memory=get_buffer())
    )
    page_visualizer: PageVisualizer = field(default_factory=PageVisualizer)

    async def run(
        self,
        url: str,
        max_iterations: int = 10,
        job_id: str = "",
        intent: str = "engage",
    ) -> dict:
        job_id = job_id or str(uuid.uuid4())
        job = jobs.get(job_id)
        if job:
            job.intent = intent
        pattern_lib = get_library()

        work_dir = os.path.join(tempfile.gettempdir(), "neurallens", job_id)
        os.makedirs(work_dir, exist_ok=True)

        try:
            # ── 1. Scrape ─────────────────────────────────────────────────
            await self._emit(job, "progress", {
                "status": "scraping",
                "message": "Simulating human browsing the page…",
                "iteration_count": 0, "max_iterations": max_iterations,
            })
            page = await scrape(url, work_dir)
            viz_components = _text_to_viz_components(page.text)

            # ── 2. Gaze analysis ──────────────────────────────────────────
            await self._emit(job, "progress", {
                "status": "gaze_analysis",
                "message": "Predicting human gaze patterns…",
                "iteration_count": 0, "max_iterations": max_iterations,
            })
            from gaze_predictor import get_gaze_predictor
            gaze_data = await get_gaze_predictor().analyze(page.screenshot_path)
            gaze_regions = gaze_data["regions"]
            if job:
                job.gaze_regions = gaze_regions
            await self._emit(job, "gaze", {
                "status": "gaze_complete",
                "message": (
                    f"Gaze: {len(gaze_regions)} salient regions mapped"
                    f" ({'live DeepGaze IIE' if gaze_data.get('gaze_live') else 'F-pattern stub'})"
                ),
                "gaze_regions": gaze_regions,
                "gaze_live": gaze_data.get("gaze_live", False),
            })

            # ── 3. Baseline ───────────────────────────────────────────────
            await self._emit(job, "progress", {
                "status": "scoring",
                "message": "Computing baseline neural activation…",
                "iteration_count": 0, "max_iterations": max_iterations,
            })
            baseline_score = await self.scorer.score(
                page.video_path, page.text, page.audio_path,
                screenshot_path=page.screenshot_path,
            )
            current_score = baseline_score.copy()
            current_text = page.text
            episode_tried: list[str] = []
            history: list[dict] = []
            accepted_edits: list[dict] = []
            accepted_comp_ids: list[str] = []
            current_screenshot_path: str = page.screenshot_path or ""
            pending_resample: Optional[asyncio.Task] = None

            # Seed viz_components with proportionally-weighted baseline scores
            n_vc = max(len(viz_components), 1)
            for i, vc in enumerate(viz_components):
                w = 1.0 - (i / n_vc) * 0.3
                vc["neural_score"] = round(
                    min(1.0, baseline_score["overall_score"] * w), 4
                )

            # ── Brain region scoring at baseline ──────────────────────────
            from brain_regions import evaluate_ethics, compute_intent_reward
            baseline_brain = await self.scorer.score_brain_regions(
                page.screenshot_path or None, text=page.text
            )
            current_brain = baseline_brain.copy()
            baseline_ethics = evaluate_ethics(baseline_brain, intent)

            await self._emit(job, "progress", {
                "status": "baseline",
                "message": f"Baseline overall: {baseline_score['overall_score']:.4f}",
                "score": baseline_score,
                "iteration_count": 0, "max_iterations": max_iterations,
            })
            await self._emit(job, "brain_regions", {
                "iteration_count": 0,
                "regions": baseline_brain,
                "ethics_flags": baseline_ethics,
                "intent": intent,
                "is_baseline": True,
            })
            if job:
                job.current_iteration = 0

            # ── 3. RL episode ─────────────────────────────────────────────
            for step in range(1, max_iterations + 1):
                if job:
                    job.current_iteration = step

                # ── Drain background resample if ready ────────────────────
                if pending_resample is not None and pending_resample.done():
                    try:
                        resample_result = pending_resample.result()
                        if resample_result:
                            current_screenshot_path, gaze_regions = resample_result
                            if job:
                                job.gaze_regions = gaze_regions
                            await self._emit(job, "gaze", {
                                "status": "gaze_resample",
                                "message": (
                                    f"[{step}/{max_iterations}] Gaze resampled — "
                                    f"{len(gaze_regions)} regions from updated page"
                                ),
                                "gaze_regions": gaze_regions,
                                "iteration_count": step,
                                "max_iterations": max_iterations,
                            })
                    except Exception as _exc:
                        print(f"[resample] drain error: {_exc}")
                    pending_resample = None

                # Epsilon-greedy: explore early, exploit later
                epsilon = max(0.1, 1.0 - step / max_iterations)
                exploit = random.random() > epsilon
                strategy = "exploit" if exploit else "explore"

                if exploit:
                    action_type = _exploitative_action()
                else:
                    action_type = _exploratory_action(episode_tried)

                # Override agent's cycling index to the selected action
                if action_type in ACTION_TYPES:
                    self.agent._action_index = ACTION_TYPES.index(action_type)

                await self._emit(job, "progress", {
                    "status": "proposing",
                    "message": (
                        f"[{step}/{max_iterations}] Agent proposing edit "
                        f"({strategy}, ε={epsilon:.2f})…"
                    ),
                    "iteration_count": step, "max_iterations": max_iterations,
                    "strategy": strategy, "epsilon": round(epsilon, 2),
                    "action_type": action_type,
                })

                current_ethics = evaluate_ethics(current_brain, intent)
                score_history = [baseline_score] + [h["score"] for h in history]
                edit = await self.agent.select_action(
                    page, score_history, step,
                    screenshot_path=current_screenshot_path,
                    gaze_regions=gaze_regions or None,
                    brain_scores=current_brain,
                    ethics_flags=current_ethics,
                    intent=intent,
                )
                episode_tried.append(edit.get("action_type", action_type))

                # Apply text edit
                updated_text = current_text
                original = edit.get("original", "")
                replacement = edit.get("replacement", "")
                if original and original in current_text:
                    updated_text = current_text.replace(original, replacement, 1)

                # Find which viz_component is being targeted
                target_id = _find_target_comp(viz_components, original)
                for vc in viz_components:
                    vc["is_target"] = vc["id"] == target_id

                await self._emit(job, "progress", {
                    "status": "scoring",
                    "message": f"[{step}/{max_iterations}] Scoring updated content…",
                    "iteration_count": step, "max_iterations": max_iterations,
                })

                # Run annotation and scoring concurrently
                _loop = asyncio.get_event_loop()
                annot_task = _loop.run_in_executor(
                    None,
                    self.page_visualizer.draw_annotated_screenshot,
                    current_screenshot_path,
                    [dict(vc) for vc in viz_components],
                    list(gaze_regions),
                    target_id,
                    list(accepted_comp_ids),
                    [],
                )
                new_score = await self.scorer.score(
                    page.video_path, updated_text, page.audio_path,
                    screenshot_path=current_screenshot_path,
                )
                annotated_b64 = await annot_task

                # Brain region scoring on current screenshot (fast local stub)
                new_brain = await self.scorer.score_brain_regions(
                    current_screenshot_path or None, text=updated_text
                )
                step_ethics = evaluate_ethics(new_brain, intent, prev_scores=current_brain)
                intent_reward = compute_intent_reward(new_brain, current_brain, intent)

                score_delta = new_score["overall_score"] - current_score["overall_score"]
                roi_deltas = {
                    "language": round(
                        new_score.get("language_roi", 0) - current_score.get("language_roi", 0), 4
                    ),
                    "attention": round(
                        new_score.get("attention_roi", 0) - current_score.get("attention_roi", 0), 4
                    ),
                    "visual": round(
                        new_score.get("visual_roi", 0) - current_score.get("visual_roi", 0), 4
                    ),
                }

                novelty = edit.get("action_type", "") not in episode_tried[:-1]
                shaped = _shaped_reward(score_delta, roi_deltas, novelty)
                accepted = score_delta > 0

                # Persist to both memory systems
                get_buffer().store(
                    url=url, action=edit,
                    before_score=current_score["overall_score"],
                    after_score=new_score["overall_score"],
                    reward=score_delta, accepted=accepted,
                    roi_deltas=roi_deltas, page_text=current_text[:500],
                )
                pattern_lib.store_experience(
                    url=url,
                    component_type=_infer_component_type(edit),
                    action_type=edit.get("action_type", "unknown"),
                    before_content=original or current_text[:200],
                    after_content=replacement or updated_text[:200],
                    before_scores=current_score,
                    after_scores=new_score,
                    reward=score_delta,
                    accepted=accepted,
                )

                if accepted:
                    current_score = new_score
                    current_text = updated_text
                    current_brain = new_brain
                    accepted_edits.append(edit)
                    accepted_comp_ids.append(target_id)
                    # Fire background re-render + gaze re-sample (non-blocking)
                    if pending_resample is None or pending_resample.done():
                        pending_resample = asyncio.create_task(
                            _resample_page(url, accepted_edits[:], work_dir, step)
                        )

                # Update viz_component neural scores after each step
                overall = new_score["overall_score"]
                for i, vc in enumerate(viz_components):
                    w = 1.0 - (i / max(n_vc, 1)) * 0.3
                    vc["neural_score"] = round(min(1.0, overall * w), 4)
                    vc["is_target"] = False  # reset after scoring

                agent_thought = {
                    "step": "accepted" if accepted else "rejected",
                    "action_type": edit.get("action_type", action_type),
                    "target": edit.get("target", ""),
                    "reasoning": edit.get("reasoning", ""),
                    "original": edit.get("original", "")[:150],
                    "replacement": edit.get("replacement", "")[:150],
                    "expected_roi_impact": edit.get("expected_roi_impact", {}),
                    "strategy": strategy,
                    "epsilon": round(epsilon, 2),
                    "score_delta": round(score_delta, 4),
                    "new_score": round(new_score["overall_score"], 4),
                    "prev_score": round(current_score["overall_score"] if not accepted else baseline_score["overall_score"], 4),
                }

                memory_count = pattern_lib.get_experience_count()

                entry = {
                    "iteration": step,
                    "action": edit,
                    "reward": round(score_delta, 4),
                    "shaped_reward": shaped,
                    "score": new_score,
                    "accepted": accepted,
                    "roi_deltas": roi_deltas,
                    "strategy": strategy,
                    "epsilon": round(epsilon, 2),
                }
                history.append(entry)

                await self._emit(job, "progress", {
                    "status": "iteration_complete",
                    "message": (
                        f"[{step}/{max_iterations}] "
                        f"{'✓ Accepted' if accepted else '✗ Rejected'} "
                        f"({score_delta:+.4f})"
                    ),
                    "iteration_count": step, "max_iterations": max_iterations,
                    "edit": edit, "reward": score_delta,
                    "shaped_reward": shaped,
                    "intent_reward": intent_reward,
                    "score": new_score, "accepted": accepted,
                    "roi_deltas": roi_deltas, "strategy": strategy,
                    "annotated_screenshot_base64": annotated_b64,
                    "agent_thought": agent_thought,
                    "viz_components": [dict(vc) for vc in viz_components],
                    "gaze_regions": gaze_regions,
                    "memory_count": memory_count,
                })
                await self._emit(job, "brain_regions", {
                    "iteration_count": step,
                    "regions": new_brain,
                    "ethics_flags": step_ethics,
                    "intent": intent,
                    "intent_reward": intent_reward,
                    "accepted": accepted,
                })

            # ── 4. Render optimized screenshot ────────────────────────────
            await self._emit(job, "progress", {
                "status": "rendering",
                "message": "Re-rendering page with accepted edits…",
                "iteration_count": max_iterations,
                "max_iterations": max_iterations,
            })
            after_shot = await _render_optimized_screenshot(url, accepted_edits, work_dir)

            # ── 5. Discover patterns from this episode ────────────────────
            pattern_lib.discover_patterns()

            # ── 6. Final result ───────────────────────────────────────────
            bv = baseline_score["overall_score"]
            fv = current_score["overall_score"]
            improvement_pct = ((fv - bv) / max(bv, 1e-6)) * 100 if bv > 0 else 0.0

            final_ethics = evaluate_ethics(current_brain, intent)
            result = {
                "job_id": job_id, "url": url,
                "baseline_score": baseline_score,
                "final_score": current_score,
                "improvement_pct": round(improvement_pct, 2),
                "iterations": max_iterations,
                "intent": intent,
                "history": history,
                "accepted_edits": accepted_edits,
                "final_content": current_text,
                "before_screenshot": page.screenshot_path,
                "after_screenshot": after_shot,
                "discovered_patterns": get_library().get_experience_count(),
                "gaze_regions": gaze_regions,
                "gaze_live": gaze_data.get("gaze_live", False),
                "baseline_brain_regions": baseline_brain,
                "final_brain_regions": current_brain,
                "ethics_flags": final_ethics,
            }

            if job:
                job.status = "complete"
                job.result = result

            await self._emit(job, "complete", result)
            return result

        except Exception as exc:
            msg = str(exc)
            if job:
                job.status = "error"
                job.error = msg
            await self._emit(job, "error", {"message": msg})
            raise

    @staticmethod
    async def _emit(job: Optional[JobState], event_type: str, data: dict) -> None:
        if job is None:
            return
        event = {"type": event_type, "data": data}
        job.events.append(event)
        await job.event_queue.put(event)


# Backward-compat alias used by main.py
OptimizationLoop = NeuralOptimizer


def _infer_component_type(edit: dict) -> str:
    action = edit.get("action_type", "")
    if "headline" in action or "hero" in action:
        return "headline"
    if "cta" in action:
        return "cta"
    if "body" in action or "paragraph" in action:
        return "body"
    if "meta" in action or "description" in action:
        return "meta"
    if "social_proof" in action or "testimonial" in action:
        return "testimonial"
    return "body"


async def _resample_page(
    url: str, accepted_edits: list, work_dir: str, step: int
) -> Optional[tuple[str, list]]:
    """Re-render page with current accepted edits, take screenshot, re-run gaze.

    Runs as a fire-and-forget background task so it never blocks the RL loop.
    Returns (screenshot_path, gaze_regions) or None on failure.
    """
    from gaze_predictor import get_gaze_predictor
    try:
        ss_path = await _render_optimized_screenshot(
            url, accepted_edits, work_dir, suffix=f"_step{step}"
        )
        if not ss_path:
            return None
        gaze_data = await get_gaze_predictor().analyze(ss_path)
        return ss_path, gaze_data.get("regions", [])
    except Exception as exc:
        print(f"[resample] step {step} failed: {exc}")
        return None


async def _render_optimized_screenshot(
    url: str, accepted_edits: list, out_dir: str, suffix: str = ""
) -> str:
    dest = str(Path(out_dir) / f"optimized_screenshot{suffix}.png")
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=["--no-sandbox"])
            ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
            page = await ctx.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30_000)
            for edit in accepted_edits:
                orig = edit.get("original", "")
                repl = edit.get("replacement", "")
                if not (orig and repl):
                    continue
                await page.evaluate(
                    """([o, r]) => {
                        const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                        const ns = []; let n;
                        while ((n = w.nextNode())) ns.push(n);
                        for (const n of ns) {
                            if (n.textContent.includes(o)) {
                                n.textContent = n.textContent.replace(o, r); break;
                            }
                        }
                    }""",
                    [orig, repl],
                )
            await page.evaluate("window.scrollTo(0, 0)")
            await page.screenshot(path=dest, full_page=True)
            await ctx.close()
            await browser.close()
        return dest
    except Exception:
        return ""
