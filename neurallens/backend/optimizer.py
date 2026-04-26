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
import base64
import os
import random
import tempfile
import uuid
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

from playwright.async_api import async_playwright  # type: ignore[import-untyped]

import re as _re

from agent import OptimizationAgent
from memory import get_buffer
from pattern_learner import get_library
from scraper import PageContent, _chromium_launch_kwargs, scrape
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
    decision_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    gaze_regions: list = field(default_factory=list)
    # Per-iteration screenshots accumulated during the live run (keyed by
    # iteration step). The /job/{id}/iteration/{step}/screenshot endpoint
    # reads this so the UI can show iteration thumbnails before the job
    # completes and result is finalized.
    iteration_screenshots: dict = field(default_factory=dict)


jobs: dict[str, JobState] = {}


# ── Reward shaping ────────────────────────────────────────────────────────────

def _shaped_reward(score_delta: float, roi_deltas: dict, novelty: bool) -> float:
    base = score_delta * 10
    lang_bonus = roi_deltas.get("language", 0) * 3
    novelty_bonus = 0.05 if novelty else 0.0
    return round(base + lang_bonus + novelty_bonus, 4)


def _action_family(action_type: str) -> str:
    if action_type in {"change_visual_hierarchy"}:
        return "visual"
    if action_type in {"reorder_sections"}:
        return "structure"
    if action_type in {"add_social_proof"}:
        return "trust"
    if action_type in {"adjust_meta_description"}:
        return "meta"
    return "text"


def _diversify_action(preferred: str, recent_actions: list[str]) -> str:
    """Prefer category variety across recent attempts."""
    if not recent_actions:
        return preferred
    recent_families = {_action_family(a) for a in recent_actions[-2:] if a}
    if _action_family(preferred) not in recent_families and preferred != recent_actions[-1]:
        return preferred

    candidates = [
        a for a in ACTION_TYPES
        if _action_family(a) not in recent_families and a != recent_actions[-1]
    ]
    if candidates:
        return random.choice(candidates)
    non_repeat = [a for a in ACTION_TYPES if a != recent_actions[-1]]
    return random.choice(non_repeat) if non_repeat else preferred


def _exploitative_action(recent_actions: list[str]) -> str:
    """Pick action type with best historical avg_reward, or random if no history."""
    stats = get_buffer().get_action_stats()
    preferred = max(stats, key=lambda k: stats[k]["avg_reward"]) if stats else random.choice(ACTION_TYPES)
    return _diversify_action(preferred, recent_actions)


def _exploratory_action(tried: list[str], recent_actions: list[str]) -> str:
    untried = [a for a in ACTION_TYPES if a not in tried]
    preferred = random.choice(untried) if untried else random.choice(ACTION_TYPES)
    return _diversify_action(preferred, recent_actions)


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

    async def _await_user_acceptance(
        self,
        job: Optional[JobState],
        *,
        iteration: int,
        default_accept: bool,
    ) -> bool:
        """Wait for explicit user accept/reject decision."""
        if job is None:
            return default_accept
        while True:
            decision = await job.decision_queue.get()
            if int(decision.get("iteration", -1)) != iteration:
                continue
            return bool(decision.get("accept", default_accept))

    async def _await_with_watchdog(
        self,
        job: Optional[JobState],
        awaitable,
        *,
        stage_label: str,
        iteration_count: int,
        max_iterations: int,
        tick_s: float = 1.0,
    ):
        """Await a task and emit a single TRIBE-latency wait event."""
        task = asyncio.create_task(awaitable)
        timeout_budget = max(1, int(getattr(self.scorer, "timeout_seconds", 90)))
        done, _ = await asyncio.wait({task}, timeout=tick_s)
        if task in done:
            return task.result()

        await self._emit(job, "progress", {
            "status": "scoring_wait",
            "message": f"Waiting on TRIBE API: {stage_label}.",
            "stage_label": stage_label,
            "wait_started_at": time.time(),
            "timeout_seconds": timeout_budget,
            "iteration_count": iteration_count,
            "max_iterations": max_iterations,
        })
        return await task

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
            from gaze_predictor import get_gaze_predictor

            predictor = get_gaze_predictor()
            gaze_task: Optional[asyncio.Task] = None

            def _start_gaze_early(screenshot_path: str) -> None:
                nonlocal gaze_task
                if gaze_task is None:
                    gaze_task = asyncio.create_task(predictor.analyze(screenshot_path))

            # ── 1. Scrape ─────────────────────────────────────────────────
            await self._emit(job, "progress", {
                "status": "scraping",
                "message": "Simulating human browsing the page…",
                "iteration_count": 0, "max_iterations": max_iterations,
            })
            page = await scrape(url, work_dir, on_screenshot_ready=_start_gaze_early)
            viz_components = _text_to_viz_components(page.text)

            # ── 2. Gaze analysis ──────────────────────────────────────────
            await self._emit(job, "progress", {
                "status": "gaze_analysis",
                "message": "Predicting human gaze patterns…",
                "iteration_count": 0, "max_iterations": max_iterations,
            })
            if gaze_task is None:
                gaze_task = asyncio.create_task(predictor.analyze(page.screenshot_path))
            gaze_data = await gaze_task
            gaze_regions = gaze_data["regions"]
            gaze_overlay_b64 = ""
            try:
                overlay_path = os.path.join(work_dir, "gaze_overlay.png")
                _loop = asyncio.get_event_loop()
                await _loop.run_in_executor(
                    None, predictor.generate_heatmap_overlay, page.screenshot_path, overlay_path
                )
                if overlay_path and os.path.exists(overlay_path):
                    import base64
                    with open(overlay_path, "rb") as fh:
                        gaze_overlay_b64 = base64.b64encode(fh.read()).decode()
            except Exception as exc:
                print(f"[gaze] overlay generation failed: {exc}")
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
                "gaze_overlay_base64": gaze_overlay_b64,
            })

            # ── 3. Baseline ───────────────────────────────────────────────
            await self._emit(job, "progress", {
                "status": "scoring",
                "message": "Computing baseline neural activation (TRIBE API)…",
                "iteration_count": 0, "max_iterations": max_iterations,
            })
            baseline_score = await self._await_with_watchdog(
                job,
                self.scorer.score(
                    page.video_path, page.text, page.audio_path,
                    screenshot_path=page.screenshot_path,
                ),
                stage_label="baseline score",
                iteration_count=0,
                max_iterations=max_iterations,
            )
            current_score = baseline_score.copy()
            current_text = page.text
            episode_tried: list[str] = []
            history: list[dict] = []
            accepted_edits: list[dict] = []
            accepted_comp_ids: list[str] = []
            current_screenshot_path: str = page.screenshot_path or ""
            pending_resample: Optional[asyncio.Task] = None
            # Per-iteration screenshots (accepted edits only). Keyed by step.
            iteration_screenshots: dict[int, str] = {}

            # Seed viz_components with proportionally-weighted baseline scores
            n_vc = max(len(viz_components), 1)
            for i, vc in enumerate(viz_components):
                w = 1.0 - (i / n_vc) * 0.3
                vc["neural_score"] = round(
                    min(1.0, baseline_score["overall_score"] * w), 4
                )

            # ── Brain region scoring at baseline ──────────────────────────
            from brain_regions import evaluate_ethics, compute_intent_reward
            # Reuse atlas_regions returned by score() — same TRIBE response,
            # no second round-trip needed. Falls back to a fresh call only
            # if score() somehow didn't include them.
            baseline_brain = (baseline_score.get("atlas_regions") or {}).copy()
            if not baseline_brain:
                baseline_brain = await self._await_with_watchdog(
                    job,
                    self.scorer.score_brain_regions(
                        page.screenshot_path or None, text=page.text
                    ),
                    stage_label="baseline brain-region score",
                    iteration_count=0,
                    max_iterations=max_iterations,
                )
            current_brain = baseline_brain.copy()
            baseline_ethics = evaluate_ethics(baseline_brain, intent)

            _baseline_ss_b64 = ""
            _bss_path = page.screenshot_path or ""
            if _bss_path and Path(_bss_path).exists():
                with open(_bss_path, "rb") as _fh:
                    _baseline_ss_b64 = base64.b64encode(_fh.read()).decode()

            await self._emit(job, "progress", {
                "status": "baseline",
                "message": f"Baseline overall: {baseline_score['overall_score']:.4f}",
                "score": baseline_score,
                "iteration_count": 0, "max_iterations": max_iterations,
                "annotated_screenshot_base64": _baseline_ss_b64,
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
                            new_path, new_gaze, resample_step = resample_result
                            current_screenshot_path = new_path
                            gaze_regions = new_gaze
                            if new_path:
                                iteration_screenshots[resample_step] = new_path
                                if job:
                                    job.iteration_screenshots[resample_step] = new_path
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
                                "iteration_screenshot_step": resample_step,
                            })
                            await self._emit(job, "iteration_screenshot", {
                                "iteration_count": resample_step,
                                "max_iterations": max_iterations,
                            })
                    except Exception as _exc:
                        print(f"[resample] drain error: {_exc}")
                    pending_resample = None

                # Epsilon-greedy: explore early, exploit later
                epsilon = max(0.1, 1.0 - step / max_iterations)
                exploit = random.random() > epsilon
                strategy = "exploit" if exploit else "explore"
                recent_actions = [h.get("action", {}).get("action_type", "") for h in history[-3:]]

                if exploit:
                    action_type = _exploitative_action(recent_actions)
                else:
                    action_type = _exploratory_action(episode_tried, recent_actions)

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
                page_for_agent = SimpleNamespace(
                    url=page.url,
                    text=current_text,
                    html=getattr(page, "html", ""),
                )

                # Build last-step outcome packet so the agent can self-correct.
                last_outcome: Optional[dict] = None
                if history:
                    last = history[-1]
                    last_action = last.get("action") or {}
                    last_outcome = {
                        "accepted": bool(last.get("accepted")),
                        "score_delta": float(last.get("reward", 0.0)),
                        "roi_deltas": last.get("roi_deltas") or {},
                        "action_type": last_action.get("action_type", ""),
                        "target": last_action.get("target", ""),
                        "original": last_action.get("original", ""),
                        "replacement": last_action.get("replacement", ""),
                    }

                edit = await self.agent.select_action(
                    page_for_agent, score_history, step,
                    screenshot_path=current_screenshot_path,
                    gaze_regions=gaze_regions or None,
                    brain_scores=current_brain,
                    ethics_flags=current_ethics,
                    intent=intent,
                    recent_actions=recent_actions,
                    last_outcome=last_outcome,
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
                    "message": f"[{step}/{max_iterations}] Scoring updated content (TRIBE API)…",
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
                new_score = await self._await_with_watchdog(
                    job,
                    self.scorer.score(
                        page.video_path, updated_text, page.audio_path,
                        screenshot_path=current_screenshot_path,
                    ),
                    stage_label=f"iteration {step} score",
                    iteration_count=step,
                    max_iterations=max_iterations,
                )
                annotated_b64 = await annot_task

                # Reuse atlas_regions from the TRIBE response we already paid for.
                # (Eliminates a redundant ~2-15s TRIBE round-trip per iteration.)
                new_brain = (new_score.get("atlas_regions") or {}).copy()
                if not new_brain:
                    new_brain = await self._await_with_watchdog(
                        job,
                        self.scorer.score_brain_regions(
                            current_screenshot_path or None, text=updated_text
                        ),
                        stage_label=f"iteration {step} brain-region score",
                        iteration_count=step,
                        max_iterations=max_iterations,
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
                auto_accept = score_delta > 0
                prev_overall = current_score["overall_score"]
                await self._emit(job, "progress", {
                    "status": "approval_needed",
                    "message": (
                        f"[{step}/{max_iterations}] Run paused — review proposed change."
                    ),
                    "iteration_count": step,
                    "max_iterations": max_iterations,
                    "edit": edit,
                    "score_delta": round(score_delta, 4),
                    "current_overall": round(prev_overall, 4),
                    "proposed_overall": round(new_score["overall_score"], 4),
                    "roi_deltas": roi_deltas,
                    "shaped_reward": shaped,
                    "intent_reward": intent_reward,
                    "default_decision": "accept" if auto_accept else "reject",
                    "annotated_screenshot_base64": annotated_b64,
                    "strategy": strategy,
                    "epsilon": round(epsilon, 2),
                })
                accepted = await self._await_user_acceptance(
                    job,
                    iteration=step,
                    default_accept=auto_accept,
                )

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
                    edit["iteration"] = step
                    accepted_edits.append(edit)
                    accepted_comp_ids.append(target_id)
                    # Fire background re-render + gaze re-sample (non-blocking).
                    # The resampled screenshot is stored per-step so the UI can
                    # walk through the page evolution in the result panel.
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
                    "prev_score": round(prev_overall, 4),
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
            # Drain any still-running resample so the last iteration screenshot
            # is captured before we render the cumulative "after".
            if pending_resample is not None and not pending_resample.done():
                try:
                    final_resample = await pending_resample
                    if final_resample:
                        rs_path, _rs_gaze, rs_step = final_resample
                        if rs_path:
                            iteration_screenshots[rs_step] = rs_path
                            if job:
                                job.iteration_screenshots[rs_step] = rs_path
                except Exception as _exc:
                    print(f"[resample] final drain error: {_exc}")
                pending_resample = None
            after_shot = await _render_optimized_screenshot(
                url, accepted_edits, work_dir, highlight=True
            )

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
                "iteration_screenshots": iteration_screenshots,
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
            msg = str(exc).strip() or f"{exc.__class__.__name__}: no error details provided"
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
) -> Optional[tuple[str, list, int]]:
    """Re-render page with current accepted edits, take screenshot, re-run gaze.

    Runs as a fire-and-forget background task so it never blocks the RL loop.
    Returns (screenshot_path, gaze_regions, step) or None on failure. The
    rendered screenshot has the changed text softly highlighted so the
    Before/After comparison makes the delta visually obvious.
    """
    from gaze_predictor import get_gaze_predictor
    try:
        ss_path = await _render_optimized_screenshot(
            url, accepted_edits, work_dir, suffix=f"_step{step}", highlight=True
        )
        if not ss_path:
            return None
        gaze_data = await get_gaze_predictor().analyze(ss_path)
        return ss_path, gaze_data.get("regions", []), step
    except Exception as exc:
        print(f"[resample] step {step} failed: {exc}")
        return None


# ── HTML file optimizer ───────────────────────────────────────────────────────

@dataclass
class HtmlOptimizationLoop:
    """RL-style optimizer that works on an uploaded HTML string instead of a URL.

    Each iteration:
      1. Re-renders current HTML via Playwright (file://)
      2. Takes a screenshot and runs gaze analysis
      3. Asks the agent to propose one HTML/CSS design edit
      4. Applies the edit as a string replacement in the HTML source
      5. Re-scores with TRIBE and accepts/rejects based on overall_score delta
      6. Emits SSE events in the same format as NeuralOptimizer (reuses the stream)
    """
    scorer: TribeScorer = field(default_factory=TribeScorer)
    agent: OptimizationAgent = field(
        default_factory=lambda: OptimizationAgent(memory=get_buffer())
    )

    async def _await_user_acceptance(
        self,
        job: Optional[JobState],
        *,
        iteration: int,
        default_accept: bool,
    ) -> bool:
        """Wait for explicit user accept/reject decision (HTML loop)."""
        if job is None:
            return default_accept
        while True:
            decision = await job.decision_queue.get()
            if int(decision.get("iteration", -1)) != iteration:
                continue
            return bool(decision.get("accept", default_accept))

    async def run(
        self,
        html_content: str,
        max_iterations: int = 10,
        job_id: str = "",
        filename: str = "upload.html",
    ) -> dict:
        job_id = job_id or str(uuid.uuid4())
        job = jobs.get(job_id)

        work_dir = os.path.join(tempfile.gettempdir(), "neurallens_html", job_id)
        os.makedirs(work_dir, exist_ok=True)

        try:
            from scraper import render_html_file
            from gaze_predictor import get_gaze_predictor

            predictor = get_gaze_predictor()
            gaze_task: Optional[asyncio.Task] = None

            def _start_gaze_early(screenshot_path: str) -> None:
                nonlocal gaze_task
                if gaze_task is None:
                    gaze_task = asyncio.create_task(predictor.analyze(screenshot_path))

            # ── 1. Initial render ─────────────────────────────────────────
            await self._emit(job, "progress", {
                "status": "scraping",
                "message": f"Rendering uploaded HTML file '{filename}'…",
                "iteration_count": 0, "max_iterations": max_iterations,
            })
            page = await render_html_file(
                html_content, work_dir, on_screenshot_ready=_start_gaze_early
            )
            before_screenshot = page.screenshot_path

            # ── 2. Gaze analysis ──────────────────────────────────────────
            await self._emit(job, "progress", {
                "status": "gaze_analysis",
                "message": "Predicting gaze patterns on uploaded page…",
                "iteration_count": 0, "max_iterations": max_iterations,
            })
            if gaze_task is None:
                gaze_task = asyncio.create_task(predictor.analyze(page.screenshot_path))
            gaze_data = await gaze_task
            gaze_regions = gaze_data["regions"]

            _gaze_ss_b64 = ""
            _gss_path = page.screenshot_path or ""
            if _gss_path and Path(_gss_path).exists():
                with open(_gss_path, "rb") as _fh:
                    _gaze_ss_b64 = base64.b64encode(_fh.read()).decode()

            _gaze_overlay_b64 = gaze_data.get("overlay_b64", "") or _gaze_ss_b64

            await self._emit(job, "gaze", {
                "status": "gaze_complete",
                "message": (
                    f"Gaze: {len(gaze_regions)} salient regions mapped"
                    f" ({'live DeepGaze IIE' if gaze_data.get('gaze_live') else 'F-pattern stub'})"
                ),
                "gaze_regions": gaze_regions,
                "gaze_live": gaze_data.get("gaze_live", False),
                "annotated_screenshot_base64": _gaze_overlay_b64,
                "gaze_overlay_base64": _gaze_overlay_b64,
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
            current_html = html_content
            current_screenshot = page.screenshot_path
            history: list[dict] = []
            accepted_edits: list[dict] = []
            iteration_screenshots: dict[int, str] = {}

            _html_ss_b64 = ""
            _hss_path = page.screenshot_path or ""
            if _hss_path and Path(_hss_path).exists():
                with open(_hss_path, "rb") as _fh:
                    _html_ss_b64 = base64.b64encode(_fh.read()).decode()

            await self._emit(job, "progress", {
                "status": "baseline",
                "message": f"Baseline overall: {baseline_score['overall_score']:.4f}",
                "score": baseline_score,
                "iteration_count": 0, "max_iterations": max_iterations,
                "annotated_screenshot_base64": _html_ss_b64,
            })

            # Emit baseline brain regions so BrainPanel shows data immediately
            from brain_regions import evaluate_ethics
            baseline_brain = baseline_score.get("atlas_regions") or {}
            if baseline_brain:
                await self._emit(job, "brain_regions", {
                    "iteration_count": 0,
                    "regions": baseline_brain,
                    "ethics_flags": evaluate_ethics(baseline_brain, "engage"),
                    "intent": "engage",
                    "is_baseline": True,
                })

            # ── 4. RL episode ─────────────────────────────────────────────
            for step in range(1, max_iterations + 1):
                await self._emit(job, "progress", {
                    "status": "proposing",
                    "message": f"[{step}/{max_iterations}] Agent proposing HTML design edit…",
                    "iteration_count": step, "max_iterations": max_iterations,
                    "action_type": "html_design",
                })

                score_history = [baseline_score] + [h["score"] for h in history]
                brain_scores = current_score.get("atlas_regions")

                edit = await self.agent.select_html_action(
                    current_html,
                    score_history,
                    step,
                    screenshot_path=current_screenshot,
                    gaze_regions=gaze_regions or None,
                    brain_scores=brain_scores,
                )

                # Apply HTML edit
                html_orig = edit.get("html_original", "")
                html_repl = edit.get("html_replacement", "")
                updated_html = current_html
                applied = False
                if html_orig and html_orig in current_html:
                    updated_html = current_html.replace(html_orig, html_repl, 1)
                    applied = True

                await self._emit(job, "progress", {
                    "status": "scoring",
                    "message": f"[{step}/{max_iterations}] Re-rendering and scoring…",
                    "iteration_count": step, "max_iterations": max_iterations,
                })

                # Re-render and re-screenshot if the edit was applied
                step_screenshot = current_screenshot
                if applied:
                    step_dir = os.path.join(work_dir, f"step_{step}")
                    os.makedirs(step_dir, exist_ok=True)
                    step_page = await render_html_file(updated_html, step_dir)
                    step_screenshot = step_page.screenshot_path
                    # Re-run gaze on the updated page
                    new_gaze = await get_gaze_predictor().analyze(step_screenshot)
                    if new_gaze["regions"]:
                        gaze_regions = new_gaze["regions"]

                new_score = await self.scorer.score(
                    page.video_path,
                    page.text,
                    page.audio_path,
                    screenshot_path=step_screenshot,
                )

                score_delta = new_score["overall_score"] - current_score["overall_score"]
                auto_accept = score_delta > 0
                prev_overall_html = current_score["overall_score"]
                await self._emit(job, "progress", {
                    "status": "approval_needed",
                    "message": (
                        f"[{step}/{max_iterations}] Run paused — review proposed HTML change."
                    ),
                    "iteration_count": step,
                    "max_iterations": max_iterations,
                    "edit": {
                        **edit,
                        "original": edit.get("html_original", ""),
                        "replacement": edit.get("html_replacement", ""),
                    },
                    "score_delta": round(score_delta, 4),
                    "current_overall": round(prev_overall_html, 4),
                    "proposed_overall": round(new_score["overall_score"], 4),
                    "default_decision": "accept" if auto_accept else "reject",
                })
                accepted = await self._await_user_acceptance(
                    job,
                    iteration=step,
                    default_accept=auto_accept,
                )

                if accepted:
                    current_score = new_score
                    current_html = updated_html
                    current_screenshot = step_screenshot
                    edit["iteration"] = step
                    accepted_edits.append(edit)
                    if step_screenshot:
                        iteration_screenshots[step] = step_screenshot
                        if job:
                            job.iteration_screenshots[step] = step_screenshot
                        await self._emit(job, "iteration_screenshot", {
                            "iteration_count": step,
                            "max_iterations": max_iterations,
                        })

                entry = {
                    "iteration": step,
                    "action": edit,
                    "reward": round(score_delta, 4),
                    "score": new_score,
                    "accepted": accepted,
                    "applied": applied,
                }
                history.append(entry)

                agent_thought = {
                    "step": "accepted" if accepted else "rejected",
                    "action_type": edit.get("action_type", "html_design"),
                    "target": edit.get("target", ""),
                    "reasoning": edit.get("reasoning", ""),
                    "original": edit.get("html_original", "")[:150],
                    "replacement": edit.get("html_replacement", "")[:150],
                    "expected_roi_impact": edit.get("expected_roi_impact", {}),
                    "score_delta": round(score_delta, 4),
                    "new_score": round(new_score["overall_score"], 4),
                }

                await self._emit(job, "progress", {
                    "status": "iteration_complete",
                    "message": (
                        f"[{step}/{max_iterations}] "
                        f"{'✓ Accepted' if accepted else '✗ Rejected'} "
                        f"({score_delta:+.4f})"
                    ),
                    "iteration_count": step, "max_iterations": max_iterations,
                    "edit": {
                        **edit,
                        "original": edit.get("html_original", ""),
                        "replacement": edit.get("html_replacement", ""),
                    },
                    "reward": score_delta,
                    "score": new_score,
                    "accepted": accepted,
                    "gaze_regions": gaze_regions,
                    "agent_thought": agent_thought,
                })

                # Emit brain regions so BrainPanel updates each iteration
                step_brain = new_score.get("atlas_regions") or {}
                if step_brain:
                    await self._emit(job, "brain_regions", {
                        "iteration_count": step,
                        "regions": step_brain,
                        "ethics_flags": evaluate_ethics(step_brain, "engage"),
                        "intent": "engage",
                        "accepted": accepted,
                    })

            # ── 5. Build result ───────────────────────────────────────────
            bv = baseline_score["overall_score"]
            fv = current_score["overall_score"]
            improvement_pct = ((fv - bv) / max(bv, 1e-6)) * 100 if bv > 0 else 0.0

            result = {
                "job_id": job_id,
                "filename": filename,
                "baseline_score": baseline_score,
                "final_score": current_score,
                "improvement_pct": round(improvement_pct, 2),
                "iterations": max_iterations,
                "history": history,
                "accepted_edits": accepted_edits,
                "before_screenshot": before_screenshot,
                "after_screenshot": current_screenshot,
                "iteration_screenshots": iteration_screenshots,
                "optimized_html": current_html,
                "gaze_regions": gaze_regions,
                "gaze_live": gaze_data.get("gaze_live", False),
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


# Robust replacement helper injected into the page once per render.
# Tries (in order) exact match, whitespace-normalized match, case-insensitive
# match. Highlights the replaced fragment so the "after" screenshot makes the
# delta visually obvious. Returns true if any replacement happened.
_DOM_REPLACE_JS = r"""
([o, r, highlight]) => {
    if (!o) return false;
    const norm = (s) => s.replace(/\s+/g, ' ').trim();
    const nO = norm(o);
    if (!nO) return false;

    const wrap = (node, original, replacement) => {
        const idx = node.textContent.indexOf(original);
        if (idx < 0) return false;
        const before = node.textContent.slice(0, idx);
        const after  = node.textContent.slice(idx + original.length);
        const span = document.createElement('span');
        span.textContent = replacement;
        if (highlight) {
            span.style.background = 'rgba(124, 58, 237, 0.18)';
            span.style.borderBottom = '2px solid rgba(124, 58, 237, 0.85)';
            span.style.padding = '0 2px';
            span.style.borderRadius = '2px';
        }
        const parent = node.parentNode;
        if (!parent) return false;
        if (before) parent.insertBefore(document.createTextNode(before), node);
        parent.insertBefore(span, node);
        if (after) parent.insertBefore(document.createTextNode(after), node);
        parent.removeChild(node);
        return true;
    };

    const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
        acceptNode: (n) => {
            if (!n.textContent || !n.textContent.trim()) return NodeFilter.FILTER_REJECT;
            const p = n.parentNode;
            if (!p) return NodeFilter.FILTER_REJECT;
            const tag = (p.tagName || '').toLowerCase();
            if (tag === 'script' || tag === 'style' || tag === 'noscript') return NodeFilter.FILTER_REJECT;
            return NodeFilter.FILTER_ACCEPT;
        }
    });
    const nodes = [];
    let n;
    while ((n = w.nextNode())) nodes.push(n);

    // Pass 1: exact substring
    for (const node of nodes) {
        if (node.textContent.includes(o)) {
            return wrap(node, o, r);
        }
    }
    // Pass 2: whitespace-normalized
    for (const node of nodes) {
        const nT = norm(node.textContent);
        if (nT.includes(nO)) {
            // replace the entire node text with normalized substitution
            const rebuilt = nT.replace(nO, r);
            const parent = node.parentNode;
            if (!parent) continue;
            const span = document.createElement('span');
            span.textContent = rebuilt;
            if (highlight) {
                span.style.background = 'rgba(124, 58, 237, 0.18)';
                span.style.borderBottom = '2px solid rgba(124, 58, 237, 0.85)';
                span.style.padding = '0 2px';
                span.style.borderRadius = '2px';
            }
            parent.replaceChild(span, node);
            return true;
        }
    }
    // Pass 3: case-insensitive exact
    const lo = o.toLowerCase();
    for (const node of nodes) {
        const idx = node.textContent.toLowerCase().indexOf(lo);
        if (idx >= 0) {
            const real = node.textContent.slice(idx, idx + o.length);
            return wrap(node, real, r);
        }
    }
    return false;
}
"""


async def _apply_edits_to_page(page, accepted_edits: list, *, highlight: bool = True) -> dict[str, int]:
    """Apply each accepted edit to the live Playwright page DOM.

    Returns a dict with counts: {applied, skipped, missing_text}. The
    optimizer can surface these so we know whether the after-screenshot
    actually reflects the requested changes.
    """
    applied = 0
    missing = 0
    skipped = 0
    for edit in accepted_edits:
        orig = (edit.get("original") or "").strip()
        repl = edit.get("replacement", "")
        if not (orig and repl):
            skipped += 1
            continue
        try:
            ok = await page.evaluate(_DOM_REPLACE_JS, [orig, repl, bool(highlight)])
        except Exception as exc:
            print(f"[render] DOM replace failed for edit {edit.get('action_type', '?')}: {exc}")
            skipped += 1
            continue
        if ok:
            applied += 1
        else:
            missing += 1
    return {"applied": applied, "skipped": skipped, "missing_text": missing}


async def _render_optimized_screenshot(
    url: str,
    accepted_edits: list,
    out_dir: str,
    suffix: str = "",
    *,
    highlight: bool = False,
) -> str:
    """Render the URL with all accepted edits applied and screenshot it.

    `highlight=True` softly underlines the replaced text so the after
    screenshot makes the delta visually obvious to the operator.
    """
    dest = str(Path(out_dir) / f"optimized_screenshot{suffix}.png")
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(**_chromium_launch_kwargs(pw))
            ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
            page = await ctx.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30_000)
            stats = await _apply_edits_to_page(page, accepted_edits, highlight=highlight)
            print(
                f"[render] applied={stats['applied']} skipped={stats['skipped']} "
                f"missing_text={stats['missing_text']} edits={len(accepted_edits)} "
                f"suffix={suffix or 'final'}"
            )
            # Give the page a tick to relayout after DOM mutations.
            try:
                await page.wait_for_load_state("networkidle", timeout=2_000)
            except Exception:
                pass
            await page.evaluate("window.scrollTo(0, 0)")
            await page.screenshot(path=dest, full_page=True)
            await ctx.close()
            await browser.close()
        return dest
    except Exception as exc:
        print(f"[render] failed: {exc}")
        return ""
