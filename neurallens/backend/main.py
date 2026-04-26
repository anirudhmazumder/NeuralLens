"""FastAPI application — NeuralLens backend."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from memory import get_buffer
from optimizer import JobState, OptimizationLoop, _infer_component_type, jobs
from pattern_learner import get_library
from scorer import TribeScorer
from stubs import ACTION_TYPES

app = FastAPI(title="NeuralLens API", version="0.3.0")

# Cache screenshot paths keyed by URL so /gaze-analysis avoids re-scraping
_screenshot_cache: dict[str, str] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    url: str
    max_iterations: int = 10
    intent: str = "engage"  # engage | trust | convert | accessibility | gamification

class ParsePageRequest(BaseModel):
    url: str

class ScoreLayoutRequest(BaseModel):
    components: list[dict]
    url: str = ""

class OptimizeBlockRequest(BaseModel):
    block: dict
    url: str = ""
    context: list[dict] = []

class ApplyEditRequest(BaseModel):
    url: str
    edit: dict
    current_text: str
    current_score: dict

class ExportRequest(BaseModel):
    components: list[dict]
    url: str = ""

class GazeAnalysisRequest(BaseModel):
    url: str = ""
    screenshot_path: str = ""


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    tribe_live = os.getenv("TRIBE_LIVE", "false").lower() == "true"
    agent_live = os.getenv("CLAUDE_LIVE", "false").lower() == "true"
    mode = "live" if (tribe_live or agent_live) else "stub"
    return {"status": "ok", "mode": mode, "agent": "claude-sonnet-4-6" if agent_live else "stub"}


# ── Optimization job ───────────────────────────────────────────────────────────

@app.post("/optimize")
async def start_optimization(req: OptimizeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job = JobState(job_id=job_id, url=req.url, max_iterations=req.max_iterations, status="running")
    jobs[job_id] = job

    loop = OptimizationLoop()
    background_tasks.add_task(loop.run, req.url, req.max_iterations, job_id, req.intent)
    return {"job_id": job_id}


@app.get("/job/{job_id}/stream")
async def stream_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    async def _generate():
        for event in list(job.events):
            yield f"data: {json.dumps(event)}\n\n"
        if job.status in ("complete", "error"):
            return
        while True:
            try:
                event = await asyncio.wait_for(job.event_queue.get(), timeout=30.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/job/{job_id}/result")
async def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.error)
    if job.status != "complete":
        raise HTTPException(status_code=202, detail="Job still running")
    return job.result


@app.get("/job/{job_id}/before-screenshot")
async def before_screenshot(job_id: str):
    return FileResponse(_screenshot_path(job_id, "before_screenshot"), media_type="image/png")


@app.get("/job/{job_id}/after-screenshot")
async def after_screenshot(job_id: str):
    return FileResponse(_screenshot_path(job_id, "after_screenshot"), media_type="image/png")


# ── Memory ─────────────────────────────────────────────────────────────────────

@app.get("/memory/stats")
async def memory_stats():
    return get_buffer().get_action_stats()


@app.get("/memory/history")
async def memory_history():
    rows = get_buffer().get_recent(50)
    for r in rows:
        for key in ("action_detail", "roi_deltas"):
            if isinstance(r.get(key), str):
                try:
                    r[key] = json.loads(r[key])
                except (json.JSONDecodeError, TypeError):
                    pass
    return rows


# ── Patterns ───────────────────────────────────────────────────────────────────

@app.get("/patterns")
async def get_patterns():
    return get_library().get_all_patterns()


# ── Visual builder ─────────────────────────────────────────────────────────────

@app.post("/parse-page")
async def parse_page(req: ParsePageRequest):
    from scraper import scrape

    work_dir = os.path.join(tempfile.gettempdir(), "neurallens_parse", str(uuid.uuid4()))
    os.makedirs(work_dir, exist_ok=True)

    scorer = TribeScorer()
    page = await scrape(req.url, work_dir)
    _screenshot_cache[req.url] = page.screenshot_path  # cache for gaze analysis

    components = _parse_into_components(page.text)
    page_score = await scorer.score(page.video_path, page.text, page.audio_path)

    screenshot_b64 = ""
    if page.screenshot_path and Path(page.screenshot_path).exists():
        with open(page.screenshot_path, "rb") as fh:
            screenshot_b64 = base64.b64encode(fh.read()).decode()

    return {
        "components": components,
        "screenshot_base64": screenshot_b64,
        "page_score": page_score,
        "url": req.url,
    }


@app.post("/score-layout")
async def score_layout(req: ScoreLayoutRequest):
    scorer = TribeScorer()
    full_text = "\n\n".join(c.get("content", "") for c in req.components)
    total_score = await scorer.score("", full_text, "")

    per_component = []
    for comp in req.components:
        comp_score = await scorer.score("", comp.get("content", ""), "")
        per_component.append({
            "id": comp.get("id"),
            "type": comp.get("type"),
            "score": comp_score,
            "neural_contribution": comp_score["overall_score"],
        })

    return {"total_score": total_score, "per_component": per_component}


@app.post("/optimize-block")
async def optimize_block_endpoint(req: OptimizeBlockRequest):
    from agent import OptimizationAgent
    from scraper import PageContent

    agent = OptimizationAgent(memory=get_buffer())

    comp_type = req.block.get("type", "body")
    action_type = _comp_type_to_action(comp_type)
    if action_type in ACTION_TYPES:
        agent._action_index = ACTION_TYPES.index(action_type)

    page = PageContent(
        url=req.url,
        text=req.block.get("content", ""),
        screenshot_path="",
        video_path="",
        audio_path=None,
        metadata={},
    )

    edit = await agent.select_action(page, [], 1, screenshot_path=None)
    return {"edit": edit, "block_id": req.block.get("id")}


@app.post("/apply-edit")
async def apply_edit(req: ApplyEditRequest):
    scorer = TribeScorer()

    original = req.edit.get("original", "")
    replacement = req.edit.get("replacement", "")
    new_text = req.current_text
    if original and original in req.current_text:
        new_text = req.current_text.replace(original, replacement, 1)

    new_score = await scorer.score("", new_text, "")
    score_delta = new_score["overall_score"] - req.current_score.get("overall_score", 0)
    accepted = score_delta > 0

    roi_deltas = {
        k: round(new_score.get(f"{k}_roi", 0) - req.current_score.get(f"{k}_roi", 0), 4)
        for k in ("language", "attention", "visual")
    }

    get_buffer().store(
        url=req.url,
        action=req.edit,
        before_score=req.current_score.get("overall_score", 0),
        after_score=new_score["overall_score"],
        reward=score_delta,
        accepted=accepted,
        roi_deltas=roi_deltas,
        page_text=req.current_text[:500],
    )
    get_library().store_experience(
        url=req.url,
        component_type=_infer_component_type(req.edit),
        action_type=req.edit.get("action_type", "unknown"),
        before_content=original or req.current_text[:200],
        after_content=replacement or new_text[:200],
        before_scores=req.current_score,
        after_scores=new_score,
        reward=score_delta,
        accepted=accepted,
    )

    return {
        "new_score": new_score,
        "score_delta": round(score_delta, 4),
        "accepted": accepted,
        "new_text": new_text,
    }


@app.post("/gaze-analysis")
async def gaze_analysis_endpoint(req: GazeAnalysisRequest):
    from gaze_predictor import get_gaze_predictor

    predictor = get_gaze_predictor()

    # Resolve screenshot path: use cache, explicit path, or re-scrape
    screenshot_path = ""
    if req.screenshot_path and Path(req.screenshot_path).exists():
        screenshot_path = req.screenshot_path
    elif req.url and req.url in _screenshot_cache and Path(_screenshot_cache[req.url]).exists():
        screenshot_path = _screenshot_cache[req.url]
    elif req.url:
        from scraper import scrape
        work_dir = os.path.join(tempfile.gettempdir(), "neurallens_gaze", str(uuid.uuid4()))
        os.makedirs(work_dir, exist_ok=True)
        page = await scrape(req.url, work_dir)
        screenshot_path = page.screenshot_path
        _screenshot_cache[req.url] = screenshot_path
    else:
        raise HTTPException(status_code=400, detail="Provide url or screenshot_path")

    gaze_data = await predictor.analyze(screenshot_path)

    # Generate heatmap overlay in executor
    overlay_path = screenshot_path.replace(".png", "_gaze_overlay.png")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, predictor.generate_heatmap_overlay, screenshot_path, overlay_path
    )

    heatmap_b64 = ""
    if Path(overlay_path).exists():
        with open(overlay_path, "rb") as fh:
            heatmap_b64 = base64.b64encode(fh.read()).decode()

    return {
        "salient_regions": gaze_data["regions"],
        "heatmap_overlay_base64": heatmap_b64,
        "gaze_live": gaze_data.get("gaze_live", False),
    }


@app.get("/brain-regions/{job_id}")
async def brain_regions_for_job(job_id: str):
    """Return the latest brain region scores and ethics flags for a running/complete job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    brain_evt = next(
        (e for e in reversed(job.events) if e.get("type") == "brain_regions"),
        None,
    )
    if not brain_evt:
        raise HTTPException(status_code=404, detail="No brain region data yet")
    return brain_evt["data"]


@app.post("/score-brain-regions")
async def score_brain_regions_endpoint(req: GazeAnalysisRequest):
    """Score all 9 HCP-MMP1 regions for a given URL or screenshot path."""
    scorer = TribeScorer()

    screenshot_path = ""
    if req.screenshot_path and Path(req.screenshot_path).exists():
        screenshot_path = req.screenshot_path
    elif req.url and req.url in _screenshot_cache and Path(_screenshot_cache[req.url]).exists():
        screenshot_path = _screenshot_cache[req.url]
    elif req.url:
        from scraper import scrape
        work_dir = os.path.join(tempfile.gettempdir(), "neurallens_brain", str(uuid.uuid4()))
        os.makedirs(work_dir, exist_ok=True)
        page = await scrape(req.url, work_dir)
        screenshot_path = page.screenshot_path
        _screenshot_cache[req.url] = screenshot_path

    regions = await scorer.score_brain_regions(screenshot_path or None)
    from brain_regions import evaluate_ethics, REGION_CATEGORIES
    ethics = evaluate_ethics(regions, "engage")
    return {
        "regions": regions,
        "categories": REGION_CATEGORIES,
        "ethics_flags": ethics,
    }


@app.post("/export")
async def export_html(req: ExportRequest):
    sections: list[str] = []
    for comp in req.components:
        comp_type = comp.get("type", "body")
        content = comp.get("content", "").replace("<", "&lt;").replace(">", "&gt;")
        if comp_type == "headline":
            sections.append(f'<h1 class="nl-headline">{content}</h1>')
        elif comp_type == "cta":
            sections.append(f'<div class="nl-cta"><button>{content}</button></div>')
        elif comp_type == "testimonial":
            sections.append(f'<blockquote class="nl-testimonial">{content}</blockquote>')
        else:
            sections.append(f'<p class="nl-body">{content}</p>')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NeuralLens Export</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem 1.5rem; color: #1a1a1a; }}
    .nl-headline {{ font-size: 2.2rem; font-weight: 800; line-height: 1.2; margin: 0 0 1.5rem; }}
    .nl-cta {{ margin: 2rem 0; }}
    .nl-cta button {{ background: #7c3aed; color: #fff; border: none; padding: 0.8rem 2rem; border-radius: 8px; font-size: 1.1rem; font-weight: 600; cursor: pointer; }}
    .nl-testimonial {{ border-left: 4px solid #7c3aed; margin: 1.5rem 0; padding: 0.75rem 1.25rem; color: #555; font-style: italic; background: #f9f7ff; border-radius: 0 8px 8px 0; }}
    .nl-body {{ line-height: 1.75; margin: 0 0 1.2rem; color: #374151; }}
    footer {{ margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid #e5e7eb; color: #9ca3af; font-size: 0.75rem; }}
  </style>
</head>
<body>
  {''.join(sections)}
  <footer>Optimized by <strong>NeuralLens</strong> — neural engagement optimization</footer>
</body>
</html>"""

    return HTMLResponse(
        content=html,
        headers={"Content-Disposition": "attachment; filename=neurallens_export.html"},
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _screenshot_path(job_id: str, key: str) -> str:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if not job.result:
        raise HTTPException(status_code=404, detail="Result not ready")
    path = job.result.get(key, "")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Screenshot not available")
    return path


def _parse_into_components(text: str) -> list[dict]:
    import re

    blocks = re.split(r"\n{2,}", text.strip())
    components: list[dict] = []

    for block in blocks:
        block = block.strip()
        if not block or len(block) < 10:
            continue

        words = block.split()
        wc = len(words)
        first_words = {w.lower() for w in words[:6]}

        if wc <= 10 and not block.endswith("."):
            comp_type = "headline"
        elif wc <= 6 and first_words & {"get", "start", "try", "join", "sign", "buy", "learn", "download", "register"}:
            comp_type = "cta"
        elif block.endswith("?") and wc <= 20:
            comp_type = "headline"
        elif any(w.lower() in {"customer", "users", "clients", "review", "rating", "trusted", "testimonial"} for w in words):
            comp_type = "testimonial"
        else:
            comp_type = "body"

        components.append({
            "id": uuid.uuid4().hex[:8],
            "type": comp_type,
            "content": block,
            "word_count": wc,
            "neural_contribution": 0.5,
        })

        if len(components) >= 20:
            break

    return components


def _comp_type_to_action(comp_type: str) -> str:
    mapping = {
        "headline": "rewrite_headline",
        "cta": "rewrite_cta",
        "body": "rewrite_body_paragraph",
        "testimonial": "add_social_proof",
        "image": "change_visual_hierarchy",
    }
    return mapping.get(comp_type, "rewrite_body_paragraph")
