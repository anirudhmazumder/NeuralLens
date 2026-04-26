# NeuralLens

**Neural-optimized web content** — simulates how a human brain experiences a webpage,
scores it using a TRIBE v2 brain-encoding model, and iteratively optimizes the content
using an AI agent in a reinforcement-learning loop.

## Architecture

```
Browser → Playwright scraper → [screenshot, scroll video, text, audio]
                                        ↓
                              TRIBE v2 multimodal scorer
                              (predicts neural activations)
                                        ↓
                              Claude vision agent proposes edit
                                        ↓
                              Apply edit → re-score → accept if reward > 0
                                        ↓ (repeat N iterations)
                              Final optimized content + report
```

### Components

| File | Role |
|---|---|
| `backend/scraper.py` | Playwright human-viewing simulator — scroll video, text, audio |
| `backend/stubs.py` | Fake TRIBE + LLM implementations (no API keys needed) |
| `backend/scorer.py` | TRIBE v2 multimodal scorer (stub or live) |
| `backend/agent.py` | Vision-aware Claude agent for text edit suggestions |
| `backend/optimizer.py` | RL loop orchestrator with per-job SSE event queue |
| `backend/main.py` | FastAPI server with SSE streaming |
| `frontend/src/App.jsx` | React SPA with real-time chart and action feed |

## Quick Start (stub mode — no API keys needed)

### Backend

```bash
cd neurallens/backend
pip install -r requirements.txt
playwright install chromium

# Copy and optionally edit env vars
cp ../.env.example .env

uvicorn main:app --reload --port 8080
```

### Frontend

```bash
cd neurallens/frontend
npm install
npm run dev
# Open http://localhost:5173
```

## Live Mode (real models)

Copy `.env.example` to `.env` and set:

```bash
# Use real Anthropic Claude for edit suggestions
CLAUDE_LIVE=true
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Use a deployed TRIBE v2 endpoint for neural scoring
TRIBE_LIVE=true
TRIBE_ENDPOINT=http://your-tribe-server:9090
TRIBE_TOKEN=your_optional_bearer_token
```

## Local-Only Demo Notes

- This backend is intended for local demo use.
- Keep real credentials only in local `.env` files, never in committed docs/code.
- If you rotate Colab/ngrok endpoints, update `.env` (not the README).
- `TRIBE_ENDPOINT` is runtime-configured via env; no ngrok link is hardcoded in backend code.

## Stub Behavior

- **TRIBE stub**: seeded deterministic scoring — same text always gets the same
  score, so text improvements produce genuine reward signals (~50% acceptance rate).
- **LLM stub**: cycles through 10 realistic edit suggestions covering headlines,
  CTAs, body copy, social proof, and visual hierarchy.
- End-to-end demo runs in ~60 seconds for 10 iterations.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/optimize` | Start a job `{url, max_iterations}` → `{job_id}` |
| `GET` | `/job/{id}/stream` | SSE stream of progress events |
| `GET` | `/job/{id}/result` | Final result when complete |
| `GET` | `/health` | `{status: ok, mode: stub\|live}` |

### SSE Event Types

```jsonc
// progress event
{"type": "progress", "data": {"status": "baseline|iteration_complete|scraping|...", ...}}

// complete event
{"type": "complete", "data": {"job_id": "...", "baseline_score": {...}, "final_score": {...},
                               "improvement_pct": 12.3, "history": [...], "accepted_edits": [...]}}

// error event
{"type": "error", "data": {"message": "..."}}
```

## Score Schema

```jsonc
{
  "overall_score": 0.7412,
  "visual_score":  0.7234,
  "text_score":    0.6891,
  "audio_score":   0.3012,
  "language_roi":  0.6544,   // language cortex activation
  "attention_roi": 0.7712,   // dorsal attention network
  "visual_roi":    0.7023    // visual cortex
}
```
