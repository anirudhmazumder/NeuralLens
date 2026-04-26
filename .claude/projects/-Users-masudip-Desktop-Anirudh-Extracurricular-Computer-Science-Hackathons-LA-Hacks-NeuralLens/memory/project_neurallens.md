---
name: NeuralLens project structure
description: Overview of the NeuralLens hackathon project — two layers: existing image optimizer (src/neurolens/) and new web optimizer (neurallens/)
type: project
---

Two-layer architecture in this repo:

1. **`src/neurolens/`** — existing image-based brain-encoding optimizer
   - `tribe.py` StubEncoder: PIL image → 9 brain region scores (FFA, V4, MT+, etc.)
   - `agent.py` ClaudeAgent: uses `claude-opus-4-7` with tool use to pick image edits
   - `edit.py` CATALOG: programmatic image edits (saturate, contrast, whitespace, etc.)
   - `loop.py`: N-iteration image optimization loop
   - `reward.py`: intent-aware reward (engage/trust/convert/accessibility/gamification)
   - `regions.py`: 9 brain regions with Glasser atlas metadata

2. **`neurallens/`** — new web content optimizer (built for the hackathon demo)
   - `backend/stubs.py`: fake TRIBE and LLM (no API keys, deterministic text scoring)
   - `backend/scraper.py`: Playwright human-viewing simulator (scroll video + text + audio)
   - `backend/scorer.py`: TribeScorer class (stub or live via TRIBE_ENDPOINT)
   - `backend/agent.py`: OptimizationAgent (stub or live via Claude API, OPENAI_LIVE env var)
   - `backend/optimizer.py`: RL loop with asyncio.Queue SSE events, `jobs` dict
   - `backend/main.py`: FastAPI — POST /optimize, GET /job/{id}/stream (SSE), GET /job/{id}/result
   - `frontend/src/App.jsx`: React + Recharts + Tailwind CDN — real-time SSE dashboard

**Why:** stub mode is default (TRIBE_LIVE=false, OPENAI_LIVE=false). Demo works without any API keys.
**How to apply:** when adding features to the web optimizer, work in neurallens/; when working on the image optimizer, work in src/neurolens/.
