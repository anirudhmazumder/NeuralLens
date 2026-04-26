# NeuralLens

NeuralLens is an AI system for optimizing webpages using predicted neural engagement, gaze saliency, and iterative agent-driven edits.

It combines:
- a web app for live URL + HTML upload optimization,
- a research/CLI package for deeper experimentation,
- and a deployed multi-agent interface in **FetchAI Agenteverse** that can call into the NeuralLens optimization algorithm.

## What We Built

NeuralLens runs an optimization loop:

1. Capture page state (screenshot + extracted text)
2. Score predicted neural response with a TRIBE-style encoder
3. Run gaze analysis to locate highest-attention regions
4. Let an LLM agent propose one focused edit
5. Re-score and accept only improvements
6. Repeat for N iterations and surface transparent metrics/events

For hackathon use, this enables teams to improve web engagement before shipping or spending on traffic.

## Multi-Agent Deployment (FetchAI Agenteverse)

We have deployed our multi-agent orchestration layer into **FetchAI Agenteverse**.  
That deployment can interface with NeuralLens by:

- receiving optimization tasks from external agent workflows,
- invoking NeuralLens optimization jobs through API-compatible task calls,
- and returning structured outputs (scores, accepted edits, and final artifacts) back into agent pipelines.

This makes NeuralLens usable both as a standalone app and as an optimization service in autonomous agent ecosystems.

## Repository Structure

### `neurallens/` (Web App: FastAPI + React)
- URL optimization flow with live SSE streaming
- HTML upload flow with direct HTML/CSS optimization and downloadable output
- Gaze analysis + heatmap overlays
- Agent vision/debug panels, score charts, memory/pattern views

### `src/neurolens/` (Python Package + CLI)
- Brain-region scoring and reward logic
- Ethics checks and transparency reporting
- Iterative optimization loop for research workflows
- CLI entry points for local experimentation

## Core Features

- Neural score-guided optimization (TRIBE endpoint + fallback encoder)
- Gaze-aware prioritization (DeepGaze/live or heuristic fallback)
- Iterative agent edits with acceptance/rejection by reward delta
- HTML upload mode that returns optimized HTML code
- Real-time progress streaming via SSE
- Cross-session memory and pattern learning

## High-Level Architecture

```text
URL or HTML input
        |
        v
Playwright render/extract
        |
        +--> TRIBE scorer ---------+
        |                          |
        +--> Gaze predictor ------>+--> Optimization agent --> apply edit --> re-score loop
                                   |
                                   +--> SSE events + summary artifacts
```

## Quick Start (Web App)

### Backend
```bash
cd neurallens/backend
pip install -r requirements.txt
playwright install chromium
uvicorn main:app --reload --port 8080
```

### Frontend
```bash
cd neurallens/frontend
npm install
npm run dev
```

Open the local Vite URL shown in terminal (typically `http://localhost:5173`).

## Environment

Use `neurallens/.env.example` as the template.

Common variables:
- `TRIBE_ENDPOINT` - live scoring endpoint
- `TRIBE_TIMEOUT_SECONDS` - live call timeout (configurable)
- `OPENAI_LIVE`, `OPENAI_API_KEY`, `OPENAI_MODEL` - agent behavior
- `GAZE_LIVE`, `CENTERBIAS_PATH`, `GAZE_DEVICE` - gaze mode and model config

## API Highlights

- `POST /optimize` - start URL optimization
- `GET /job/{id}/stream` - live SSE event stream
- `GET /job/{id}/result` - final result payload
- `POST /upload-html` - upload HTML and get baseline analysis
- `POST /optimize-html` - start HTML optimization job
- `GET /html-job/{id}/download` - download optimized HTML output

## Technical Notes

- TRIBE scoring falls back to local encoder if remote endpoint is unavailable/slow.
- DeepGaze weights are cached locally to avoid repeated model downloads across restarts.
- All optimization steps are observable through streamed events and frontend telemetry.

## Ethics + Guardrails

NeuralLens is built for engagement quality, not manipulative dark patterns:
- intent-aware optimization
- penalty-aware scoring and constraints
- transparent action history and score deltas

## Current Status

The platform is fully demoable end-to-end:
- live URL optimization,
- uploaded HTML optimization with downloadable output,
- and multi-agent integration via FetchAI Agenteverse.

## Submission-Oriented Project Rundown

For a fuller hackathon narrative, see:

- `neurallens/HACKATHON_SUBMISSION_DRAFT.md`
