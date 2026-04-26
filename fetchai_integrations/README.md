# NeuralLens 🧠

> Neuromarketing optimization for every business — not just the Fortune 500.

[![Fetch.ai](https://img.shields.io/badge/Fetch.ai-Track%201-blue)](https://fetch.ai)
[![Agentverse](https://img.shields.io/badge/Agentverse-Registered-green)](https://agentverse.ai)
[![ASI:One](https://img.shields.io/badge/ASI%3AOne-Discoverable-purple)](https://asi1.ai)
[![LA Hacks 2026](https://img.shields.io/badge/LA%20Hacks-2026-orange)](https://lahacks.com)

Send NeuralLens an image URL + your goal. Get back an optimized image,
new copy, brain heatmaps, and a Neural Engagement Score — all delivered
through ASI:One chat. No frontend required.

Powered by **Meta TRIBE v2** (fMRI brain simulation) + **DeepGaze**
(eye tracking) + **Google Gemma 3** (optimization) + **Cloudinary AI**
(image transforms) + **Fetch.ai uAgents** (multi-agent pipeline).

This multi-agent layer is the Agentverse-facing interface for NeuralLens.
It can trigger the same core optimization logic used in the main app and return structured optimization outputs back into agent workflows.

---

## Architecture

```
ASI:One / Agentverse
        │
        ▼
┌─────────────────────┐
│  ORCHESTRATOR       │  Chat Protocol, ASI:One LLM intent parsing
│  agents/orchestrator│  Port 8000
└──────────┬──────────┘
           │  SensorRequest
           ▼
┌─────────────────────┐
│  SENSOR             │  TRIBE v2 /infer + DeepGaze /deepgaze
│  agents/sensor_agent│  + /heatmap → Cloudinary upload
│  Port 8001          │
└──────────┬──────────┘
           │  SensorResult (70k activations + saliency)
           ▼
┌─────────────────────┐
│  INTERPRETER        │  extract_roi_values → compute_nes
│  agents/interpreter │  → analyze_intersection (9 zones)
│  Port 8002          │
└──────────┬──────────┘
           │  InterpreterResult (NES, ROI, zones, issues)
           ▼
┌─────────────────────┐
│  STRATEGIST         │  Google Gemma 3 27B → 3 visual + 2 text
│  agents/strategist  │  changes + new copy + strategy
│  Port 8003          │
└──────────┬──────────┘
           │  StrategistResult (Cloudinary transforms + copy)
           ▼
┌─────────────────────┐
│  EXECUTOR           │  Cloudinary AI transforms → optimized URL
│  agents/executor    │  → NES delta estimation → FinalResult
│  Port 8004          │
└──────────┬──────────┘
           │  FinalResult
           ▼
┌─────────────────────┐
│  ORCHESTRATOR       │  Formats and delivers to original user
│  (receives result)  │
└─────────────────────┘
           │
           ▼
      ASI:One user
  (formatted chat reply)
```

---

## Quick Start

### 1. Clone and setup

```bash
git clone <repo> neurallens
cd neurallens
bash setup.sh
source .venv/bin/activate
```

### 2. Fill in `.env`

```bash
# Already filled with real keys for demo — verify they work:
cat .env
```

### 3. Discover agent addresses

```bash
python tests/test_addresses.py
# Prints all 5 agent1q... addresses
# Copy them into .env under ORCHESTRATOR_ADDRESS etc.
```

### 4. Start all agents

```bash
bash run_all.sh
# Opens 5 Terminal windows, one per agent
```

### 5. Register on Agentverse

Go to [agentverse.ai](https://agentverse.ai) → Register agent →
paste the value from `ORCHESTRATOR_ADDRESS`.

### 6. Test via ASI:One

Open [asi1.ai](https://asi1.ai) and chat with your agent:

```
https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/
  Camponotus_flavomarginatus_ant.jpg/320px-...jpg
optimize this Instagram post for my bakery to increase foot traffic
```

---

## Running Tests

```bash
# No API keys needed:
python tests/test_addresses.py      # print all 5 addresses
python tests/test_nes_math.py       # pure math, no network

# Requires real API keys:
python tests/test_tribe_api.py      # tests Colab /infer /deepgaze /heatmap
python tests/test_gemma.py          # tests Gemma 3 prompt + parsing
python tests/test_cloudinary.py     # tests upload + transforms
```

---

## Agent Addresses

```
ORCHESTRATOR_ADDRESS=  (fill after running python tests/test_addresses.py)
SENSOR_ADDRESS=
INTERPRETER_ADDRESS=
STRATEGIST_ADDRESS=
EXECUTOR_ADDRESS=
```

---

## Key Files

| File | Role |
|------|------|
| `agents/orchestrator.py` | Chat Protocol, ASI:One LLM parsing, result delivery |
| `agents/sensor_agent.py` | TRIBE v2 + DeepGaze API calls, heatmap upload |
| `agents/interpreter_agent.py` | NES math, zone analysis |
| `agents/strategist_agent.py` | Gemma 3 optimization strategy |
| `agents/executor_agent.py` | Cloudinary transforms, FinalResult |
| `agents/models.py` | All shared uAgents Model classes |
| `pipeline/nes_math.py` | Pure NES math (no agents) |
| `integrations/cloudinary_client.py` | Upload + transform functions |

---

## Hackathon Track

**LA Hacks 2026 — Fetch.ai Track 1: Agentverse Search & Discovery**

NeuralLens demonstrates:
- ✅ Multi-agent pipeline (5 specialized agents)
- ✅ ASI:One LLM for intent parsing (OpenAI-compatible SDK)
- ✅ Chat Protocol with correct `uagents_core` imports
- ✅ `publish_manifest=True` for Agentverse discoverability
- ✅ `mailbox=True` on all agents for ASI:One reachability
- ✅ Permanent deterministic addresses via fixed seed phrases
- ✅ Zero custom frontend — entire demo runs through ASI:One
