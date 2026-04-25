# NeuralLens — NeuroUI Optimizer

AI-powered UI design optimization using brain activation prediction.

Pipeline: UI screenshot → predicted fMRI activation (TRIBE v2) → ROI scoring (HCP atlas) → reward signal → Claude agent suggests one concrete edit → repeat 5–10 times.

## Status

Hackathon scaffolding. The pipeline runs end-to-end today using a **deterministic stub encoder** in place of TRIBE v2 inference, so you can iterate on reward design, ethics guardrails, and the agent loop without a GPU. Real TRIBE v2 / nilearn HCP integration lands behind the `neuro` extra (see `src/neurolens/tribe.py` and `src/neurolens/rois.py` TODOs).

## Install

```bash
pip install -e .             # core: agent + reward + stub encoder
pip install -e '.[neuro]'    # adds torch, transformers, nilearn for real TRIBE v2
pip install -e '.[dev]'      # tests
```

## Run a demo loop

```bash
export ANTHROPIC_API_KEY=sk-ant-...
neurolens optimize path/to/screenshot.png --intent engage --iters 5
```

Outputs land in `runs/<timestamp>/`: per-iteration screenshots, region score JSON, and a plain-language transparency report.

## Architecture

| Component | Module | Notes |
|-----------|--------|-------|
| Brain encoder | `tribe.py` | TRIBE v2 wrapper; deterministic stub by default |
| ROI aggregation | `rois.py` | Voxels → 8 named regions via HCP atlas (stubbed) |
| Reward | `reward.py` | Intent-aware targets + penalties + Yerkes-Dodson ceiling |
| Ethics | `ethics.py` | Dark pattern detector, valence check, intent gating |
| Agent | `agent.py` | Claude (Opus 4.7) suggests one edit per iteration |
| Edits | `edit.py` | PIL programmatic image transforms |
| Transparency | `transparency.py` | Per-iteration plain-language report |
| Loop | `loop.py` | End-to-end optimization driver |
| CLI | `cli.py` | `neurolens optimize ...` |

## Brain regions targeted

Engagement: FFA, V4, MT+, Hippocampus
Trust: PFC (DLPFC)
Penalty: ACC, Amygdala, Insula
Dual: Nucleus Accumbens (target only when intent=gamification)
Language channel: Broca's / Wernicke's (text encoder, optional)

See `docs/regions.md` (coming) for activation drivers and agent prompts per region.

## Ethics

Built in, not bolted on:

- **Intent declaration** required at run start (`engage` / `trust` / `convert` / `accessibility` / `gamification`)
- **Dark pattern detector** — flags edits that push amygdala > 0.6 or NAcc > 0.7 without matching intent
- **Yerkes-Dodson ceiling** — penalty when any region exceeds 0.85 (avoid runaway stimulation)
- **Valence check** — never optimize toward high-arousal-negative states
- **Transparency report** — every run dumps which regions changed, by how much, what edit drove it, and any ethical flags raised

## What's honest

TRIBE v2 predicts population-average responses, not individual brains. fMRI is an indirect (blood-flow) proxy. Applying region-level neuroscience to UI is a novel inference layer — the regions themselves are well-replicated; the UI mapping is informed hypothesis. The amygdala/insula UI inferences are the most extrapolated.
