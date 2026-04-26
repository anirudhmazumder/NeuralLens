# NeuralLens — NeuroUI Optimizer

AI-powered UI design optimization using brain activation prediction.

Pipeline: UI screenshot → predicted fMRI activation (TRIBE v2) → ROI scoring (HCP atlas) → reward signal → Claude agent suggests one concrete edit → repeat 5–10 times.

## Status

Hackathon scaffolding. The pipeline runs end-to-end today with an **atlas-backed demo encoder** (real HCP-MMP1 masks plus synthetic voxel activations), so you can exercise ROI aggregation and reward dynamics without a GPU. Full TRIBE v2 inference remains behind the `neuro` extra (see `src/neurolens/tribe.py`).

## Install

```bash
pip install -e .             # core: agent + reward + stub encoder
pip install -e '.[neuro]'    # adds torch, transformers, nilearn for real TRIBE v2
pip install -e '.[dev]'      # tests
```

## Run a demo loop

```bash
export ANTHROPIC_API_KEY=sk-ant-...
neurolens fetch-atlas
neurolens optimize path/to/screenshot.png --intent engage --iters 5
```

`neurolens optimize` now defaults to `--encoder atlas`. If the atlas file is missing, it falls back to a synthetic mini-atlas and prints a notice.

Outputs land in `runs/<timestamp>/`: per-iteration screenshots, region score JSON, and a plain-language transparency report.

### Use a remote TRIBE server (e.g. Colab + ngrok)

If you already host TRIBE inference elsewhere, call it with the `remote` encoder:

```bash
neurolens optimize path/to/screenshot.png \
  --encoder remote \
  --remote-endpoint "https://deluge-glucose-rework.ngrok-free.dev/encode" \
  --remote-timeout 45
```

Optional auth:

```bash
neurolens optimize path/to/screenshot.png \
  --encoder remote \
  --remote-endpoint "https://<your-host>/encode" \
  --remote-token "<bearer-token>"
```

If your server expects **raw PNG bytes** in the POST body (Flask `request.data` style), set:

```bash
--remote-request-mode raw
```

Default `auto` first tries JSON (`image_base64`) and falls back to raw PNG when the server returns a 400 indicating raw bytes are required.

Expected response JSON is either:

- `{"scores": {"FFA": ..., "V4": ..., ...}}`, or
- `{"region_scores": {"FFA": ..., "V4": ..., ...}, "vertices": {"lh": [...], "rh": [...]}, "meta": {...}}`, or
- direct region map `{"FFA": ..., "V4": ..., ...}`.

Or voxel output:

- `{"voxels": [...]}` (1D flattened atlas grid) or `{"voxels": [[[...]]]}` (3D grid)
- optional subcortical passthrough values: `Hippocampus`, `Amygdala`, `NAcc`

Use `--remote-response-mode voxels` to force voxel aggregation (default `auto` prefers explicit region scores).

Required keys: `FFA`, `V4`, `MT+`, `Hippocampus`, `PFC`, `ACC`, `Amygdala`, `Insula`, `NAcc`.

If remote output is cortical-only (`FFA`, `V4`, `MT+`, `PFC`, `ACC`, `Insula`), NeuralLens will estimate
`Hippocampus`, `Amygdala`, and `NAcc` locally from image/cortical cues (instead of fixed constants).

Use `--remote-subcortical-mode estimate` to always compute those three locally even if API provides placeholder values.

## Architecture

| Component | Module | Notes |
|-----------|--------|-------|
| Brain encoder | `tribe.py` | Atlas-backed demo encoder by default; TRIBE v2 wrapper in progress |
| ROI aggregation | `rois.py` | Voxels → named regions via HCP-MMP1 masks |
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
