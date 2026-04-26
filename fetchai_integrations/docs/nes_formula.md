# Neural Engagement Score — Formula & Methodology

## Overview

The Neural Engagement Score (NES) is a 0–100 composite metric that predicts
how strongly a real human brain would engage with a webpage, based on
simulated fMRI activations from Meta FAIR's TRIBE v2 model.

## Scoring formula

```
NES = clamp(Σ roi[r] × weight[r], 0, 100)
```

| Region | Weight | Rationale |
|--------|--------|-----------|
| Striatum (ventral) | **+0.25** | Reward/desire circuitry; strongest purchase-intent signal |
| Amygdala | **+0.20** | Emotional arousal; drives click-through and sharing |
| Hippocampus | **+0.15** | Memory encoding; ensures brand recall |
| dlPFC | **−0.20** | Cognitive load; high activation → page is confusing |
| IPS | **+0.10** | Visual attention salience; ensures CTA is noticed |
| mPFC | **+0.10** | Self-relevance; "this product is for me" signal |
| Insula | **±0.00** | Context-dependent distrust; currently neutral |

## Affective dimensions

### Valence (−1 to +1)

```
valence = (
    (striatum − 50) × 0.4
  + (amygdala − 50) × 0.3
  + (insula   − 50) × −0.3
) / 100
```

Positive valence → pleasant, appealing page experience.
Negative valence → repulsive, off-putting, or distressing.

### Arousal (−1 to +1)

```
arousal = (
    (amygdala − 50) × 0.4
  + (dlPFC    − 50) × 0.3
  + (IPS      − 50) × 0.3
) / 100
```

High arousal + high valence → excitement (ideal for impulse purchase CTAs).
High arousal + low valence → anxiety (common on overcrowded landing pages).

## ROI voxel index ranges

> **These ranges must be verified against the TRIBE v2 paper appendix and the
> MNI152 brain atlas used during model training before production use.**

| Region | Voxel indices | Anatomical reference |
|--------|--------------|---------------------|
| Amygdala | 1200–1450 | Bilateral basolateral amygdala (BLA) |
| Striatum | 2100–2380 | Ventral striatum / nucleus accumbens (NAcc) |
| Hippocampus | 3400–3700 | Bilateral CA1/CA3 + subiculum |
| dlPFC | 8200–8600 | BA 9/46, bilateral |
| IPS | 5100–5400 | Intraparietal sulcus, bilateral |
| mPFC | 7800–8100 | Medial prefrontal cortex, BA 10/11 |
| Insula | 4200–4500 | Bilateral anterior insula |

## TRIBE v2 output field resolution

When loading the model via HuggingFace `transformers`, the pipeline checks
output fields in this priority order:

1. `brain_activations` — most specific; likely the intended field name
2. `fmri_predictions` — alternative naming used in some model checkpoints
3. `logits` — generic fallback for transformer-style heads
4. First available tensor — last-resort fallback with a warning

**Action required before production:** Run `print(outputs.keys())` after a
real inference call and verify which field name is correct, then hard-code
it in `tribe_inference.py` to remove the resolution loop.

## References

- TRIBE v2 paper: Meta FAIR (to be released — check HuggingFace model card)
- MNI152 brain atlas: https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152Lin
- Russell's circumplex model of affect: Russell, J.A. (1980). *J. Personality & Social Psychology*, 39(6), 1161–1178.
