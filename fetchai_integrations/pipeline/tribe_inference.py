"""
pipeline/tribe_inference.py

Wraps Meta FAIR's TRIBE v2 fMRI brain-simulation model.
Converts a webpage screenshot + text into a voxel-level brain activation
array, and renders those activations as a heatmap overlay.

IMPORTANT: The HuggingFace model id is "facebook/tribev2".
Output field names ("brain_activations", "fmri_predictions", "logits")
must be verified against the official model card — see docs/nes_formula.md.
"""

# stdlib
import io
import warnings

# third-party
import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Model loading — happens once at import time
# ---------------------------------------------------------------------------

USE_MOCK_TRIBE: bool = False

_processor = None
_model = None

def load_tribe_model():
    """
    Load the TRIBE v2 processor and model from HuggingFace.

    Returns:
        tuple: (processor, model) both on CPU in eval mode.

    Side effects:
        Sets the module-level USE_MOCK_TRIBE flag to True if loading fails,
        and prints a warning so the caller knows mock data will be used.
    """
    global USE_MOCK_TRIBE

    print("Loading TRIBE v2... (this takes ~60 seconds)")
    try:
        # Import lazily so the module still loads without transformers
        from transformers import AutoProcessor, AutoModel  # type: ignore

        processor = AutoProcessor.from_pretrained("facebook/tribev2")
        model = AutoModel.from_pretrained("facebook/tribev2")
        model.eval()
        print("TRIBE v2 loaded successfully.")
        return processor, model

    except (OSError, EnvironmentError, Exception) as exc:
        warnings.warn(
            f"[NeuralLens] TRIBE v2 failed to load ({exc}). "
            "Falling back to mock inference — results are illustrative only.",
            RuntimeWarning,
            stacklevel=2,
        )
        USE_MOCK_TRIBE = True
        return None, None


try:
    _processor, _model = load_tribe_model()
except Exception:
    USE_MOCK_TRIBE = True
    _processor, _model = None, None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def run_tribe_inference(screenshot_bytes: bytes, text: str) -> np.ndarray:
    """
    Run TRIBE v2 (or mock) on a webpage screenshot and accompanying text.

    Args:
        screenshot_bytes: Full-page PNG image as raw bytes.
        text: Clean visible text extracted from the page (will be truncated
              to 512 chars internally).

    Returns:
        np.ndarray of shape (N,) containing voxel-level brain activations,
        where N is determined by the model (typically ~70 000 voxels).
    """
    if USE_MOCK_TRIBE or _processor is None or _model is None:
        return mock_tribe_inference()

    import torch  # type: ignore

    image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    truncated_text = text[:512]

    inputs = _processor(images=image, text=truncated_text, return_tensors="pt")

    with torch.no_grad():
        outputs = _model(**inputs)

    # -----------------------------------------------------------------------
    # Field-name resolution — MUST be verified against the TRIBE v2 model card
    # The order of preference follows likely naming conventions in the paper.
    # -----------------------------------------------------------------------
    activation_tensor = None
    for field in ("brain_activations", "fmri_predictions", "logits"):
        if hasattr(outputs, field):
            activation_tensor = getattr(outputs, field)
            break

    if activation_tensor is None:
        # Last resort: grab the first tensor in the output object
        for val in outputs.values():
            import torch as _torch
            if isinstance(val, _torch.Tensor):
                activation_tensor = val
                break

    if activation_tensor is None:
        raise ValueError(
            "Could not locate brain activation tensor in TRIBE v2 output. "
            "Check model card for correct output field names."
        )

    activations: np.ndarray = activation_tensor.squeeze().detach().cpu().numpy()
    return activations.flatten()


def activations_to_heatmap(activations: np.ndarray, screenshot_bytes: bytes) -> bytes:
    """
    Render brain activations as a hot-colormap heatmap sized to the screenshot.

    Args:
        activations: 1-D array of voxel activation values.
        screenshot_bytes: Original page PNG used only to determine output size.

    Returns:
        PNG bytes of the heatmap overlay (same pixel dimensions as the screenshot).
    """
    original = Image.open(io.BytesIO(screenshot_bytes))
    width_px, height_px = original.size

    # Normalize to [0, 1]
    act_min, act_max = activations.min(), activations.max()
    if act_max - act_min < 1e-9:
        normalized = np.zeros_like(activations, dtype=float)
    else:
        normalized = (activations - act_min) / (act_max - act_min)

    # Build the nearest square grid that fits within the activation array
    side = int(np.floor(np.sqrt(len(normalized))))
    grid = normalized[: side * side].reshape(side, side)

    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100))
    ax.imshow(grid, cmap="hot", alpha=0.7, interpolation="bilinear",
              aspect="auto", extent=[0, width_px, height_px, 0])
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Mock fallback
# ---------------------------------------------------------------------------

def mock_tribe_inference() -> np.ndarray:
    """
    Generate realistic-looking random brain activations for development/testing.

    Returns:
        np.ndarray of shape (70000,) sampled from a Beta(2, 5) distribution,
        which produces a right-skewed distribution similar to real fMRI data.
    """
    return np.random.beta(2, 5, 70_000).astype(np.float32)
