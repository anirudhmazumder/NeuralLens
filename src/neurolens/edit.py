"""Programmatic image edits.

The agent picks one edit per iteration via a structured tool call. This module
defines the catalog and applies the chosen edit. Each edit is small,
reversible (the loop saves prev/next pairs), and targets a specific region's
known activation drivers.

Catalog deliberately stays narrow — fancier edits (face insertion, layout
restructuring) are stretch goals; the hackathon demo runs on these primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class EditSpec:
    name: str
    description: str
    targets_regions: tuple[str, ...]
    apply: Callable[[Image.Image, dict[str, Any]], Image.Image]


def _saturate(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    factor = float(params.get("factor", 1.3))
    return ImageEnhance.Color(img).enhance(factor)


def _contrast(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    factor = float(params.get("factor", 1.2))
    return ImageEnhance.Contrast(img).enhance(factor)


def _brightness(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    factor = float(params.get("factor", 1.1))
    return ImageEnhance.Brightness(img).enhance(factor)


def _sharpen(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    factor = float(params.get("factor", 1.5))
    return ImageEnhance.Sharpness(img).enhance(factor)


def _add_whitespace(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Pad with white border — proxy for breathing room (PFC↑, ACC↓)."""
    pad = int(params.get("pad_pct", 8))
    w, h = img.size
    border = (max(1, w * pad // 100), max(1, h * pad // 100))
    return ImageOps.expand(img, border=(border[0], border[1]), fill="white")


def _motion_blur(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Subtle directional blur — implied motion for MT+."""
    radius = float(params.get("radius", 1.5))
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _desaturate_reds(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Pull saturation off red-dominant regions — amygdala-soothing."""
    factor = float(params.get("factor", 0.7))
    rgb = img.convert("RGB")
    r, g, b = rgb.split()
    # Reduce red channel where it dominates
    from PIL import ImageMath

    new_r = ImageMath.eval("convert(min(r, max(g, b) + 30), 'L')", r=r, g=g, b=b)
    merged = Image.merge("RGB", (new_r, g, b))
    return ImageEnhance.Color(merged).enhance(factor)


def _grayscale_partial(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Pull saturation down — reduces V4 and insula. Use sparingly."""
    factor = float(params.get("factor", 0.6))
    return ImageEnhance.Color(img).enhance(factor)


CATALOG: dict[str, EditSpec] = {
    "increase_saturation": EditSpec(
        name="increase_saturation",
        description="Boost color saturation. Drives V4 (color cortex) up.",
        targets_regions=("V4",),
        apply=_saturate,
    ),
    "decrease_saturation": EditSpec(
        name="decrease_saturation",
        description="Reduce overall saturation. Calms V4 and insula.",
        targets_regions=("V4", "Insula"),
        apply=_grayscale_partial,
    ),
    "increase_contrast": EditSpec(
        name="increase_contrast",
        description="Raise luma contrast. Improves PFC readability and overall clarity.",
        targets_regions=("PFC", "Insula"),
        apply=_contrast,
    ),
    "increase_brightness": EditSpec(
        name="increase_brightness",
        description="Lift overall brightness. Adds whitespace-feel; calms ACC.",
        targets_regions=("PFC", "ACC"),
        apply=_brightness,
    ),
    "sharpen": EditSpec(
        name="sharpen",
        description="Sharpen edges and text. Helps PFC; clarifies hierarchy.",
        targets_regions=("PFC",),
        apply=_sharpen,
    ),
    "add_whitespace": EditSpec(
        name="add_whitespace",
        description="Pad with white border. Breathing room — PFC up, ACC down.",
        targets_regions=("PFC", "ACC"),
        apply=_add_whitespace,
    ),
    "motion_blur": EditSpec(
        name="motion_blur",
        description="Subtle blur, implies motion. Drives MT+ on static images.",
        targets_regions=("MT+",),
        apply=_motion_blur,
    ),
    "desaturate_reds": EditSpec(
        name="desaturate_reds",
        description="Reduce red dominance. Lowers amygdala (urgency/threat coding).",
        targets_regions=("Amygdala",),
        apply=_desaturate_reds,
    ),
}


def apply_edit(img: Image.Image, edit_name: str, params: dict[str, Any] | None = None) -> Image.Image:
    spec = CATALOG.get(edit_name)
    if spec is None:
        raise ValueError(f"unknown edit: {edit_name!r}. Choices: {sorted(CATALOG)}")
    return spec.apply(img, params or {})


def catalog_for_prompt() -> str:
    """Render the catalog as a string for the agent's system prompt."""
    lines = []
    for spec in CATALOG.values():
        regions = ", ".join(spec.targets_regions)
        lines.append(f"- {spec.name} (targets {regions}): {spec.description}")
    return "\n".join(lines)
