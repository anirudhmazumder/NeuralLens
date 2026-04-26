"""
integrations/__init__.py

NeuralLens external service integrations.
"""

from integrations.cloudinary_client import (
    upload_image,
    create_heatmap_overlay,
    apply_visual_transformations,
)

__all__ = [
    "upload_image",
    "create_heatmap_overlay",
    "apply_visual_transformations",
]
