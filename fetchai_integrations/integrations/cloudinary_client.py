"""
Cloudinary upload and transformation functions.
Uses credentials from environment variables.
"""

import cloudinary
import cloudinary.uploader
import cloudinary.utils
import os
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"],
    secure=True,
)


def upload_image(image_bytes: bytes, public_id: str) -> str:
    """
    Uploads PNG/JPG bytes to Cloudinary.
    Returns the secure HTTPS URL of the uploaded image.

    Args:
        image_bytes: raw image bytes
        public_id: unique identifier for this upload
                   use timestamps to avoid collisions
    Returns:
        str: secure Cloudinary URL
    """
    result = cloudinary.uploader.upload(
        image_bytes,
        public_id=public_id,
        overwrite=True,
        resource_type="image",
    )
    return result["secure_url"]


def create_heatmap_overlay(
    base_public_id: str,
    overlay_public_id: str,
) -> str:
    """
    Composites heatmap on top of original screenshot.
    Returns URL of composited image.
    Uses Cloudinary URL-based transformation — no extra code.

    Args:
        base_public_id: Cloudinary public_id of base image
        overlay_public_id: Cloudinary public_id of heatmap
    Returns:
        str: URL of composited image
    """
    url, _ = cloudinary.utils.cloudinary_url(
        base_public_id,
        transformation=[
            {"overlay": overlay_public_id},
            {"opacity": 60},
            {"effect": "colorize", "color": "red"},
            {"flags": "layer_apply"},
        ],
    )
    return url


def apply_visual_transformations(
    public_id: str,
    transformations: list,
) -> str:
    """
    Applies a list of Cloudinary transformation dicts
    to an already-uploaded image.
    Returns the transformed image URL.

    Args:
        public_id: Cloudinary public_id of image to transform
        transformations: list of Cloudinary transformation dicts
    Returns:
        str: URL of transformed image
    """
    url, _ = cloudinary.utils.cloudinary_url(
        public_id,
        transformation=transformations,
    )
    return url
