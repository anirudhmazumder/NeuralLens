"""ROI aggregation: voxel activations -> named region scores.

The real flow:
    voxels (np.ndarray, shape ~ (70000,)) in MNI space
        -> nilearn loads HCP-MMP1 parcellation (Glasser 2016) as a labels image
        -> for each region, mean(voxels[mask_for_region])
        -> normalized to [0, 1] for downstream reward weighting

This module exposes a single `aggregate(voxels)` function. The TRIBE v2 backend
calls it after inference; the stub encoder bypasses it (it produces region
scores directly from image features).

The HCP atlas object is loaded lazily and cached. If `nilearn` isn't
installed, calling `aggregate` raises with a clear install hint.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .regions import REGIONS

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


@lru_cache(maxsize=1)
def _load_hcp_atlas():  # pragma: no cover — network + nilearn required
    try:
        from nilearn import datasets
    except ImportError as e:
        raise ImportError(
            "ROI aggregation needs nilearn. Install with: pip install -e '.[neuro]'"
        ) from e
    # The HCP-MMP1 parcellation isn't shipped directly by nilearn; the closest
    # built-in is the Glasser 360 surface atlas. For volumetric MNI work,
    # download glasser_2016_parcellation.nii.gz once and point this to it.
    # TODO: replace with a real loader that returns (labels_img, region_index_map).
    raise NotImplementedError(
        "Wire up HCP-MMP1 atlas loader: download "
        "https://balsa.wustl.edu/file/show/3VLx (Glasser 2016 supplementary) "
        "and use nilearn.image.load_img + a labels lookup."
    )


def aggregate(voxels: "np.ndarray") -> dict[str, float]:
    """Average voxel activations within each region's HCP indices.

    Args:
        voxels: 1D array of activations in MNI space, length ~70k.

    Returns:
        {region_name: mean_activation_in_[0,1]} for every region in REGIONS.
    """
    import numpy as np

    labels_img, index_map = _load_hcp_atlas()  # noqa: F841 — wired up later
    out: dict[str, float] = {}
    for name, region in REGIONS.items():
        # TODO: real implementation looks like:
        #   mask = np.isin(labels_array, region.hcp_indices)
        #   out[name] = float(voxels[mask].mean())
        # For now, fail loudly so callers know to use StubEncoder.
        raise NotImplementedError(
            f"aggregate() needs HCP atlas wired up; cannot score region {name!r}."
        )
    return _normalize(out)


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Map raw voxel means into [0, 1] via min-max with a soft floor.

    Real TRIBE v2 outputs are roughly z-scored; this rescales them to the
    [0, 1] range the reward function expects.
    """
    if not scores:
        return scores
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-6:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
