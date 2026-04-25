"""Atlas tests using the synthetic mini-atlas (no network required)."""

import numpy as np

from neurolens.regions import REGIONS, cortical_names
from neurolens.rois import KNOWN_LH_INDICES, RegionMasks, aggregate, synthetic_atlas


def test_synthetic_atlas_has_expected_labels():
    atlas = synthetic_atlas(shape=(32, 32, 32))
    present = set(np.unique(atlas.labels)) - {0}
    expected = set()
    for lh in KNOWN_LH_INDICES.values():
        expected.update({lh, lh + 180})
    overlap = present & expected
    assert len(overlap) >= len(KNOWN_LH_INDICES), (
        f"only {len(overlap)} known indices present; "
        "synthetic atlas slab packing dropped too many."
    )


def test_region_masks_cover_cortical_regions():
    atlas = synthetic_atlas(shape=(64, 8, 8))
    masks = RegionMasks.build(atlas)
    cortical = set(cortical_names())
    have = set(masks.masks)
    assert have == cortical, f"missing cortical regions: {cortical - have}"
    for r, region in REGIONS.items():
        if not region.cortical:
            assert r not in masks.masks


def test_aggregate_picks_up_targeted_voxel_spike():
    atlas = synthetic_atlas(shape=(64, 8, 8))
    masks = RegionMasks.build(atlas)
    voxels = np.full(atlas.labels.shape, 0.1, dtype=np.float32).ravel()
    voxels[masks.masks["FFA"]] = 0.9
    voxels[masks.masks["V4"]] = 0.5
    scores = aggregate(voxels, masks, normalize=False)
    assert scores["FFA"] > scores["V4"] > scores["PFC"]
    assert abs(scores["FFA"] - 0.9) < 0.01
    assert abs(scores["V4"] - 0.5) < 0.01


def test_aggregate_normalize_maps_to_unit_range():
    atlas = synthetic_atlas(shape=(64, 8, 8))
    masks = RegionMasks.build(atlas)
    voxels = np.full(atlas.labels.shape, 0.0, dtype=np.float32).ravel()
    voxels[masks.masks["FFA"]] = 1.0
    voxels[masks.masks["V4"]] = 0.3
    scores = aggregate(voxels, masks, normalize=True)
    assert 0.0 <= min(scores.values()) <= 1e-6
    assert abs(max(scores.values()) - 1.0) < 1e-6
