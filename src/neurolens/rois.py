"""HCP-MMP1 atlas wiring: voxel activations -> named region scores.

Pipeline:

    voxels (np.ndarray, shape matches atlas grid) in MNI space
        -> labels array from HCP-MMP1 (Glasser 2016) NIfTI
        -> for each region in REGIONS, mean(voxels[mask]) where mask covers
           every Glasser sub-region (both hemispheres) listed under that
           region's `glasser_names`
        -> normalize across regions for the reward function

The atlas is downloaded on demand and cached under
`~/.cache/neurolens/atlases/`.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .regions import REGIONS, cortical_names

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


@dataclass(frozen=True)
class AtlasManifest:
    name: str
    nifti_url: str
    nifti_filename: str
    nifti_sha256: str | None


HCP_MMP1_MNI = AtlasManifest(
    name="HCP-MMP1 in MNI152",
    nifti_url="https://ndownloader.figshare.com/files/5594363",
    nifti_filename="MMP_in_MNI_corr.nii.gz",
    nifti_sha256=None,
)


KNOWN_LH_INDICES: dict[str, int] = {
    # These indices are for the MMP_in_MNI_corr.nii.gz NIfTI from figshare
    # (https://ndownloader.figshare.com/files/5594363), where LH parcels are
    # labelled 1–180 and RH = LH + 180. If you use a different atlas version
    # (e.g. the HCP-MMP1 .dlabel.nii surface file) the indices will differ.
    #
    # To verify: load the NIfTI, extract unique non-zero labels, and cross-
    # reference against Table S1 of Glasser et al. 2016 (Nature).
    #
    # ⚠ UNVERIFIED: "MST": 2  — in the surface parcellation, parcel 2 is
    #   typically near V1. MST is lateral-temporal and may be higher-indexed.
    #   If atlas coverage reports say MST is empty, check this value first.
    #
    # ⚠ UNVERIFIED: "p24": 179  — p24 (posterior BA24, perigenual ACC) is a
    #   medial-wall parcel. An LH index of 179 puts it at the very boundary of
    #   the 1–180 range, which is implausible for a medial cingulate area.
    #   Recommended: run `coverage_report()` after loading the atlas and
    #   verify the p24 voxel count is non-zero. Expected LH index ≈ 30–70.
    "V1": 1,
    "MST": 2,    # ⚠ unverified — see note above
    "V2": 4,
    "V3": 5,
    "V4": 6,
    "FFC": 18,
    "MT": 23,
    "24dd": 40,
    "24dv": 41,
    "a24": 61,
    "p32": 63,
    "s32": 64,
    "p9-46v": 83,
    "46": 84,
    "a9-46v": 85,
    "9-46d": 86,
    "MI": 109,
    "PI": 110,
    "AVI": 111,
    "AAIC": 112,
    "V4t": 156,
    "FST": 157,
    "p24": 179,  # ⚠ unverified — see note above; expected ≈ 30–70
}


def _both_hemis(lh_idx: int) -> tuple[int, int]:
    return (lh_idx, lh_idx + 180)


def cache_dir() -> Path:
    p = Path.home() / ".cache" / "neurolens" / "atlases"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _verify(path: Path, expected_sha256: str | None) -> bool:
    if expected_sha256 is None:
        return True
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha256


def download_atlas(manifest: AtlasManifest = HCP_MMP1_MNI, force: bool = False) -> Path:
    target = cache_dir() / manifest.nifti_filename
    if target.exists() and not force:
        if _verify(target, manifest.nifti_sha256):
            return target
        target.unlink()
    tmp = target.with_suffix(target.suffix + ".tmp")
    urllib.request.urlretrieve(manifest.nifti_url, tmp)
    if not _verify(tmp, manifest.nifti_sha256):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {manifest.nifti_url}")
    tmp.rename(target)
    return target


@dataclass
class Atlas:
    labels: "np.ndarray"
    affine: "np.ndarray"
    name_to_index: dict[str, int] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.labels.shape)  # type: ignore[return-value]

    def index_for(self, glasser_name: str) -> int | None:
        return self.name_to_index.get(glasser_name)


def _parse_labels_tsv(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        name = parts[1]
        if name.startswith("L_") and name.endswith("_ROI"):
            out[name[2:-4]] = idx
    return out


def load_atlas(nifti_path: str | Path, labels_tsv: str | Path | None = None) -> Atlas:
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError("Atlas loading needs nibabel. Install with: pip install nibabel") from e
    import numpy as np

    img = nib.load(str(nifti_path))
    labels = np.asarray(img.get_fdata(), dtype=np.int32)
    affine = np.asarray(img.affine, dtype=np.float64)
    name_to_index: dict[str, int] = dict(KNOWN_LH_INDICES)
    if labels_tsv is not None:
        name_to_index.update(_parse_labels_tsv(Path(labels_tsv)))
    return Atlas(labels=labels, affine=affine, name_to_index=name_to_index)


@dataclass
class RegionMasks:
    masks: dict[str, "np.ndarray"]
    coverage: dict[str, list[str]]
    missing: dict[str, list[str]]

    @classmethod
    def build(cls, atlas: Atlas) -> "RegionMasks":
        import numpy as np

        flat = atlas.labels.ravel()
        masks: dict[str, np.ndarray] = {}
        coverage: dict[str, list[str]] = {}
        missing: dict[str, list[str]] = {}
        for region_name, region in REGIONS.items():
            if not region.cortical:
                continue
            indices: list[int] = []
            found: list[str] = []
            absent: list[str] = []
            for gname in region.glasser_names:
                lh = atlas.index_for(gname)
                if lh is None:
                    absent.append(gname)
                    continue
                lh_i, rh_i = _both_hemis(lh)
                indices.extend((lh_i, rh_i))
                found.append(gname)
            if not indices:
                missing[region_name] = absent
                continue
            mask = np.isin(flat, indices)
            if int(mask.sum()) == 0:
                missing[region_name] = absent + [f"<no voxels for indices {indices}>"]
                continue
            masks[region_name] = mask
            coverage[region_name] = found
            if absent:
                missing[region_name] = absent
        return cls(masks=masks, coverage=coverage, missing=missing)


def aggregate(
    voxels: "np.ndarray",
    masks: RegionMasks,
    *,
    normalize: bool = True,
) -> dict[str, float]:
    import numpy as np

    flat = np.asarray(voxels).ravel()
    out: dict[str, float] = {}
    for region_name, mask in masks.masks.items():
        if mask.shape[0] != flat.shape[0]:
            raise ValueError(
                f"voxel array length {flat.shape[0]} != mask length {mask.shape[0]}; "
                "did you pass voxels on a different grid than the atlas?"
            )
        out[region_name] = float(flat[mask].mean())
    if normalize:
        out = _normalize(out)
    return out


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return scores
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-6:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def coverage_report(masks: RegionMasks) -> str:
    lines = ["Atlas coverage:"]
    for name in REGIONS:
        if name in masks.masks:
            n = int(masks.masks[name].sum())
            found = ", ".join(masks.coverage.get(name, []))
            miss = ", ".join(masks.missing.get(name, []))
            extra = f"  (missing Glasser names: {miss})" if miss else ""
            lines.append(f"  ok {name:14s} - {n:>6} voxels - {found}{extra}")
        elif name in cortical_names():
            miss = ", ".join(masks.missing.get(name, ["<unknown>"]))
            lines.append(f"  missing {name:14s} - NOT FOUND in atlas - {miss}")
        else:
            lines.append(f"  skip {name:14s} - subcortical, requires separate atlas")
    return "\n".join(lines)


def synthetic_atlas(shape: tuple[int, int, int] = (16, 16, 16)) -> Atlas:
    import numpy as np

    nx, ny, nz = shape
    labels = np.zeros(shape, dtype=np.int32)
    items = list(KNOWN_LH_INDICES.items())
    slabs = max(1, nx // (len(items) + 1))
    for i, (_name, lh_idx) in enumerate(items):
        start = i * slabs
        end = start + slabs
        if end > nx:
            break
        mid = (start + end) // 2
        labels[start:mid, :, :] = lh_idx
        labels[mid:end, :, :] = lh_idx + 180
    affine = _identity_affine()
    return Atlas(labels=labels, affine=affine, name_to_index=dict(KNOWN_LH_INDICES))


def scores_to_volume(scores: dict[str, float], atlas: Atlas, masks: RegionMasks) -> "np.ndarray":
    """Project region scores back into atlas voxel space."""
    import numpy as np

    vol = np.zeros(atlas.shape, dtype=np.float32).ravel()
    for region_name, mask in masks.masks.items():
        val = float(scores.get(region_name, 0.0))
        vol[mask] = val
    return vol.reshape(atlas.shape)


def render_heatmap_png(
    scores: dict[str, float],
    atlas: Atlas,
    masks: RegionMasks,
    out_path: str | Path,
    *,
    threshold: float = 0.05,
) -> Path:
    """Render an fMRI-like stat map PNG with nilearn.

    Raises ImportError if nilearn/nibabel/matplotlib are unavailable.
    """
    from pathlib import Path as _Path

    try:
        import nibabel as nib
        from nilearn import plotting
    except ImportError as e:
        raise ImportError(
            "Heatmap rendering needs nilearn + nibabel. Install with: pip install -e '.[neuro]'"
        ) from e

    volume = scores_to_volume(scores, atlas, masks)
    img = nib.Nifti1Image(volume, atlas.affine)
    out = _Path(out_path)
    display = plotting.plot_stat_map(
        img,
        bg_img=None,
        threshold=threshold,
        display_mode="z",
        cut_coords=7,
        cmap="inferno",
        colorbar=True,
        black_bg=True,
    )
    display.savefig(str(out))
    display.close()
    return out


def _identity_affine():
    import numpy as np

    return np.eye(4, dtype=np.float64)
