"""TRIBE v2 brain encoder wrapper.

The real model (Meta FAIR, CC BY-NC, HuggingFace) outputs ~70k cortical voxel
activations per image. We expose a uniform `encode(image) -> dict[region, score]`
surface so callers don't care which backend produced the numbers.

Two backends:

* `StubEncoder` — deterministic, GPU-free, derives plausible region activations
  from cheap image features (face-likeness via skin-tone clustering, color
  saturation, edge orientation, contrast, etc.). Good enough to drive the
  agent loop while real TRIBE v2 is being wired up.
* `TribeV2Encoder` — TODO. Loads the HF checkpoint, runs inference, then
  delegates voxel→region aggregation to `rois.aggregate`.

The stub is calibrated so its scalar outputs land in roughly [0, 1] and respond
to the same visual properties TRIBE v2 would (faces↑FFA, saturation↑V4, etc.).
This means edits suggested against the stub are still neuroscientifically
plausible — they're just based on hand-rolled image features instead of
learned voxel predictors.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageStat

from .regions import REGIONS, all_names


class Encoder(Protocol):
    def encode(self, image: Image.Image) -> dict[str, float]: ...


# ---------- Stub backend ---------------------------------------------------


def _luma_contrast(img: Image.Image) -> float:
    g = img.convert("L")
    stat = ImageStat.Stat(g)
    # stddev / 128 gives a 0..1-ish contrast proxy
    return min(1.0, stat.stddev[0] / 96.0)


def _saturation(img: Image.Image) -> float:
    hsv = img.convert("HSV")
    s_band = hsv.split()[1]
    return ImageStat.Stat(s_band).mean[0] / 255.0


def _skin_fraction(img: Image.Image) -> float:
    """Cheap face/skin proxy: fraction of pixels in a skin-tone HSV envelope."""
    small = img.convert("HSV").resize((64, 64))
    pixels = small.load()
    hits = 0
    for y in range(64):
        for x in range(64):
            h, s, v = pixels[x, y]
            # Skin tones: H ~ 0..30 (red-orange-yellow), moderate S and V
            if 0 <= h <= 30 and 50 <= s <= 200 and 80 <= v <= 230:
                hits += 1
    return hits / (64 * 64)


def _edge_diagonality(img: Image.Image) -> float:
    """Proxy for implied motion: ratio of diagonal vs axis-aligned edges."""
    from PIL import ImageFilter

    edges = img.convert("L").filter(ImageFilter.FIND_EDGES).resize((64, 64))
    px = edges.load()
    diag = axis = 0
    for y in range(1, 63):
        for x in range(1, 63):
            c = px[x, y]
            if c < 40:
                continue
            # Axis-aligned: strong edge if neighbour up/down or left/right matches
            if abs(px[x, y - 1] - px[x, y + 1]) > abs(px[x - 1, y - 1] - px[x + 1, y + 1]):
                axis += 1
            else:
                diag += 1
    if diag + axis == 0:
        return 0.0
    return diag / (diag + axis)


def _whitespace_fraction(img: Image.Image) -> float:
    """Pixels close to white (proxy for breathing room / clean PFC layouts)."""
    g = img.convert("L").resize((64, 64))
    px = g.load()
    light = sum(1 for y in range(64) for x in range(64) if px[x, y] > 230)
    return light / (64 * 64)


def _palette_clash(img: Image.Image) -> float:
    """High mean saturation + high hue variance -> insula-style visceral unease."""
    hsv = img.convert("HSV").resize((64, 64))
    h_band, s_band, _ = hsv.split()
    h_stats = ImageStat.Stat(h_band)
    s_mean = ImageStat.Stat(s_band).mean[0] / 255.0
    hue_var = h_stats.stddev[0] / 128.0
    return min(1.0, s_mean * hue_var * 1.6)


def _novelty_hash(img: Image.Image) -> float:
    """Stable pseudo-novelty: hash the downscaled image. Same input → same score."""
    small = img.convert("RGB").resize((16, 16))
    h = hashlib.sha256(small.tobytes()).digest()
    # Map first 4 bytes to 0..1 — deterministic, varies between distinct designs
    n = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
    return 0.3 + 0.6 * n  # bias into a useful range


def _sigmoid(x: float, center: float = 0.5, k: float = 6.0) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - center)))


@dataclass
class StubEncoder:
    """Deterministic, GPU-free encoder.

    Maps cheap image features to plausible region activations. Same image →
    same scores, so the loop's improvement signal is real.
    """

    def encode(self, image: Image.Image) -> dict[str, float]:
        sat = _saturation(image)
        contrast = _luma_contrast(image)
        skin = _skin_fraction(image)
        diag = _edge_diagonality(image)
        white = _whitespace_fraction(image)
        clash = _palette_clash(image)
        novelty = _novelty_hash(image)

        # Engagement
        ffa = _sigmoid(skin * 6.0, center=0.5)            # faces drive FFA
        v4 = _sigmoid(sat, center=0.4, k=8.0)             # color saturation drives V4
        mtp = _sigmoid(diag, center=0.45, k=8.0)          # diagonal/implied motion
        hippo = novelty                                   # distinctiveness

        # Trust
        # PFC rewards whitespace + readable contrast, punishes clutter (high clash)
        pfc = _sigmoid(0.6 * white + 0.4 * contrast - 0.5 * clash, center=0.3)

        # Penalties
        # ACC = "confusion" proxy: low whitespace + many edges
        acc = _sigmoid(diag * (1.0 - white), center=0.45, k=7.0)
        # Amygdala: high red saturation + low whitespace ≈ aggressive urgency
        red_dom = _red_dominance(image)
        amyg = _sigmoid(0.6 * red_dom + 0.4 * (1.0 - white), center=0.55, k=8.0)
        insula = _sigmoid(clash, center=0.4, k=8.0)

        # Dual
        # NAcc fires on bright reward signals — very high sat + high contrast spots
        nacc = _sigmoid(sat * contrast, center=0.35, k=8.0)

        scores = {
            "FFA": ffa,
            "V4": v4,
            "MT+": mtp,
            "Hippocampus": hippo,
            "PFC": pfc,
            "ACC": acc,
            "Amygdala": amyg,
            "Insula": insula,
            "NAcc": nacc,
        }
        # Sanity: every region in REGIONS should be covered.
        assert set(scores) == set(all_names()), "stub encoder missing regions"
        return scores


def _red_dominance(img: Image.Image) -> float:
    """Fraction of pixels where red channel dominates strongly."""
    rgb = img.convert("RGB").resize((64, 64))
    px = rgb.load()
    hits = 0
    for y in range(64):
        for x in range(64):
            r, g, b = px[x, y]
            if r > 160 and r > g + 40 and r > b + 40:
                hits += 1
    return hits / (64 * 64)


# ---------- Real TRIBE v2 backend (stub for later) -------------------------


@dataclass
class TribeV2Encoder:
    """Real TRIBE v2 inference. Requires the `[neuro]` extra.

    Loads the HuggingFace checkpoint once, runs image → 70k voxel activations,
    then delegates voxel → region aggregation to `rois.aggregate`.
    """

    checkpoint: str = "facebook/tribe-v2"  # TODO: confirm canonical HF id
    device: str = "cuda"

    def __post_init__(self) -> None:  # pragma: no cover — requires GPU
        raise NotImplementedError(
            "TribeV2Encoder is a placeholder. Wire up: "
            "(1) `transformers.AutoModel.from_pretrained(self.checkpoint)`, "
            "(2) image preprocessing per the TRIBE v2 model card, "
            "(3) voxel output → rois.aggregate(voxels). "
            "Until then, pass `StubEncoder()` to the optimizer."
        )

    def encode(self, image: Image.Image) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError


@dataclass
class RemoteTribeEncoder:
    """Call a remote TRIBE inference endpoint that returns region scores."""

    endpoint: str
    token: str | None = None
    timeout_s: float = 30.0
    image_field: str = "image_base64"
    request_mode: str = "auto"  # auto | json | raw
    response_mode: str = "auto"  # auto | scores | voxels
    masks: object | None = None
    voxel_shape: tuple[int, int, int] | None = None
    normalize_voxels: bool = True
    subcortical_mode: str = "auto"  # auto | api | estimate

    def __post_init__(self) -> None:
        self.endpoint = self.endpoint.strip()
        if not self.endpoint:
            raise ValueError("remote endpoint cannot be empty")
        if not (self.endpoint.startswith("http://") or self.endpoint.startswith("https://")):
            raise ValueError(f"remote endpoint must be http(s), got: {self.endpoint!r}")
        if self.request_mode not in {"auto", "json", "raw"}:
            raise ValueError(f"remote request_mode must be one of auto/json/raw, got {self.request_mode!r}")
        if self.response_mode not in {"auto", "scores", "voxels"}:
            raise ValueError(f"remote response_mode must be one of auto/scores/voxels, got {self.response_mode!r}")
        if self.subcortical_mode not in {"auto", "api", "estimate"}:
            raise ValueError(
                f"remote subcortical_mode must be one of auto/api/estimate, got {self.subcortical_mode!r}"
            )

    def _headers(self, content_type: str) -> dict[str, str]:
        headers = {"Content-Type": content_type, "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _post(self, body: bytes, content_type: str) -> bytes:
        req = Request(
            self.endpoint,
            data=body,
            headers=self._headers(content_type),
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_s) as resp:
            return resp.read()

    @staticmethod
    def _lookup_num(d: dict[str, object], key: str) -> float | None:
        v = d.get(key)
        if v is None:
            v = d.get(key.lower())
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _fill_subcortical(
        self,
        out: dict[str, float],
        image: Image.Image,
        *,
        preferred: dict[str, object] | None = None,
        fallback: dict[str, object] | None = None,
    ) -> dict[str, float]:
        """
        Ensure Hippocampus/Amygdala/NAcc exist.

        Priority:
        1) explicit values from response (preferred, then fallback dicts)
        2) local image-driven estimates (no fixed constants)
        """
        preferred = preferred or {}
        fallback = fallback or {}
        pref_sub = preferred.get("subcortical_estimates")
        pref_vals = pref_sub.get("values") if isinstance(pref_sub, dict) else {}
        fb_sub = fallback.get("subcortical_estimates")
        fb_vals = fb_sub.get("values") if isinstance(fb_sub, dict) else {}

        use_api = self.subcortical_mode in {"auto", "api"}
        use_est = self.subcortical_mode in {"auto", "estimate"}

        hip = None
        if use_api:
            hip = self._lookup_num(preferred, "Hippocampus")
            if hip is None and isinstance(pref_vals, dict):
                hip = self._lookup_num(pref_vals, "Hippocampus")
            if hip is None:
                hip = self._lookup_num(fallback, "Hippocampus")
            if hip is None and isinstance(fb_vals, dict):
                hip = self._lookup_num(fb_vals, "Hippocampus")
        if hip is None and use_est:
            hip = _novelty_hash(image)
        if hip is None:
            hip = 0.5

        amyg = None
        if use_api:
            amyg = self._lookup_num(preferred, "Amygdala")
            if amyg is None and isinstance(pref_vals, dict):
                amyg = self._lookup_num(pref_vals, "Amygdala")
            if amyg is None:
                amyg = self._lookup_num(fallback, "Amygdala")
            if amyg is None and isinstance(fb_vals, dict):
                amyg = self._lookup_num(fb_vals, "Amygdala")
        if amyg is None and use_est:
            red = _red_dominance(image)
            ins = float(out.get("Insula", _sigmoid(_palette_clash(image), center=0.4, k=8.0)))
            acc = float(out.get("ACC", 0.5))
            amyg = _sigmoid(0.55 * red + 0.30 * ins + 0.15 * acc, center=0.48, k=7.5)
        if amyg is None:
            amyg = 0.3

        nacc = None
        if use_api:
            nacc = self._lookup_num(preferred, "NAcc")
            if nacc is None and isinstance(pref_vals, dict):
                nacc = self._lookup_num(pref_vals, "NAcc")
            if nacc is None:
                nacc = self._lookup_num(fallback, "NAcc")
            if nacc is None and isinstance(fb_vals, dict):
                nacc = self._lookup_num(fb_vals, "NAcc")
        if nacc is None and use_est:
            nacc = _sigmoid(_saturation(image) * _luma_contrast(image), center=0.35, k=8.0)
        if nacc is None:
            nacc = 0.4

        out["Hippocampus"] = float(max(0.0, min(1.0, hip)))
        out["Amygdala"] = float(max(0.0, min(1.0, amyg)))
        out["NAcc"] = float(max(0.0, min(1.0, nacc)))
        return out

    def encode(self, image: Image.Image) -> dict[str, float]:
        import numpy as np

        buf = BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        png_bytes = buf.getvalue()
        payload = {
            self.image_field: base64.b64encode(png_bytes).decode("ascii"),
            "image_format": "PNG",
        }
        json_body = json.dumps(payload).encode("utf-8")
        raw: bytes
        try:
            if self.request_mode == "raw":
                raw = self._post(png_bytes, "image/png")
            elif self.request_mode == "json":
                raw = self._post(json_body, "application/json")
            else:
                try:
                    raw = self._post(json_body, "application/json")
                except HTTPError as e:
                    detail = ""
                    try:
                        detail = e.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    # Compatibility with Flask endpoints that expect raw PNG in request body.
                    if e.code == 400 and ("raw PNG bytes" in detail or "request body" in detail):
                        raw = self._post(png_bytes, "image/png")
                    else:
                        raise
        except HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            msg = f"remote TRIBE HTTP {e.code}"
            if detail:
                msg += f": {detail}"
            raise RuntimeError(msg) from e
        except URLError as e:
            raise RuntimeError(f"remote TRIBE request failed: {e}") from e

        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise RuntimeError("remote TRIBE returned non-JSON response") from e

        if not isinstance(data, dict):
            raise RuntimeError("remote TRIBE response must be a JSON object")

        def _try_scores(d: dict) -> dict[str, float] | None:
            required_cortical = {"FFA", "V4", "MT+", "PFC", "ACC", "Insula"}
            if isinstance(d.get("scores"), dict):
                sraw = d["scores"]
            elif isinstance(d.get("region_scores"), dict):
                sraw = d["region_scores"]
            else:
                sraw = d
            required = set(all_names())
            if required.issubset(set(sraw.keys())):
                out = {k: float(sraw[k]) for k in all_names()}
                return self._fill_subcortical(out, image, preferred=sraw, fallback=d)
            # Accept cortical-only responses and estimate missing subcortical values locally.
            if required_cortical.issubset(set(sraw.keys())):
                out = {k: float(sraw[k]) for k in required_cortical}
                return self._fill_subcortical(out, image, preferred=sraw, fallback=d)
            return None

        def _try_voxels(d: dict) -> dict[str, float] | None:
            if self.masks is None:
                raise RuntimeError(
                    "remote response contains voxels but no atlas masks were provided; "
                    "pass masks=... when creating RemoteTribeEncoder"
                )
            vox = d.get("voxels")
            if vox is None:
                return None
            arr = np.asarray(vox, dtype=np.float32)
            if arr.ndim == 1:
                pass
            elif arr.ndim == 3:
                arr = arr.ravel()
            else:
                raise RuntimeError(f"remote voxels must be 1D or 3D, got shape {arr.shape}")
            if self.voxel_shape is not None and tuple(np.asarray(vox).shape) == self.voxel_shape:
                arr = np.asarray(vox, dtype=np.float32).ravel()
            from .rois import aggregate

            cortical_scores = aggregate(arr, self.masks, normalize=self.normalize_voxels)
            return self._fill_subcortical(dict(cortical_scores), image, fallback=d)

        if self.response_mode == "scores":
            scores = _try_scores(data)
            if scores is None:
                raise RuntimeError("remote response_mode=scores but response does not contain all region scores")
            return scores
        if self.response_mode == "voxels":
            scores = _try_voxels(data)
            if scores is None:
                raise RuntimeError("remote response_mode=voxels but response has no 'voxels' field")
            return scores

        # auto: prefer explicit region scores; else aggregate voxels.
        scores = _try_scores(data)
        if scores is not None:
            return scores
        scores = _try_voxels(data)
        if scores is not None:
            return scores

        required = set(all_names())
        missing = sorted(required - set(data.keys()))
        raise RuntimeError(
            "remote TRIBE response did not include full region scores "
            "and had no usable voxels field; missing regions: "
            f"{missing}"
        )


@dataclass
class AtlasStubEncoder:
    """Synthetic-voxels-through-real-atlas encoder."""

    atlas: "object"
    masks: "object"

    def encode(self, image: Image.Image) -> dict[str, float]:
        import numpy as np

        feats = {
            "FFA": _sigmoid(_skin_fraction(image) * 6.0, center=0.5),
            "V4": _sigmoid(_saturation(image), center=0.4, k=8.0),
            "MT+": _sigmoid(_edge_diagonality(image), center=0.45, k=8.0),
            "PFC": _sigmoid(
                0.6 * _whitespace_fraction(image)
                + 0.4 * _luma_contrast(image)
                - 0.5 * _palette_clash(image),
                center=0.3,
            ),
            "ACC": _sigmoid(
                _edge_diagonality(image) * (1.0 - _whitespace_fraction(image)),
                center=0.45,
                k=7.0,
            ),
            "Insula": _sigmoid(_palette_clash(image), center=0.4, k=8.0),
        }

        voxels = np.zeros(self.atlas.labels.shape, dtype=np.float32).ravel()
        rng = np.random.default_rng(42)
        for region_name, base in feats.items():
            mask = self.masks.masks.get(region_name)
            if mask is None:
                continue
            n = int(mask.sum())
            voxels[mask] = base + rng.normal(0.0, 0.03, size=n).astype(np.float32)

        from .rois import aggregate

        cortical_scores = aggregate(voxels, self.masks, normalize=False)
        sat = _saturation(image)
        contrast = _luma_contrast(image)
        white = _whitespace_fraction(image)
        red = _red_dominance(image)

        out: dict[str, float] = dict(cortical_scores)
        out["Hippocampus"] = _novelty_hash(image)
        out["Amygdala"] = _sigmoid(0.6 * red + 0.4 * (1.0 - white), center=0.55, k=8.0)
        out["NAcc"] = _sigmoid(sat * contrast, center=0.35, k=8.0)
        assert set(out) == set(all_names()), f"AtlasStubEncoder missing: {set(all_names()) - set(out)}"
        return out


def load_encoder(name: str = "stub", **kwargs) -> Encoder:
    if name == "stub":
        return StubEncoder()
    if name == "atlas":
        atlas = kwargs.get("atlas")
        masks = kwargs.get("masks")
        if atlas is None or masks is None:
            raise ValueError("atlas encoder needs atlas= and masks= kwargs")
        return AtlasStubEncoder(atlas=atlas, masks=masks)
    if name == "tribe":
        return TribeV2Encoder()
    if name == "remote":
        endpoint = kwargs.get("endpoint")
        if not endpoint:
            raise ValueError("remote encoder needs endpoint= kwargs")
        return RemoteTribeEncoder(
            endpoint=endpoint,
            token=kwargs.get("token"),
            timeout_s=float(kwargs.get("timeout_s", 30.0)),
            image_field=str(kwargs.get("image_field", "image_base64")),
            request_mode=str(kwargs.get("request_mode", "auto")),
            response_mode=str(kwargs.get("response_mode", "auto")),
            masks=kwargs.get("masks"),
            voxel_shape=kwargs.get("voxel_shape"),
            normalize_voxels=bool(kwargs.get("normalize_voxels", True)),
            subcortical_mode=str(kwargs.get("subcortical_mode", "auto")),
        )
    raise ValueError(f"unknown encoder: {name!r}")


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")
