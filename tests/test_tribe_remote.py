from __future__ import annotations

import json
from io import BytesIO
from urllib.error import HTTPError

from PIL import Image

from neurolens.regions import all_names
from neurolens.rois import RegionMasks, synthetic_atlas
from neurolens.tribe import RemoteTribeEncoder


class _FakeResponse:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._raw


def _region_payload(v: float = 0.33) -> dict[str, float]:
    return {k: v for k in all_names()}


def test_remote_encoder_accepts_scores_object(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        body = json.loads(req.data.decode("utf-8"))
        assert "image_base64" in body
        return _FakeResponse({"scores": _region_payload(0.5)})

    monkeypatch.setattr("neurolens.tribe.urlopen", _fake_urlopen)
    enc = RemoteTribeEncoder(endpoint="https://example.com/encode ")
    scores = enc.encode(Image.new("RGB", (8, 8), "white"))
    assert captured["url"] == "https://example.com/encode"
    assert captured["timeout"] == 30.0
    assert scores["FFA"] == 0.5


def test_remote_encoder_accepts_region_scores_object(monkeypatch):
    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        del req, timeout
        return _FakeResponse(
            {
                "vertices": {"lh": [0.0] * 4, "rh": [0.0] * 4},
                "region_scores": _region_payload(0.61),
                "meta": {"n_vertices_per_hemi": 4, "n_timesteps": 2, "surface": "fsaverage5"},
            }
        )

    monkeypatch.setattr("neurolens.tribe.urlopen", _fake_urlopen)
    enc = RemoteTribeEncoder(endpoint="https://example.com/encode")
    scores = enc.encode(Image.new("RGB", (8, 8), "white"))
    assert scores["FFA"] == 0.61
    assert scores["NAcc"] == 0.61


def test_remote_encoder_reads_nested_subcortical_estimates(monkeypatch):
    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        del req, timeout
        return _FakeResponse(
            {
                "region_scores": {
                    "FFA": 0.82,
                    "V4": 0.61,
                    "MT+": 0.74,
                    "PFC": 0.45,
                    "ACC": 0.38,
                    "Insula": 0.51,
                },
                "subcortical_estimates": {
                    "values": {"Hippocampus": 0.43, "Amygdala": 0.67, "NAcc": 0.29},
                    "method": "nearest_cortical_vertex",
                },
            }
        )

    monkeypatch.setattr("neurolens.tribe.urlopen", _fake_urlopen)
    enc = RemoteTribeEncoder(endpoint="https://example.com/encode", subcortical_mode="api")
    scores = enc.encode(Image.new("RGB", (8, 8), "white"))
    assert scores["Hippocampus"] == 0.43
    assert scores["Amygdala"] == 0.67
    assert scores["NAcc"] == 0.29


def test_remote_encoder_estimates_missing_subcorticals(monkeypatch):
    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        del req, timeout
        # Cortical-only payload.
        return _FakeResponse(
            {
                "region_scores": {
                    "FFA": 0.8,
                    "V4": 0.4,
                    "MT+": 0.3,
                    "PFC": 0.6,
                    "ACC": 0.2,
                    "Insula": 0.25,
                }
            }
        )

    monkeypatch.setattr("neurolens.tribe.urlopen", _fake_urlopen)
    enc = RemoteTribeEncoder(endpoint="https://example.com/encode")
    scores = enc.encode(Image.new("RGB", (8, 8), "white"))
    assert set(scores) == set(all_names())
    # Missing subcorticals are estimated, not fixed old constants.
    assert scores["Hippocampus"] != 0.5
    assert scores["Amygdala"] != 0.3
    assert scores["NAcc"] != 0.4


def test_remote_encoder_can_force_estimated_subcorticals(monkeypatch):
    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        del req, timeout
        return _FakeResponse(
            {
                "region_scores": {
                    "FFA": 0.8,
                    "V4": 0.4,
                    "MT+": 0.3,
                    "PFC": 0.6,
                    "ACC": 0.2,
                    "Insula": 0.25,
                    "Hippocampus": 0.5,
                    "Amygdala": 0.3,
                    "NAcc": 0.4,
                }
            }
        )

    monkeypatch.setattr("neurolens.tribe.urlopen", _fake_urlopen)
    enc = RemoteTribeEncoder(
        endpoint="https://example.com/encode",
        subcortical_mode="estimate",
    )
    scores = enc.encode(Image.new("RGB", (8, 8), "white"))
    assert scores["Hippocampus"] != 0.5
    assert scores["Amygdala"] != 0.3
    assert scores["NAcc"] != 0.4


def test_remote_encoder_accepts_flat_region_object(monkeypatch):
    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        del req, timeout
        return _FakeResponse(_region_payload(0.25))

    monkeypatch.setattr("neurolens.tribe.urlopen", _fake_urlopen)
    enc = RemoteTribeEncoder(endpoint="https://example.com/encode")
    scores = enc.encode(Image.new("RGB", (8, 8), "black"))
    assert set(scores) == set(all_names())
    assert scores["PFC"] == 0.25


def test_remote_encoder_auto_falls_back_to_raw_png(monkeypatch):
    calls: list[str] = []

    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        del timeout
        ct = req.headers.get("Content-type", "")
        calls.append(ct)
        if ct == "application/json":
            body = BytesIO(b'{"error":"POST raw PNG bytes as request body"}')
            raise HTTPError(req.full_url, 400, "Bad Request", hdrs=None, fp=body)
        assert ct == "image/png"
        return _FakeResponse({"scores": _region_payload(0.7)})

    monkeypatch.setattr("neurolens.tribe.urlopen", _fake_urlopen)
    enc = RemoteTribeEncoder(endpoint="https://example.com/encode", request_mode="auto")
    scores = enc.encode(Image.new("RGB", (8, 8), "red"))
    assert calls == ["application/json", "image/png"]
    assert scores["V4"] == 0.7


def test_remote_encoder_aggregates_voxel_response(monkeypatch):
    atlas = synthetic_atlas(shape=(8, 8, 8))
    masks = RegionMasks.build(atlas)
    flat_n = atlas.labels.size
    vox = [0.1] * flat_n
    # Create a clear cortical hotspot for FFA mask.
    for idx, on in enumerate(masks.masks["FFA"]):
        if bool(on):
            vox[idx] = 0.9

    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        del req, timeout
        return _FakeResponse({"voxels": vox, "Hippocampus": 0.55, "Amygdala": 0.25, "NAcc": 0.45})

    monkeypatch.setattr("neurolens.tribe.urlopen", _fake_urlopen)
    enc = RemoteTribeEncoder(
        endpoint="https://example.com/encode",
        response_mode="voxels",
        masks=masks,
        voxel_shape=atlas.shape,
    )
    scores = enc.encode(Image.new("RGB", (8, 8), "black"))
    assert scores["FFA"] > scores["V4"]
    assert scores["Hippocampus"] == 0.55
    assert scores["Amygdala"] == 0.25
    assert scores["NAcc"] == 0.45
