"""Command-line entry: `neurolens optimize`, `neurolens view`, `neurolens fetch-atlas`,
`neurolens demo-atlas`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neurolens")
    sub = parser.add_subparsers(dest="cmd", required=True)

    view = sub.add_parser("view", help="Serve the demo viewer for a completed run.")
    view.add_argument("run_dir", type=Path)
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--no-open", action="store_true")

    opt = sub.add_parser("optimize", help="Run the optimization loop on an image.")
    opt.add_argument("image", type=Path, help="Path to UI screenshot (png/jpg).")
    opt.add_argument(
        "--intent",
        choices=["engage", "trust", "convert", "accessibility", "gamification"],
        default="engage",
    )
    opt.add_argument("--iters", type=int, default=5)
    opt.add_argument("--out", type=Path, default=Path("runs"))
    opt.add_argument(
        "--agent",
        choices=["claude", "mock"],
        default="mock",
        help="Use 'claude' for real Anthropic API; 'mock' is offline.",
    )
    opt.add_argument(
        "--encoder",
        choices=["stub", "atlas", "tribe"],
        default="atlas",
        help="'stub' = image-feature only, 'atlas' = synthetic voxels through real "
        "HCP-MMP1, 'tribe' = real TRIBE v2 (TODO).",
    )
    opt.add_argument(
        "--atlas-path",
        type=Path,
        default=None,
        help="Override path to HCP-MMP1 NIfTI. If unset, uses the cached download.",
    )
    opt.add_argument(
        "--labels-tsv",
        type=Path,
        default=None,
        help="Optional Glasser labels TSV (index<TAB>L_<name>_ROI ...).",
    )

    fa = sub.add_parser("fetch-atlas", help="Download and cache the HCP-MMP1 atlas.")
    fa.add_argument("--force", action="store_true", help="Re-download even if cached.")
    fa.add_argument(
        "--no-download",
        action="store_true",
        help="Print the cache path that would be used; don't fetch.",
    )

    da = sub.add_parser(
        "demo-atlas",
        help="Workflow demo: load atlas, build masks, run synthetic voxels through "
        "aggregate(), prove FFA-spike voxels yield a high FFA score.",
    )
    da.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip the download and use the synthetic mini-atlas (always works).",
    )
    da.add_argument("--atlas-path", type=Path, default=None)
    da.add_argument("--labels-tsv", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "view":
        from .viewer.server import serve

        serve(args.run_dir, port=args.port, open_browser=not args.no_open)
        return 0

    if args.cmd == "fetch-atlas":
        from .rois import HCP_MMP1_MNI, cache_dir, download_atlas

        if args.no_download:
            print(cache_dir() / HCP_MMP1_MNI.nifti_filename)
            return 0
        path = download_atlas(force=args.force)
        print(f"\natlas ready: {path}")
        return 0

    if args.cmd == "demo-atlas":
        return _demo_atlas(synthetic=args.synthetic, atlas_path=args.atlas_path, labels_tsv=args.labels_tsv)

    if args.cmd == "optimize":
        return _optimize(args)
    return 1


def _demo_atlas(synthetic: bool, atlas_path: Path | None, labels_tsv: Path | None) -> int:
    import numpy as np

    from .rois import (
        HCP_MMP1_MNI,
        RegionMasks,
        aggregate,
        cache_dir,
        coverage_report,
        load_atlas,
        synthetic_atlas,
    )

    if synthetic:
        atlas = synthetic_atlas(shape=(32, 32, 32))
        print("using synthetic mini-atlas (32^3 voxels)")
    else:
        path = atlas_path or (cache_dir() / HCP_MMP1_MNI.nifti_filename)
        if not path.exists():
            print(f"atlas not cached at {path}")
            print("run `neurolens fetch-atlas` first, or pass --synthetic for the smoke test.")
            return 2
        print(f"loading atlas: {path}")
        atlas = load_atlas(path, labels_tsv=labels_tsv)

    print(f"atlas shape: {atlas.shape}, max label: {int(atlas.labels.max())}")
    masks = RegionMasks.build(atlas)
    print()
    print(coverage_report(masks))
    print()
    if "FFA" not in masks.masks:
        print("FFA mask not built; pass --labels-tsv if your atlas uses different indices.")
        return 3

    rng = np.random.default_rng(0)
    voxels = np.full(atlas.labels.shape, 0.10, dtype=np.float32).ravel()
    voxels += rng.normal(0, 0.02, size=voxels.shape).astype(np.float32)
    voxels[masks.masks["FFA"]] = 0.90
    if "V4" in masks.masks:
        voxels[masks.masks["V4"]] = 0.50

    scores = aggregate(voxels, masks, normalize=False)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print("voxel spike -> region scores:")
    for name, score in ranked:
        bar = "#" * int(round(score * 30))
        print(f"  {name:14s} {score:+.3f}  {bar}")
    top = ranked[0][0]
    if top != "FFA":
        print(f"\nFAIL: expected FFA at the top; got {top}")
        return 4
    print("\nPASS: FFA-targeted voxels produced the highest FFA region score.")
    return 0


def _optimize(args) -> int:
    from .agent import ClaudeAgent, MockAgent
    from .loop import LoopConfig, run
    from .tribe import StubEncoder

    if not args.image.exists():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2
    if args.encoder == "stub":
        encoder = StubEncoder()
    elif args.encoder == "atlas":
        from .rois import HCP_MMP1_MNI, RegionMasks, cache_dir, load_atlas, synthetic_atlas
        from .tribe import AtlasStubEncoder

        atlas_path = args.atlas_path or (cache_dir() / HCP_MMP1_MNI.nifti_filename)
        if atlas_path.exists():
            print(f"using atlas encoder with: {atlas_path}")
            atlas = load_atlas(atlas_path, labels_tsv=args.labels_tsv)
        else:
            print(f"atlas not cached at {atlas_path}; using synthetic mini-atlas.")
            atlas = synthetic_atlas(shape=(32, 32, 32))
        masks = RegionMasks.build(atlas)
        encoder = AtlasStubEncoder(atlas=atlas, masks=masks)
    else:
        from .tribe import TribeV2Encoder

        encoder = TribeV2Encoder()

    agent = ClaudeAgent() if args.agent == "claude" else MockAgent()
    result = run(
        args.image,
        LoopConfig(iterations=args.iters, intent=args.intent, out_dir=args.out),
        encoder=encoder,
        agent=agent,
    )
    print(f"\nrun complete -> {result['run_dir']}")
    first = result["records"][0]["reward_total"]
    last = result["records"][-1]["reward_total"]
    print(f"reward: {first:+.3f} -> {last:+.3f}  ({last - first:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
