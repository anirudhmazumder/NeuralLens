"""Command-line entry: `neurolens optimize <image> --intent engage --iters 5`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import ClaudeAgent, MockAgent
from .loop import LoopConfig, run
from .tribe import StubEncoder


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
        choices=["stub", "tribe"],
        default="stub",
        help="'stub' = deterministic image-feature encoder; 'tribe' = real TRIBE v2 (TODO).",
    )

    args = parser.parse_args(argv)

    if args.cmd == "view":
        from .viewer.server import serve
        serve(args.run_dir, port=args.port, open_browser=not args.no_open)
        return 0

    if args.cmd == "optimize":
        if not args.image.exists():
            print(f"image not found: {args.image}", file=sys.stderr)
            return 2
        encoder = StubEncoder() if args.encoder == "stub" else None
        if encoder is None:
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
    return 1


if __name__ == "__main__":
    sys.exit(main())
