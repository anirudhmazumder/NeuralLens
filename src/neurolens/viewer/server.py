"""Tiny static server for the demo viewer.

Serves a run directory at http://localhost:<port>:

    /              -> index.html (copied from this package)
    /index.json    -> {run_name, frames: [iter_00.png, iter_01_<edit>.png, ...]}
    /summary.json  -> the iterations log produced by the loop
    /<png>         -> per-iteration screenshots
    /report.md     -> raw markdown report

Frames are sorted so that index 0 is the baseline, index N is the final image,
and intermediate indices align with the iteration log.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


_ITER_RE = re.compile(r"^iter_(\d+)_")


def _frames_in(run_dir: Path) -> list[str]:
    """Return iteration screenshots ordered: baseline, then iter_01, iter_02, …"""
    pngs = list(run_dir.glob("*.png"))
    baseline = [p for p in pngs if p.name.startswith("iter_00")]
    iters = sorted(
        (p for p in pngs if _ITER_RE.match(p.name) and not p.name.startswith("iter_00")),
        key=lambda p: int(_ITER_RE.match(p.name).group(1)),
    )
    final = [p for p in pngs if p.name == "final.png"]
    ordered = baseline + iters + final
    seen: set[str] = set()
    out: list[str] = []
    for p in ordered:
        if p.name in seen:
            continue
        seen.add(p.name)
        out.append(p.name)
    return out


def _write_index_json(run_dir: Path) -> None:
    (run_dir / "index.json").write_text(
        json.dumps({"run_name": run_dir.name, "frames": _frames_in(run_dir)}, indent=2)
    )


def serve(run_dir: Path, port: int = 8765, open_browser: bool = True) -> None:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"run directory not found: {run_dir}")
    if not (run_dir / "summary.json").exists():
        raise SystemExit(
            f"{run_dir} has no summary.json — did the optimization loop finish?"
        )

    # Copy index.html into the run dir so the static server serves it from /
    pkg_html = Path(__file__).parent / "index.html"
    shutil.copyfile(pkg_html, run_dir / "index.html")
    _write_index_json(run_dir)

    handler_cls = type(
        "RunHandler",
        (SimpleHTTPRequestHandler,),
        {"directory": str(run_dir),
         "__init__": lambda self, *a, **kw: SimpleHTTPRequestHandler.__init__(
             self, *a, directory=str(run_dir), **kw)},
    )

    httpd = HTTPServer(("127.0.0.1", port), handler_cls)
    url = f"http://127.0.0.1:{port}/"
    print(f"NeuralLens viewer serving {run_dir}")
    print(f"  -> {url}")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="neurolens-view")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args(argv)
    serve(args.run_dir, port=args.port, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    sys.exit(main())
