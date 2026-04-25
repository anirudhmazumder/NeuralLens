"""Smoke test: full loop runs end-to-end with stub encoder + mock agent."""

from pathlib import Path

from PIL import Image

from neurolens.loop import LoopConfig, run


def test_loop_runs_endtoend(tmp_path):
    img_path = tmp_path / "fake_ui.png"
    # 256x256 colorful synthetic image — gives the stub encoder real signal
    img = Image.new("RGB", (256, 256), "white")
    px = img.load()
    for y in range(256):
        for x in range(256):
            px[x, y] = ((x * 5) % 256, (y * 3) % 256, ((x + y) * 2) % 256)
    img.save(img_path)

    out = tmp_path / "runs"
    result = run(img_path, LoopConfig(iterations=3, intent="engage", out_dir=out))

    assert len(result["records"]) == 4  # 3 edits + final eval
    run_dir = Path(result["run_dir"])
    assert (run_dir / "report.md").exists()
    assert (run_dir / "iterations.jsonl").exists()
    assert (run_dir / "final.png").exists()
