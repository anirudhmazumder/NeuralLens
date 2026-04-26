from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from neurolens.deepgaze_attn import DeepGazePredictor


def make_dummy_webpage(path: Path) -> None:
    img = Image.new("RGB", (1200, 760), "#f5f7fb")
    d = ImageDraw.Draw(img)

    # Header/nav
    d.rectangle((0, 0, 1200, 90), fill="#ffffff")
    d.rectangle((40, 25, 180, 65), fill="#1f2937")  # logo block
    for i in range(4):
        x = 320 + i * 130
        d.rectangle((x, 30, x + 90, 55), fill="#c7d2fe")

    # Hero text area
    d.rectangle((80, 150, 560, 510), fill="#ffffff")
    d.rectangle((120, 200, 500, 230), fill="#94a3b8")
    d.rectangle((120, 250, 460, 272), fill="#cbd5e1")
    d.rectangle((120, 286, 430, 308), fill="#cbd5e1")

    # Strong call-to-action region (expected attention hotspot)
    d.rectangle((150, 360, 420, 440), fill="#ef4444")

    # Right-side product visual with contrast
    d.rectangle((650, 130, 1110, 540), fill="#111827")
    d.ellipse((760, 210, 1010, 460), fill="#22d3ee")
    d.rectangle((730, 490, 1030, 525), fill="#fde047")

    # Footer cards
    for i in range(3):
        x0 = 120 + i * 330
        d.rectangle((x0, 590, x0 + 280, 700), fill="#ffffff")
        d.rectangle((x0 + 20, 615, x0 + 220, 637), fill="#cbd5e1")

    img.save(path)


def main() -> int:
    out_dir = Path("runs/deepgaze-dummy-example")
    out_dir.mkdir(parents=True, exist_ok=True)

    input_img = out_dir / "dummy_webpage.png"
    make_dummy_webpage(input_img)

    img = Image.open(input_img).convert("RGB")
    pred = DeepGazePredictor(max_side=768)
    metrics = pred.metrics(img)
    heat = out_dir / "dummy_saliency_heatmap.png"
    overlay = out_dir / "dummy_saliency_overlay.png"
    pred.save_heatmap_png(img, heat)
    pred.save_overlay_png(img, overlay)

    print("DeepGaze dummy demo complete")
    print(f"- input:   {input_img}")
    print(f"- heatmap: {heat}")
    print(f"- overlay: {overlay}")
    print(
        "- peak fixation (x,y): "
        f"({metrics['peak_x']}, {metrics['peak_y']}) | "
        f"p_peak={metrics['peak_prob_model']:.6f} | "
        f"entropy={metrics['entropy_nats']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
