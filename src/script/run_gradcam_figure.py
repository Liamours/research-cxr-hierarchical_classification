"""Regenerate Fig. 5 (Grad-CAM qualitative examples) against the corrected
NIH boxes. Loads the bucket selection from run_localization_eval.py's
gradcam_bucket_examples.json (run that first) and draws one panel per
selected image: flat condition's Grad-CAM heatmap as background (the
hierarchical model only contributes its predicted box, per the archived
figure's own convention -- confirmed by inspection, not re-derived here),
ground truth / flat / hierarchical boxes in three distinct line styles so
the figure supports grayscale printing per the reviewer's accessibility
ask.

    uv run python src/script/run_gradcam_figure.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image, ImageDraw, ImageFont

from src.data.label_space import CANONICAL_LABELS
from src.script.run_localization_eval import CONDITIONS, OUT_DIR, load_condition_model
from src.script.run_xai import blend_heatmap
from src.xai.gradcam import GradCAM, resolve_target_layer

BUCKET_JSON = OUT_DIR / "gradcam_bucket_examples.json"
OUT_PNG = Path(r"C:\rifqi\research-cxr-hierarchical_classification\draft\revision_1\components\fig5_gradcam.png")

# Sampled via Image.getpixel() from the archived figure's own legend
# swatches, not eyeballed -- see context/task/experiments.md for the
# sampling command.
GT_COLOR = (255, 212, 0)
FLAT_COLOR = (195, 166, 232)
HIER_COLOR = (166, 227, 184)

BUCKET_ORDER = ("both_miss", "both_hit", "flat_only", "hier_only")
COLUMN_HEADERS = {
    "both_miss": "Both miss",
    "both_hit": "Both localize\n(IoU > 0)",
    "flat_only": "Flat localizes,\nhierarchical misses",
    "hier_only": "Hierarchical localizes,\nflat misses",
}

PANEL_SIZE = 400
TITLE_H = 70
HEADER_H = 100
LEGEND_H = 80
BOX_WIDTH = 4
# 1 example per bucket (4 panels, single row): page-budget cut from the
# original 2 (8 panels, 4x2), approved 2026-09-02. gradcam_bucket_examples.json
# still selects 2 per bucket; only the first (same "first file-order
# occurrence" pick) is drawn here -- bump this back to 2 to restore the
# original 4x2 figure without touching the bucketing logic at all.
PANELS_PER_BUCKET = 1


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "timesbd.ttf" if bold else "times.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def draw_style_rect(draw: ImageDraw.ImageDraw, box, color, style: str, width: int = BOX_WIDTH) -> None:
    """PIL has no dashed-line primitive: solid uses draw.rectangle, dashed
    and dotted walk each edge drawing dash/gap segments (dotted = shorter
    dashes, tighter gaps)."""
    if style == "solid":
        draw.rectangle(box, outline=color, width=width)
        return
    x1, y1, x2, y2 = box
    dash, gap = (14, 8) if style == "dashed" else (3, 6)
    for (sx, sy), (ex, ey) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                                ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = math.hypot(ex - sx, ey - sy)
        if length == 0:
            continue
        step = dash + gap
        n = int(length // step) + 1
        for i in range(n):
            t0 = min(1.0, i * step / length)
            t1 = min(1.0, t0 + dash / length)
            if t0 >= 1.0:
                break
            draw.line([(sx + (ex - sx) * t0, sy + (ey - sy) * t0),
                       (sx + (ex - sx) * t1, sy + (ey - sy) * t1)], fill=color, width=width)


def make_panel(record: dict, cam: GradCAM, transform, image_size: int, device) -> Image.Image:
    class_idx = CANONICAL_LABELS.index(record["canonical"])
    img = Image.open(record["image_path"]).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    heatmap = cam(x, class_idx=class_idx)[0].cpu().numpy()

    blended = blend_heatmap(record["image_path"], heatmap, alpha=0.5, image_size=image_size)
    panel_img = Image.fromarray(blended).resize((PANEL_SIZE, PANEL_SIZE), Image.LANCZOS)
    draw = ImageDraw.Draw(panel_img)

    scale = PANEL_SIZE / image_size
    def scaled(box):
        x1, y1, x2, y2 = box
        return (x1 * scale, y1 * scale, x2 * scale, y2 * scale)

    draw_style_rect(draw, scaled(record["gt_box"]), GT_COLOR, "solid")
    if record["flat_pred_box"] is not None:
        draw_style_rect(draw, scaled(record["flat_pred_box"]), FLAT_COLOR, "dashed")
    if record["hier_pred_box"] is not None:
        draw_style_rect(draw, scaled(record["hier_pred_box"]), HIER_COLOR, "dotted")

    canvas = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE + TITLE_H), (255, 255, 255))
    canvas.paste(panel_img, (0, TITLE_H))
    d = ImageDraw.Draw(canvas)
    title = record["canonical"].replace("_", " ")
    subtitle = f"flat={record['flat_iou']:.3f}, hier={record['hier_iou']:.3f}"
    d.text((PANEL_SIZE / 2, 18), title, fill=(0, 0, 0), font=_font(22), anchor="mm")
    d.text((PANEL_SIZE / 2, 48), subtitle, fill=(0, 0, 0), font=_font(18), anchor="mm")
    return canvas


def blank_panel() -> Image.Image:
    return Image.new("RGB", (PANEL_SIZE, PANEL_SIZE + TITLE_H), (255, 255, 255))


def draw_legend(canvas: Image.Image, y0: int) -> None:
    d = ImageDraw.Draw(canvas)
    items = [(GT_COLOR, "solid", "Ground truth"), (FLAT_COLOR, "dashed", "Flat prediction"),
             (HIER_COLOR, "dotted", "Hierarchical prediction")]
    x = 60
    for color, style, label in items:
        swatch = (x, y0 + 20, x + 70, y0 + 50)
        draw_style_rect(d, swatch, color, style)
        d.text((x + 85, y0 + 35), label, fill=(0, 0, 0), font=_font(22), anchor="lm")
        x += 85 + d.textlength(label, font=_font(22)) + 60


def main() -> None:
    with open(BUCKET_JSON, encoding="utf-8") as f:
        buckets = json.load(f)["examples"]

    device = torch.device("cpu")
    model, transform = load_condition_model(CONDITIONS["flat"], device)
    cam = GradCAM(model, resolve_target_layer(model))
    # Read the transform's real output size rather than assuming 224.
    image_size = transform(Image.new("RGB", (224, 224))).shape[-1]

    total_w = PANEL_SIZE * len(BUCKET_ORDER)
    total_h = HEADER_H + (PANEL_SIZE + TITLE_H) * PANELS_PER_BUCKET + LEGEND_H
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    for col, bucket_name in enumerate(BUCKET_ORDER):
        header = COLUMN_HEADERS[bucket_name]
        cx = col * PANEL_SIZE + PANEL_SIZE / 2
        lines = header.split("\n")
        y = HEADER_H / 2 - (len(lines) - 1) * 14
        for line in lines:
            d.text((cx, y), line, fill=(0, 0, 0), font=_font(24, bold=True), anchor="mm")
            y += 28

        examples = buckets[bucket_name]
        for row in range(PANELS_PER_BUCKET):
            panel = (make_panel(examples[row], cam, transform, image_size, device)
                     if row < len(examples) else blank_panel())
            canvas.paste(panel, (col * PANEL_SIZE, HEADER_H + row * (PANEL_SIZE + TITLE_H)))

    cam.remove()
    draw_legend(canvas, HEADER_H + (PANEL_SIZE + TITLE_H) * PANELS_PER_BUCKET)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PNG)
    print(f"Written: {OUT_PNG} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
