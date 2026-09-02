"""Localization metrics against ground-truth bounding boxes, from a Grad-CAM
heatmap. Matches the manuscript's protocol (main.tex, Sec. II-D / IV-B):
predicted box = largest connected component above the heatmap's own 90th
percentile (a fixed threshold produced degenerate whole-image boxes); IoU,
IoBB (intersection over the ground-truth box), pointing-game hit rate
(does the heatmap's argmax pixel fall inside the ground-truth box), and AP
at a given IoU threshold using model confidence as the detection score.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2)


def largest_component_box(heatmap: np.ndarray, percentile: float = 90.0) -> Box | None:
    """Threshold at the heatmap's own percentile, return the bounding box of
    the largest connected component. None if nothing is above the threshold.

    Strict greater-than, matching "above the Nth percentile" literally: a
    heatmap with a large flat region at its minimum (common after Grad-CAM's
    own ReLU, before interpolation smooths it) can otherwise put the
    percentile value AT the minimum, making >= select most of the image."""
    thresh = np.percentile(heatmap, percentile)
    mask = heatmap > thresh
    if not mask.any():
        return None
    labeled, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    largest = 1 + int(np.argmax(sizes))
    ys, xs = np.where(labeled == largest)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_iobb(pred: Box, gt: Box) -> float:
    """Intersection over the ground-truth box area (not the union)."""
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    return inter / gt_area if gt_area > 0 else 0.0


def pointing_game_hit(heatmap: np.ndarray, gt: Box) -> bool:
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    x1, y1, x2, y2 = gt
    return x1 <= x < x2 and y1 <= y < y2


def average_precision_at_iou(scores: list[float], hits: list[bool]) -> float:
    """AP over one category: each image contributes one (score, hit) pair,
    hit = predicted box's IoU against that image's ground-truth box >= the
    threshold used to build `hits`. Standard single-instance-per-image AP,
    computed via the precision-recall curve (sklearn average_precision_score)."""
    if not scores or not any(hits):
        return 0.0
    return float(average_precision_score(np.asarray(hits, dtype=int), np.asarray(scores)))
