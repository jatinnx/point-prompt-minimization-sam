"""Figures for the Step 0 joint inspection.

The panel that matters is the last one: it splits SAM's automatic result into
found / missed / spilled pixels, which is what makes a *partial* failure legible
rather than just a number.
"""
from __future__ import annotations

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import Config  # noqa: E402
from .regions import Region  # noqa: E402

FOUND = (60, 200, 90)      # gt AND matched
MISSED = (230, 60, 60)     # gt AND NOT matched
SPILL = (70, 130, 240)     # matched AND NOT gt


def _rng_colours(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(60, 256, size=(max(n, 1), 3), dtype=np.uint8)


def outline_regions(image: np.ndarray, regions: list[Region],
                    thickness: int = 1) -> np.ndarray:
    """Draw each region's boundary and label it with class and area."""
    canvas = image.copy()
    colours = _rng_colours(len(regions))
    for i, region in enumerate(regions):
        colour = tuple(int(c) for c in colours[i])
        contours, _ = cv2.findContours(region.gt_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, thickness)
    return canvas


def colourise_masks(masks: np.ndarray, shape: tuple[int, int],
                    seed: int = 42) -> np.ndarray:
    """Paint a mask stack, smallest last so small masks stay visible."""
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    if len(masks) == 0:
        return canvas
    order = np.argsort([-int(m.sum()) for m in masks])
    colours = _rng_colours(len(masks), seed)
    for i in order:
        canvas[masks[i]] = colours[i]
    return canvas


def error_overlay(image: np.ndarray, regions: list[Region],
                  alpha: float = 0.55) -> np.ndarray:
    """found / missed / spilled pixels across every region in the image."""
    overlay = image.copy().astype(np.float32)
    paint = np.zeros_like(overlay)
    touched = np.zeros(image.shape[:2], dtype=bool)

    for region in regions:
        gt = region.gt_mask
        pred = region.matched_sam_mask
        if pred is None:
            paint[gt] = MISSED
            touched |= gt
            continue
        paint[np.logical_and(gt, pred)] = FOUND
        paint[np.logical_and(gt, ~pred)] = MISSED
        spill = np.logical_and(pred, ~gt)
        paint[spill] = SPILL
        touched |= gt | pred

    overlay[touched] = (1 - alpha) * overlay[touched] + alpha * paint[touched]
    return overlay.astype(np.uint8)


def inspection_figure(image: np.ndarray, colour_gt: np.ndarray,
                      regions: list[Region], auto_masks: np.ndarray,
                      image_id: str, backend_name: str, out_path):
    """Five-panel Step 0 figure for one image."""
    fig, axes = plt.subplots(1, 5, figsize=(23, 5.2))
    n_unrec = sum(1 for r in regions if r.status == "unrecognized")

    panels = [
        (image, f"{image_id}\nimage"),
        (colour_gt, f"ground truth\n{len({r.class_id for r in regions})} classes"),
        (outline_regions(image, regions),
         f"regions (connected components)\n{len(regions)} regions, "
         f"{n_unrec} unrecognized"),
        (colourise_masks(auto_masks, image.shape[:2]),
         f"SAM automatic masks ({backend_name})\n{len(auto_masks)} masks"),
        (error_overlay(image, regions),
         "matched vs GT\ngreen found / red missed / blue spill"),
    ]
    for ax, (data, title) in zip(axes, panels):
        ax.imshow(data)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def coverage_iou_scatter(regions: list[Region], cfg: Config, out_path,
                         title: str = "region coverage vs IoU"):
    """Where the regions actually sit relative to the provisional thresholds."""
    if not regions:
        return None
    cov = np.array([r.coverage for r in regions])
    iou = np.array([r.iou for r in regions])
    area = np.array([r.area_px for r in regions], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sizes = 12 + 90 * (area / max(area.max(), 1)) ** 0.5
    ax.scatter(cov, iou, s=sizes, alpha=0.55, edgecolor="none")
    ax.axvline(cfg.coverage_threshold, ls="--", lw=1, color="crimson")
    ax.axhline(cfg.iou_threshold, ls="--", lw=1, color="crimson")
    ax.set_xlabel("coverage")
    ax.set_ylabel("IoU")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{title}\n{len(regions)} regions, marker size ~ area; "
                 f"dashed = provisional thresholds", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
