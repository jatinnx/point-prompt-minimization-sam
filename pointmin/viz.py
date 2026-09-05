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

from .class_map import CLASS_NAMES, PALETTE  # noqa: E402
from .config import Config  # noqa: E402
from .regions import RECOGNIZED, Region  # noqa: E402

FOUND = (60, 200, 90)      # gt AND matched
MISSED = (230, 60, 60)     # gt AND NOT matched
SPILL = (70, 130, 240)     # matched AND NOT gt

RECOGNIZED_OUTLINE = (60, 220, 90)
UNRECOGNIZED_OUTLINE = (235, 55, 55)

# Labels are drawn on an upscaled copy: a 256x256 tile has no room for thirty
# legible captions, and the panel is displayed several times that size anyway.
LABEL_SCALE = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45


def _rng_colours(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(60, 256, size=(max(n, 1), 3), dtype=np.uint8)


def class_colour(name: str) -> tuple[int, int, int]:
    """The official DLRSD palette colour for a class name.

    SAM's masks are painted from the same palette as the ground-truth panel, so
    "SAM called this grass" and "this is grass" are the same green: a mislabel
    shows up as a colour mismatch between two panels instead of something you
    have to read off a table.
    """
    try:
        return tuple(int(c) for c in PALETTE[CLASS_NAMES.index(name)])
    except ValueError:
        return (200, 200, 200)


def _upscale(a: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return a
    return cv2.resize(a, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_NEAREST)


def _anchor(mask: np.ndarray) -> tuple[int, int]:
    """The most interior pixel, so a caption never lands outside its own shape."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(dist.argmax()), dist.shape)
    return int(x), int(y)


def _put_label(canvas: np.ndarray, text: str, xy: tuple[int, int],
               colour: tuple[int, int, int]) -> None:
    """Centred text with a dark stroke behind it, legible on any background."""
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
    x = int(np.clip(xy[0] - tw // 2, 1, max(canvas.shape[1] - tw - 1, 1)))
    y = int(np.clip(xy[1] + th // 2, th + 1, canvas.shape[0] - 2))
    for c, t in (((0, 0, 0), 3), (colour, 1)):
        cv2.putText(canvas, text, (x, y), FONT, FONT_SCALE, c, t, cv2.LINE_AA)


def outline_regions(image: np.ndarray, regions: list[Region],
                    scale: int = LABEL_SCALE, colour_by: str = "status",
                    label: bool = True) -> np.ndarray:
    """Each region's boundary, coloured by whether SAM already recognized it.

    Green is recognized, red is the point selector's workload. The panel used to
    give an arbitrary colour per region and the count only in its title, so *which*
    regions were the workload had to be read off the console table.

    Captions are ``<region_id> <class name>``, cut back to the id alone where the
    region is too narrow for the name to fit beside it. ``colour_by="index"``
    restores the arbitrary colours, for regions that have not been scored yet.
    """
    canvas = _upscale(image, scale)
    fallback = _rng_colours(len(regions))
    for i, region in enumerate(regions):
        if colour_by == "status":
            colour = (RECOGNIZED_OUTLINE if region.status == RECOGNIZED
                      else UNRECOGNIZED_OUTLINE)
        else:
            colour = tuple(int(c) for c in fallback[i])
        mask = _upscale(region.gt_mask.astype(np.uint8), scale)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, max(1, scale - 1))
        if not label:
            continue
        text = f"{region.region_id} {region.class_name}"
        (tw, _), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
        if tw > region.bbox[2] * scale:
            text = str(region.region_id)
        _put_label(canvas, text, _anchor(mask), colour)
    return canvas


def colourise_masks(masks: np.ndarray, shape: tuple[int, int], seed: int = 42,
                    labels: list[str] | None = None,
                    scale: int = 1) -> np.ndarray:
    """Paint a mask stack, smallest last so small masks stay visible.

    ``labels`` is one class name per mask -- what SAM answered for it. Given them,
    colour comes from the DLRSD palette rather than a random one, so this panel
    can be read straight across against the ground-truth panel beside it, and each
    name is captioned once on its largest instance with a count. Without them
    colour only separates neighbours, which is all a class-agnostic pass supports.
    """
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    if len(masks) == 0:
        return canvas
    order = np.argsort([-int(m.sum()) for m in masks])      # largest first
    if labels is None:
        colours = _rng_colours(len(masks), seed)
        for i in order:
            canvas[masks[i]] = colours[i]
        return canvas

    canvas = _upscale(canvas, scale)
    by_name: dict[str, list[int]] = {}
    for i in order:
        mask = _upscale(masks[i].astype(np.uint8), scale).astype(bool)
        canvas[mask] = class_colour(labels[i])
        by_name.setdefault(labels[i], []).append(i)
    for name, idx in by_name.items():
        biggest = _upscale(masks[idx[0]].astype(np.uint8), scale)
        text = name if len(idx) == 1 else f"{name} x{len(idx)}"
        _put_label(canvas, text, _anchor(biggest), (255, 255, 255))
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
                      image_id: str, backend_name: str, out_path,
                      masks_label: str = "automatic masks",
                      mask_labels: list[str] | None = None):
    """Five-panel Step 0 figure for one image.

    ``masks_label`` names what panel 4 is showing, because the same figure serves
    both recognition modes -- a class-agnostic segment-everything pass and the
    union of what the 17 class-name prompts returned. Mislabelling those two is
    the easiest way to misread the figure.

    ``mask_labels`` gives one class name per mask, available in text mode only,
    and turns panel 4 from arbitrary colours into the DLRSD palette: the same
    colours as panel 2, so what SAM named a thing can be compared with what it is.
    """
    fig, axes = plt.subplots(1, 5, figsize=(23, 5.2))
    n_unrec = sum(1 for r in regions if r.status == "unrecognized")
    named = " coloured by the name SAM gave" if mask_labels is not None else ""

    panels = [
        (image, f"{image_id}\nimage"),
        (colour_gt, f"ground truth\n{len({r.class_id for r in regions})} classes"),
        (outline_regions(image, regions),
         f"regions: {len(regions)} total, {n_unrec} unrecognized\n"
         f"green = SAM already got it, red = point selector's job"),
        (colourise_masks(auto_masks, image.shape[:2], labels=mask_labels,
                         scale=LABEL_SCALE),
         f"SAM {masks_label} ({backend_name})\n{len(auto_masks)} masks{named}"),
        (error_overlay(image, regions),
         "matched vs GT\ngreen found / red missed / blue spill"),
    ]
    for ax, (data, title) in zip(axes, panels):
        ax.imshow(data)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=155, bbox_inches="tight")
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
