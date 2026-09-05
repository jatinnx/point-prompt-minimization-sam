"""Coverage, IoU, and region-to-mask matching (Section 4, items 4 and 5).

Matching is reported two ways because SAM routinely splits one ground-truth
object into parts -- a building into roof and wall, a harbour into separate
docks. Taking only the single best-overlapping mask understates coverage on
exactly those regions and would mark them unrecognised for the wrong reason.

    best_single   the single automatic mask with the highest IoU
    greedy_union  start from that mask, then keep adding whichever remaining
                  mask improves IoU most, until none does
"""
from __future__ import annotations

import numpy as np

from .regions import RECOGNIZED, UNRECOGNIZED, Region


def coverage(gt_mask: np.ndarray, pred_mask: np.ndarray | None) -> float:
    """Share of the ground-truth region that the prediction found (recall)."""
    gt_area = int(gt_mask.sum())
    if gt_area == 0:
        return 0.0
    if pred_mask is None:
        return 0.0
    return float(np.logical_and(gt_mask, pred_mask).sum()) / gt_area


def iou(gt_mask: np.ndarray, pred_mask: np.ndarray | None) -> float:
    """Intersection over union of two boolean masks."""
    if pred_mask is None:
        return 0.0
    inter = int(np.logical_and(gt_mask, pred_mask).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(gt_mask, pred_mask).sum())
    return inter / union if union else 0.0


def score(gt_mask: np.ndarray, pred_mask: np.ndarray | None) -> tuple[float, float]:
    """(coverage, iou) in one pass."""
    return coverage(gt_mask, pred_mask), iou(gt_mask, pred_mask)


def is_recognized(cov: float, region_iou: float,
                  coverage_threshold: float, iou_threshold: float) -> bool:
    """A region counts as recognised only if it clears *both* thresholds."""
    return cov >= coverage_threshold and region_iou >= iou_threshold


def match_best_single(gt_mask: np.ndarray, masks: np.ndarray):
    """Highest-IoU single mask.

    ``masks`` is (N, H, W) bool. Returns (mask | None, coverage, iou, indices).
    """
    if masks is None or len(masks) == 0:
        return None, 0.0, 0.0, ()
    gt = gt_mask.astype(bool)
    gt_area = int(gt.sum())
    if gt_area == 0:
        return None, 0.0, 0.0, ()

    flat_gt = gt.reshape(-1)
    flat = masks.reshape(len(masks), -1)
    inter = flat[:, flat_gt].sum(axis=1).astype(np.int64)
    areas = flat.sum(axis=1).astype(np.int64)
    union = gt_area + areas - inter
    ious = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    best = int(np.argmax(ious))
    if inter[best] == 0:
        return None, 0.0, 0.0, ()
    return masks[best].astype(bool), float(inter[best]) / gt_area, float(ious[best]), (best,)


def match_greedy_union(gt_mask: np.ndarray, masks: np.ndarray, max_masks: int = 8):
    """Union of automatic masks, grown while IoU keeps improving."""
    if masks is None or len(masks) == 0:
        return None, 0.0, 0.0, ()
    gt = gt_mask.astype(bool)
    gt_area = int(gt.sum())
    if gt_area == 0:
        return None, 0.0, 0.0, ()

    current, cov, best_iou, chosen = match_best_single(gt, masks)
    if current is None:
        return None, 0.0, 0.0, ()

    picked = set(chosen)
    union_mask = current.copy()
    while len(picked) < min(max_masks, len(masks)):
        gain_idx, gain_iou, gain_mask = -1, best_iou, None
        for i in range(len(masks)):
            if i in picked:
                continue
            candidate = np.logical_or(union_mask, masks[i])
            cand_iou = iou(gt, candidate)
            if cand_iou > gain_iou + 1e-12:
                gain_idx, gain_iou, gain_mask = i, cand_iou, candidate
        if gain_idx < 0:
            break
        picked.add(gain_idx)
        union_mask = gain_mask
        best_iou = gain_iou

    return (union_mask, coverage(gt, union_mask), best_iou,
            tuple(sorted(picked)))


MATCHERS = {
    "best_single": match_best_single,
    "greedy_union": match_greedy_union,
}


def score_regions(
    regions: list[Region],
    masks: np.ndarray,
    coverage_threshold: float,
    iou_threshold: float,
    match_mode: str = "greedy_union",
) -> list[Region]:
    """Fill in matched_sam_mask, coverage, iou and status for every region.

    Mutates and returns the same Region objects.
    """
    try:
        matcher = MATCHERS[match_mode]
    except KeyError:
        raise ValueError(f"match_mode must be one of {sorted(MATCHERS)}, "
                         f"got {match_mode!r}") from None

    for region in regions:
        mask, cov, region_iou, indices = matcher(region.gt_mask, masks)
        region.matched_sam_mask = mask
        region.coverage = cov
        region.iou = region_iou
        region.matched_mask_indices = indices
        region.status = (RECOGNIZED
                         if is_recognized(cov, region_iou,
                                          coverage_threshold, iou_threshold)
                         else UNRECOGNIZED)
    return regions


def distribution(values, percentiles=(5, 25, 50, 75, 95)) -> dict:
    """Summary statistics, so thresholds get calibrated from data in Step 6."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"count": 0}
    out = {"count": int(arr.size), "mean": float(arr.mean()),
           "min": float(arr.min()), "max": float(arr.max())}
    for p in percentiles:
        out[f"p{p}"] = float(np.percentile(arr, p))
    return out
