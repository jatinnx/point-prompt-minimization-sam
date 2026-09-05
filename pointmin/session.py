"""Prompting sessions -- the revised Section 5 contract.

The plan document specifies ``prompt_sam3(image, points)``, taking a raw image
on every call. Measured on this machine, that costs a 349 ms encoder pass each
time, while a cached embedding makes each extra prompt 12 ms -- a 28x
difference that an iterative point-selection loop pays on every iteration. So
the contract becomes a session that encodes once:

    with harness.open(image_id) as sess:
        mask = sess.predict([(x, y)], region=region)

Two other changes come with it. ``labels`` follows SAM's convention (1
positive, 0 negative) and defaults to all-positive, which gives the caller a
way to *shrink* an over-large mask -- something a positive-only contract cannot
express. And ``region`` carries the ground truth, which is what the document's
"picks the best one by comparing against ground truth" actually needs; the
original signature had no way to pass it.

``FakeSession`` implements the same contract without any model, so point
selection can be written and unit-tested before SAM weights are available.
"""
from __future__ import annotations

import numpy as np

from .metrics import iou as mask_iou
from .regions import Region


def _as_points(points) -> np.ndarray:
    """Normalise a point list to an (N, 2) int array of (x, y) pairs."""
    arr = np.asarray(points, dtype=np.int64).reshape(-1, 2)
    if arr.size == 0:
        raise ValueError("predict() needs at least one point")
    return arr


def _as_labels(labels, n: int) -> np.ndarray:
    if labels is None:
        return np.ones(n, dtype=np.int64)
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if arr.size != n:
        raise ValueError(f"got {n} points but {arr.size} labels")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("labels must be 1 (positive) or 0 (negative)")
    return arr


class SamSession:
    """One image, encoded once, prompted many times.

    Coordinates are pixel ``(x, y)`` = (column, row) with the origin at the top
    left, so a point maps into a mask as ``mask[y, x]``.
    """

    def __init__(self, backend, image: np.ndarray, image_id: str):
        self.backend = backend
        self.image = image
        self.image_id = image_id
        self.num_sam_calls = 0
        self._closed = False
        backend.set_image(image)

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]

    def predict(self, points, labels=None, region: Region | None = None,
                multimask: bool = True) -> np.ndarray:
        """Segment with the given prompts. Returns an (H, W) bool mask.

        With ``region``, the best of SAM's candidate masks is chosen by IoU
        against that region's ground truth. Without it, SAM's own predicted-IoU
        score decides.
        """
        if self._closed:
            raise RuntimeError("session is closed")
        pts = _as_points(points)
        lbl = _as_labels(labels, len(pts))

        h, w = self.shape
        if ((pts[:, 0] < 0) | (pts[:, 0] >= w) | (pts[:, 1] < 0) | (pts[:, 1] >= h)).any():
            raise ValueError(f"point outside image bounds (w={w}, h={h}): {pts.tolist()}")

        masks, scores = self.backend.predict_points(pts, lbl, multimask=multimask)
        self.num_sam_calls += 1

        if len(masks) == 0:
            return np.zeros((h, w), dtype=bool)
        if region is None:
            return masks[int(np.argmax(scores))].astype(bool)
        ious = [mask_iou(region.gt_mask, m) for m in masks]
        return masks[int(np.argmax(ious))].astype(bool)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "SamSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"SamSession({self.image_id}, backend={self.backend.name}, "
                f"num_sam_calls={self.num_sam_calls})")


class FakeSession:
    """Deterministic stand-in for SamSession, no model required.

    Each positive point reveals a disk of the region it lands in, sized so that
    one point recovers roughly half of a compact region and more points recover
    more. Negative points subtract a disk. That is enough for an iterative
    selection loop to make progress, converge, and be unit-tested, while being
    fully reproducible.

    ``leak_px`` optionally dilates the result past the region boundary, so that
    coverage can reach 1.0 while IoU does not -- useful for exercising code
    that has to react to an over-large mask.
    """

    name = "fake"

    def __init__(self, image: np.ndarray, image_id: str, regions: list[Region],
                 radius_frac: float = 0.40, leak_px: int = 0):
        self.image = image
        self.image_id = image_id
        self.regions = regions
        self.radius_frac = radius_frac
        self.leak_px = leak_px
        self.num_sam_calls = 0
        self._closed = False

        h, w = image.shape[:2]
        self._yy, self._xx = np.mgrid[0:h, 0:w]
        # owner[y, x] = 1 + index into self.regions, or 0 for no region
        self._owner = np.zeros((h, w), dtype=np.int32)
        for i, r in enumerate(regions):
            self._owner[r.gt_mask] = i + 1

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]

    def _disk(self, x: int, y: int, radius: float) -> np.ndarray:
        return ((self._xx - x) ** 2 + (self._yy - y) ** 2) <= radius ** 2

    def _radius_for(self, region: Region) -> float:
        return max(3.0, self.radius_frac * float(np.sqrt(region.area_px)))

    def predict(self, points, labels=None, region: Region | None = None,
                multimask: bool = True) -> np.ndarray:
        if self._closed:
            raise RuntimeError("session is closed")
        pts = _as_points(points)
        lbl = _as_labels(labels, len(pts))
        h, w = self.shape
        if ((pts[:, 0] < 0) | (pts[:, 0] >= w) | (pts[:, 1] < 0) | (pts[:, 1] >= h)).any():
            raise ValueError(f"point outside image bounds (w={w}, h={h}): {pts.tolist()}")
        self.num_sam_calls += 1

        mask = np.zeros((h, w), dtype=bool)
        for (x, y), label in zip(pts, lbl):
            if label != 1:
                continue
            owner = region
            if owner is None:
                idx = int(self._owner[y, x])
                if idx == 0:
                    continue
                owner = self.regions[idx - 1]
            elif not owner.gt_mask[y, x]:
                continue
            mask |= owner.gt_mask & self._disk(int(x), int(y), self._radius_for(owner))

        if self.leak_px > 0 and mask.any():
            import cv2
            k = np.ones((2 * self.leak_px + 1,) * 2, np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)

        for (x, y), label in zip(pts, lbl):
            if label == 0:
                mask &= ~self._disk(int(x), int(y), 6.0)
        return mask

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"FakeSession({self.image_id}, "
                f"num_sam_calls={self.num_sam_calls})")
