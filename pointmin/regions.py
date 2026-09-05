"""Ground-truth regions (Section 4, item 3) and the Region record.

A region is one connected component of one class in the ground truth, so two
separate buildings are two regions even though both are class "buildings".

One deviation from the plan document, deliberate: ``region_id`` is unique
within the *image*, not within the class. The document describes it as "which
connected blob of that class" while ``PointSelectionResult`` keys results on
``(image_id, region_id)`` alone -- with a per-class counter those two cannot
both hold, because class 3 region 0 and class 9 region 0 would collide. The
per-class ordinal is still available as ``class_region_index``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

import cv2
import numpy as np

from .config import Config
from .dlrsd import class_name, classes_present

RECOGNIZED = "recognized"
UNRECOGNIZED = "unrecognized"

# Section 5's ``Region``, field for field and in its order. The dataclass carries
# five additive fields after these; ``to_dict(spec_only=True)`` drops them for a
# consumer that wants the document's dict and nothing else.
SPEC_FIELDS = ("image_id", "class_id", "region_id", "gt_mask",
               "matched_sam_mask", "coverage", "iou", "status")


@dataclass(eq=False)
class Region:
    """One connected ground-truth blob, scored against SAM's automatic output.

    Field names follow the plan document's Section 5 ``Region`` exactly;
    ``area_px``, ``bbox``, ``class_name`` and ``class_region_index`` are
    additive.
    """

    image_id: str                       # filename without extension
    class_id: int                       # native DLRSD class id, 1..17
    region_id: int                      # unique within the image, 0, 1, 2, ...
    gt_mask: np.ndarray                 # (H, W) bool, True = in this region
    matched_sam_mask: np.ndarray | None  # (H, W) bool, or None if nothing matched
    coverage: float                     # 0..1, share of gt_mask that SAM found
    iou: float                          # 0..1, IoU of gt_mask and matched mask
    status: str                         # RECOGNIZED | UNRECOGNIZED

    area_px: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)   # x, y, w, h in pixels
    class_name: str = ""
    class_region_index: int = 0         # ordinal among blobs of the same class
    matched_mask_indices: tuple[int, ...] = ()       # which automatic masks matched

    @property
    def key(self) -> str:
        return f"{self.image_id}#{self.region_id}"

    @property
    def shape(self) -> tuple[int, int]:
        return self.gt_mask.shape        # type: ignore[return-value]

    def to_dict(self, include_masks: bool = True, spec_only: bool = False) -> dict:
        """This region as a dict.

        ``spec_only`` returns exactly ``SPEC_FIELDS``, in the document's order,
        with the additive fields dropped.
        """
        d = asdict(self)
        if spec_only:
            d = {name: d[name] for name in SPEC_FIELDS}
        if not include_masks:
            d.pop("gt_mask")
            d.pop("matched_sam_mask")
        return d

    def summary(self) -> dict:
        """JSON-serialisable record without the mask arrays."""
        return {
            "image_id": self.image_id,
            "region_id": self.region_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "class_region_index": self.class_region_index,
            "area_px": self.area_px,
            "bbox": list(self.bbox),
            "coverage": round(float(self.coverage), 4),
            "iou": round(float(self.iou), 4),
            "status": self.status,
            "matched_mask_indices": list(self.matched_mask_indices),
        }

    def __repr__(self) -> str:
        return (f"Region({self.key} class={self.class_id}:{self.class_name} "
                f"area={self.area_px} cov={self.coverage:.3f} "
                f"iou={self.iou:.3f} {self.status})")


def extract_regions(
    labels: np.ndarray,
    image_id: str,
    min_area: int = 24,
    connectivity: int = 8,
) -> list[Region]:
    """Split a ground-truth label map into per-class connected components.

    Regions smaller than ``min_area`` pixels are dropped. Ordering is
    deterministic: classes ascending, then components in cv2 label order.
    Scores start unset; ``harness``/``metrics`` fill them in once SAM has run.
    """
    regions: list[Region] = []
    next_id = 0
    for class_id in classes_present(labels):
        binary = (labels == class_id).astype(np.uint8)
        count, comp, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=connectivity)
        class_index = 0
        for label in range(1, count):          # 0 is background of this binary map
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            regions.append(Region(
                image_id=image_id,
                class_id=class_id,
                region_id=next_id,
                gt_mask=(comp == label),
                matched_sam_mask=None,
                coverage=0.0,
                iou=0.0,
                status=UNRECOGNIZED,
                area_px=area,
                bbox=(int(stats[label, cv2.CC_STAT_LEFT]),
                      int(stats[label, cv2.CC_STAT_TOP]),
                      int(stats[label, cv2.CC_STAT_WIDTH]),
                      int(stats[label, cv2.CC_STAT_HEIGHT])),
                class_name=class_name(class_id),
                class_region_index=class_index,
            ))
            next_id += 1
            class_index += 1
    return regions


def region_area_stats(regions: Iterable[Region]) -> dict:
    """Area distribution, for choosing ``min_region_area`` from data."""
    areas = np.asarray([r.area_px for r in regions], dtype=np.int64)
    if areas.size == 0:
        return {"count": 0}
    return {
        "count": int(areas.size),
        "min": int(areas.min()),
        "p25": int(np.percentile(areas, 25)),
        "median": int(np.median(areas)),
        "p75": int(np.percentile(areas, 75)),
        "max": int(areas.max()),
        "mean": float(areas.mean()),
    }


def interior_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erode a region so sampled points avoid its boundary.

    Falls back to the original mask when erosion would empty it, which happens
    for thin structures such as roads and docks.
    """
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
    return eroded.astype(bool) if eroded.any() else mask.astype(bool)
