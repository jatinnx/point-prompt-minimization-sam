"""Persisting scored regions, so point selection can be developed without a GPU.

Section 4 says Jatin "hands off to Raven a list of Region objects". Handing over
live Python objects means Raven needs the SAM checkpoint, a GPU and a few
seconds of automatic-mode inference per image. Writing them to disk instead
means he needs neither, and we both look at byte-identical regions.

Masks are bit-packed, so a full image of regions is a few kB rather than a few
hundred. Everything else travels as JSON in the same file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .regions import Region

FORMAT_VERSION = 1


def _pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(np.ascontiguousarray(mask, dtype=bool).ravel())


def _unpack(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.unpackbits(packed, count=h * w).reshape(h, w).astype(bool)


def _meta(region: Region) -> dict:
    """Full-precision record; ``Region.summary()`` rounds for human reading."""
    return {
        "image_id": region.image_id,
        "class_id": int(region.class_id),
        "region_id": int(region.region_id),
        "coverage": float(region.coverage),
        "iou": float(region.iou),
        "status": region.status,
        "area_px": int(region.area_px),
        "bbox": [int(v) for v in region.bbox],
        "class_name": region.class_name,
        "class_region_index": int(region.class_region_index),
        "matched_mask_indices": [int(v) for v in region.matched_mask_indices],
        "shape": [int(v) for v in region.gt_mask.shape],
        "has_matched": region.matched_sam_mask is not None,
    }


def save_regions(regions: list[Region], path: str | Path) -> Path:
    """Write one image's scored regions to a single compressed .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    meta = []
    for i, region in enumerate(regions):
        arrays[f"gt_{i}"] = _pack(region.gt_mask)
        if region.matched_sam_mask is not None:
            arrays[f"matched_{i}"] = _pack(region.matched_sam_mask)
        meta.append(_meta(region))
    np.savez_compressed(
        path,
        meta=np.array(json.dumps({"version": FORMAT_VERSION, "regions": meta})),
        **arrays,
    )
    return path


def load_regions(path: str | Path) -> list[Region]:
    """Rebuild the exact Region list that ``save_regions`` was given."""
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(data["meta"].item())
        if header.get("version") != FORMAT_VERSION:
            raise ValueError(f"{path}: format version {header.get('version')}, "
                             f"expected {FORMAT_VERSION}")
        out = []
        for i, m in enumerate(header["regions"]):
            shape = tuple(m["shape"])
            matched = _unpack(data[f"matched_{i}"], shape) if m["has_matched"] else None
            out.append(Region(
                image_id=m["image_id"],
                class_id=m["class_id"],
                region_id=m["region_id"],
                gt_mask=_unpack(data[f"gt_{i}"], shape),
                matched_sam_mask=matched,
                coverage=m["coverage"],
                iou=m["iou"],
                status=m["status"],
                area_px=m["area_px"],
                bbox=tuple(m["bbox"]),
                class_name=m["class_name"],
                class_region_index=m["class_region_index"],
                matched_mask_indices=tuple(m["matched_mask_indices"]),
            ))
    return out


def dump_dataset(regions_by_image: dict[str, list[Region]],
                 out_dir: str | Path, extra: dict | None = None) -> Path:
    """Write one .npz per image plus an index.json describing the whole dump."""
    out_dir = Path(out_dir)
    (out_dir / "regions").mkdir(parents=True, exist_ok=True)
    index = {
        "version": FORMAT_VERSION,
        "images": sorted(regions_by_image),
        "n_regions": {k: len(v) for k, v in regions_by_image.items()},
        **(extra or {}),
    }
    for image_id, regions in regions_by_image.items():
        save_regions(regions, out_dir / "regions" / f"{image_id}.npz")
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    return out_dir


def load_dataset(out_dir: str | Path) -> dict[str, list[Region]]:
    """Counterpart of ``dump_dataset``; no SAM checkpoint or GPU needed."""
    out_dir = Path(out_dir)
    index = json.loads((out_dir / "index.json").read_text())
    return {image_id: load_regions(out_dir / "regions" / f"{image_id}.npz")
            for image_id in index["images"]}
