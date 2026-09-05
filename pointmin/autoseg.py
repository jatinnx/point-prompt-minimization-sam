"""Automatic ("segment everything") mode (Section 4, item 2).

The area-fraction filter runs here rather than inside a backend so that both
backends are filtered identically and their numbers stay comparable.

Automatic masks are cached to ``artifacts/auto_masks/`` because the pass is the
expensive part of the harness and Step 0 and Step 1 both need the same output.
The cache key includes the settings that change the output, not just the image
and backend -- Step 6 tunes those settings, and a key that ignored them would
quietly serve stale masks and make the tuning look like it did nothing.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .config import Config
from .sam_backend import SamBackend

# Config fields that change what automatic mode returns.
CACHE_KEY_FIELDS = (
    "auto_points_per_side",
    "auto_pred_iou_thresh",
    "auto_stability_score_thresh",
    "auto_box_nms_thresh",
    "auto_min_area_frac",
    "auto_max_area_frac",
)


def filter_by_area(masks: np.ndarray, min_area_frac: float,
                   max_area_frac: float) -> np.ndarray:
    """Drop masks that are a speck or that cover essentially the whole tile."""
    if len(masks) == 0:
        return masks
    h, w = masks.shape[1:]
    total = h * w
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    keep = (areas >= min_area_frac * total) & (areas <= max_area_frac * total)
    return masks[keep]


def automatic_masks(backend: SamBackend, image: np.ndarray,
                    cfg: Config) -> np.ndarray:
    """Run the backend's segment-everything pass and area-filter the result.

    ``configure_automatic`` runs first so that cfg, and therefore the cache key
    derived from cfg, always describes the settings that actually ran.
    """
    backend.configure_automatic(cfg)
    masks = backend.automatic_masks(image)
    return filter_by_area(masks, cfg.auto_min_area_frac, cfg.auto_max_area_frac)


def settings_key(cfg: Config) -> str:
    """Short stable digest of the automatic-mode settings."""
    blob = ";".join(f"{f}={getattr(cfg, f)!r}" for f in CACHE_KEY_FIELDS)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def cache_path(cfg: Config, image_id: str, backend_name: str):
    d = cfg.artifacts / "auto_masks" / f"{backend_name}-{settings_key(cfg)}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{image_id}.npz"


def load_cached(cfg: Config, image_id: str, backend_name: str) -> np.ndarray | None:
    path = cache_path(cfg, image_id, backend_name)
    if not path.is_file():
        return None
    with np.load(path) as data:
        return data["masks"].astype(bool)


def save_cached(cfg: Config, image_id: str, backend_name: str,
                masks: np.ndarray) -> None:
    np.savez_compressed(cache_path(cfg, image_id, backend_name),
                        masks=masks.astype(bool))


def automatic_masks_cached(backend: SamBackend, image: np.ndarray, cfg: Config,
                           image_id: str, use_cache: bool = True) -> np.ndarray:
    """``automatic_masks`` with an on-disk cache keyed by image and backend."""
    if use_cache:
        cached = load_cached(cfg, image_id, backend.name)
        if cached is not None:
            return cached
    masks = automatic_masks(backend, image, cfg)
    if use_cache:
        save_cached(cfg, image_id, backend.name, masks)
    return masks
