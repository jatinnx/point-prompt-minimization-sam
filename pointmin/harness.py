"""The harness -- the single object the point-selection code talks to.

    from pointmin import Harness

    h = Harness()
    for region in h.regions("harbo_451"):
        if region.status == "unrecognized":
            with h.open(region.image_id) as sess:
                mask = sess.predict([(x, y)], region=region)
                cov, iou = h.score(region.gt_mask, mask)
                if h.is_recognized(cov, iou):
                    ...

Nothing above depends on which SAM backend is active.
"""
from __future__ import annotations

import numpy as np

from . import autoseg, dlrsd, metrics
from .config import Config
from .regions import Region, extract_regions
from .sam_backend import SamBackend, build_backend
from .session import FakeSession, SamSession


class Harness:
    def __init__(self, cfg: Config | None = None,
                 backend: SamBackend | None = None,
                 use_cache: bool = True, verbose: bool = True):
        self.cfg = cfg or Config()
        self.use_cache = use_cache
        self.verbose = verbose
        self._backend = backend
        self._regions: dict[str, list[Region]] = {}
        self._images: dict[str, np.ndarray] = {}

    # ---- backend ----------------------------------------------------------
    @property
    def backend(self) -> SamBackend:
        """Built lazily so region extraction and metrics work with no model."""
        if self._backend is None:
            self._backend = build_backend(self.cfg, verbose=self.verbose)
        return self._backend

    @property
    def backend_name(self) -> str:
        return self.backend.name

    # ---- data -------------------------------------------------------------
    def image_ids(self) -> list[str]:
        return dlrsd.image_ids(self.cfg)

    def categories(self) -> dict[str, list[str]]:
        return dlrsd.categories(self.cfg)

    def load_image(self, image_id: str) -> np.ndarray:
        if image_id not in self._images:
            self._images[image_id] = dlrsd.load_image(self.cfg, image_id)
        return self._images[image_id]

    def load_labels(self, image_id: str) -> np.ndarray:
        return dlrsd.load_labels(self.cfg, image_id)

    def load_colour(self, image_id: str) -> np.ndarray:
        return dlrsd.load_colour(self.cfg, image_id)

    # ---- regions ----------------------------------------------------------
    def gt_regions(self, image_id: str, min_area: int | None = None) -> list[Region]:
        """Regions straight from the ground truth, unscored. No model needed."""
        return extract_regions(
            self.load_labels(image_id), image_id,
            min_area=self.cfg.min_region_area if min_area is None else min_area,
            connectivity=self.cfg.connectivity,
        )

    def automatic_masks(self, image_id: str) -> np.ndarray:
        """(N, H, W) bool -- SAM's segment-everything output for this image."""
        return autoseg.automatic_masks_cached(
            self.backend, self.load_image(image_id), self.cfg, image_id,
            use_cache=self.use_cache)

    def regions(self, image_id: str) -> list[Region]:
        """Ground-truth regions scored against SAM's automatic output.

        This is what the plan document's Section 4 hands over: coverage, IoU and
        recognised/unrecognised filled in. Cached per image.
        """
        if image_id not in self._regions:
            regions = self.gt_regions(image_id)
            masks = self.automatic_masks(image_id)
            self._regions[image_id] = metrics.score_regions(
                regions, masks,
                coverage_threshold=self.cfg.coverage_threshold,
                iou_threshold=self.cfg.iou_threshold,
                match_mode=self.cfg.match_mode,
            )
        return self._regions[image_id]

    def unrecognized(self, image_id: str) -> list[Region]:
        return [r for r in self.regions(image_id) if r.status == "unrecognized"]

    # ---- prompting --------------------------------------------------------
    def open(self, image_id: str) -> SamSession:
        """Open a prompting session; encodes the image once."""
        return SamSession(self.backend, self.load_image(image_id), image_id)

    def open_fake(self, image_id: str, **kwargs) -> FakeSession:
        """Model-free session for developing point selection without weights."""
        return FakeSession(self.load_image(image_id), image_id,
                           self.gt_regions(image_id), **kwargs)

    # ---- scoring ----------------------------------------------------------
    def score(self, gt_mask: np.ndarray,
              pred_mask: np.ndarray | None) -> tuple[float, float]:
        return metrics.score(gt_mask, pred_mask)

    def is_recognized(self, coverage: float, iou: float) -> bool:
        return metrics.is_recognized(coverage, iou,
                                     self.cfg.coverage_threshold,
                                     self.cfg.iou_threshold)

    def __repr__(self) -> str:
        state = self._backend.name if self._backend is not None else "not built"
        return f"Harness(backend={state}, images={len(self.image_ids())})"
