"""The plan document's literal ``prompt_sam3`` signature.

Section 5 specifies ``prompt_sam3(image, points) -> mask``. The harness offers a
session instead, because a session encodes the image once (measured on this
machine: 349 ms) and then answers each prompt in 12 ms, a 28x difference that an
iterative loop pays on *every* iteration.

This module exists so that code already written against the document's signature
runs without edits. It is a wrapper, not a second implementation: every call
opens a one-shot session, so every call re-encodes. Use it to get moving, then
switch the loop to ``harness.open(image_id)``.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .regions import Region
from .sam_backend import SamBackend, build_backend
from .session import SamSession

_BACKEND: SamBackend | None = None


def _shared_backend(cfg: Config | None = None, verbose: bool = False) -> SamBackend:
    """One process-wide backend, so the checkpoint loads once rather than per call."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = build_backend(cfg or Config(), verbose=verbose)
    return _BACKEND


def prompt_sam3(image: np.ndarray, points: list[tuple[int, int]],
                region: Region | None = None,
                backend: SamBackend | None = None) -> np.ndarray:
    """Segment ``image`` from ``points``; returns an (H, W) bool mask.

    image:  (H, W, 3) uint8 RGB array -- the raw image, not a path
    points: list of (x, y) pixel coordinates, all treated as positive
    region: optional; when given, the best of SAM's candidate masks is chosen by
            IoU against ``region.gt_mask`` -- the "already picks the best one by
            comparing against ground truth" clause of the document. Without it
            SAM's own predicted-IoU score decides, which is the best available
            answer when there is no ground truth to compare against.

    Re-encodes the image on every call. ``harness.open(image_id)`` does not.
    """
    with SamSession(backend or _shared_backend(), image, "<ad-hoc>") as sess:
        return sess.predict(points, region=region)
