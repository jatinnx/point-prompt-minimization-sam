"""SAM backends behind one interface.

The plan document specifies SAM 3, but its weights are gated behind manual
approval on Hugging Face and need transformers >= 5.0. Both backends therefore
live behind the same protocol and ``build_backend`` picks whichever is actually
usable, so the harness and everything downstream of it are model-agnostic.

Two SAM 3 details that are easy to get wrong and are baked in below: point
prompts do *not* go through ``Sam3Model`` (that is text/box concept prompting)
-- they need ``Sam3TrackerModel``. And Meta's ``sam3`` repo ships no automatic
mask generator at all; the transformers ``mask-generation`` pipeline is the
equivalent, and it runs a dense point grid against the tracker branch.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .config import Config


class BackendUnavailable(RuntimeError):
    """Raised when a backend's weights or packages are not usable here."""


@runtime_checkable
class SamBackend(Protocol):
    name: str

    def set_image(self, image: np.ndarray) -> None:
        """Encode an image once; later prompts reuse the cached embedding."""

    def predict_points(self, points: np.ndarray, labels: np.ndarray,
                       multimask: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Prompt the cached image. Returns (masks (K,H,W) bool, scores (K,))."""

    def configure_automatic(self, cfg: Config) -> None:
        """Apply cfg's automatic-mode settings to the next automatic pass.

        Called by ``autoseg`` immediately before every pass. Without it a
        backend built from one Config and reused with another would silently
        keep the first Config's grid density and thresholds, while the on-disk
        cache key said otherwise.
        """

    def automatic_masks(self, image: np.ndarray) -> np.ndarray:
        """Segment-everything pass. Returns (N, H, W) bool."""


# Settings that require a new SamAutomaticMaskGenerator.
_AMG_FIELDS = ("auto_points_per_side", "auto_points_per_batch",
               "auto_pred_iou_thresh", "auto_stability_score_thresh",
               "auto_box_nms_thresh")


def _amg_key(cfg: Config) -> tuple:
    return tuple(getattr(cfg, f) for f in _AMG_FIELDS)


class Sam1Backend:
    """SAM 1 ViT-B via segment_anything. Verified working on this machine."""

    name = "sam1_vit_b"

    def __init__(self, cfg: Config):
        from pathlib import Path
        if not Path(cfg.sam1_checkpoint).is_file():
            raise BackendUnavailable(
                f"SAM 1 checkpoint not found: {cfg.sam1_checkpoint}")
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise BackendUnavailable(f"segment_anything not importable: {exc}") from exc

        self.cfg = cfg
        self.model = sam_model_registry[cfg.sam1_model_type](
            checkpoint=cfg.sam1_checkpoint).to(cfg.device)
        self.model.eval()
        self._predictor = SamPredictor(self.model)
        self._amg = None
        self._amg_key: tuple | None = None

    def set_image(self, image: np.ndarray) -> None:
        # set_image runs the ViT encoder (~350 ms) and caches the result;
        # each later predict() is a decoder-only pass (~12 ms).
        self._predictor.set_image(image)

    def predict_points(self, points: np.ndarray, labels: np.ndarray,
                       multimask: bool = True) -> tuple[np.ndarray, np.ndarray]:
        masks, scores, _ = self._predictor.predict(
            point_coords=np.asarray(points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=multimask,
        )
        return masks.astype(bool), np.asarray(scores, dtype=np.float32)

    def configure_automatic(self, cfg: Config) -> None:
        key = _amg_key(cfg)
        if self._amg is not None and key == self._amg_key:
            return
        from segment_anything import SamAutomaticMaskGenerator
        self._amg = SamAutomaticMaskGenerator(
            self.model,
            points_per_side=cfg.auto_points_per_side,
            points_per_batch=cfg.auto_points_per_batch,
            pred_iou_thresh=cfg.auto_pred_iou_thresh,
            stability_score_thresh=cfg.auto_stability_score_thresh,
            box_nms_thresh=cfg.auto_box_nms_thresh,
            output_mode="binary_mask",
        )
        self._amg_key = key

    def automatic_masks(self, image: np.ndarray) -> np.ndarray:
        if self._amg is None:
            self.configure_automatic(self.cfg)
        records = self._amg.generate(image)
        if not records:
            h, w = image.shape[:2]
            return np.zeros((0, h, w), dtype=bool)
        return np.stack([r["segmentation"].astype(bool) for r in records])


class Sam3Backend:
    """SAM 3 via transformers.

    Untested here: ``facebook/sam3`` is gated and this machine has no approved
    token, so construction raises BackendUnavailable rather than half-working.
    Once access is granted this becomes the default with no other code change.
    """

    name = "sam3"

    def __init__(self, cfg: Config):
        try:
            import transformers
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        except ImportError as exc:
            raise BackendUnavailable(
                "SAM 3 needs transformers >= 5.0 with Sam3Tracker* classes "
                f"({exc})") from exc

        self.cfg = cfg
        try:
            self.model = Sam3TrackerModel.from_pretrained(
                cfg.sam3_model_id).to(cfg.device).eval()
            self.processor = Sam3TrackerProcessor.from_pretrained(cfg.sam3_model_id)
        except Exception as exc:                       # gated 401, network, etc.
            raise BackendUnavailable(
                f"could not load {cfg.sam3_model_id} (transformers "
                f"{transformers.__version__}). The repo is gated: request "
                f"access on Hugging Face, then `hf auth login`. Original "
                f"error: {type(exc).__name__}: {exc}") from exc

        self._pipeline = None
        self._pipeline_key: tuple | None = None
        self._auto_kwargs: dict = {}
        self._image: np.ndarray | None = None
        self._embeddings = None

    def set_image(self, image: np.ndarray) -> None:
        import torch
        self._image = image
        self._embeddings = None
        # Cache the vision embedding if this checkpoint exposes it, matching the
        # encode-once/decode-many pattern. If it does not, predict_points still
        # works, just slower -- callers see no difference.
        getter = getattr(self.model, "get_image_embeddings", None)
        if getter is None:
            return
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self._embeddings = getter(inputs["pixel_values"])

    def predict_points(self, points: np.ndarray, labels: np.ndarray,
                       multimask: bool = True) -> tuple[np.ndarray, np.ndarray]:
        import torch
        if self._image is None:
            raise RuntimeError("call set_image() before predict_points()")

        # input_points is 4-D (image, object, point_per_object, xy) and
        # input_labels is 3-D (image, object, label).
        pts = [[[[int(x), int(y)] for x, y in np.asarray(points).reshape(-1, 2)]]]
        lbl = [[[int(v) for v in np.asarray(labels).reshape(-1)]]]

        inputs = self.processor(
            images=self._image, input_points=pts, input_labels=lbl,
            return_tensors="pt").to(self.model.device)
        if self._embeddings is not None:
            inputs["image_embeddings"] = self._embeddings
            inputs.pop("pixel_values", None)

        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=multimask)

        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        masks = np.asarray(masks).reshape(-1, *self._image.shape[:2]).astype(bool)
        scores = np.asarray(outputs.iou_scores.detach().cpu()).reshape(-1)
        return masks, scores.astype(np.float32)[: len(masks)]

    def configure_automatic(self, cfg: Config) -> None:
        # The mask-generation pipeline takes its grid density and thresholds as
        # call arguments rather than constructor arguments, so only a device or
        # model change forces a rebuild.
        key = (cfg.sam3_model_id, cfg.device)
        if key != self._pipeline_key:
            self._pipeline = None
            self._pipeline_key = key
        self._auto_kwargs = {
            "points_per_batch": cfg.auto_points_per_batch,
            "points_per_crop": cfg.auto_points_per_side ** 2,
            "pred_iou_thresh": cfg.auto_pred_iou_thresh,
            "stability_score_thresh": cfg.auto_stability_score_thresh,
        }

    def automatic_masks(self, image: np.ndarray) -> np.ndarray:
        from transformers import pipeline
        if not self._auto_kwargs:
            self.configure_automatic(self.cfg)
        if self._pipeline is None:
            device = 0 if self.cfg.device.startswith("cuda") else -1
            self._pipeline = pipeline(
                "mask-generation", model=self.cfg.sam3_model_id, device=device)
        out = self._pipeline(image, **self._auto_kwargs)
        records = out["masks"] if isinstance(out, dict) else out
        if not len(records):
            h, w = image.shape[:2]
            return np.zeros((0, h, w), dtype=bool)
        return np.stack([np.asarray(m).astype(bool) for m in records])


_ORDER = {"auto": ("sam3", "sam1"), "sam3": ("sam3",), "sam1": ("sam1",)}
_BACKENDS = {"sam3": Sam3Backend, "sam1": Sam1Backend}


def build_backend(cfg: Config, verbose: bool = True) -> SamBackend:
    """Instantiate the first usable backend for ``cfg.backend``."""
    try:
        order = _ORDER[cfg.backend]
    except KeyError:
        raise ValueError(f"cfg.backend must be one of {sorted(_ORDER)} "
                         f"(or 'fake'), got {cfg.backend!r}") from None

    failures = []
    for key in order:
        try:
            backend = _BACKENDS[key](cfg)
        except BackendUnavailable as exc:
            failures.append(f"  {key}: {exc}")
            if verbose:
                print(f"[backend] {key} unavailable, falling back")
            continue
        if verbose:
            print(f"[backend] using {backend.name} on {cfg.device}")
        return backend

    raise BackendUnavailable(
        "no usable SAM backend:\n" + "\n".join(failures))
