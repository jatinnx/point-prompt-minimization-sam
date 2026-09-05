"""SAM 3, behind one interface.

The model the plan specifies, and the only one here. Everything above this file
talks to the ``SamBackend`` protocol rather than to transformers, so the harness,
the caches and the scripts never name a class from ``transformers``.

Two SAM 3 details that are easy to get wrong and are baked in below: point
prompts do *not* go through ``Sam3Model`` (that is text/box concept prompting)
-- they need ``Sam3TrackerModel``. And Meta's ``sam3`` repo ships no automatic
mask generator at all; the transformers ``mask-generation`` pipeline is the
equivalent, and it runs a dense point grid against the tracker branch.

``supports_text`` is what ``Config.recognition="auto"`` consults. SAM 3 sets it
True -- it exists because the protocol is also what the stub backends in
``tests/`` implement, and a stub that cannot be handed a phrase must be refused
rather than silently downgraded to the class-agnostic pass: the two modes'
numbers are not comparable.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .config import (SAM3_HUB_ID, SAM3_LOCAL_DIR,
                    SAM3_NATIVE_CHECKPOINT_CANDIDATES,
                    SAM3_NATIVE_REPO_CANDIDATES, Config)


class BackendUnavailable(RuntimeError):
    """Raised when a backend's weights or packages are not usable here."""


@runtime_checkable
class SamBackend(Protocol):
    name: str

    # True only if the backend can be prompted with a class name. Consulted by
    # Harness to resolve ``Config.recognition="auto"``.
    supports_text: bool

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

    def concept_masks(self, image: np.ndarray,
                      prompts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Text-prompt one image with each phrase in ``prompts``.

        Returns ``(masks (N,H,W) bool, prompt_index (N,) int32,
        scores (N,) float32)`` -- one row per returned instance, with
        ``prompt_index`` saying which phrase produced it. Only implemented where
        ``supports_text`` is True.
        """


def _to_numpy(x) -> np.ndarray:
    """Torch tensor or array-like -> numpy, without importing torch here."""
    detach = getattr(x, "detach", None)
    if detach is not None:
        x = detach().cpu()
    return np.asarray(x)


class Sam3Backend:
    """SAM 3 via transformers -- both of its prompt paths.

    SAM 3 is two models behind one checkpoint id, and they are not
    interchangeable:

    * ``Sam3Model``         text ("concept") prompting. Image + a phrase, back
                            come every instance of that phrase. This is the path
                            the project is built on: SAM gets the image and the
                            17 DLRSD class names, nothing else.
    * ``Sam3TrackerModel``  geometric prompting -- points, boxes, coarse masks.
                            Text never reaches it. This is what point selection
                            needs from Step 2 on.

    Each branch is its own multi-GB load, so they load on first use rather than
    in ``__init__``: recognition touches only the concept branch, point selection
    only the tracker, and on a shared GPU loading both when one is wanted is the
    difference between fitting and an OOM.

    ``__init__`` therefore proves availability by loading the *config*, which is
    cheap and, on the hub path, passes through the same gate as the weights -- so
    the failures that actually happen (transformers too old, no local copy and no
    access to the gated repo) surface immediately instead of after Step 1 has
    already extracted regions for ten images.
    """

    name = "sam3"
    supports_text = True

    def __init__(self, cfg: Config):
        try:
            import transformers
            from transformers import (Sam3Config, Sam3Model,  # noqa: F401
                                      Sam3Processor, Sam3TrackerModel,
                                      Sam3TrackerProcessor)
        except ImportError as exc:
            raise BackendUnavailable(
                "SAM 3 needs transformers >= 5.0 with the Sam3* classes "
                f"({exc})") from exc

        try:
            Sam3Config.from_pretrained(cfg.sam3_model_id)
        except Exception as exc:              # missing dir, gated 401, network
            raise BackendUnavailable(
                f"could not load SAM 3 from {cfg.sam3_model_id!r} (transformers "
                f"{transformers.__version__}). Expected the in-project copy at "
                f"{SAM3_LOCAL_DIR}; see README 'Where the weights live'. Set "
                f"POINTMIN_SAM3_PATH to point elsewhere, or fall back to the hub "
                f"id {SAM3_HUB_ID!r}, which is gated: request access on Hugging "
                f"Face, then `hf auth login`. Original error: "
                f"{type(exc).__name__}: {exc}") from exc

        self.cfg = cfg
        self._tracker = None            # geometric prompts, lazy
        self._tracker_proc = None
        self._concept = None            # text prompts, lazy
        self._concept_proc = None
        self._pipeline = None
        self._pipeline_key: tuple | None = None
        self._auto_kwargs: dict = {}
        self._image: np.ndarray | None = None
        self._embeddings = None

    # ---- lazy branch loading ---------------------------------------------
    def _load_tracker(self):
        if self._tracker is None:
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
            self._tracker = Sam3TrackerModel.from_pretrained(
                self.cfg.sam3_model_id).to(self.cfg.device).eval()
            self._tracker_proc = Sam3TrackerProcessor.from_pretrained(
                self.cfg.sam3_model_id)
        return self._tracker, self._tracker_proc

    def _load_concept(self):
        if self._concept is None:
            from transformers import Sam3Model, Sam3Processor
            self._concept = Sam3Model.from_pretrained(
                self.cfg.sam3_model_id).to(self.cfg.device).eval()
            self._concept_proc = Sam3Processor.from_pretrained(
                self.cfg.sam3_model_id)
        return self._concept, self._concept_proc

    # ---- geometric prompting (points) -------------------------------------
    def set_image(self, image: np.ndarray) -> None:
        import torch
        model, processor = self._load_tracker()
        self._image = image
        self._embeddings = None
        # Cache the vision embedding if this checkpoint exposes it, matching the
        # encode-once/decode-many pattern. If it does not, predict_points still
        # works, just slower -- callers see no difference.
        getter = getattr(model, "get_image_embeddings", None)
        if getter is None:
            return
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            self._embeddings = getter(inputs["pixel_values"])

    def predict_points(self, points: np.ndarray, labels: np.ndarray,
                       multimask: bool = True) -> tuple[np.ndarray, np.ndarray]:
        import torch
        if self._image is None:
            raise RuntimeError("call set_image() before predict_points()")
        model, processor = self._load_tracker()

        # input_points is 4-D (image, object, point_per_object, xy) and
        # input_labels is 3-D (image, object, label).
        pts = [[[[int(x), int(y)] for x, y in np.asarray(points).reshape(-1, 2)]]]
        lbl = [[[int(v) for v in np.asarray(labels).reshape(-1)]]]

        inputs = processor(
            images=self._image, input_points=pts, input_labels=lbl,
            return_tensors="pt").to(model.device)
        if self._embeddings is not None:
            inputs["image_embeddings"] = self._embeddings
            inputs.pop("pixel_values", None)

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=multimask)

        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        masks = np.asarray(masks).reshape(-1, *self._image.shape[:2]).astype(bool)
        scores = np.asarray(outputs.iou_scores.detach().cpu()).reshape(-1)
        return masks, scores.astype(np.float32)[: len(masks)]

    # ---- automatic ("segment everything") --------------------------------
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
        from PIL import Image
        pil_image = Image.fromarray(image)      # pipeline needs PIL, not ndarray
        out = self._pipeline(pil_image, **self._auto_kwargs)
        records = out["masks"] if isinstance(out, dict) else out
        if not len(records):
            h, w = image.shape[:2]
            return np.zeros((0, h, w), dtype=bool)
        return np.stack([np.asarray(m).astype(bool) for m in records])

    # ---- text ("concept") prompting ---------------------------------------
    def concept_masks(self, image: np.ndarray,
                      prompts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prompt this image with each phrase in ``prompts``.

        One image encode serves every phrase. The processor upsamples a 256x256
        DLRSD tile to 1008x1008 before the ViT, so the encoder dominates the
        cost and paying for it once instead of 17 times is most of the runtime.

        Phrases that return nothing simply contribute no rows -- an empty
        result for "airplane" over farmland is the correct answer, not an error.
        """
        import torch
        model, processor = self._load_concept()
        prompts = list(prompts)
        h, w = image.shape[:2]

        img_inputs = processor(images=image, return_tensors="pt").to(model.device)
        target_sizes = img_inputs["original_sizes"].tolist()
        with torch.no_grad():
            vision = model.get_vision_features(img_inputs["pixel_values"])

        masks, index, scores = [], [], []
        for i, phrase in enumerate(prompts):
            text_inputs = processor(text=phrase, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(vision_embeds=vision,
                                input_ids=text_inputs["input_ids"],
                                attention_mask=text_inputs.get("attention_mask"))
            found = processor.post_process_instance_segmentation(
                outputs,
                threshold=self.cfg.sam3_score_threshold,
                mask_threshold=self.cfg.sam3_mask_threshold,
                target_sizes=target_sizes)[0]
            m = _to_numpy(found["masks"]).astype(bool).reshape(-1, h, w)
            if not len(m):
                continue
            masks.append(m)
            index.extend([i] * len(m))
            scores.extend(_to_numpy(found["scores"]).reshape(-1).tolist())

        if not masks:
            return (np.zeros((0, h, w), dtype=bool),
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0,), dtype=np.float32))
        return (np.concatenate(masks),
                np.asarray(index, dtype=np.int32),
                np.asarray(scores, dtype=np.float32))


class Sam3NativeBackend:
    """Native SAM 3 via Meta's repository and sam3.pt checkpoint.

    This runs 100% offline without Hugging Face transformers or gated hub
    authentication. It implements the SamBackend protocol:
    - Text ("concept") prompting with the 17 DLRSD class names via Sam3Processor
    - Point prompts via SAM3InteractiveImagePredictor
    """

    name = "sam3_native"
    supports_text = True

    def __init__(self, cfg: Config):
        from pathlib import Path
        self.cfg = cfg
        ckpt = cfg.sam3_native_checkpoint
        if not ckpt or not Path(ckpt).is_file():
            raise BackendUnavailable(
                f"Native SAM 3 checkpoint not found. Looked in candidates: "
                f"{SAM3_NATIVE_CHECKPOINT_CANDIDATES}. Set "
                f"POINTMIN_SAM3_NATIVE_CHECKPOINT=/path/to/sam3.pt to specify it."
            )

        repo = cfg.sam3_native_repo
        if repo and Path(repo).is_dir():
            import sys
            repo_str = str(Path(repo).resolve())
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)
            inner = Path(repo) / "sam3"
            if inner.is_dir() and str(inner.resolve()) not in sys.path:
                sys.path.insert(0, str(inner.resolve()))

        try:
            from sam3.model_builder import (build_sam3_image_model,  # noqa: F401
                                            build_tracker)
            from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
            from sam3.model.sam1_task_predictor import (  # noqa: F401
                SAM3InteractiveImagePredictor)
        except ImportError as exc:
            raise BackendUnavailable(
                f"Native SAM 3 packages could not be imported ({exc}). Ensure "
                f"iopath, einops, timm are installed and POINTMIN_SAM3_NATIVE_REPO "
                f"points to the sam3 source tree."
            ) from exc

        self._checkpoint_path = str(Path(ckpt).resolve())
        self._concept = None
        self._concept_proc = None
        self._tracker = None
        self._tracker_pred = None
        self._image: np.ndarray | None = None
        self._auto_kwargs: dict = {}

    def _load_concept(self):
        if self._concept is None:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            self._concept = build_sam3_image_model(
                device=self.cfg.device,
                eval_mode=True,
                img_size=1008,
                load_from_HF=False,
                checkpoint_path=self._checkpoint_path,
            )
            self._concept.eval()
            self._concept_proc = Sam3Processor(
                model=self._concept, device=self.cfg.device, resolution=1008
            )
        return self._concept, self._concept_proc

    def _load_tracker(self):
        if self._tracker_pred is None:
            import torch
            from sam3.model_builder import build_tracker
            from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
            tracker = build_tracker(
                apply_temporal_disambiguation=False, with_backbone=True
            )
            checkpoint = torch.load(
                self._checkpoint_path, map_location="cpu", weights_only=False
            )
            state = checkpoint.get("model", checkpoint)
            tracker_state = {}
            for k, v in state.items():
                if "tracker" in k:
                    tracker_state[k.replace("tracker.", "")] = v
                elif "detector" in k and "freqs_cis" not in k:
                    tracker_state[k.replace("detector.", "")] = v
            tracker.load_state_dict(tracker_state, strict=False)
            tracker.eval()
            pred = SAM3InteractiveImagePredictor(tracker)
            pred.to(self.cfg.device)
            self._tracker = tracker
            self._tracker_pred = pred
        return self._tracker_pred

    def set_image(self, image: np.ndarray) -> None:
        self._image = image
        pred = self._load_tracker()
        pred.set_image(image)

    def predict_points(self, points: np.ndarray, labels: np.ndarray,
                       multimask: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if self._image is None:
            raise RuntimeError("call set_image() before predict_points()")
        pred = self._load_tracker()
        pts = np.asarray(points).reshape(-1, 2)
        lbl = np.asarray(labels).reshape(-1)
        masks, scores, _ = pred.predict(
            point_coords=pts,
            point_labels=lbl,
            multimask_output=multimask,
            normalize_coords=True,
        )
        return masks.astype(bool), scores.astype(np.float32)[: len(masks)]

    def configure_automatic(self, cfg: Config) -> None:
        self._auto_kwargs = {
            "points_per_side": cfg.auto_points_per_side,
            "pred_iou_thresh": cfg.auto_pred_iou_thresh,
            "stability_score_thresh": cfg.auto_stability_score_thresh,
        }

    def automatic_masks(self, image: np.ndarray) -> np.ndarray:
        pred = self._load_tracker()
        pred.set_image(image)
        h, w = image.shape[:2]
        pts_per_side = self.cfg.auto_points_per_side
        xs = np.linspace(0, w - 1, pts_per_side, dtype=np.float32)
        ys = np.linspace(0, h - 1, pts_per_side, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        masks = []
        for pt in points:
            m, sc, _ = pred.predict(
                point_coords=pt[None, :],
                point_labels=np.array([1]),
                multimask_output=True,
                normalize_coords=True,
            )
            best_idx = int(np.argmax(sc))
            if sc[best_idx] >= self.cfg.auto_pred_iou_thresh:
                masks.append(m[best_idx].astype(bool))
        if not masks:
            return np.zeros((0, h, w), dtype=bool)
        return np.stack(masks)

    def concept_masks(self, image: np.ndarray,
                      prompts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import cv2
        import torch
        from PIL import Image

        _model, processor = self._load_concept()
        prompts = list(prompts)
        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)

        with torch.no_grad():
            with torch.autocast(device_type=self.cfg.device, dtype=torch.bfloat16):
                base_state = processor.set_image(pil_image)

        masks, index, scores = [], [], []
        for i, phrase in enumerate(prompts):
            processor.reset_all_prompts(base_state)
            with torch.no_grad():
                with torch.autocast(device_type=self.cfg.device, dtype=torch.bfloat16):
                    state = processor.set_text_prompt(prompt=phrase, state=base_state)
            found_masks = state.get("masks", None)
            found_scores = state.get("scores", None)
            if found_masks is None or len(found_masks) == 0:
                continue

            for j, m_tensor in enumerate(found_masks):
                m = m_tensor[0].cpu().numpy()
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    m = m.astype(bool)
                if not np.any(m):
                    continue
                score = (float(found_scores[j].item())
                         if found_scores is not None and len(found_scores) > j
                         else 1.0)
                if score < self.cfg.sam3_score_threshold:
                    continue
                masks.append(m)
                index.append(i)
                scores.append(score)

        if not masks:
            return (np.zeros((0, h, w), dtype=bool),
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0,), dtype=np.float32))
        return (np.stack(masks),
                np.asarray(index, dtype=np.int32),
                np.asarray(scores, dtype=np.float32))


_BACKENDS = {
    "sam3_native": Sam3NativeBackend,
    "sam3": Sam3Backend,
}
_ORDER = {
    "auto": ("sam3_native", "sam3"),
    "sam3_native": ("sam3_native",),
    "sam3": ("sam3",),
}


def build_backend(cfg: Config, verbose: bool = True) -> SamBackend:
    """Build the selected or first usable SAM backend."""
    target = getattr(cfg, "backend", "auto")
    try:
        order = _ORDER[target]
    except KeyError:
        raise ValueError(
            f"cfg.backend must be one of {sorted(_ORDER)}, got {target!r}"
        ) from None

    failures = []
    for key in order:
        try:
            backend = _BACKENDS[key](cfg)
        except BackendUnavailable as exc:
            failures.append(f"  {key}: {exc}")
            if verbose and target == "auto":
                print(f"[backend] {key} unavailable, trying fallback: {exc}")
            continue
        if verbose:
            source = (
                cfg.sam3_native_checkpoint
                if key == "sam3_native"
                else cfg.sam3_model_id
            )
            print(f"[backend] using {backend.name} on {cfg.device} from {source}")
        return backend

    raise BackendUnavailable(
        "no usable SAM backend:\n" + "\n".join(failures)
    )
