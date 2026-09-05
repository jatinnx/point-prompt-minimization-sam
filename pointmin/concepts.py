"""Text ("concept") prompting: what SAM finds when told only the class name.

This is the project's recognition step. SAM 3 gets an image and the 17 DLRSD
class names -- no points, no boxes, no ground truth -- and returns instances per
name. A ground-truth region then counts as recognised only if the masks returned
for *its own* class name cover it.

That is strictly harder than the ``autoseg`` path, and deliberately so. Automatic
mode credits a region whenever any blob happens to overlap it, which measures
whether SAM can outline the thing. Here SAM also has to name it. Both live in the
tree because they are the two sides of ``Config.recognition``: this one is the
hand-off, and the image-only pass is what separates "cannot outline it" from
"can outline it but does not call it that".

Caching mirrors ``autoseg``: the encoder pass over a 1008x1008 upsample is the
expensive part, Step 0 and Step 1 both want the same output, and the key includes
the prompt text and thresholds so Step 6 retuning cannot be served stale masks.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .class_map import CLASS_NAMES
from .config import Config
from .sam_backend import SamBackend

# Config fields that change what text prompting returns.
CACHE_KEY_FIELDS = ("sam3_prompt_template", "sam3_score_threshold",
                    "sam3_mask_threshold")


def class_prompts(cfg: Config) -> list[tuple[int, str]]:
    """``(native DLRSD class id 1..17, phrase)`` for all 17 classes, in order.

    The pairing is explicit rather than positional because everything else keys
    on the native id while ``CLASS_NAMES`` is indexed 0..16; letting those two
    drift by one would silently credit every region to its neighbour class.
    """
    return [(i + 1, cfg.sam3_prompt_template.format(name=name))
            for i, name in enumerate(CLASS_NAMES)]


@dataclass
class ConceptMasks:
    """Every instance SAM returned for an image, tagged with the class asked for.

    ``masks[i]`` came from prompting with the phrase for ``class_ids[i]``. Rows
    from different classes sit in one array so a region's match indices stay
    meaningful across the whole set, the way ``matched_mask_indices`` already
    works for automatic masks.
    """

    masks: np.ndarray                       # (N, H, W) bool
    class_ids: np.ndarray                   # (N,) native class id 1..17
    scores: np.ndarray                      # (N,) float32
    prompts: dict[int, str] = field(default_factory=dict)   # class id -> phrase

    def __len__(self) -> int:
        return len(self.masks)

    @property
    def shape(self) -> tuple[int, int]:
        return self.masks.shape[1:]         # type: ignore[return-value]

    def indices_for_class(self, class_id: int) -> np.ndarray:
        """Positions in ``masks`` that came from ``class_id``'s prompt."""
        return np.flatnonzero(self.class_ids == class_id)

    def for_class(self, class_id: int) -> np.ndarray:
        return self.masks[self.indices_for_class(class_id)]

    def counts(self) -> dict[int, int]:
        """class id -> how many instances that prompt returned."""
        return {int(c): int((self.class_ids == c).sum())
                for c in sorted(set(self.prompts) | set(self.class_ids.tolist()))}

    def empty_prompts(self) -> list[int]:
        """Class ids whose prompt returned nothing at all."""
        return [c for c, n in self.counts().items() if n == 0]


def concept_masks(backend: SamBackend, image: np.ndarray,
                  cfg: Config) -> ConceptMasks:
    """Prompt ``image`` with all 17 class names. No area filter.

    ``autoseg`` drops specks and whole-tile masks because a dense point grid
    produces plenty of both. A named concept does not: if SAM says "this 30 px
    blob is a car", that is an answer about cars and dropping it would hide a
    real recognition. Small regions are already excluded upstream by
    ``min_region_area``.
    """
    if not getattr(backend, "supports_text", False):
        raise TypeError(
            f"model {backend.name!r} cannot be prompted with a class name; "
            f"use recognition='automatic' for the image-only pass instead")

    pairs = class_prompts(cfg)
    masks, prompt_index, scores = backend.concept_masks(
        image, [phrase for _, phrase in pairs])
    ids = np.asarray([pairs[i][0] for i in np.asarray(prompt_index, dtype=int)],
                     dtype=np.int32)
    return ConceptMasks(masks=np.asarray(masks, dtype=bool), class_ids=ids,
                        scores=np.asarray(scores, dtype=np.float32),
                        prompts={cid: phrase for cid, phrase in pairs})


def settings_key(cfg: Config) -> str:
    """Short stable digest of the prompt text and thresholds."""
    blob = ";".join(f"{f}={getattr(cfg, f)!r}" for f in CACHE_KEY_FIELDS)
    blob += ";names=" + ",".join(CLASS_NAMES)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def cache_path(cfg: Config, image_id: str, backend_name: str):
    d = cfg.artifacts / "concept_masks" / f"{backend_name}-{settings_key(cfg)}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{image_id}.npz"


def save_cached(cfg: Config, image_id: str, backend_name: str,
                concept: ConceptMasks) -> None:
    ids = sorted(concept.prompts)
    np.savez_compressed(
        cache_path(cfg, image_id, backend_name),
        masks=concept.masks.astype(bool),
        class_ids=concept.class_ids.astype(np.int32),
        scores=concept.scores.astype(np.float32),
        prompt_class_ids=np.asarray(ids, dtype=np.int32),
        prompt_texts=np.asarray([concept.prompts[c] for c in ids]),
    )


def load_cached(cfg: Config, image_id: str,
                backend_name: str) -> ConceptMasks | None:
    path = cache_path(cfg, image_id, backend_name)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return ConceptMasks(
            masks=data["masks"].astype(bool),
            class_ids=data["class_ids"].astype(np.int32),
            scores=data["scores"].astype(np.float32),
            prompts={int(c): str(t) for c, t in zip(data["prompt_class_ids"],
                                                    data["prompt_texts"])},
        )


def concept_masks_cached(backend: SamBackend, image: np.ndarray, cfg: Config,
                         image_id: str, use_cache: bool = True) -> ConceptMasks:
    """``concept_masks`` with an on-disk cache keyed by image, backend, prompts."""
    if use_cache:
        cached = load_cached(cfg, image_id, backend.name)
        if cached is not None:
            return cached
    concept = concept_masks(backend, image, cfg)
    if use_cache:
        save_cached(cfg, image_id, backend.name, concept)
    return concept
