"""DLRSD image and ground-truth loading (Section 4, item 1).

Label convention, verified exhaustively over all 2100 files in the archive:

    labels/  pixel values are exactly 1..17, PIL mode L
             0, 18 and 255 never occur, so there is no background class and no
             ignore index. Class ids stay in this native 1..17 space throughout
             the project; ``CLASS_NAMES[class_id - 1]`` gives the name.

    colour/  the same indices as mode P (paletted) PNGs. cv2.imread with
             IMREAD_GRAYSCALE on a paletted PNG silently returns palette
             luminance instead of class ids, so colour masks go through PIL.

Both directories use identical basenames, which makes them easy to swap by
mistake -- ``load_labels`` cross-checks the cv2 and PIL reads so that failure
is loud rather than silent.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .class_map import CLASS_NAMES, NUM_CLASSES
from .config import Config

CLASS_ID_MIN = 1
CLASS_ID_MAX = NUM_CLASSES   # 17


def class_name(class_id: int) -> str:
    """Name for a native DLRSD class id (1..17)."""
    if not CLASS_ID_MIN <= class_id <= CLASS_ID_MAX:
        raise ValueError(f"class_id {class_id} outside DLRSD range "
                         f"{CLASS_ID_MIN}..{CLASS_ID_MAX}")
    return CLASS_NAMES[class_id - 1]


def image_ids(cfg: Config) -> list[str]:
    """Sorted image ids (filename without extension), e.g. "harbo_451"."""
    d = cfg.images
    if not d.is_dir():
        raise FileNotFoundError(
            f"{d} not found -- run scripts/00_extract_data.py first")
    return sorted(p.stem for p in d.glob("*.png"))


def load_image(cfg: Config, image_id: str) -> np.ndarray:
    """RGB uint8 array, shape (H, W, 3)."""
    path = cfg.images / f"{image_id}.png"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_labels(cfg: Config, image_id: str, verify: bool = True) -> np.ndarray:
    """Ground-truth label map, uint8 (H, W), values in 1..17."""
    path = cfg.labels / f"{image_id}.png"
    labels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if labels is None:
        raise FileNotFoundError(f"could not read labels {path}")

    if verify:
        # A paletted PNG here would decode to luminance under cv2 but to class
        # indices under PIL. Disagreement means the wrong directory is wired up.
        pil = np.array(Image.open(path))
        if pil.shape != labels.shape or not np.array_equal(pil, labels):
            raise ValueError(
                f"{path}: cv2 and PIL reads disagree, so this is probably a "
                f"paletted (mode P) mask. cv2 uniques={np.unique(labels)[:12]}, "
                f"PIL uniques={np.unique(pil)[:12]}. Point labels_dir at the "
                f"single-channel masks (full_1cmasks), not the colour ones.")
        lo, hi = int(labels.min()), int(labels.max())
        if lo < CLASS_ID_MIN or hi > CLASS_ID_MAX:
            raise ValueError(
                f"{path}: label values {lo}..{hi} outside the DLRSD range "
                f"{CLASS_ID_MIN}..{CLASS_ID_MAX}; uniques={np.unique(labels)}")
    return labels


def load_colour(cfg: Config, image_id: str) -> np.ndarray:
    """Official DLRSD colour-coded ground truth as RGB uint8 (H, W, 3).

    Read through PIL because these are paletted; cv2 would return luminance.
    """
    path = cfg.colour / f"{image_id}.png"
    if not path.is_file():
        raise FileNotFoundError(f"could not read colour mask {path}")
    return np.array(Image.open(path).convert("RGB"))


def classes_present(labels: np.ndarray) -> list[int]:
    """Native class ids (1..17) present in a label map, ascending."""
    return [int(c) for c in np.unique(labels)
            if CLASS_ID_MIN <= int(c) <= CLASS_ID_MAX]


@lru_cache(maxsize=1)
def _category_index(images_dir: str) -> dict[str, list[str]]:
    """Group image ids by the 5-character land-use prefix DLRSD filenames use."""
    groups: dict[str, list[str]] = {}
    for p in sorted(Path(images_dir).glob("*.png")):
        groups.setdefault(p.stem.split("_", 1)[0], []).append(p.stem)
    return groups


def categories(cfg: Config) -> dict[str, list[str]]:
    """Land-use category prefix -> image ids, e.g. "harbo" -> [...]."""
    return _category_index(str(cfg.images))
