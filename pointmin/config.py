"""Configuration and path resolution.

Relative paths in Config resolve against the project root, not the current
working directory, so scripts work from anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The ViT-B checkpoint already exists in several places on this machine. We
# reference one in place rather than adding a seventh 375 MB copy.
SAM1_CHECKPOINT_CANDIDATES = (
    "/home/cse-sdpl/Documents/123ad0032-paper/PointonlySAM-R03/checkpoints/sam_vit_b_01ec64.pth",
    "/home/cse-sdpl/Downloads/point_only_semseg/sam_vit_b_01ec64.pth",
    "/home/cse-sdpl/Documents/123ad0032-paper/Geminisam/weights/sam_vit_b_01ec64.pth",
)


def resolve(path: str | Path) -> Path:
    """Absolute paths pass through; relative ones anchor to the project root."""
    p = Path(path)
    if p.is_absolute():
        return p
    anchored = PROJECT_ROOT / p
    if anchored.exists():
        return anchored
    if p.exists():
        return p.resolve()
    return anchored


def _first_existing(candidates) -> str:
    """First checkpoint that exists here.

    ``POINTMIN_SAM1_CHECKPOINT`` wins if set, since the paths below are this
    machine's and a clone elsewhere needs a way to say where its copy is.
    """
    override = os.environ.get("POINTMIN_SAM1_CHECKPOINT")
    if override:
        return override
    for c in candidates:
        if Path(c).is_file():
            return c
    return candidates[0]


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    images_dir: str = "data/dlrsd/images"    # RGB PNG, 256x256
    labels_dir: str = "data/dlrsd/labels"    # mode L PNG, pixel values 1..17
    colour_dir: str = "data/dlrsd/colour"    # mode P PNG, DLRSD palette
    artifacts_dir: str = "artifacts"

    # ---- region extraction ------------------------------------------------
    num_classes: int = 17
    # DLRSD has no background class and no ignore index: every pixel is one of
    # 1..17. Verified exhaustively over all 2100 label files.
    background_class: int | None = None
    min_region_area: int = 24       # px; from the min_size=24 used on DLRSD in prior work
    connectivity: int = 8           # cv2 connectedComponents connectivity

    # ---- recognised / unrecognised ---------------------------------------
    # PROVISIONAL. Section 6 Step 6 calibrates these from pilot numbers; they
    # are deliberately not tuned here. Report distributions, not just counts.
    coverage_threshold: float = 0.90
    iou_threshold: float = 0.75
    match_mode: str = "greedy_union"   # "best_single" | "greedy_union"

    # ---- SAM backend ------------------------------------------------------
    backend: str = "auto"           # "auto" | "sam3" | "sam1" | "fake"
    device: str = "cuda"
    sam1_model_type: str = "vit_b"
    sam1_checkpoint: str = field(
        default_factory=lambda: _first_existing(SAM1_CHECKPOINT_CANDIDATES))
    sam3_model_id: str = "facebook/sam3"

    # ---- automatic ("segment everything") mode ----------------------------
    # Chosen from scripts/03_auto_sweep.py, not carried over. The stricter
    # values used by prior work on this machine (24 / 0.84 / 0.90) reached mean
    # coverage 0.575 on the Step 1 pilot against 0.765 here, which would have
    # overstated how much of the dataset needs point prompts. Loosening further
    # (48 / 0.60 / 0.80) buys 0.043 more coverage for roughly twice the runtime.
    auto_points_per_side: int = 32
    auto_pred_iou_thresh: float = 0.70
    auto_stability_score_thresh: float = 0.85
    auto_box_nms_thresh: float = 0.70
    auto_min_area_frac: float = 0.0004
    auto_max_area_frac: float = 0.92
    auto_points_per_batch: int = 96

    seed: int = 42

    def __post_init__(self) -> None:
        if self.device.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"

    # ---- resolved paths ---------------------------------------------------
    @property
    def images(self) -> Path:
        return resolve(self.images_dir)

    @property
    def labels(self) -> Path:
        return resolve(self.labels_dir)

    @property
    def colour(self) -> Path:
        return resolve(self.colour_dir)

    @property
    def artifacts(self) -> Path:
        p = resolve(self.artifacts_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
