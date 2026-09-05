"""Configuration and path resolution.

Relative paths in Config resolve against the project root, not the current
working directory, so scripts work from anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# SAM 3 lives in the project rather than in the Hugging Face cache. The cache is
# outside the tree, is shared with unrelated work, and holds the weights behind
# symlinked blobs; a `hf cache prune` elsewhere on this machine would break a run
# here. SAM-modals/sam3/REVISION.txt records which revision the copy is.
SAM3_LOCAL_DIR = "SAM-modals/sam3"
SAM3_HUB_ID = "facebook/sam3"        # fallback: gated, needs `hf auth login`


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


def sam3_source(local_dir: str | Path = SAM3_LOCAL_DIR,
                hub_id: str = SAM3_HUB_ID) -> str:
    """Where ``from_pretrained`` should load SAM 3 from.

    The in-project copy if it is there, the hub id otherwise, and
    ``POINTMIN_SAM3_PATH`` ahead of both -- a clone elsewhere needs a way to say
    where its copy is without editing this file. Presence is decided by
    ``config.json``, not by the directory existing, so a half-finished copy falls
    through to the hub instead of failing deep inside transformers.
    """
    override = os.environ.get("POINTMIN_SAM3_PATH")
    if override:
        return override
    local = resolve(local_dir)
    return str(local) if (local / "config.json").is_file() else hub_id


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

    # How SAM is asked what it can already find, before any point prompt.
    #
    #   "text"       image + the 17 DLRSD class names, nothing else. A region
    #                counts as recognised only if the masks returned for *its
    #                own* class name cover it. This is the project's design:
    #                SAM has to name the thing, not just outline it.
    #   "automatic"  the image and nothing else -- segment-everything on a
    #                point grid, no class names. Class-agnostic, so a region is
    #                credited for any overlapping blob: it asks whether SAM can
    #                outline the thing, not whether it can name it.
    #   "auto"       "text" if the model can be prompted with a phrase, else
    #                "automatic". Resolved by the harness, since only it knows
    #                what actually built.
    #
    # Both sides are SAM 3. Their numbers differ by a lot (see README), so the
    # mode is recorded in every artifact and never averaged across.
    recognition: str = "auto"          # "auto" | "text" | "automatic"

    # ---- SAM 3 ------------------------------------------------------------
    device: str = "cuda"
    # Path or hub id; both SAM 3 branches load from it. Defaults to the copy
    # under SAM-modals/sam3, so a run needs neither the HF cache nor the network.
    sam3_model_id: str = field(default_factory=sam3_source)

    # ---- text ("concept") prompting, SAM 3 only ---------------------------
    # One prompt per class name per image. The template exists so Step 6 can
    # try "an aerial photo of {name}" against the bare name without a code
    # change; the bare name is what the project promises SAM, so it is default.
    sam3_prompt_template: str = "{name}"
    sam3_score_threshold: float = 0.5   # drop instances below this confidence
    sam3_mask_threshold: float = 0.5    # logit -> binary mask cut

    # ---- automatic ("segment everything") mode ----------------------------
    # These govern recognition="automatic" only; the hand-off runs in text mode
    # and never sees a point grid. 16 is what produced artifacts/step0_sam3 and
    # artifacts/step1_sam3, so a bare image-only run reproduces them instead of
    # spending an hour recomputing at a density nothing on disk was measured at.
    # Cost is quadratic in this number: SAM 3 gets points_per_side**2 prompts per
    # tile. scripts/03_auto_sweep.py re-derives the trade-off if Step 6 wants it.
    auto_points_per_side: int = 16
    auto_pred_iou_thresh: float = 0.70
    auto_stability_score_thresh: float = 0.85
    auto_box_nms_thresh: float = 0.70
    auto_min_area_frac: float = 0.0004
    auto_max_area_frac: float = 0.92
    auto_points_per_batch: int = 96

    seed: int = 42

    RECOGNITION_MODES = ("auto", "text", "automatic")

    def __post_init__(self) -> None:
        if self.recognition not in self.RECOGNITION_MODES:
            raise ValueError(
                f"recognition must be one of {list(self.RECOGNITION_MODES)}, "
                f"got {self.recognition!r}")
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
