"""Point Prompt Minimization for SAM -- data and evaluation harness.

Jatin's half of the project: Section 6 Steps 0 and 1 of point-plan.md.
See INTERFACE.md for the contract the point-selection half codes against.
"""
from .class_map import CLASS_NAMES, DLRSD_COLORS, NUM_CLASSES, PALETTE
from .compat import prompt_sam3
from .concepts import ConceptMasks, class_prompts, concept_masks
from .config import Config, resolve
from .harness import Harness
from .metrics import (coverage, iou, is_recognized, score, score_regions,
                      score_regions_by_class)
from .regions import (RECOGNIZED, SPEC_FIELDS, UNRECOGNIZED, Region,
                      extract_regions, interior_mask)
from .results import (PointSelectionResult, load_results, points_by_image,
                      save_results, summarise)
from .sam_backend import BackendUnavailable, build_backend
from .session import FakeSession, SamSession
from .store import dump_dataset, load_dataset, load_regions, save_regions

__all__ = [
    "CLASS_NAMES", "DLRSD_COLORS", "NUM_CLASSES", "PALETTE",
    "Config", "resolve", "Harness",
    "coverage", "iou", "is_recognized", "score", "score_regions",
    "score_regions_by_class", "ConceptMasks", "class_prompts", "concept_masks",
    "Region", "extract_regions", "interior_mask", "RECOGNIZED", "UNRECOGNIZED",
    "SPEC_FIELDS",
    "BackendUnavailable", "build_backend", "SamSession", "FakeSession",
    "save_regions", "load_regions", "dump_dataset", "load_dataset",
    "PointSelectionResult", "save_results", "load_results", "points_by_image",
    "summarise", "prompt_sam3",
]
