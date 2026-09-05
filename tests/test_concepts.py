"""Text ("concept") prompting: the 17 class names, and class-restricted scoring.

The point of these tests is the one thing that separates this recognition mode
from automatic mode: a mask only counts for a region if SAM produced it under
that region's own class name. A perfect outline filed under the wrong name earns
nothing.
"""
import numpy as np
import pytest

from pointmin.class_map import CLASS_NAMES
from pointmin.concepts import (ConceptMasks, cache_path, class_prompts,
                               concept_masks, concept_masks_cached,
                               load_cached, save_cached, settings_key)
from pointmin.config import Config
from pointmin.metrics import score_regions, score_regions_by_class
from pointmin.regions import RECOGNIZED, UNRECOGNIZED, Region

H = W = 8


def blob(y0, y1, x0, x1, shape=(H, W)) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def region(gt: np.ndarray, class_id: int, region_id: int = 0) -> Region:
    return Region(image_id="t", class_id=class_id, region_id=region_id,
                  gt_mask=gt, matched_sam_mask=None, coverage=0.0, iou=0.0,
                  status=UNRECOGNIZED, area_px=int(gt.sum()),
                  class_name=CLASS_NAMES[class_id - 1])


# ---- prompts ---------------------------------------------------------------
def test_one_prompt_per_class_in_native_id_order():
    pairs = class_prompts(Config(device="cpu"))
    assert len(pairs) == len(CLASS_NAMES) == 17
    assert [cid for cid, _ in pairs] == list(range(1, 18))
    assert [name for _, name in pairs] == CLASS_NAMES


def test_template_wraps_the_name_without_shifting_ids():
    cfg = Config(device="cpu", sam3_prompt_template="an aerial photo of {name}")
    pairs = class_prompts(cfg)
    assert pairs[0] == (1, "an aerial photo of airplane")
    assert pairs[-1] == (17, "an aerial photo of water")


def test_settings_key_tracks_prompt_text_and_thresholds():
    base = settings_key(Config(device="cpu"))
    assert base != settings_key(Config(device="cpu",
                                       sam3_prompt_template="a photo of {name}"))
    assert base != settings_key(Config(device="cpu", sam3_score_threshold=0.3))
    assert base == settings_key(Config(device="cpu"))


# ---- ConceptMasks ----------------------------------------------------------
def test_indices_for_class_are_positions_in_the_pooled_array():
    cm = ConceptMasks(masks=np.stack([blob(0, 2, 0, 2), blob(2, 4, 2, 4),
                                      blob(4, 6, 4, 6)]),
                      class_ids=np.array([3, 15, 3], np.int32),
                      scores=np.array([0.9, 0.8, 0.7], np.float32),
                      prompts={3: "buildings", 15: "trees"})
    assert cm.indices_for_class(3).tolist() == [0, 2]
    assert cm.indices_for_class(15).tolist() == [1]
    assert cm.indices_for_class(7).tolist() == []
    assert len(cm.for_class(3)) == 2
    assert cm.shape == (H, W)


def test_counts_include_prompts_that_returned_nothing():
    cm = ConceptMasks(masks=np.stack([blob(0, 2, 0, 2)]),
                      class_ids=np.array([3], np.int32),
                      scores=np.array([0.9], np.float32),
                      prompts={3: "buildings", 15: "trees"})
    assert cm.counts() == {3: 1, 15: 0}
    assert cm.empty_prompts() == [15]


# ---- class-restricted scoring ----------------------------------------------
def test_perfect_mask_under_the_wrong_name_earns_nothing():
    gt = blob(0, 4, 0, 4)
    cm = ConceptMasks(masks=np.stack([gt.copy()]),
                      class_ids=np.array([10], np.int32),   # pavement
                      scores=np.array([0.99], np.float32),
                      prompts={3: "buildings", 10: "pavement"})

    r = region(gt, class_id=3)                              # buildings
    score_regions_by_class([r], cm, 0.90, 0.75)
    assert r.status == UNRECOGNIZED
    assert (r.coverage, r.iou) == (0.0, 0.0)
    assert r.matched_sam_mask is None

    # the same mask, judged class-agnostically, is a perfect match
    same = region(gt, class_id=3)
    score_regions([same], cm.masks, 0.90, 0.75)
    assert same.status == RECOGNIZED
    assert same.iou == pytest.approx(1.0)


def test_matched_indices_point_into_the_pooled_array():
    gt = blob(0, 4, 0, 4)
    cm = ConceptMasks(masks=np.stack([blob(6, 8, 6, 8), gt.copy()]),
                      class_ids=np.array([10, 3], np.int32),
                      scores=np.array([0.9, 0.9], np.float32),
                      prompts={3: "buildings", 10: "pavement"})
    r = region(gt, class_id=3)
    score_regions_by_class([r], cm, 0.90, 0.75)
    assert r.status == RECOGNIZED
    assert r.matched_mask_indices == (1,)          # global position, not 0
    assert cm.class_ids[r.matched_mask_indices[0]] == 3


# ---- prompting a backend ---------------------------------------------------
class FakeTextBackend:
    """Answers "buildings" with two instances, "trees" with one, rest silent."""

    name = "fake_text"
    supports_text = True

    def __init__(self):
        self.calls = 0
        self.seen_prompts: list[str] = []

    def concept_masks(self, image, prompts):
        self.calls += 1
        self.seen_prompts = list(prompts)
        answers = {"buildings": [blob(0, 2, 0, 2), blob(2, 4, 2, 4)],
                   "trees": [blob(4, 6, 4, 6)]}
        masks, index, scores = [], [], []
        for i, phrase in enumerate(prompts):
            for m in answers.get(phrase, []):
                masks.append(m)
                index.append(i)
                scores.append(0.75)
        if not masks:
            return (np.zeros((0, H, W), bool), np.zeros((0,), np.int32),
                    np.zeros((0,), np.float32))
        return (np.stack(masks), np.asarray(index, np.int32),
                np.asarray(scores, np.float32))


def test_concept_masks_tags_every_instance_with_its_own_class_id():
    cfg = Config(device="cpu")
    cm = concept_masks(FakeTextBackend(), np.zeros((H, W, 3), np.uint8), cfg)
    buildings = CLASS_NAMES.index("buildings") + 1
    trees = CLASS_NAMES.index("trees") + 1
    assert len(cm) == 3
    assert cm.class_ids.tolist() == [buildings, buildings, trees]
    assert cm.counts()[buildings] == 2 and cm.counts()[trees] == 1
    assert len(cm.prompts) == 17            # all 17 asked, silence recorded
    assert buildings not in cm.empty_prompts()
    assert len(cm.empty_prompts()) == 15


def test_the_backend_is_handed_the_17_names_and_nothing_else():
    backend = FakeTextBackend()
    concept_masks(backend, np.zeros((H, W, 3), np.uint8), Config(device="cpu"))
    assert backend.seen_prompts == CLASS_NAMES


def test_a_model_that_cannot_be_named_at_refuses_rather_than_guessing():
    """Text and image-only numbers differ by a lot, so a model that cannot take a
    phrase must raise here rather than quietly return the class-agnostic pass."""

    class NoText:
        name = "geometry_only"
        supports_text = False

    with pytest.raises(TypeError, match="class name"):
        concept_masks(NoText(), np.zeros((H, W, 3), np.uint8), Config(device="cpu"))


# ---- cache -----------------------------------------------------------------
def test_cache_round_trip_preserves_masks_ids_scores_and_prompts(tmp_path):
    cfg = Config(device="cpu", artifacts_dir=str(tmp_path))
    cm = concept_masks(FakeTextBackend(), np.zeros((H, W, 3), np.uint8), cfg)

    assert load_cached(cfg, "t", "fake_text") is None
    save_cached(cfg, "t", "fake_text", cm)
    back = load_cached(cfg, "t", "fake_text")

    assert np.array_equal(back.masks, cm.masks)
    assert back.masks.dtype == bool
    assert back.class_ids.tolist() == cm.class_ids.tolist()
    assert back.scores.tolist() == pytest.approx(cm.scores.tolist())
    assert back.prompts == cm.prompts


def test_cached_call_hits_the_backend_once_and_a_new_template_misses(tmp_path):
    cfg = Config(device="cpu", artifacts_dir=str(tmp_path))
    backend, image = FakeTextBackend(), np.zeros((H, W, 3), np.uint8)

    first = concept_masks_cached(backend, image, cfg, "t")
    second = concept_masks_cached(backend, image, cfg, "t")
    assert backend.calls == 1
    assert np.array_equal(first.masks, second.masks)

    retuned = Config(device="cpu", artifacts_dir=str(tmp_path),
                     sam3_score_threshold=0.8)
    assert cache_path(retuned, "t", "fake_text") != cache_path(cfg, "t", "fake_text")
    concept_masks_cached(backend, image, retuned, "t")
    assert backend.calls == 2            # retuning cannot be served stale masks


def test_use_cache_false_neither_reads_nor_writes(tmp_path):
    cfg = Config(device="cpu", artifacts_dir=str(tmp_path))
    backend, image = FakeTextBackend(), np.zeros((H, W, 3), np.uint8)
    concept_masks_cached(backend, image, cfg, "t", use_cache=False)
    concept_masks_cached(backend, image, cfg, "t", use_cache=False)
    assert backend.calls == 2
    assert load_cached(cfg, "t", "fake_text") is None
