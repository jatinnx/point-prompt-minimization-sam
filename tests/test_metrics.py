"""Coverage, IoU, and region-to-mask matching."""
import numpy as np
import pytest

from pointmin.metrics import (coverage, iou, is_recognized, match_best_single,
                              match_greedy_union, score_regions)
from pointmin.regions import RECOGNIZED, UNRECOGNIZED, Region


def _region(gt: np.ndarray) -> Region:
    return Region(image_id="t", class_id=3, region_id=0, gt_mask=gt.astype(bool),
                  matched_sam_mask=None, coverage=0.0, iou=0.0,
                  status=UNRECOGNIZED, area_px=int(gt.sum()))


def test_coverage_and_iou_hand_computed():
    gt = np.zeros((4, 4), bool)
    gt[0:2, 0:2] = True                 # 4 px
    pred = np.zeros((4, 4), bool)
    pred[1:3, 1:3] = True               # 4 px, overlap 1 px
    assert coverage(gt, pred) == pytest.approx(1 / 4)
    assert iou(gt, pred) == pytest.approx(1 / 7)


def test_identical_masks_score_one():
    gt = np.zeros((5, 5), bool)
    gt[1:4, 1:4] = True
    assert coverage(gt, gt) == pytest.approx(1.0)
    assert iou(gt, gt) == pytest.approx(1.0)


def test_coverage_ignores_spill_but_iou_punishes_it():
    """The pair of metrics has to disagree on an over-large mask, or there is
    no way to tell 'found the whole thing' from 'found the whole thing plus
    half the neighbours'."""
    gt = np.zeros((10, 10), bool)
    gt[2:5, 2:5] = True                 # 9 px
    pred = np.zeros((10, 10), bool)
    pred[0:8, 0:8] = True               # 64 px, fully contains gt
    assert coverage(gt, pred) == pytest.approx(1.0)
    assert iou(gt, pred) == pytest.approx(9 / 64)


def test_degenerate_cases():
    gt = np.zeros((3, 3), bool)
    gt[0, 0] = True
    empty = np.zeros((3, 3), bool)
    assert coverage(gt, empty) == 0.0
    assert iou(gt, empty) == 0.0
    assert coverage(gt, None) == 0.0
    assert iou(gt, None) == 0.0
    assert coverage(empty, gt) == 0.0   # empty ground truth must not divide by zero


def test_recognized_needs_both_thresholds():
    assert is_recognized(0.95, 0.80, 0.90, 0.75)
    assert not is_recognized(0.95, 0.10, 0.90, 0.75)   # coverage alone is not enough
    assert not is_recognized(0.50, 0.80, 0.90, 0.75)


def test_best_single_picks_highest_iou():
    gt = np.zeros((10, 10), bool)
    gt[0:6, 0:6] = True
    poor = np.zeros((10, 10), bool)
    poor[0:2, 0:2] = True
    good = np.zeros((10, 10), bool)
    good[0:6, 0:7] = True
    masks = np.stack([poor, good])
    mask, cov, region_iou, idx = match_best_single(gt, masks)
    assert idx == (1,)
    assert region_iou == pytest.approx(36 / 42)
    assert cov == pytest.approx(1.0)
    assert mask is not None and mask.dtype == bool


def test_greedy_union_beats_best_single_on_a_split_object():
    """SAM splitting one object in half is the case that makes single-best
    matching understate coverage; the union matcher has to recover it."""
    gt = np.zeros((10, 10), bool)
    gt[0:8, 0:8] = True
    top = np.zeros((10, 10), bool)
    top[0:4, 0:8] = True
    bottom = np.zeros((10, 10), bool)
    bottom[4:8, 0:8] = True
    masks = np.stack([top, bottom])

    _, single_cov, single_iou, single_idx = match_best_single(gt, masks)
    _, union_cov, union_iou, union_idx = match_greedy_union(gt, masks)

    assert single_cov == pytest.approx(0.5)
    assert single_iou == pytest.approx(0.5)
    assert len(single_idx) == 1
    assert union_cov == pytest.approx(1.0)
    assert union_iou == pytest.approx(1.0)
    assert set(union_idx) == {0, 1}


def test_greedy_union_refuses_a_mask_that_hurts():
    gt = np.zeros((12, 12), bool)
    gt[0:4, 0:4] = True
    exact = gt.copy()
    junk = np.zeros((12, 12), bool)
    junk[8:12, 8:12] = True
    masks = np.stack([exact, junk])
    _, cov, region_iou, idx = match_greedy_union(gt, masks)
    assert idx == (0,)
    assert region_iou == pytest.approx(1.0)
    assert cov == pytest.approx(1.0)


def test_no_masks_at_all_leaves_region_unmatched():
    gt = np.zeros((6, 6), bool)
    gt[1:3, 1:3] = True
    empty_stack = np.zeros((0, 6, 6), bool)
    for matcher in (match_best_single, match_greedy_union):
        mask, cov, region_iou, idx = matcher(gt, empty_stack)
        assert mask is None and cov == 0.0 and region_iou == 0.0 and idx == ()


def test_score_regions_sets_status_and_masks():
    gt = np.zeros((10, 10), bool)
    gt[0:5, 0:5] = True
    other = np.zeros((10, 10), bool)
    other[7:9, 7:9] = True

    hit = _region(gt)
    miss = _region(other)
    masks = np.stack([gt])

    score_regions([hit, miss], masks, coverage_threshold=0.9, iou_threshold=0.75)
    assert hit.status == RECOGNIZED
    assert hit.coverage == pytest.approx(1.0)
    assert hit.matched_sam_mask is not None
    assert miss.status == UNRECOGNIZED
    assert miss.matched_sam_mask is None


def test_score_regions_rejects_unknown_match_mode():
    with pytest.raises(ValueError, match="match_mode"):
        score_regions([], np.zeros((0, 4, 4), bool), 0.9, 0.75,
                      match_mode="nonsense")
