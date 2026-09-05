"""Ground-truth region extraction."""
import numpy as np
import pytest

from pointmin.regions import extract_regions, interior_mask, region_area_stats


def test_two_blobs_of_one_class_are_two_regions():
    labels = np.ones((20, 20), np.uint8)        # class 1 everywhere
    labels[2:8, 2:8] = 3                        # buildings blob A
    labels[12:18, 12:18] = 3                    # buildings blob B, not touching
    regions = extract_regions(labels, "t", min_area=1)
    buildings = [r for r in regions if r.class_id == 3]
    assert len(buildings) == 2
    assert [r.area_px for r in buildings] == [36, 36]
    assert {r.class_region_index for r in buildings} == {0, 1}


def test_region_id_is_unique_within_the_image():
    """PointSelectionResult keys on (image_id, region_id) alone, so a per-class
    counter would let class 3 region 0 collide with class 9 region 0."""
    labels = np.full((20, 20), 9, np.uint8)
    labels[1:5, 1:5] = 3
    labels[10:14, 10:14] = 3
    labels[1:5, 10:14] = 6
    regions = extract_regions(labels, "t", min_area=1)
    ids = [r.region_id for r in regions]
    assert len(ids) == len(set(ids))
    assert ids == list(range(len(regions)))


def test_min_area_filter_drops_specks():
    labels = np.ones((20, 20), np.uint8)
    labels[0, 0] = 4                            # 1 px car
    labels[10:15, 10:15] = 4                    # 25 px car
    assert len([r for r in extract_regions(labels, "t", min_area=1)
                if r.class_id == 4]) == 2
    kept = [r for r in extract_regions(labels, "t", min_area=24) if r.class_id == 4]
    assert len(kept) == 1
    assert kept[0].area_px == 25


def test_diagonal_touch_depends_on_connectivity():
    labels = np.ones((10, 10), np.uint8)
    labels[2, 2] = 5
    labels[3, 3] = 5                            # touches only at a corner
    assert len([r for r in extract_regions(labels, "t", min_area=1, connectivity=8)
                if r.class_id == 5]) == 1
    assert len([r for r in extract_regions(labels, "t", min_area=1, connectivity=4)
                if r.class_id == 5]) == 2


def test_bbox_is_x_y_w_h_on_a_non_square_region():
    labels = np.ones((30, 30), np.uint8)
    labels[5:9, 20:26] = 7                      # rows 5..8, cols 20..25
    region = [r for r in extract_regions(labels, "t", min_area=1)
              if r.class_id == 7][0]
    x, y, w, h = region.bbox
    assert (x, y, w, h) == (20, 5, 6, 4)
    assert region.gt_mask[y, x]


def test_class_names_and_gt_masks_are_consistent():
    labels = np.full((10, 10), 3, np.uint8)
    region = extract_regions(labels, "img_1", min_area=1)[0]
    assert region.class_name == "buildings"     # DLRSD id 3, native 1..17 space
    assert region.gt_mask.dtype == bool
    assert region.gt_mask.sum() == 100
    assert region.key == "img_1#0"
    assert region.summary()["class_name"] == "buildings"


def test_summary_is_json_serialisable():
    import json
    labels = np.full((8, 8), 12, np.uint8)
    region = extract_regions(labels, "t", min_area=1)[0]
    json.dumps(region.summary())                 # must not raise


def test_area_stats_handles_empty_input():
    assert region_area_stats([]) == {"count": 0}


def test_interior_mask_falls_back_when_erosion_empties_a_thin_region():
    """Roads and docks are one pixel wide in places; eroding them to nothing
    would leave no valid point to sample."""
    thin = np.zeros((10, 10), bool)
    thin[5, :] = True                            # 1 px tall road
    assert interior_mask(thin).sum() == thin.sum()

    fat = np.zeros((10, 10), bool)
    fat[2:8, 2:8] = True
    eroded = interior_mask(fat)
    assert 0 < eroded.sum() < fat.sum()
    assert not eroded[2, 2]                      # boundary pixel is gone


def test_labels_outside_the_dlrsd_range_are_ignored():
    labels = np.zeros((10, 10), np.uint8)        # 0 is not a DLRSD class
    labels[0:5, 0:5] = 3
    regions = extract_regions(labels, "t", min_area=1)
    assert [r.class_id for r in regions] == [3]
