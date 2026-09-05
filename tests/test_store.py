"""Serialising scored regions must survive the trip to disk exactly.

The point-selection half loads these files instead of running SAM, so a silent
change in a mask, a coverage value or the (x, y, w, h) bbox order here would
show up as a mysterious algorithm bug there.
"""
from __future__ import annotations

import numpy as np
import pytest

from pointmin import load_dataset, load_regions, save_regions
from pointmin.regions import RECOGNIZED, UNRECOGNIZED, extract_regions
from pointmin.store import dump_dataset


def make_regions(image_id="test_00"):
    """Two regions on a deliberately non-square 40x90 label map."""
    labels = np.zeros((90, 40), dtype=np.uint8)
    labels[10:20, 5:15] = 3        # buildings
    labels[60:80, 20:35] = 16      # trees
    regions = extract_regions(labels, image_id, min_area=24)
    regions[0].matched_sam_mask = regions[0].gt_mask.copy()
    regions[0].coverage = 1.0
    regions[0].iou = 1.0
    regions[0].status = RECOGNIZED
    regions[0].matched_mask_indices = (2,)
    regions[1].matched_sam_mask = None
    regions[1].coverage = 0.0
    regions[1].iou = 0.0
    regions[1].status = UNRECOGNIZED
    return regions


def test_round_trip_preserves_masks_and_scores(tmp_path):
    original = make_regions()
    path = save_regions(original, tmp_path / "test_00.npz")
    reloaded = load_regions(path)

    assert len(reloaded) == len(original)
    for a, b in zip(original, reloaded):
        assert b.image_id == a.image_id
        assert b.region_id == a.region_id
        assert b.class_id == a.class_id
        assert b.class_name == a.class_name
        assert b.class_region_index == a.class_region_index
        assert b.area_px == a.area_px
        assert b.bbox == a.bbox
        assert b.status == a.status
        assert b.matched_mask_indices == a.matched_mask_indices
        assert b.coverage == pytest.approx(a.coverage, abs=0)
        assert b.iou == pytest.approx(a.iou, abs=0)
        assert b.gt_mask.dtype == np.bool_
        assert np.array_equal(b.gt_mask, a.gt_mask)


def test_round_trip_keeps_non_square_shape_unrotated(tmp_path):
    """A packed/unpacked mask on a 40x90 image would silently transpose if the
    shape were rebuilt in the wrong order."""
    original = make_regions()
    reloaded = load_regions(save_regions(original, tmp_path / "r.npz"))
    assert reloaded[0].gt_mask.shape == (90, 40)
    y, x = 15, 10
    assert reloaded[0].gt_mask[y, x]
    assert not reloaded[0].gt_mask[x, y]


def test_none_matched_mask_stays_none(tmp_path):
    reloaded = load_regions(save_regions(make_regions(), tmp_path / "r.npz"))
    assert reloaded[0].matched_sam_mask is not None
    assert reloaded[1].matched_sam_mask is None


def test_empty_region_list(tmp_path):
    assert load_regions(save_regions([], tmp_path / "empty.npz")) == []


def test_dataset_dump_and_load(tmp_path):
    by_image = {"a_00": make_regions("a_00"), "b_01": make_regions("b_01")}
    dump_dataset(by_image, tmp_path, extra={"backend": "unit_test"})
    reloaded = load_dataset(tmp_path)

    assert sorted(reloaded) == ["a_00", "b_01"]
    assert [r.image_id for r in reloaded["b_01"]] == ["b_01", "b_01"]
    assert (tmp_path / "index.json").is_file()


def test_rejects_unknown_format_version(tmp_path):
    import json

    path = tmp_path / "bad.npz"
    np.savez_compressed(path, meta=np.array(json.dumps({"version": 99,
                                                        "regions": []})))
    with pytest.raises(ValueError, match="format version"):
        load_regions(path)
