"""PointSelectionResult -- the format the point-selection half hands back.

These are contract tests. If one fails, the two halves would not fit together at
Step 7, which is exactly the failure Section 5 exists to prevent.
"""
from __future__ import annotations

import json

import pytest

from pointmin.results import (PointSelectionResult, load_results,
                              points_by_image, save_results, summarise)


def make(image_id="harbo_401", region_id=3, points=((10, 20), (30, 40)),
         cov=0.97, iou=0.81, calls=4, labels=None):
    return PointSelectionResult(image_id=image_id, region_id=region_id,
                                points=list(points), final_coverage=cov,
                                final_iou=iou, num_sam_calls=calls, labels=labels)


def test_round_trip_through_json(tmp_path):
    results = [make(), make(image_id="agric_1901", region_id=0,
                            points=[(1, 2)], labels=[1])]
    path = save_results(results, tmp_path / "r.json", extra={"step": 5})
    reloaded = load_results(path)

    assert len(reloaded) == 2
    # sorted by (image_id, region_id), so agric comes first
    assert [r.key for r in reloaded] == ["agric_1901#0", "harbo_401#3"]
    assert reloaded[1].points == [(10, 20), (30, 40)]
    assert reloaded[1].final_coverage == pytest.approx(0.97)
    assert reloaded[1].num_sam_calls == 4
    assert json.loads(path.read_text())["step"] == 5


def test_points_are_coerced_to_int_tuples():
    r = PointSelectionResult("x", 0, [[10.0, 20.0]], 1.0, 1.0, 1)
    assert r.points == [(10, 20)]
    assert all(isinstance(v, int) for v in r.points[0])


def test_labels_default_to_all_positive():
    r = make(labels=None)
    assert r.labels is None
    assert r.positive_points() == [(10, 20), (30, 40)]


def test_negative_points_are_excluded_from_positive_points():
    r = make(labels=[1, 0])
    assert r.positive_points() == [(10, 20)]


def test_label_count_must_match_point_count():
    with pytest.raises(ValueError, match="2 points but 1 labels"):
        make(labels=[1])


def test_labels_must_be_zero_or_one():
    with pytest.raises(ValueError, match="labels must be"):
        make(labels=[1, 2])


@pytest.mark.parametrize("cov,iou", [(1.5, 0.5), (0.5, -0.1)])
def test_scores_outside_unit_range_are_rejected(cov, iou):
    with pytest.raises(ValueError, match="outside 0..1"):
        make(cov=cov, iou=iou)


def test_labels_omitted_from_json_when_none(tmp_path):
    path = save_results([make(labels=None)], tmp_path / "r.json")
    assert "labels" not in json.loads(path.read_text())["results"][0]


def test_unknown_format_version_is_rejected(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"version": 99, "results": []}))
    with pytest.raises(ValueError, match="format version 99"):
        load_results(path)


def test_points_by_image_merges_regions_and_drops_duplicates():
    results = [
        make(image_id="a", region_id=1, points=[(5, 5)]),
        make(image_id="a", region_id=0, points=[(1, 1), (5, 5)]),
        make(image_id="b", region_id=0, points=[(9, 9), (2, 2)], labels=[1, 0]),
    ]
    merged = points_by_image(results)
    # region 0 sorts before region 1, and (5, 5) appears once
    assert merged["a"] == [(1, 1), (5, 5)]
    # the negative point is not part of the deliverable point set
    assert merged["b"] == [(9, 9)]


def test_points_by_image_can_keep_negatives():
    results = [make(image_id="b", region_id=0, points=[(9, 9), (2, 2)],
                    labels=[1, 0])]
    assert points_by_image(results, positive_only=False)["b"] == [(9, 9), (2, 2)]


def test_summarise_reports_point_budget():
    s = summarise([make(points=[(1, 1)], calls=2),
                   make(region_id=4, points=[(1, 1), (2, 2), (3, 3)], calls=6)])
    assert s["count"] == 2
    assert s["total_points"] == 4
    assert s["points_per_region_mean"] == 2.0
    assert s["points_per_region_max"] == 3
    assert s["total_sam_calls"] == 8


def test_summarise_handles_empty():
    assert summarise([]) == {"count": 0}
