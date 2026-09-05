"""The (x, y) versus mask[y, x] contract.

DLRSD tiles are square, so a row/column transpose bug produces plausible-looking
output on every real image and would only surface as mysteriously bad scores.
Everything here therefore runs on a deliberately non-square image.
"""
import numpy as np
import pytest

from pointmin.regions import extract_regions
from pointmin.session import FakeSession

W, H = 40, 90                     # width != height, and H > W on purpose
BLOB_ROWS = slice(60, 80)         # y
BLOB_COLS = slice(5, 15)          # x
INSIDE = (10, 70)                 # (x, y)


def _scene():
    labels = np.ones((H, W), np.uint8)
    labels[BLOB_ROWS, BLOB_COLS] = 3
    image = np.zeros((H, W, 3), np.uint8)
    regions = extract_regions(labels, "nonsquare", min_area=1)
    blob = [r for r in regions if r.class_id == 3][0]
    return image, regions, blob


def test_shapes_are_h_w_not_w_h():
    image, _, blob = _scene()
    assert image.shape == (H, W, 3)
    assert blob.gt_mask.shape == (H, W)


def test_point_maps_into_the_mask_as_y_then_x():
    _, _, blob = _scene()
    x, y = INSIDE
    assert blob.gt_mask[y, x]
    # The transposed lookup is out of bounds here, which is the whole point of
    # using a non-square image.
    with pytest.raises(IndexError):
        blob.gt_mask[x, y]


def test_bbox_reports_x_y_w_h():
    _, _, blob = _scene()
    assert blob.bbox == (BLOB_COLS.start, BLOB_ROWS.start,
                         BLOB_COLS.stop - BLOB_COLS.start,
                         BLOB_ROWS.stop - BLOB_ROWS.start)


def test_session_predicts_at_the_point_it_was_given():
    image, regions, blob = _scene()
    sess = FakeSession(image, "nonsquare", regions)
    mask = sess.predict([INSIDE], region=blob)
    x, y = INSIDE
    assert mask.shape == (H, W)
    assert mask[y, x]
    assert not (mask & ~blob.gt_mask).any()      # never leaks outside the region
    assert sess.num_sam_calls == 1


def test_a_transposed_point_is_rejected_rather_than_silently_wrong():
    image, regions, blob = _scene()
    sess = FakeSession(image, "nonsquare", regions)
    x, y = INSIDE
    with pytest.raises(ValueError, match="outside image bounds"):
        sess.predict([(y, x)], region=blob)      # (70, 10): 70 >= width 40


def test_bounds_are_checked_per_axis():
    image, regions, blob = _scene()
    sess = FakeSession(image, "nonsquare", regions)
    for bad in [(W, 10), (10, H), (-1, 10), (10, -1)]:
        with pytest.raises(ValueError, match="outside image bounds"):
            sess.predict([bad], region=blob)
    sess.predict([(W - 1, H - 1)])               # the far corner is valid


def test_more_points_recover_more_of_a_region():
    """FakeSession has to reward additional points, or an iterative selection
    loop written against it could never make progress."""
    image, regions, blob = _scene()
    sess = FakeSession(image, "nonsquare", regions)
    one = sess.predict([(10, 63)], region=blob).sum()
    two = sess.predict([(10, 63), (10, 76)], region=blob).sum()
    assert two > one


def test_negative_points_shrink_the_mask():
    image, regions, blob = _scene()
    sess = FakeSession(image, "nonsquare", regions)
    positive_only = sess.predict([INSIDE], region=blob)
    with_negative = sess.predict([INSIDE, (10, 74)], labels=[1, 0], region=blob)
    assert with_negative.sum() < positive_only.sum()


def test_labels_must_be_zero_or_one_and_match_the_point_count():
    image, regions, blob = _scene()
    sess = FakeSession(image, "nonsquare", regions)
    with pytest.raises(ValueError, match="labels"):
        sess.predict([INSIDE, (11, 71)], labels=[1])
    with pytest.raises(ValueError, match="1 .positive. or 0"):
        sess.predict([INSIDE], labels=[2])


def test_empty_point_list_is_an_error():
    image, regions, blob = _scene()
    sess = FakeSession(image, "nonsquare", regions)
    with pytest.raises(ValueError, match="at least one point"):
        sess.predict([])


def test_session_closes_and_refuses_further_prompts():
    image, regions, blob = _scene()
    with FakeSession(image, "nonsquare", regions) as sess:
        sess.predict([INSIDE], region=blob)
    with pytest.raises(RuntimeError, match="closed"):
        sess.predict([INSIDE], region=blob)
