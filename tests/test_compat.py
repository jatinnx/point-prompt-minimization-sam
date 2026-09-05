"""The plan document's literal prompt_sam3 signature still works.

Point selection written against Section 5 as published must run unchanged
against this harness, otherwise the "revised to a session" decision silently
breaks the other half's existing code.
"""
from __future__ import annotations

import numpy as np
import pytest

from pointmin.compat import prompt_sam3
from pointmin.regions import Region


class StubBackend:
    """Two candidate masks: a small correct one and a big sloppy one.

    The small mask has the *lower* score, so a caller that goes by SAM's score
    picks the big one and a caller that goes by ground truth picks the small one.
    That is the difference the ``region`` argument is there to make.
    """

    name = "stub"

    def __init__(self):
        self.encodes = 0
        self.prompts: list[list[tuple[int, int]]] = []

    def set_image(self, image):
        self.encodes += 1
        self.shape = image.shape[:2]

    def predict_points(self, points, labels, multimask=True):
        self.prompts.append([(int(x), int(y)) for x, y in points])
        h, w = self.shape
        tight = np.zeros((h, w), dtype=bool)
        tight[2:6, 2:6] = True
        sloppy = np.zeros((h, w), dtype=bool)
        sloppy[0:10, 0:10] = True
        return np.stack([tight, sloppy]), np.array([0.4, 0.9], dtype=np.float32)

    def configure_automatic(self, cfg):
        pass

    def automatic_masks(self, image):
        return np.zeros((0, *image.shape[:2]), dtype=bool)


@pytest.fixture
def image():
    return np.zeros((12, 12, 3), dtype=np.uint8)


def gt_region(shape=(12, 12)) -> Region:
    gt = np.zeros(shape, dtype=bool)
    gt[2:6, 2:6] = True
    return Region(image_id="stub", class_id=1, region_id=0, gt_mask=gt,
                  matched_sam_mask=None, coverage=0.0, iou=0.0,
                  status="unrecognized", area_px=int(gt.sum()))


def test_documented_signature_returns_a_bool_mask(image):
    backend = StubBackend()
    mask = prompt_sam3(image, [(3, 4)], backend=backend)

    assert mask.shape == image.shape[:2]
    assert mask.dtype == bool
    assert backend.prompts == [[(3, 4)]]


def test_without_region_sams_own_score_decides(image):
    mask = prompt_sam3(image, [(3, 4)], backend=StubBackend())
    assert int(mask.sum()) == 100          # the sloppy mask, score 0.9


def test_with_region_ground_truth_decides(image):
    mask = prompt_sam3(image, [(3, 4)], region=gt_region(), backend=StubBackend())
    assert int(mask.sum()) == 16           # the tight mask, IoU 1.0


def test_every_call_re_encodes_which_is_why_sessions_exist(image):
    backend = StubBackend()
    for _ in range(3):
        prompt_sam3(image, [(3, 4)], backend=backend)
    assert backend.encodes == 3


def test_points_are_x_y_not_row_column(image):
    backend = StubBackend()
    prompt_sam3(image, [(1, 9)], backend=backend)
    assert backend.prompts == [[(1, 9)]]   # passed through unswapped


def test_out_of_bounds_point_is_rejected(image):
    with pytest.raises(ValueError, match="outside image bounds"):
        prompt_sam3(image, [(12, 0)], backend=StubBackend())


def test_empty_point_list_is_rejected(image):
    with pytest.raises(ValueError, match="at least one point"):
        prompt_sam3(image, [], backend=StubBackend())
