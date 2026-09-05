"""The automatic-mask cache must not outlive the settings that produced it.

Step 6 of the plan tunes the automatic-mode thresholds. A cache keyed only on the
image would keep serving the old masks, and the tuning would look like it had no
effect at all.
"""
from __future__ import annotations

import numpy as np

from pointmin import Config
from pointmin.autoseg import (automatic_masks_cached, cache_path, filter_by_area,
                              settings_key)


class StubBackend:
    """Counts calls, so a cache hit is distinguishable from a cache miss.

    It also records the settings it was configured with, which is what caught
    the real bug here: a backend reused across two Configs kept the first
    Config's grid density while the cache key claimed the second's.
    """

    name = "stub"

    def __init__(self, n_masks: int = 3, shape=(16, 16)):
        self.calls = 0
        self.configured: list[int] = []
        self.n_masks = n_masks
        self.shape = shape

    def set_image(self, image):
        pass

    def predict_points(self, points, labels, multimask_output=True):
        raise NotImplementedError

    def configure_automatic(self, cfg):
        self.configured.append(cfg.auto_points_per_side)

    def automatic_masks(self, image):
        self.calls += 1
        masks = np.zeros((self.n_masks, *self.shape), dtype=bool)
        for i in range(self.n_masks):
            masks[i, i, :] = True          # one full row each, area = width
        return masks


def cfg_in(tmp_path, **overrides) -> Config:
    """Config with every cache-relevant field pinned, so these tests keep
    testing the key and not the current defaults."""
    pinned = dict(auto_points_per_side=20, auto_pred_iou_thresh=0.80,
                  auto_stability_score_thresh=0.88, auto_box_nms_thresh=0.70,
                  auto_min_area_frac=0.0004, auto_max_area_frac=0.92)
    return Config(artifacts_dir=str(tmp_path), **{**pinned, **overrides})


def test_settings_key_changes_with_each_relevant_field(tmp_path):
    base = cfg_in(tmp_path)
    assert settings_key(base) == settings_key(cfg_in(tmp_path))
    assert settings_key(base) != settings_key(cfg_in(tmp_path, auto_points_per_side=32))
    assert settings_key(base) != settings_key(
        cfg_in(tmp_path, auto_pred_iou_thresh=0.70))
    assert settings_key(base) != settings_key(
        cfg_in(tmp_path, auto_stability_score_thresh=0.85))
    assert settings_key(base) != settings_key(cfg_in(tmp_path, auto_min_area_frac=0.01))
    assert settings_key(base) != settings_key(cfg_in(tmp_path, auto_max_area_frac=1.0))
    assert settings_key(base) != settings_key(cfg_in(tmp_path, auto_box_nms_thresh=0.5))


def test_settings_key_ignores_unrelated_fields(tmp_path):
    """Changing the device must not throw away a perfectly good cache."""
    assert settings_key(cfg_in(tmp_path)) == settings_key(cfg_in(tmp_path,
                                                                device="cpu"))


def test_cache_path_includes_the_model_and_the_settings(tmp_path):
    cfg = cfg_in(tmp_path)
    path = cache_path(cfg, "beach_01", "sam3")
    assert path.parent.name == f"sam3-{settings_key(cfg)}"
    assert path.name == "beach_01.npz"


def test_second_call_hits_the_cache(tmp_path):
    cfg = cfg_in(tmp_path, auto_min_area_frac=0.0, auto_max_area_frac=1.0)
    backend, image = StubBackend(), np.zeros((16, 16, 3), np.uint8)

    first = automatic_masks_cached(backend, image, cfg, "img_00")
    second = automatic_masks_cached(backend, image, cfg, "img_00")

    assert backend.calls == 1
    assert np.array_equal(first, second)
    assert second.dtype == np.bool_


def test_changed_settings_force_a_recompute(tmp_path):
    backend, image = StubBackend(), np.zeros((16, 16, 3), np.uint8)
    dense = cfg_in(tmp_path, auto_min_area_frac=0.0, auto_max_area_frac=1.0,
                   auto_points_per_side=16)
    denser = cfg_in(tmp_path, auto_min_area_frac=0.0, auto_max_area_frac=1.0,
                    auto_points_per_side=48)

    automatic_masks_cached(backend, image, dense, "img_00")
    automatic_masks_cached(backend, image, denser, "img_00")

    assert backend.calls == 2


def test_backend_is_reconfigured_before_every_pass(tmp_path):
    """The cache key and the settings that ran must never disagree."""
    backend, image = StubBackend(), np.zeros((16, 16, 3), np.uint8)
    a = cfg_in(tmp_path, auto_min_area_frac=0.0, auto_max_area_frac=1.0,
               auto_points_per_side=16)
    b = cfg_in(tmp_path, auto_min_area_frac=0.0, auto_max_area_frac=1.0,
               auto_points_per_side=48)

    automatic_masks_cached(backend, image, a, "img_00")
    automatic_masks_cached(backend, image, b, "img_00")

    assert backend.configured == [16, 48]


def test_use_cache_false_never_reads_or_writes(tmp_path):
    cfg = cfg_in(tmp_path, auto_min_area_frac=0.0, auto_max_area_frac=1.0)
    backend, image = StubBackend(), np.zeros((16, 16, 3), np.uint8)

    automatic_masks_cached(backend, image, cfg, "img_00", use_cache=False)
    automatic_masks_cached(backend, image, cfg, "img_00", use_cache=False)

    assert backend.calls == 2
    assert not cache_path(cfg, "img_00", backend.name).is_file()


def test_area_filter_drops_specks_and_whole_tile_masks():
    masks = np.zeros((3, 10, 10), dtype=bool)
    masks[0, 0, 0] = True          # 1 px, 1% of the tile
    masks[1, :5, :] = True         # 50 px, 50%
    masks[2, :, :] = True          # 100 px, 100%

    kept = filter_by_area(masks, min_area_frac=0.02, max_area_frac=0.92)

    assert len(kept) == 1
    assert np.array_equal(kept[0], masks[1])


def test_area_filter_passes_empty_stack_through():
    empty = np.zeros((0, 8, 8), dtype=bool)
    assert len(filter_by_area(empty, 0.01, 0.9)) == 0
