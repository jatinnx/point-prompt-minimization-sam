"""How much of the unrecognized rate is the task, and how much is our settings?

Section 4 item 2 is "a wrapper around SAM's automatic mode", and the wrapper has
a point grid and two score filters that decide how much it finds. Handing the
point-selection half a workload figure produced by needlessly strict settings
would make the task look harder than it is, so this sweeps them.

This is the *image-only* side of the toggle only: text mode has no point grid.
It is also the expensive script here -- cost is quadratic in points_per_side and
nothing is cached below the preset -- so it is a tool for Step 6 rather than part
of the Step 0/1 run. Nothing here touches the *recognition* thresholds
(coverage/IoU); those are Step 6 and belong to both of us.

    python scripts/03_auto_sweep.py
    python scripts/03_auto_sweep.py --n 21
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pointmin import Config, Harness                        # noqa: E402
from pointmin.autoseg import settings_key                   # noqa: E402
from pointmin.config import resolve                         # noqa: E402
from pointmin.metrics import distribution, score_regions    # noqa: E402

# points_per_side, pred_iou_thresh, stability_score_thresh. "default" is what
# Config ships and what artifacts/step0_sam3 and artifacts/step1_sam3 were run
# at, so it is the only row that is free on a machine with the cache.
SETTINGS = {
    "strict": dict(auto_points_per_side=16, auto_pred_iou_thresh=0.88,
                   auto_stability_score_thresh=0.92),
    "default": dict(auto_points_per_side=16, auto_pred_iou_thresh=0.70,
                    auto_stability_score_thresh=0.85),
    "denser": dict(auto_points_per_side=24, auto_pred_iou_thresh=0.70,
                   auto_stability_score_thresh=0.85),
    "loose": dict(auto_points_per_side=32, auto_pred_iou_thresh=0.60,
                  auto_stability_score_thresh=0.80),
}


def pick_images(harness: Harness, n: int) -> list[str]:
    prefixes = sorted(harness.categories())
    if n < len(prefixes):
        step = len(prefixes) / n
        prefixes = [prefixes[int(i * step)] for i in range(n)]
    cats = harness.categories()
    return [cats[p][0] for p in prefixes if cats[p]]


def evaluate(name: str, overrides: dict, image_ids: list[str],
             backend) -> dict:
    cfg = Config(**overrides)
    harness = Harness(cfg, backend=backend, verbose=False)
    t0 = time.perf_counter()
    n_masks, regions = 0, []
    for image_id in image_ids:
        masks = harness.automatic_masks(image_id)
        n_masks += len(masks)
        regions += score_regions(harness.gt_regions(image_id), masks,
                                 cfg.coverage_threshold, cfg.iou_threshold,
                                 cfg.match_mode)
    elapsed = time.perf_counter() - t0
    n_rec = sum(1 for r in regions if r.status == "recognized")
    cov = distribution([r.coverage for r in regions])
    iou = distribution([r.iou for r in regions])
    return {
        "name": name,
        "settings": overrides,
        "cache_key": settings_key(cfg),
        "n_masks": int(n_masks),
        "masks_per_image": round(n_masks / len(image_ids), 1),
        "n_regions": len(regions),
        "n_recognized": n_rec,
        "recognized_frac": round(n_rec / len(regions), 3),
        "mean_coverage": round(cov["mean"], 3),
        "median_coverage": round(cov["p50"], 3),
        "mean_iou": round(iou["mean"], 3),
        "median_iou": round(iou["p50"], 3),
        "n_no_match": sum(1 for r in regions if r.matched_sam_mask is None),
        "seconds_total": round(elapsed, 1),
        "seconds_per_image": round(elapsed / len(image_ids), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--out", default="artifacts/step1_sam3")
    args = ap.parse_args()

    probe = Harness(Config(recognition="automatic"))
    image_ids = args.images or pick_images(probe, args.n)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"automatic-mode sweep on {len(image_ids)} images, "
          f"backend={probe.backend_name} (image only, no class names)")
    print("timings include a cached pass as 0s, so re-runs look free")
    print(f"\n  {'preset':<11} {'pps':>4} {'iou':>5} {'stab':>5} {'masks/img':>10} "
          f"{'recog':>7} {'cov':>6} {'IoU':>6} {'s/img':>7}")

    rows = []
    for name, overrides in SETTINGS.items():
        row = evaluate(name, overrides, image_ids, probe.backend)
        rows.append(row)
        print(f"  {name:<11} {overrides['auto_points_per_side']:>4} "
              f"{overrides['auto_pred_iou_thresh']:>5.2f} "
              f"{overrides['auto_stability_score_thresh']:>5.2f} "
              f"{row['masks_per_image']:>10.1f} "
              f"{row['recognized_frac']:>7.1%} {row['mean_coverage']:>6.3f} "
              f"{row['mean_iou']:>6.3f} {row['seconds_per_image']:>7.2f}")

    best = max(rows, key=lambda r: r["mean_coverage"])
    print(f"\nhighest mean coverage: {best['name']} at {best['mean_coverage']:.3f}")
    print("coverage buys less per preset than runtime costs -- read the s/img "
          "column beside it before moving Config.auto_points_per_side")

    (out_dir / "auto_mode_sweep.json").write_text(json.dumps({
        "backend": probe.backend_name,
        "images": image_ids,
        "recognition_thresholds": {
            "coverage": probe.cfg.coverage_threshold,
            "iou": probe.cfg.iou_threshold,
            "note": "held fixed here; calibrating these is Step 6, jointly",
        },
        "presets": rows,
    }, indent=2))
    print(f"wrote auto_mode_sweep.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
