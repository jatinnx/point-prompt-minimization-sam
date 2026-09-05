"""Step 0 -- manual inspection.

Section 6 marks this step "Both, together": the point is for two people to look
at real output and agree on what a region is, and on what a partial SAM failure
looks like. This script produces the things to look at, plus the measurements
that settle the three open questions in Section 8:

  * does one-connected-component-per-class hold up, or do sprawling classes
    such as freeways and rivers break it?
  * what does a partial failure actually look like?
  * is a 24 px minimum region area sensible?

Images are chosen to stress the region definition rather than flatter it.

    python scripts/01_inspect.py
    python scripts/01_inspect.py --per-category 2 --backend sam1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pointmin import Config, Harness, viz                   # noqa: E402
from pointmin.config import PROJECT_ROOT, resolve           # noqa: E402
from pointmin.regions import Region, region_area_stats      # noqa: E402

# 5-character DLRSD filename prefixes, picked for the failure mode each exposes.
STRESS_CATEGORIES = {
    "dense": "dense residential -- many small buildings packed together",
    "harbo": "harbour -- ships and docks, thin repeated structures",
    "freew": "freeway -- sprawling pavement, the connected-component worry",
    "agric": "agricultural -- one large homogeneous field",
    "parki": "parking lot -- many tiny cars near the area floor",
    "beach": "beach -- large smooth sand and sea regions",
}

AREA_FLOORS = (0, 8, 16, 24, 48, 96)


def pick_images(harness: Harness, per_category: int) -> list[tuple[str, str]]:
    available = harness.categories()
    chosen: list[tuple[str, str]] = []
    for prefix, why in STRESS_CATEGORIES.items():
        ids = available.get(prefix)
        if not ids:
            print(f"  WARNING: no images with prefix {prefix!r}, skipping")
            continue
        chosen.extend((image_id, why) for image_id in ids[:per_category])
    return chosen


def shape_stats(region: Region) -> dict:
    """How road-like a region is.

    A compact blob fills much of its bounding box and merely shrinks under
    erosion. A sprawling one-pixel-wide road fills little of its box and can be
    erased entirely by a 3x3 erosion -- the case that makes "one connected
    component" a questionable unit of work, and that leaves no interior pixel to
    place a point in.
    """
    _, _, w, h = region.bbox
    eroded = cv2.erode(region.gt_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    return {
        "fill_ratio": round(region.area_px / max(w * h, 1), 3),
        "interior_px": int(eroded.sum()),
        "erodes_to_nothing": not bool(eroded.any()),
    }


def report_image(harness: Harness, image_id: str, why: str, out_dir: Path) -> dict:
    image = harness.load_image(image_id)
    colour = harness.load_colour(image_id)
    masks = harness.automatic_masks(image_id)
    regions = harness.regions(image_id)

    fig_path = out_dir / f"{image_id}.png"
    viz.inspection_figure(image, colour, regions, masks, image_id,
                          harness.backend_name, fig_path)

    print(f"\n{image_id}  ({why})")
    print(f"  {len(masks)} automatic masks, {len(regions)} regions "
          f"at min_area={harness.cfg.min_region_area} px")
    print(f"  {'id':>3} {'class':<12} {'area':>6} {'fill':>5} {'cov':>6} "
          f"{'IoU':>6} {'masks':>5}  status")
    rows = []
    for r in sorted(regions, key=lambda r: -r.area_px):
        s = shape_stats(r)
        flag = "  <- no interior" if s["erodes_to_nothing"] else ""
        print(f"  {r.region_id:>3} {r.class_name:<12} {r.area_px:>6} "
              f"{s['fill_ratio']:>5.2f} {r.coverage:>6.3f} {r.iou:>6.3f} "
              f"{len(r.matched_mask_indices):>5}  {r.status}{flag}")
        rows.append({**r.summary(), **s})

    n_px = image.shape[0] * image.shape[1]
    return {
        "image_id": image_id,
        "why_chosen": why,
        "n_auto_masks": int(len(masks)),
        "n_regions": len(regions),
        "n_unrecognized": sum(1 for r in regions if r.status == "unrecognized"),
        "gt_fraction_in_regions": round(sum(r.area_px for r in regions) / n_px, 4),
        "figure": str(fig_path.relative_to(PROJECT_ROOT)),
        "regions": rows,
    }


def area_floor_sweep(harness: Harness, image_ids: list[str]) -> dict:
    """Region counts at several area floors, so 24 px is a choice not a guess."""
    total_px = sum(harness.load_labels(i).size for i in image_ids)
    sweep = {}
    for floor in AREA_FLOORS:
        counts, kept_px = [], 0
        for image_id in image_ids:
            regions = harness.gt_regions(image_id, min_area=floor)
            counts.append(len(regions))
            kept_px += sum(r.area_px for r in regions)
        sweep[floor] = {
            "regions_total": int(sum(counts)),
            "per_image_mean": round(float(np.mean(counts)), 1),
            "per_image_max": int(max(counts)),
            "gt_pixels_retained": round(kept_px / total_px, 4),
        }
    return sweep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-category", type=int, default=1,
                    help="images per stress category (default 1 -> 6 images)")
    ap.add_argument("--backend", default="auto", choices=["auto", "sam3", "sam1"])
    ap.add_argument("--out", default="artifacts/step0")
    args = ap.parse_args()

    cfg = Config(backend=args.backend)
    harness = Harness(cfg)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = pick_images(harness, args.per_category)
    if not chosen:
        print("no images found -- run scripts/00_extract_data.py first")
        return 1

    print(f"Step 0 inspection: {len(chosen)} images")
    print(f"backend={harness.backend_name} device={cfg.device}")
    print(f"thresholds (PROVISIONAL, Step 6 calibrates them): "
          f"coverage >= {cfg.coverage_threshold}, IoU >= {cfg.iou_threshold}, "
          f"match_mode={cfg.match_mode}")

    per_image = [report_image(harness, image_id, why, out_dir)
                 for image_id, why in chosen]
    image_ids = [image_id for image_id, _ in chosen]
    all_regions = [r for i in image_ids for r in harness.regions(i)]

    viz.coverage_iou_scatter(all_regions, cfg, out_dir / "coverage_vs_iou.png",
                             title="Step 0 regions: coverage vs IoU")

    sweep = area_floor_sweep(harness, image_ids)
    print("\nminimum-area sweep -- is 24 px sensible?")
    print(f"  {'floor':>5} {'regions':>8} {'per img':>8} {'max':>5} {'GT px kept':>11}")
    for floor, s in sweep.items():
        print(f"  {floor:>5} {s['regions_total']:>8} {s['per_image_mean']:>8.1f} "
              f"{s['per_image_max']:>5} {s['gt_pixels_retained']:>11.4f}")

    shapes = [(r, shape_stats(r)) for r in all_regions]
    vanishing = [r for r, s in shapes if s["erodes_to_nothing"]]
    print("\nsprawl check -- does one component per class hold up?")
    print(f"  {len(vanishing)}/{len(all_regions)} regions erode to nothing under a "
          f"3x3 kernel, i.e. they are one pixel wide everywhere")
    print("  eight lowest bbox fill ratios:")
    for r, s in sorted(shapes, key=lambda rs: rs[1]["fill_ratio"])[:8]:
        print(f"    {r.key:<16} {r.class_name:<12} area={r.area_px:>6} "
              f"fill={s['fill_ratio']:.2f} cov={r.coverage:.3f} iou={r.iou:.3f}")

    n_unrec = sum(1 for r in all_regions if r.status == "unrecognized")
    print(f"\n{n_unrec}/{len(all_regions)} regions unrecognized -- these are what "
          f"the point selector has to fix")

    summary = {
        "backend": harness.backend_name,
        "config": {
            "min_region_area": cfg.min_region_area,
            "connectivity": cfg.connectivity,
            "coverage_threshold": cfg.coverage_threshold,
            "iou_threshold": cfg.iou_threshold,
            "match_mode": cfg.match_mode,
            "auto_points_per_side": cfg.auto_points_per_side,
            "auto_pred_iou_thresh": cfg.auto_pred_iou_thresh,
            "auto_stability_score_thresh": cfg.auto_stability_score_thresh,
        },
        "n_images": len(chosen),
        "n_regions": len(all_regions),
        "n_unrecognized": n_unrec,
        "area_stats": region_area_stats(all_regions),
        "min_area_sweep": {str(k): v for k, v in sweep.items()},
        "regions_eroding_to_nothing": [r.key for r in vanishing],
        "images": per_image,
    }
    (out_dir / "step0_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(chosen)} figures, coverage_vs_iou.png and "
          f"step0_summary.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
