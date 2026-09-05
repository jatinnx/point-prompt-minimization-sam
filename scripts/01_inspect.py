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
    python scripts/01_inspect.py --images harbo_401 chapa_1701   # explicit ids

The toggle -- one model, SAM 3, asked two ways:

    --recognition text        image + the 17 DLRSD class names, and a region
                              counts as found only via its own name. The
                              default, and the project's design.
    --recognition automatic   the image alone: segment everything on a point
                              grid, then match class-agnostically. Says whether
                              SAM can outline the region at all, ignoring what
                              it would call it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pointmin import Config, Harness, dlrsd, viz                # noqa: E402
from pointmin.concepts import class_prompts                 # noqa: E402
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


def default_out(step: str, harness) -> str:
    """``artifacts/<step>_sam3_text`` for the hand-off, ``artifacts/<step>_sam3``
    for the image-only side.

    Both sides of the toggle are SAM 3 and their numbers are far apart, so the
    mode is in the directory name: a run in one mode cannot land on top of the
    other's figures and reports under a name that no longer says which.
    """
    suffix = "_text" if harness.recognition == "text" else ""
    return f"artifacts/{step}_sam3{suffix}"


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


def figure_ref(path: Path) -> str:
    """Project-relative if it can be, absolute otherwise.

    ``--out`` takes any path, including one outside the tree. Recording the
    reference used to be an unguarded ``relative_to(PROJECT_ROOT)``, which threw
    *after* every image had been segmented -- the whole run lost to the choice of
    output directory.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def report_image(harness: Harness, image_id: str, why: str, out_dir: Path) -> dict:
    image = harness.load_image(image_id)
    colour = harness.load_colour(image_id)
    regions = harness.regions(image_id)

    text_mode = harness.recognition == "text"
    concept = harness.concept_masks(image_id) if text_mode else None
    masks = concept.masks if text_mode else harness.automatic_masks(image_id)
    label = "masks for the 17 class names" if text_mode else "automatic masks"
    # One class name per mask -- only text mode knows which name produced which
    # mask, so only text mode can colour the panel by what SAM called things.
    mask_labels = ([dlrsd.class_name(int(c)) for c in concept.class_ids]
                   if text_mode else None)

    fig_path = out_dir / f"{image_id}.png"
    viz.inspection_figure(image, colour, regions, masks, image_id,
                          harness.backend_name, fig_path, masks_label=label,
                          mask_labels=mask_labels)

    print(f"\n{image_id}  ({why})")
    print(f"  {len(masks)} {label}, {len(regions)} regions "
          f"at min_area={harness.cfg.min_region_area} px")
    if text_mode:
        answered = {c: n for c, n in concept.counts().items() if n}
        present = sorted({r.class_id for r in regions})
        print(f"  SAM answered {len(answered)}/17 names: "
              + ", ".join(f"{concept.prompts[c]}={n}" for c, n in answered.items()))
        silent = [concept.prompts[c] for c in present if not answered.get(c)]
        if silent:
            print(f"  silent on {len(silent)} name(s) that ARE in the ground "
                  f"truth: {', '.join(silent)}")
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
    record = {
        "image_id": image_id,
        "why_chosen": why,
        "n_sam_masks": int(len(masks)),
        "n_regions": len(regions),
        "n_unrecognized": sum(1 for r in regions if r.status == "unrecognized"),
        "gt_fraction_in_regions": round(sum(r.area_px for r in regions) / n_px, 4),
        "figure": figure_ref(fig_path),
        "regions": rows,
    }
    if text_mode:
        record["prompt_instance_counts"] = {
            concept.prompts[c]: n for c, n in concept.counts().items()}
        record["prompts_silent_but_present"] = [
            concept.prompts[c] for c in sorted({r.class_id for r in regions})
            if not concept.counts().get(c)]
    else:
        record["n_auto_masks"] = int(len(masks))    # kept: older summaries use it
    return record


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


def require_image(known_ids: set, image_id: str) -> None:
    """Fail fast on a typo in an explicit --images list."""
    if image_id not in known_ids:
        known = sorted({i.split("_")[0] for i in known_ids})
        raise SystemExit(
            f"unknown image id {image_id!r} -- known prefixes: {', '.join(known)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-category", type=int, default=1,
                    help="images per stress category (default 1 -> 6 images)")
    ap.add_argument("--images", nargs="*", default=None,
                    help="explicit image ids, overrides --per-category")
    ap.add_argument("--points-per-side", type=int, default=None,
                    help="override cfg.auto_points_per_side; only used by "
                         "--recognition automatic. Cost is quadratic in it, and "
                         "the committed figures used the default 16")
    ap.add_argument("--recognition", default="auto",
                    choices=["auto", "text", "automatic"],
                    help="how SAM 3 is asked what it can find: 'text' = image + "
                         "the 17 class names, 'automatic' = the image alone, "
                         "segment everything (default: text)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: artifacts/step0_sam3_text,\n"
                         "or artifacts/step0_sam3 under --recognition automatic,\n"
                         "so one mode cannot overwrite the other)")
    args = ap.parse_args()

    cfg = Config(recognition=args.recognition)
    if args.points_per_side is not None:
        cfg.auto_points_per_side = args.points_per_side
    harness = Harness(cfg)
    out_dir = resolve(args.out or default_out("step0", harness))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.images:
        known_ids = set(harness.image_ids())
        chosen = []
        for image_id in args.images:
            prefix = image_id.split("_")[0]
            why = STRESS_CATEGORIES.get(prefix,
                                        f"{prefix} -- explicit list, no stress rationale")
            require_image(known_ids, image_id)
            chosen.append((image_id, why))
    else:
        chosen = pick_images(harness, args.per_category)
    if not chosen:
        print("no images found -- run scripts/00_extract_data.py first")
        return 1

    print(f"Step 0 inspection: {len(chosen)} images")
    print(f"backend={harness.backend_name} device={cfg.device} "
          f"recognition={harness.recognition}")
    if harness.recognition == "text":
        print(f"  SAM gets the image and these 17 names, nothing else: "
              f"{', '.join(name for _, name in class_prompts(cfg))}")
        print(f"  a region is recognized only by masks for its OWN class name "
              f"(score >= {cfg.sam3_score_threshold})")
    else:
        print(f"  SAM gets the image and a {cfg.auto_points_per_side}x"
              f"{cfg.auto_points_per_side} point grid, no class names; matching "
              f"is class-agnostic")
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
        "recognition": harness.recognition,
        "class_names": [name for _, name in class_prompts(cfg)],
        "config": {
            "min_region_area": cfg.min_region_area,
            "connectivity": cfg.connectivity,
            "coverage_threshold": cfg.coverage_threshold,
            "iou_threshold": cfg.iou_threshold,
            "match_mode": cfg.match_mode,
            "recognition": cfg.recognition,
            "sam3_prompt_template": cfg.sam3_prompt_template,
            "sam3_score_threshold": cfg.sam3_score_threshold,
            "sam3_mask_threshold": cfg.sam3_mask_threshold,
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
