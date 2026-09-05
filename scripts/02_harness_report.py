"""Step 1 -- the evaluation harness, measured on a real pilot.

Section 6 Step 1 is "build region matching + coverage/IoU scoring, test on 5-10
images". Running it is what turns the provisional thresholds in Config into
something Step 6 can calibrate, so this reports distributions rather than only
pass/fail counts, and scores every region under *both* matching modes.

It also writes the scored regions to disk (regions/*.npz under --out) so the
point-selection half can load real Region objects with no weights and no GPU.

    python scripts/02_harness_report.py
    python scripts/02_harness_report.py --n 21 --out artifacts/step1_full
    python scripts/02_harness_report.py --images beach_1101 harbo_401

--recognition is the toggle, and both sides are SAM 3. Under ``text`` it is handed
the image and the 17 DLRSD class names and nothing else, and a region is credited
only by masks returned for its own class name. Under ``automatic`` it gets the
image alone -- a point grid, no names -- and matching is class-agnostic. The two
are not comparable as one number, so the mode is recorded in every artifact this
writes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pointmin import Config, Harness, store, viz                  # noqa: E402
from pointmin.config import resolve                               # noqa: E402
from pointmin.concepts import class_prompts                        # noqa: E402
from pointmin.metrics import (MATCHERS, distribution, score_regions,  # noqa: E402
                             score_regions_by_class)
from pointmin.regions import region_area_stats                    # noqa: E402

COVERAGE_GRID = (0.80, 0.85, 0.90, 0.95)
IOU_GRID = (0.50, 0.60, 0.70, 0.75, 0.80)
AREA_FLOORS = (0, 24)


def default_out(step: str, harness) -> str:
    """``artifacts/<step>_sam3_text`` for the hand-off, ``artifacts/<step>_sam3``
    for the image-only side.

    Both sides of the toggle are SAM 3 and their numbers are far apart, so the
    mode is in the directory name: a run in one mode cannot land on top of the
    other's figures and reports under a name that no longer says which.
    """
    suffix = "_text" if harness.recognition == "text" else ""
    return f"artifacts/{step}_sam3{suffix}"


def pick_images(harness: Harness, n: int) -> list[str]:
    """One image from each of n land-use categories, evenly spaced, no randomness."""
    prefixes = sorted(harness.categories())
    if n >= len(prefixes):
        picked = prefixes
    else:
        step = len(prefixes) / n
        picked = [prefixes[int(i * step)] for i in range(n)]
    cats = harness.categories()
    return [cats[p][0] for p in picked if cats[p]]


def fmt_dist(name: str, d: dict) -> str:
    if not d.get("count"):
        return f"  {name:<10} (empty)"
    return (f"  {name:<10} mean={d['mean']:.3f}  min={d['min']:.3f}  "
            f"p5={d['p5']:.3f}  p25={d['p25']:.3f}  p50={d['p50']:.3f}  "
            f"p75={d['p75']:.3f}  p95={d['p95']:.3f}  max={d['max']:.3f}")


def score_all_modes(harness: Harness, image_ids: list[str]) -> dict[str, dict]:
    """Fresh regions scored under each matching mode. Fresh because the scorers
    mutate the Region objects they are handed.

    Both recognition modes go through the same two matchers and the same
    thresholds; the only difference is whether a region may match any mask or
    only the masks SAM returned for that region's own class name.
    """
    cfg = harness.cfg
    text_mode = harness.recognition == "text"
    out: dict[str, dict] = {}
    for mode in MATCHERS:
        common = dict(coverage_threshold=cfg.coverage_threshold,
                      iou_threshold=cfg.iou_threshold, match_mode=mode)
        if text_mode:
            out[mode] = {
                image_id: score_regions_by_class(
                    harness.gt_regions(image_id),
                    harness.concept_masks(image_id), **common)
                for image_id in image_ids
            }
        else:
            out[mode] = {
                image_id: score_regions(
                    harness.gt_regions(image_id),
                    harness.automatic_masks(image_id), **common)
                for image_id in image_ids
            }
    return out


def per_class_table(regions) -> dict:
    by_class = defaultdict(list)
    for r in regions:
        by_class[(r.class_id, r.class_name)].append(r)
    table = {}
    for (class_id, name), rs in sorted(by_class.items()):
        n_rec = sum(1 for r in rs if r.status == "recognized")
        table[name] = {
            "class_id": class_id,
            "n_regions": len(rs),
            "n_recognized": n_rec,
            "recognized_frac": round(n_rec / len(rs), 3),
            "mean_coverage": round(float(np.mean([r.coverage for r in rs])), 3),
            "mean_iou": round(float(np.mean([r.iou for r in rs])), 3),
            "median_area_px": int(np.median([r.area_px for r in rs])),
            "total_area_px": int(sum(r.area_px for r in rs)),
        }
    return table


def threshold_grid(regions) -> dict:
    """Recognised fraction across a threshold grid -- the input to Step 6."""
    cov = np.array([r.coverage for r in regions])
    iou = np.array([r.iou for r in regions])
    grid = {}
    for ct in COVERAGE_GRID:
        for it in IOU_GRID:
            n = int(np.sum((cov >= ct) & (iou >= it)))
            grid[f"cov>={ct:.2f},iou>={it:.2f}"] = {
                "recognized": n,
                "unrecognized": len(regions) - n,
                "recognized_frac": round(n / max(len(regions), 1), 3),
            }
    return grid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10, help="number of pilot images")
    ap.add_argument("--images", nargs="*", default=None,
                    help="explicit image ids, overrides --n")
    ap.add_argument("--recognition", default="auto",
                    choices=["auto", "text", "automatic"],
                    help="how SAM 3 is asked what it can find: 'text' = image + "
                         "the 17 class names, 'automatic' = the image alone, "
                         "segment everything (default: text)")
    ap.add_argument("--points-per-side", type=int, default=None,
                    help="override cfg.auto_points_per_side; only used by "
                         "--recognition automatic")
    ap.add_argument("--out", default=None,
                    help="output directory (default: artifacts/step1_sam3_text,\n"
                         "or artifacts/step1_sam3 under --recognition automatic,\n"
                         "so one mode cannot overwrite the other)")
    args = ap.parse_args()

    cfg = Config(recognition=args.recognition)
    if args.points_per_side is not None:
        cfg.auto_points_per_side = args.points_per_side
    harness = Harness(cfg)
    out_dir = resolve(args.out or default_out("step1", harness))
    out_dir.mkdir(parents=True, exist_ok=True)

    image_ids = args.images or pick_images(harness, args.n)
    print(f"Step 1 harness report: {len(image_ids)} images")
    print(f"backend={harness.backend_name} device={cfg.device} "
          f"recognition={harness.recognition}")
    text_mode = harness.recognition == "text"
    if text_mode:
        print(f"  SAM gets the image and these 17 names, nothing else: "
              f"{', '.join(name for _, name in class_prompts(cfg))}")
        print(f"  a region is recognized only by masks for its OWN class name "
              f"(score >= {cfg.sam3_score_threshold})")
    else:
        print(f"  SAM gets a {cfg.auto_points_per_side}x{cfg.auto_points_per_side} "
              f"point grid and no class names; matching is class-agnostic")
    print("  " + ", ".join(image_ids))

    timings, prompt_totals = {}, defaultdict(int)
    for image_id in image_ids:
        t0 = time.perf_counter()
        if text_mode:
            concept = harness.concept_masks(image_id)
            n_masks = len(concept)
            for cid, n in concept.counts().items():
                prompt_totals[concept.prompts[cid]] += n
        else:
            n_masks = len(harness.automatic_masks(image_id))
        timings[image_id] = {"recognition_s": round(time.perf_counter() - t0, 3),
                             "n_masks": int(n_masks)}

    scored = score_all_modes(harness, image_ids)

    # ---- matching mode comparison ----------------------------------------
    print("\nmatching mode comparison (thresholds "
          f"cov>={cfg.coverage_threshold}, iou>={cfg.iou_threshold})")
    mode_summary = {}
    for mode, by_image in scored.items():
        regions = [r for rs in by_image.values() for r in rs]
        n_rec = sum(1 for r in regions if r.status == "recognized")
        n_empty = sum(1 for r in regions if r.matched_sam_mask is None)
        masks_used = [len(r.matched_mask_indices) for r in regions]
        print(f"\n{mode}: {n_rec}/{len(regions)} recognized "
              f"({n_rec / len(regions):.1%}), {n_empty} with no matching mask at all, "
              f"{np.mean(masks_used):.2f} masks per region on average")
        print(fmt_dist("coverage", distribution([r.coverage for r in regions])))
        print(fmt_dist("IoU", distribution([r.iou for r in regions])))
        mode_summary[mode] = {
            "n_regions": len(regions),
            "n_recognized": n_rec,
            "recognized_frac": round(n_rec / len(regions), 3),
            "n_no_match": n_empty,
            "masks_per_region_mean": round(float(np.mean(masks_used)), 2),
            "coverage": distribution([r.coverage for r in regions]),
            "iou": distribution([r.iou for r in regions]),
            "threshold_grid": threshold_grid(regions),
            "per_class": per_class_table(regions),
        }

    # ---- the configured mode, in detail ----------------------------------
    active = scored[cfg.match_mode]
    active_regions = [r for rs in active.values() for r in rs]

    print(f"\nper class, mode={cfg.match_mode}")
    print(f"  {'class':<12} {'regions':>7} {'recog':>6} {'cov':>6} {'IoU':>6} "
          f"{'median area':>11}")
    for name, row in mode_summary[cfg.match_mode]["per_class"].items():
        print(f"  {name:<12} {row['n_regions']:>7} "
              f"{row['n_recognized']:>6} {row['mean_coverage']:>6.3f} "
              f"{row['mean_iou']:>6.3f} {row['median_area_px']:>11}")

    print(f"\nthreshold grid, mode={cfg.match_mode} "
          f"(recognized fraction of {len(active_regions)} regions)")
    header = "  cov\\IoU " + "".join(f"{it:>8.2f}" for it in IOU_GRID)
    print(header)
    grid = mode_summary[cfg.match_mode]["threshold_grid"]
    for ct in COVERAGE_GRID:
        cells = "".join(
            f"{grid[f'cov>={ct:.2f},iou>={it:.2f}']['recognized_frac']:>8.2f}"
            for it in IOU_GRID)
        print(f"  {ct:>7.2f}{cells}")

    # ---- region counts at the area floors --------------------------------
    floor_cols = "".join(f"{'floor ' + str(f):>9}" for f in AREA_FLOORS)
    print("\nregion counts per image")
    print(f"  {'image':<16} {'masks':>6}{floor_cols} {'unrec':>6}")
    counts = {}
    for image_id in image_ids:
        per_floor = {f: len(harness.gt_regions(image_id, min_area=f))
                     for f in AREA_FLOORS}
        n_unrec = sum(1 for r in active[image_id] if r.status == "unrecognized")
        counts[image_id] = {**{f"floor_{f}": n for f, n in per_floor.items()},
                            "n_unrecognized": n_unrec,
                            **timings[image_id]}
        cells = "".join(f"{per_floor[f]:>9}" for f in AREA_FLOORS)
        print(f"  {image_id:<16} {timings[image_id]['n_masks']:>6}{cells} "
              f"{n_unrec:>6}")
    floor_totals = {f: sum(counts[i][f"floor_{f}"] for i in image_ids)
                    for f in AREA_FLOORS}
    total_cells = "".join(f"{floor_totals[f]:>9}" for f in AREA_FLOORS)
    print(f"  {'TOTAL':<16} {'':>6}{total_cells} "
          f"{sum(c['n_unrecognized'] for c in counts.values()):>6}")

    # ---- what each class name actually returned ---------------------------
    if text_mode:
        print(f"\nper class name, over all {len(image_ids)} images")
        print(f"  {'name':<12} {'instances':>9} {'GT regions':>10} {'recog':>6} "
              f"{'cov':>6} {'IoU':>6}")
        per_class = mode_summary[cfg.match_mode]["per_class"]
        for _, name in class_prompts(cfg):
            row = per_class.get(name)
            gt = f"{row['n_regions']:>10}" if row else f"{'-':>10}"
            rec = f"{row['n_recognized']:>6}" if row else f"{'-':>6}"
            cov = f"{row['mean_coverage']:>6.3f}" if row else f"{'-':>6}"
            i = f"{row['mean_iou']:>6.3f}" if row else f"{'-':>6}"
            print(f"  {name:<12} {prompt_totals.get(name, 0):>9} {gt}{rec}{cov}{i}")
        never = [n for _, n in class_prompts(cfg) if not prompt_totals.get(n)]
        if never:
            print(f"  never answered on any image: {', '.join(never)}")

    # ---- figures and hand-off --------------------------------------------
    viz.coverage_iou_scatter(active_regions, cfg, out_dir / "coverage_vs_iou.png",
                             title=f"Step 1 pilot ({len(image_ids)} images, "
                                   f"{cfg.match_mode})")
    store.dump_dataset(active, out_dir, extra={
        "backend": harness.backend_name,
        "recognition": harness.recognition,
        "match_mode": cfg.match_mode,
        "min_region_area": cfg.min_region_area,
        "coverage_threshold": cfg.coverage_threshold,
        "iou_threshold": cfg.iou_threshold,
        "note": "scored regions from Step 1; load with pointmin.load_dataset()",
    })

    report = {
        "backend": harness.backend_name,
        "recognition": harness.recognition,
        "class_names": [name for _, name in class_prompts(cfg)],
        "device": cfg.device,
        "images": image_ids,
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
            "auto_min_area_frac": cfg.auto_min_area_frac,
            "auto_max_area_frac": cfg.auto_max_area_frac,
        },
        "area_stats": region_area_stats(active_regions),
        "region_counts_by_area_floor": {str(f): floor_totals[f] for f in AREA_FLOORS},
        "per_image": counts,
        "by_match_mode": mode_summary,
    }
    if text_mode:
        report["instances_per_class_name"] = dict(prompt_totals)
    (out_dir / "step1_report.json").write_text(json.dumps(report, indent=2))

    n_unrec = sum(1 for r in active_regions if r.status == "unrecognized")
    print(f"\n{n_unrec}/{len(active_regions)} regions unrecognized under the "
          f"provisional thresholds -- the point selector's workload")
    print(f"wrote step1_report.json, coverage_vs_iou.png and "
          f"{len(image_ids)} region files to {out_dir}")

    reloaded = store.load_dataset(out_dir)
    same = sum(len(v) for v in reloaded.values()) == len(active_regions)
    print(f"round-trip check: reloaded {sum(len(v) for v in reloaded.values())} "
          f"regions from disk, matches in-memory count: {same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
