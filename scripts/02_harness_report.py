"""Step 1 -- the evaluation harness, measured on a real pilot.

Section 6 Step 1 is "build region matching + coverage/IoU scoring, test on 5-10
images". Running it is what turns the provisional thresholds in Config into
something Step 6 can calibrate, so this reports distributions rather than only
pass/fail counts, and scores every region under *both* matching modes.

It also writes the scored regions to disk (artifacts/step1/regions/*.npz) so the
point-selection half can load real Region objects with no checkpoint and no GPU.

    python scripts/02_harness_report.py
    python scripts/02_harness_report.py --n 21 --out artifacts/step1_full
    python scripts/02_harness_report.py --images beach_1101 harbo_401
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
from pointmin.metrics import MATCHERS, distribution, score_regions  # noqa: E402
from pointmin.regions import region_area_stats                    # noqa: E402

COVERAGE_GRID = (0.80, 0.85, 0.90, 0.95)
IOU_GRID = (0.50, 0.60, 0.70, 0.75, 0.80)
AREA_FLOORS = (0, 24)


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
    """Fresh regions scored under each matching mode. Fresh because
    score_regions mutates the Region objects it is handed."""
    cfg = harness.cfg
    out: dict[str, dict] = {}
    for mode in MATCHERS:
        out[mode] = {
            image_id: score_regions(
                harness.gt_regions(image_id), harness.automatic_masks(image_id),
                coverage_threshold=cfg.coverage_threshold,
                iou_threshold=cfg.iou_threshold, match_mode=mode)
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
    ap.add_argument("--backend", default="auto", choices=["auto", "sam3", "sam1"])
    ap.add_argument("--out", default="artifacts/step1")
    args = ap.parse_args()

    cfg = Config(backend=args.backend)
    harness = Harness(cfg)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_ids = args.images or pick_images(harness, args.n)
    print(f"Step 1 harness report: {len(image_ids)} images")
    print(f"backend={harness.backend_name} device={cfg.device}")
    print("  " + ", ".join(image_ids))

    timings = {}
    for image_id in image_ids:
        t0 = time.perf_counter()
        n_masks = len(harness.automatic_masks(image_id))
        timings[image_id] = {"automatic_s": round(time.perf_counter() - t0, 3),
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

    # ---- figures and hand-off --------------------------------------------
    viz.coverage_iou_scatter(active_regions, cfg, out_dir / "coverage_vs_iou.png",
                             title=f"Step 1 pilot ({len(image_ids)} images, "
                                   f"{cfg.match_mode})")
    store.dump_dataset(active, out_dir, extra={
        "backend": harness.backend_name,
        "match_mode": cfg.match_mode,
        "min_region_area": cfg.min_region_area,
        "coverage_threshold": cfg.coverage_threshold,
        "iou_threshold": cfg.iou_threshold,
        "note": "scored regions from Step 1; load with pointmin.load_dataset()",
    })

    report = {
        "backend": harness.backend_name,
        "device": cfg.device,
        "images": image_ids,
        "config": {
            "min_region_area": cfg.min_region_area,
            "connectivity": cfg.connectivity,
            "coverage_threshold": cfg.coverage_threshold,
            "iou_threshold": cfg.iou_threshold,
            "match_mode": cfg.match_mode,
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
