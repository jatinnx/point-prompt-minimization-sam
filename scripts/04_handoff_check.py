"""Hand-off check -- run this first, from the point-selection side.

This does not select points. It answers one question: *is what Jatin handed over
actually loadable and shaped the way Section 5 of point-plan.md says?* Run it
before writing any point-selection code, and again after any pull, so an
interface drift shows up here instead of at Step 7.

    python scripts/04_handoff_check.py                 # no GPU, no weights
    python scripts/04_handoff_check.py --with-sam      # also exercise prompting
    python scripts/04_handoff_check.py --in artifacts/step1_sam3   # image-only side

Where a point is needed for the prompting smoke test it is the region's first
interior pixel in raster order -- deliberately the dumbest possible choice, so
nobody mistakes this for the center-point baseline. Choosing points well is
Steps 2-5 and belongs to Raven.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pointmin import (Config, PointSelectionResult, load_dataset,  # noqa: E402
                      load_results, points_by_image, save_results, summarise)
from pointmin.config import resolve                               # noqa: E402
from pointmin.metrics import score                                # noqa: E402
from pointmin.regions import SPEC_FIELDS as REQUIRED_FIELDS       # noqa: E402


class CheckFailed(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def check_region_contract(regions: list, image_id: str) -> None:
    """Every promise INTERFACE.md makes about a Region, asserted."""
    require(len(regions) > 0, f"{image_id}: no regions")
    shape = regions[0].gt_mask.shape
    seen_ids = set()

    for r in regions:
        for name in REQUIRED_FIELDS:
            require(hasattr(r, name), f"{r.image_id}: missing field {name!r}")

        require(r.image_id == image_id,
                f"{r.key}: image_id {r.image_id!r} != file name {image_id!r}")
        require(r.region_id not in seen_ids,
                f"{r.key}: region_id repeats inside one image")
        seen_ids.add(r.region_id)

        require(1 <= r.class_id <= 17,
                f"{r.key}: class_id {r.class_id} outside DLRSD's native 1..17")
        require(r.gt_mask.dtype == bool, f"{r.key}: gt_mask is {r.gt_mask.dtype}, not bool")
        require(r.gt_mask.shape == shape,
                f"{r.key}: gt_mask {r.gt_mask.shape} != {shape}")
        require(r.gt_mask.any(), f"{r.key}: gt_mask is empty")
        require(int(r.gt_mask.sum()) == r.area_px,
                f"{r.key}: area_px {r.area_px} != gt_mask.sum() {int(r.gt_mask.sum())}")
        require(r.status in ("recognized", "unrecognized"),
                f"{r.key}: status {r.status!r}")
        require(0.0 <= r.coverage <= 1.0 and 0.0 <= r.iou <= 1.0,
                f"{r.key}: coverage/iou outside 0..1")
        require(r.iou <= r.coverage + 1e-9,
                f"{r.key}: iou {r.iou} > coverage {r.coverage}, impossible")

        if r.matched_sam_mask is None:
            require(r.coverage == 0.0 and r.iou == 0.0,
                    f"{r.key}: no matched mask but nonzero scores")
        else:
            require(r.matched_sam_mask.dtype == bool,
                    f"{r.key}: matched_sam_mask is not bool")
            require(r.matched_sam_mask.shape == shape,
                    f"{r.key}: matched_sam_mask {r.matched_sam_mask.shape} != {shape}")
            cov, iou = score(r.gt_mask, r.matched_sam_mask)
            require(abs(cov - r.coverage) < 1e-6,
                    f"{r.key}: stored coverage {r.coverage} != recomputed {cov}")
            require(abs(iou - r.iou) < 1e-6,
                    f"{r.key}: stored iou {r.iou} != recomputed {iou}")

        # bbox is (x, y, w, h); gt_mask is indexed [y, x]. A transpose bug is
        # invisible on square masks, so check the bbox against the mask instead.
        x, y, w, h = r.bbox
        ys, xs = np.nonzero(r.gt_mask)
        require((int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                 int(ys.max() - ys.min() + 1)) == (x, y, w, h),
                f"{r.key}: bbox {r.bbox} disagrees with gt_mask extent")


def first_interior_point(region) -> tuple[int, int]:
    """First True pixel in raster order, as (x, y). Not a point-selection strategy."""
    ys, xs = np.nonzero(region.gt_mask)
    return int(xs[0]), int(ys[0])


def check_prompting(image_ids: list[str], data: dict, cfg: Config) -> dict:
    """Open a real session and prompt it, the way Raven's loop will.

    Needs the SAM 3 weights and the DLRSD tiles under data/, unlike everything
    else here.
    """
    from pointmin import Harness

    harness = Harness(cfg)
    print(f"\nprompting check   backend={harness.backend_name} device={cfg.device}")
    rows = []
    for image_id in image_ids:
        targets = [r for r in data[image_id] if r.status == "unrecognized"]
        if not targets:
            continue
        region = max(targets, key=lambda r: r.area_px)
        point = first_interior_point(region)
        with harness.open(image_id) as sess:
            require(sess.shape == region.gt_mask.shape,
                    f"{image_id}: session shape {sess.shape} != region shape")
            mask = sess.predict([point], region=region)
            require(mask.shape == region.gt_mask.shape, f"{image_id}: mask shape")
            require(mask.dtype == bool, f"{image_id}: mask dtype {mask.dtype}")
            cov, iou = harness.score(region.gt_mask, mask)
            calls = sess.num_sam_calls
        print(f"  {region.key:<16} {region.class_name:<10} area={region.area_px:>6} "
              f"point={str(point):<10} auto: cov={region.coverage:.3f} "
              f"iou={region.iou:.3f}  ->  1 point: cov={cov:.3f} iou={iou:.3f}")
        rows.append({"key": region.key, "point": point, "coverage": cov,
                     "iou": iou, "calls": calls})
    return {"n": len(rows), "rows": rows}


def check_fake_session(data: dict) -> None:
    """The model-free session, which is what makes Steps 2-5 startable today.

    Built straight from the stored regions rather than through the harness, so
    this runs on a bare clone: FakeSession only reads the image's shape, and the
    DLRSD tiles themselves are not in the repository.
    """
    from pointmin.session import FakeSession

    image_id = sorted(data)[0]
    regions = data[image_id]
    region = max(regions, key=lambda r: r.area_px)
    h, w = region.gt_mask.shape
    stand_in = np.zeros((h, w, 3), dtype=np.uint8)

    with FakeSession(stand_in, image_id, regions) as fake:
        # FakeSession needs a point inside the region; the centroid of a compact
        # region is inside it, unlike the first raster pixel on a curved edge.
        ys, xs = np.nonzero(region.gt_mask)
        cx, cy = int(round(xs.mean())), int(round(ys.mean()))
        if not region.gt_mask[cy, cx]:
            cx, cy = first_interior_point(region)
        mask = fake.predict([(cx, cy)], region=region)
        cov, iou = score(region.gt_mask, mask)
        require(mask.shape == region.gt_mask.shape, "fake session shape")
        require(fake.num_sam_calls == 1, "fake session call count")
    print(f"\nmodel-free check  FakeSession on {region.key} at ({cx}, {cy}): "
          f"cov={cov:.3f} iou={iou:.3f}  (no weights, no GPU, no image files)")


def check_result_format(data: dict, out_dir: Path) -> Path:
    """Write, reload and aggregate a PointSelectionResult -- Raven's return leg."""
    image_id = sorted(data)[0]
    region = data[image_id][0]
    point = first_interior_point(region)
    result = PointSelectionResult(
        image_id=region.image_id, region_id=region.region_id, points=[point],
        final_coverage=region.coverage, final_iou=region.iou, num_sam_calls=1)

    path = save_results([result], out_dir / "example_results.json",
                        extra={"note": "format example written by "
                                       "scripts/04_handoff_check.py, not a result"})
    reloaded = load_results(path)
    require(len(reloaded) == 1 and reloaded[0].to_dict() == result.to_dict(),
            "PointSelectionResult did not survive a round trip")
    merged = points_by_image(reloaded)
    require(merged == {region.image_id: [point]}, "points_by_image mismatch")

    print(f"\nreturn-leg check  {reloaded[0]!r}")
    print(f"  points_by_image -> {merged}")
    print(f"  summarise       -> {summarise(reloaded)}")
    print(f"  wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", default="artifacts/step1_sam3_text",
                    help="hand-off directory containing index.json (default: the "
                         "text-mode hand-off; artifacts/step1_sam3 is the "
                         "image-only side of the same 10 images)")
    ap.add_argument("--with-sam", action="store_true",
                    help="also open a real session (needs the SAM 3 weights)")
    args = ap.parse_args()

    in_dir = resolve(args.in_dir)
    if not (in_dir / "index.json").is_file():
        print(f"no hand-off at {in_dir} -- run scripts/02_harness_report.py first")
        siblings = sorted(p.parent.name for p in
                          resolve("artifacts").glob("*/index.json"))
        if siblings:
            print(f"  hand-offs that do exist: {', '.join(siblings)}")
            print(f"  pass one with --in artifacts/<name>")
        return 1

    cfg = Config()
    data = load_dataset(in_dir)
    index = json.loads((in_dir / "index.json").read_text())
    image_ids = sorted(data)
    all_regions = [r for image_id in image_ids for r in data[image_id]]
    # Which recognition mode produced these scores decides what the numbers mean:
    # text mode credits a region only via its own class name, automatic mode via
    # any overlapping mask, and the two differ by a factor of three. Printing it
    # keeps a saved console log from being quoted against the wrong baseline.
    recognition = index.get("recognition", "automatic")

    print(f"hand-off: {in_dir}")
    print(f"  backend={index.get('backend', '?')} recognition={recognition} "
          f"match_mode={index.get('match_mode', '?')}")
    print(f"  {len(image_ids)} images, {len(all_regions)} regions, "
          f"{sum(1 for r in all_regions if r.status == 'unrecognized')} unrecognized")

    for image_id in image_ids:
        check_region_contract(data[image_id], image_id)
    print(f"  Region contract holds for all {len(all_regions)} regions "
          f"({len(REQUIRED_FIELDS)} required fields, scores recomputed from the "
          f"masks, bbox checked against gt_mask extent)")

    print("\nworkload per image")
    print(f"  {'image':<16} {'regions':>7} {'unrec':>6} {'mean cov':>9} {'mean IoU':>9}")
    for image_id in image_ids:
        rs = data[image_id]
        print(f"  {image_id:<16} {len(rs):>7} "
              f"{sum(1 for r in rs if r.status == 'unrecognized'):>6} "
              f"{np.mean([r.coverage for r in rs]):>9.3f} "
              f"{np.mean([r.iou for r in rs]):>9.3f}")

    no_match = [r.key for r in all_regions if r.matched_sam_mask is None]
    why = ("SAM returned nothing under this region's own class name"
           if recognition == "text" else "SAM's automatic pass found nothing there")
    print(f"\n{len(no_match)} regions have no matching mask at all "
          f"({why}): {', '.join(no_match) or 'none'}")
    print("the rest are partial failures -- a mask exists, it is just wrong")

    check_fake_session(data)
    if args.with_sam:
        check_prompting(image_ids, data, cfg)
    else:
        print("\nprompting check   skipped (pass --with-sam to run it; it needs "
              "the SAM 3 weights and the DLRSD tiles under data/)")

    check_result_format(data, in_dir)
    print("\nhand-off OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
