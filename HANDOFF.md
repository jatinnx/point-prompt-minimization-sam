# HANDOFF.md — Jatin → Raven

Steps 0 and 1 are done. This is the runbook for turning them into the input of
your Steps 2–5: what you get, how to load it, how to call SAM, what to hand back,
and what the pilot numbers say you should design around.

`INTERFACE.md` is the reference — every field, every convention, every deviation
from `point-plan.md`. This document is the procedure. Read this first, then that.

---

## 1. What you are getting

Everything below is committed, so `git clone` is the whole install as far as the
hand-off is concerned. No checkpoint and no GPU needed to consume it.

| Path | What it is |
| --- | --- |
| `artifacts/step1/regions/*.npz` | **The hand-off.** 10 images, 179 scored `Region` objects, masks bit-packed. 188 kB total |
| `artifacts/step1/index.json` | What settings produced them: backend, match mode, thresholds, area floor |
| `artifacts/step1/step1_report.json` | Full Step 1 numbers: distributions, per class, threshold grid, per image |
| `artifacts/step1/step1_console.txt` | The Step 1 run, verbatim — the same numbers, readable |
| `artifacts/step1/coverage_vs_iou.png` | Where the 179 regions sit against the provisional thresholds |
| `artifacts/step1/example_results.json` | A valid `PointSelectionResult` file, so §6 is copyable not aspirational |
| `artifacts/step0/*.png` | Step 0's six figures — image, GT, regions, automatic masks, found/missed/spilled |
| `artifacts/step0/step0_summary.json` | Per-region shape stats and the minimum-area sweep |
| `pointmin/` | The harness. `Harness`, `Region`, sessions, metrics, the store |
| `scripts/04_handoff_check.py` | Verifies the hand-off is intact. Run it before you write anything |

Produced with SAM 1 ViT-B on cuda, `greedy_union` matching, `min_region_area=24`,
and **provisional** thresholds coverage ≥ 0.90 / IoU ≥ 0.75. Why SAM 1 and not
SAM 3: §9 of `INTERFACE.md`.

---

## 2. First: prove the hand-off arrived intact

```bash
git clone <repo> && cd Point-project
pip install -r requirements.txt          # numpy, opencv, matplotlib, pytest
python -m pytest -q                      # 74 passed
python scripts/04_handoff_check.py       # no GPU, no checkpoint
```

`04_handoff_check.py` re-derives `coverage` and `iou` from the stored masks and
compares them to the stored floats, checks `bbox` against the actual mask extent,
and asserts all eight Section 5 fields exist on all 179 regions. If it prints
`hand-off OK`, the two halves agree. If it fails, that is my bug — send me the
line it printed. Run it again after every pull.

Its actual output on this machine, so you know what "intact" looks like:

```
hand-off: /home/cse-sdpl/Videos/Point-project/artifacts/step1
  10 images, 179 regions, 131 unrecognized
  Region contract holds for all 179 regions (8 required fields, scores recomputed
  from the masks, bbox checked against gt_mask extent)

workload per image
  image            regions  unrec  mean cov  mean IoU
  agric_1901             2      0     0.978     0.923
  baseb_1000            16     11     0.814     0.747
  build_1601            18     12     0.853     0.677
  dense_1401            32     23     0.770     0.661
  freew_301             21     17     0.744     0.542
  harbo_401             24     15     0.822     0.741
  mediu_1201            22     18     0.761     0.621
  overp_1501            11      8     0.920     0.701
  river_1301            12     11     0.347     0.270
  spars_101             21     16     0.744     0.627

2 regions have no matching automatic mask at all (SAM found nothing there):
freew_301#12, river_1301#3
the rest are partial failures -- a mask exists, it is just wrong
```

---

## 3. Load the regions — one function

```python
from pointmin import load_dataset

regions_by_image = load_dataset("artifacts/step1")   # dict[str, list[Region]]

targets = [r for rs in regions_by_image.values()
             for r in rs if r.status == "unrecognized"]
print(len(targets))                                  # 131
```

These are the same `Region` objects the live harness produces — same class, same
fields, full-precision `coverage` and `iou`. They come off disk in milliseconds
with no model involved, which is the point: you can build and debug Steps 2–5
today, and we both look at byte-identical regions when we compare numbers.

What each one gives you:

```python
r = regions_by_image["freew_301"][5]
r.key                 # 'freew_301#5'
r.image_id            # 'freew_301'
r.region_id           # 5           unique within the image (see INTERFACE.md §2)
r.class_id            # 4           native DLRSD 1..17, NOT 0..16
r.class_name          # 'cars'      for printing only, never for logic
r.gt_mask             # (256, 256) bool  -- what SAM should have found
r.matched_sam_mask    # (256, 256) bool or None -- what it did find
r.coverage, r.iou     # 1.0, 0.3758   automatic mode's score for this region
r.status              # 'unrecognized'
r.area_px, r.bbox     # 59, (90, 61, 11, 11)   bbox is (x, y, w, h)
```

The two masks are all you need to find the error: `r.gt_mask & ~r.matched_sam_mask`
is what was missed, `r.matched_sam_mask & ~r.gt_mask` is what spilled. Step 4 says
place the next point "in the remaining missed area" — that is the first of those
two arrays.

---

## 4. Call SAM — a session, or the document's `prompt_sam3`

Both work. The session is faster and is what you want inside a loop.

```python
from pointmin import Harness

h = Harness()                                   # builds SAM lazily
region = regions_by_image["freew_301"][5]

with h.open("freew_301") as sess:                # encodes the image once, 349 ms
    m1 = sess.predict([(95, 66)], region=region)                     # ~12 ms
    m2 = sess.predict([(95, 66), (95, 72)], labels=[1, 0], region=region)
    print(sess.num_sam_calls)                    # 2  -> PointSelectionResult
```

The encode is 349 ms and a warm prompt is 12 ms, a 28× gap that an iterative loop
would otherwise pay on every iteration. That is the only reason the document's
signature changed; `INTERFACE.md` §4 has the full argument.

If you have already written code against `prompt_sam3(image, points)`, it runs
unchanged — the shim is real, tested, and exported:

```python
from pointmin import prompt_sam3

mask = prompt_sam3(h.load_image("freew_301"), [(95, 66)], region=region)
```

It re-encodes on every call. Fine for a one-off, wrong for a loop.

`region=` is optional and does the thing the document asked for: with it, the best
of SAM's candidate masks is chosen by IoU against `region.gt_mask`, so you never
compare candidates yourself. Without it, SAM's own confidence decides.

### You get negative points, and you will need them

`labels` follows SAM's convention: `1` positive, `0` negative, defaulting to all
positive. A positive-only interface cannot shrink a mask that is already too big,
and 29 `cars` regions in the pilot are exactly that. Measured on `freew_301#5`:

| prompt | coverage | IoU |
| --- | --- | --- |
| 1 positive at (95, 66) | 1.000 | 0.373 |
| + 1 negative at (95, 72) | 1.000 | **0.621** |

Coverage was already perfect and could not improve; only a negative point moved
IoU. If your result uses negatives, record them — `PointSelectionResult` has a
`labels` field for it, because a mixed point set replayed as all-positive gives a
different mask.

### No GPU? Use `FakeSession`

```python
with h.open_fake("freew_301") as sess:
    mask = sess.predict([(95, 66)], region=region)
```

Same contract, no model, fully deterministic: each positive point reveals a disk
of the region it lands in, negatives subtract one. Enough for a loop to make
progress and converge, so you can write and unit-test Steps 2–5 before touching a
checkpoint. `open_fake(image_id, leak_px=3)` makes the mask spill past the region
so coverage hits 1.0 while IoU does not — use it to test the over-large branch.

Write the loop against `open_fake`, then change one word to `open`.

---

## 5. Score it and decide when to stop

```python
cov, iou = h.score(region.gt_mask, mask)     # both, one pass
if h.is_recognized(cov, iou):                # cov >= 0.90 AND iou >= 0.75
    break
```

Use `h.is_recognized` rather than your own comparison. When Step 6 recalibrates
the thresholds it changes `Config`, and everything that calls this follows
automatically — anything with the numbers hardcoded quietly keeps the old rule.

Both functions work with no checkpoint, so your stopping rule is testable without
a model.

---

## 6. Hand back a `PointSelectionResult`

Section 5 defines the format; it is now a real class with validation, so a
malformed result fails at construction rather than at Step 7.

```python
from pointmin import PointSelectionResult, save_results, points_by_image, summarise

results = [
    PointSelectionResult(
        image_id=region.image_id,
        region_id=region.region_id,
        points=[(95, 66), (95, 72)],     # AFTER pruning
        labels=[1, 0],                   # omit entirely if all positive
        final_coverage=cov,
        final_iou=iou,
        num_sam_calls=sess.num_sam_calls,
    ),
]

save_results(results, "artifacts/step5/results.json")
```

It rejects a label count that does not match the point count, labels other than
0/1, and scores outside 0..1. Points are coerced to `int` tuples, so passing
numpy scalars is safe. Output is sorted by `(image_id, region_id)`, so two runs of
the same code produce identical files and `git diff` means something.

Reading them back and reducing to the deliverable:

```python
from pointmin import load_results

results = load_results("artifacts/step5/results.json")
summarise(results)          # points per region, total SAM calls, mean final scores
points_by_image(results)    # {'freew_301': [(95, 66)], ...}  <- Section 3's deliverable
```

`points_by_image` drops negative points and de-duplicates, because Section 3's
final output is "each image paired with a set of point locations, no class labels
attached". That is the shape Step 7 writes out.

`artifacts/step1/example_results.json` is a valid file if you want to see one.

---

## 7. The whole loop, as plumbing

This is the shape of Steps 2–5 with the two decisions left blank, because those
decisions are the actual project and they are yours. Everything else — loading,
prompting, scoring, stopping, recording — is already built and shown here.

```python
from pointmin import (Harness, PointSelectionResult, load_dataset,
                      save_results, summarise)

POINT_CAP = 8          # Step 6 sets this from data; a safety cap until then

def choose_first_point(region):
    """Step 2 (random) / Step 3 (centre). YOUR CODE."""
    raise NotImplementedError

def choose_next_point(region, mask):
    """Step 4: somewhere in region.gt_mask & ~mask, the remaining missed area.
    YOUR CODE."""
    raise NotImplementedError

def prune(sess, region, points, labels):
    """Step 5: drop each point in turn, keep the drop if the region still passes.
    YOUR CODE -- sess.predict is how you re-test."""
    raise NotImplementedError


h = Harness()
results = []

for image_id, regions in load_dataset("artifacts/step1").items():
    todo = [r for r in regions if r.status == "unrecognized"]
    if not todo:
        continue
    with h.open(image_id) as sess:                    # one encode for the image
        for region in todo:
            points, labels = [choose_first_point(region)], [1]
            mask = sess.predict(points, labels, region=region)
            cov, iou = h.score(region.gt_mask, mask)

            while not h.is_recognized(cov, iou) and len(points) < POINT_CAP:
                points.append(choose_next_point(region, mask))
                labels.append(1)
                mask = sess.predict(points, labels, region=region)
                cov, iou = h.score(region.gt_mask, mask)

            points, labels = prune(sess, region, points, labels)
            mask = sess.predict(points, labels, region=region)
            cov, iou = h.score(region.gt_mask, mask)

            results.append(PointSelectionResult(
                image_id=image_id, region_id=region.region_id,
                points=points, labels=labels if 0 in labels else None,
                final_coverage=cov, final_iou=iou,
                num_sam_calls=sess.num_sam_calls))

save_results(results, "artifacts/step5/results.json")
print(summarise(results))
```

Two notes on the structure. The session is opened per *image* and reused across
that image's regions — reopening per region would pay the 349 ms encode 179 times
instead of 10. And `sess.num_sam_calls` counts calls since the session opened, so
if you share a session across regions like this, subtract the count you had before
the region started to get a per-region figure.

---

## 8. What the pilot found, and what it means for your design

Ten images, one per land-use category, 179 regions, SAM 1 ViT-B, `greedy_union`,
coverage ≥ 0.90 / IoU ≥ 0.75. Full numbers in `artifacts/step1/step1_report.json`.

**131 of 179 regions (73%) are unrecognized — that is your workload.** Only **2**
of them have no matching automatic mask at all. The rest are partial failures: a
mask exists, it is just wrong. An algorithm built around "SAM found nothing here"
addresses 1.5% of the problem.

**Placement matters more than count.** `04_handoff_check.py --with-sam` puts one
deliberately stupid point — the first interior pixel in raster order, i.e. a
boundary pixel — on the largest unrecognized region of each image:

```
  baseb_1000#0   bare soil  area= 24276  auto: cov=0.679 iou=0.677  ->  1 point: cov=0.165 iou=0.154
  build_1601#4   buildings  area= 20265  auto: cov=0.755 iou=0.755  ->  1 point: cov=0.513 iou=0.460
  dense_1401#11  buildings  area= 12055  auto: cov=0.583 iou=0.577  ->  1 point: cov=0.253 iou=0.203
  freew_301#13   pavement   area= 50060  auto: cov=0.101 iou=0.099  ->  1 point: cov=0.941 iou=0.849
  harbo_401#17   water      area= 19118  auto: cov=0.320 iou=0.318  ->  1 point: cov=0.845 iou=0.586
  mediu_1201#13  grass      area= 34769  auto: cov=0.261 iou=0.253  ->  1 point: cov=0.045 iou=0.043
  overp_1501#6   pavement   area= 43047  auto: cov=0.820 iou=0.803  ->  1 point: cov=0.003 iou=0.003
  river_1301#9   trees      area= 38099  auto: cov=0.026 iou=0.026  ->  1 point: cov=0.997 iou=0.627
  spars_101#0    buildings  area= 15203  auto: cov=0.491 iou=0.490  ->  1 point: cov=0.229 iou=0.227
```

Two regions improve enormously (`river_1301#9` from 0.026 to 0.997 coverage), seven
get **worse than doing nothing**. A point on a boundary pixel lands on whichever
object owns the other side of the edge. `pointmin.interior_mask(mask)` erodes a
region so you can sample away from its border; use it, and consider keeping the
automatic mask when your point set scores worse than it did.

**Sprawling background classes are the hard cases, not the small objects:**

| class | regions | recognized | mean coverage | mean IoU |
| --- | --- | --- | --- | --- |
| field | 1 | 1 | 0.994 | 0.990 |
| ship | 15 | 4 | 0.908 | 0.791 |
| sand | 1 | 1 | 0.906 | 0.883 |
| cars | 29 | 4 | 0.875 | 0.661 |
| buildings | 35 | 13 | 0.861 | 0.726 |
| trees | 41 | 12 | 0.740 | 0.605 |
| grass | 23 | 4 | 0.669 | 0.530 |
| water | 10 | 5 | 0.650 | 0.638 |
| pavement | 9 | 2 | 0.647 | 0.569 |
| bare soil | 14 | 2 | 0.532 | 0.495 |
| dock | 1 | 0 | 0.418 | 0.344 |

`cars` is the spill case: coverage 0.875, IoU 0.661, only 4 of 29 recognized. More
positive points cannot fix a mask that is already too big — that is §4's negative
points. `bare soil`, `pavement`, `dock` and `grass` are the sprawl case: one DLRSD
`grass` component snakes between buildings and SAM has no reason to return it as
one mask. `harbo_401#0` is a dock of 9907 px filling 15% of its bounding box.

**Do not tune to the thresholds.** They are provisional and the choice dominates
the headline number: at coverage ≥ 0.80 / IoU ≥ 0.50, 54% of regions are already
recognized; at 0.95 / 0.80, 8%. The whole grid is in the report JSON. Step 6
calibrates it jointly, from your Step 4–5 numbers.

**Step 0 settled the two open questions you depend on.** Across 122 regions in six
deliberately awkward images, **0 erode to nothing** under a 3×3 kernel — every
region has an interior pixel to put a point in, roads and docks included. So one
connected component per class holds up as the unit of work. And the 24 px area
floor barely matters: 124 regions at floor 0, 122 at 24, and 100.00% of labelled
pixels retained either way.

---

## 9. If you need more images, or different ones

The 10 pilot images are one per land-use category. To regenerate the hand-off with
different or more images — needs the checkpoint and a GPU, so ask me if you do not
have them set up:

```bash
python scripts/02_harness_report.py --images harbo_401 beach_1101   # specific ones
python scripts/02_harness_report.py --n 21 --out artifacts/step1_full  # all categories
python scripts/04_handoff_check.py --in artifacts/step1_full
```

Then `load_dataset("artifacts/step1_full")`. The automatic-mask pass is cached on
disk under `artifacts/auto_masks/`, keyed by backend *and* by the settings that
change its output, so re-running is cheap and changing a setting is not silently
ignored.

The checkpoint paths in `config.py` are my machine's. On yours:

```bash
export POINTMIN_SAM1_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth
```

Nothing that only reads the hand-off needs it — `load_dataset`, `h.score` and
`h.is_recognized` all work with no checkpoint present. Only `h.open`,
`h.regions` and `h.automatic_masks` do.

Live regions instead of stored ones, if you do have a GPU:

```python
h.regions("harbo_401")        # scored, needs the checkpoint
h.unrecognized("harbo_401")   # just the ones needing points
```

Identical objects either way. `load_dataset` exists so you are not blocked on that.

---

## 10. Five things that will bite you

**Coordinates are `(x, y)`, masks index `[y, x]`.** DLRSD tiles are 256×256, so a
transposed point is invisible — no crash, no visual clue, just wrong results.
`sess.predict` raises `ValueError` on an out-of-bounds point specifically to catch
this on the first call. My coordinate tests run on a non-square 40×90 image for the
same reason; if you add coordinate handling, test it that way.

**`class_id` is 1..17, not 0..16.** Native DLRSD numbering, matching the label
PNGs and the plan document's own `3 = "buildings"` example. Never index
`CLASS_NAMES` with a raw `class_id`; call `class_name(class_id)`, which does the
`- 1`. An earlier remap to 0..16 on this machine produced a real
class-0-vs-background bug.

**`region_id` is unique within the *image*, not within the class.** The document
says "which connected blob of that class", but `PointSelectionResult` keys on
`(image_id, region_id)` alone — with a per-class counter, class 3 blob 0 and class 9
blob 0 collide and your results overwrite each other. The per-class ordinal is
still there as `class_region_index`.

**`matched_sam_mask` can be `None`.** Twice in the pilot. Guard it before using it
as an array; `coverage` and `iou` are both 0.0 in that case.

**A region's `coverage`/`iou` describe automatic mode, not your prompt.** They are
the baseline you are trying to beat. Recompute with `h.score` after every predict.

---

## 11. Sync checklist

Section 7 of the plan makes an interface change a "stop and tell the other person
immediately" moment, so:

- `pytest -q` → 74 passed, and `scripts/04_handoff_check.py` → `hand-off OK`. If
  either breaks after a pull, that is on me; send me the failing line.
- Adding a field to `Region` or `PointSelectionResult` is cheap. Changing what an
  existing one means is not — message me before, not after.
- Sync point 2 in the plan is "Raven's Steps 2–3 run against Jatin's real Step 1
  output". You can hit that now: `load_dataset("artifacts/step1")` is that output.
- Thresholds stay where they are until Step 6, when we set them together from your
  Step 4–5 numbers. Report distributions, not just recognized counts.

Backend caveat worth knowing before you quote any absolute number: the plan says
SAM 3, `facebook/sam3` is gated and this machine has no approved token, so
**SAM 1 ViT-B is what produced everything here**. `Sam3Backend` is written and
preferred by `build_backend`; access would make it the default with no other code
change. The interface would not move — the numbers in §8 would.






