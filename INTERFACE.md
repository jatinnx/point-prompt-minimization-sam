# INTERFACE.md — what Jatin hands to Raven

This is the concrete version of Section 5 of `point-plan.md`. Where it differs
from the document, the difference is called out and explained. Nothing here
changes without telling you first, as agreed in Section 7.

Scope: this repository is Jatin's half only — Section 6 **Steps 0 and 1**.
Steps 2–5 (random baseline, centre point, iterative loop, backward pruning) are
yours and are deliberately not implemented here.

`HANDOFF.md` is the companion to this file: this one is the reference, that one is
the runbook — clone, verify, load, loop, hand back, with the pilot numbers you
should design around. Start there if you have not read either.

---

## 1. Sixty-second version

```python
from pointmin import Harness

h = Harness()                                # loads DLRSD, builds SAM lazily

for region in h.unrecognized("harbo_401"):   # regions SAM's automatic pass missed
    with h.open(region.image_id) as sess:    # encodes the image once
        mask = sess.predict([(x, y)], region=region)
        cov, iou = h.score(region.gt_mask, mask)
        if h.is_recognized(cov, iou):
            break
    print(region.key, sess.num_sam_calls)
```

No GPU and no checkpoint? Replace two lines:

```python
from pointmin import load_dataset
regions_by_image = load_dataset("artifacts/step1")   # real scored regions, from disk
with h.open_fake(image_id) as sess:                  # deterministic model-free SAM
    ...
```

---

## 2. `Region` — what you receive

Field names match the plan document exactly. The four extra fields are additive;
ignore them if you do not need them.

| Field | Type | Meaning |
| --- | --- | --- |
| `image_id` | `str` | filename without extension, e.g. `"harbo_401"` |
| `class_id` | `int` | **DLRSD class id, 1..17** — see the warning below |
| `region_id` | `int` | unique within the image: 0, 1, 2, … |
| `gt_mask` | `np.ndarray` `(H, W)` bool | True = this pixel belongs to the region |
| `matched_sam_mask` | `np.ndarray (H, W)` bool or `None` | automatic mode's best guess |
| `coverage` | `float` 0..1 | share of `gt_mask` that `matched_sam_mask` found |
| `iou` | `float` 0..1 | IoU of `gt_mask` and `matched_sam_mask` |
| `status` | `str` | `"recognized"` or `"unrecognized"` |
| `area_px` | `int` | `gt_mask.sum()`, precomputed |
| `bbox` | `(x, y, w, h)` | **x and y first**, matching the coordinate convention |
| `class_name` | `str` | e.g. `"buildings"` — for printing, never for logic |
| `class_region_index` | `int` | ordinal among blobs of the *same class* |
| `matched_mask_indices` | `tuple[int, ...]` | which automatic masks were matched |

Convenience: `region.key` is `f"{image_id}#{region_id}"`, and `region.summary()`
returns a JSON-safe dict with no arrays in it.

### Two deviations from the document, both deliberate

**`region_id` is unique within the image, not within the class.** The document
describes it as "which connected blob of *that class*", but `PointSelectionResult`
keys results on `(image_id, region_id)` alone. Both cannot be true at once: with a
per-class counter, class 3 blob 0 and class 9 blob 0 would both be
`(image_id, 0)` and your results would overwrite each other. The per-class
ordinal is still there as `class_region_index`.

**`class_id` is in native DLRSD 1..17 space, not 0..16.** The document's own
example says `3 = "buildings"`, which is the 1-based numbering, and the label
PNGs store 1..17. Earlier code on this machine remapped to 0..16 and that
produced a genuine class-0-versus-background bug. `class_name(class_id)` does the
`- 1` for you; never index `CLASS_NAMES` with a raw `class_id`.

---

## 3. `PointSelectionResult` — what you hand back

Field-for-field as the document specifies it, but now a real class with
validation, so a malformed result fails where you built it rather than at Step 7.

```python
from pointmin import PointSelectionResult, save_results, load_results

r = PointSelectionResult(
    image_id="freew_301",
    region_id=5,
    points=[(95, 66), (95, 72)],   # final list, AFTER pruning
    labels=[1, 0],                 # additive; omit entirely if all positive
    final_coverage=1.000,
    final_iou=0.621,
    num_sam_calls=2,               # read it off sess.num_sam_calls
)
save_results([r], "artifacts/step5/results.json")
```

It rejects a label count that disagrees with the point count, labels other than
0/1, and scores outside 0..1, and coerces points to `int` tuples so numpy scalars
are safe. Files are sorted by `(image_id, region_id)`, so re-running produces an
identical file.

`labels` is the one addition. The document's `prompt_sam3` treats every point as
positive, but `predict` accepts negative points and you will want them (§7), and a
mixed point set recorded as coordinates alone cannot be replayed — all-positive
gives a different mask. `labels=None` means all positive, so results written
against the document's literal format stay valid.

`points_by_image(results)` collapses these into Section 3's actual deliverable —
each image paired with a set of point locations, negatives dropped, duplicates
removed. `summarise(results)` gives points per region and total SAM calls for the
report.

### The document's `prompt_sam3`, if you already wrote against it

```python
from pointmin import prompt_sam3
mask = prompt_sam3(image, [(95, 66)], region=region)   # region optional
```

A tested wrapper with the exact published signature. It re-encodes the image on
every call, which is what §4 is about — use it to get moving, not in a loop.

---

## 4. Prompting SAM — a session, not `prompt_sam3`

The document specifies `prompt_sam3(image, points)`, taking a raw image every
call. Measured here: the encoder pass is **349 ms**, a warm decode is **12 ms** —
**28×**. An iterative loop pays that on every iteration, so the encode moved out
of the call.

```python
with h.open(image_id) as sess:            # 349 ms once
    m1 = sess.predict([(120, 64)], region=region)             # ~12 ms
    m2 = sess.predict([(120, 64), (130, 70)], region=region)  # ~12 ms
print(sess.num_sam_calls)                 # 2
```

`predict(points, labels=None, region=None, multimask=True) -> (H, W) bool`

- `points` — a list of `(x, y)` pairs, or anything reshapeable to `(N, 2)`.
- `labels` — SAM's convention, `1` positive and `0` negative. Defaults to all
  positive, so you can ignore it. It exists so you can *shrink* an over-large
  mask, which a positive-only signature cannot express. You will want this: see
  the `cars` row in §7.
- `region` — pass it and the best of SAM's candidate masks is chosen by IoU
  against `region.gt_mask`. This is the "already picks the best one by comparing
  against ground truth" clause from the document; the original signature had
  nowhere to put the ground truth. Omit it and SAM's own score decides.
- Returns exactly one mask, never a list.

Out-of-bounds points raise `ValueError` rather than returning something wrong.
That is on purpose — it catches a transposed `(y, x)` immediately instead of
silently on the 256×256 tiles where a transpose is invisible.

### `FakeSession` — same contract, no model

```python
with h.open_fake(image_id) as sess:
    mask = sess.predict([(x, y)], region=region)
```

Each positive point reveals a disk of whichever region it lands in, sized so one
point recovers roughly half a compact region and more points recover more.
Negative points subtract a radius-6 disk. Fully deterministic. Pass
`leak_px=3` to `open_fake` and the mask spills past the region boundary, so
coverage can reach 1.0 while IoU does not — use it to test the branch that has to
notice an over-large mask.

Write your loop against `FakeSession`, then swap `open_fake` for `open`. Nothing
else changes.

---

## 5. Coordinate conventions — the part that silently corrupts results

| Thing | Convention |
| --- | --- |
| Point order | `(x, y)` = (column, row), origin top-left |
| Indexing a mask | `mask[y, x]` — the order flips |
| Array shape | `(H, W)` = (rows, cols) |
| `bbox` | `(x, y, w, h)` |
| Units | integer pixels, never normalised 0–1 |
| Image | `np.ndarray (H, W, 3)` uint8 **RGB**, not BGR, not a path |

DLRSD tiles are 256×256, so a transposed point is undetectable by eye. Every
coordinate test in `tests/test_coords.py` therefore runs on a deliberately
non-square 40×90 image. If you add coordinate handling, test it the same way.

---

## 6. The harness API you will actually use

```python
h = Harness()                        # Config() defaults; pass Config(...) to change
h.image_ids()                        # all 2100 ids
h.categories()                       # {"harbo": [ids...], "beach": [...], ...}
h.load_image(image_id)               # (H, W, 3) uint8 RGB
h.load_labels(image_id)              # (H, W) uint8, values 1..17
h.regions(image_id)                  # scored regions — the Section 4 hand-off
h.unrecognized(image_id)             # just the ones needing points
h.gt_regions(image_id, min_area=24)  # unscored, needs no model at all
h.automatic_masks(image_id)          # (N, H, W) bool, cached on disk
h.open(image_id) / h.open_fake(image_id)
h.score(gt_mask, pred_mask)          # -> (coverage, iou)
h.is_recognized(coverage, iou)       # both thresholds, per the document
```

`Harness()` builds SAM lazily, so `gt_regions`, `score` and `is_recognized` work
with no checkpoint present. Only `regions`, `automatic_masks` and `open` need it.

### Loading regions from disk instead

`artifacts/step1/` holds the pilot's scored regions, bit-packed, a few kB each:

```python
from pointmin import load_dataset
regions_by_image = load_dataset("artifacts/step1")   # dict[str, list[Region]]
```

Identical `Region` objects, no checkpoint, no GPU, no inference. Use this for
Steps 2–3 so we are both looking at the same numbers.

---

## 7. What the Step 1 pilot actually found — read this before designing

Ten images, one per land-use category, 179 regions, SAM 1 ViT-B, thresholds
coverage ≥ 0.90 and IoU ≥ 0.75. Full numbers in `artifacts/step1/step1_report.json`.

**131 of 179 regions (73%) are unrecognized.** That is your workload. Only 2 have
no matching automatic mask at all — the overwhelming majority are *partial*
failures, so an algorithm that only handles "SAM found nothing" addresses about
1.5% of the problem.

Mean coverage 0.765, mean IoU 0.637 (`greedy_union`). Per class:

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

Three things worth designing around:

**`cars`: coverage 0.875 but IoU 0.661, and only 4 of 29 recognized.** SAM finds
the cars and then spills well past them. `freew_301#5` is the clearest instance:
59 px of cars, coverage **1.000**, IoU **0.376**. Adding positive points cannot fix
this — that is what `labels=0` is for, and it works: one positive point inside
that region plus one negative point on a spilled pixel takes IoU from 0.373 to
**0.615** with coverage still at 1.000.

**Sprawling background classes are the hard cases,** not the small objects.
`bare soil`, `pavement`, `dock` and `grass` are worst. A single DLRSD `grass`
component snakes between buildings, and SAM has no reason to return that as one
mask. Region `harbo_401#0` is a dock of 9907 px that fills 15% of its bounding
box, coverage 0.418.

**Thresholds are provisional and the choice is enormously consequential.** At
coverage ≥ 0.80 / IoU ≥ 0.50, 54% of regions are already recognized; at
0.95 / 0.80, only 8%. The whole grid is in the report JSON. Calibrating this is
Step 6, jointly, from your Step 4–5 numbers — don't tune to it in the meantime.

---

## 8. Matching, and why there are two modes

A ground-truth region often corresponds to several automatic masks, because SAM
splits one object into parts. Two modes, in `Config.match_mode`:

- `best_single` — highest-IoU single mask. 17.3% recognized, mean coverage 0.668.
- `greedy_union` — start there, keep adding whichever mask improves IoU most,
  stop when none does. **26.8% recognized, mean coverage 0.765. This is the
  default.** 2.79 masks per region on average.

`best_single` would mark a correctly-but-piecewise segmented region as
unrecognized and send you to fix something that is not broken.

---

## 9. Backend status

The plan says SAM 3. `facebook/sam3` is gated on Hugging Face and this machine
has no approved token, so **SAM 1 ViT-B is what runs** — per Jatin's instruction
to use what is available. `Sam3Backend` is written and `build_backend` prefers it;
granting access makes it the default with no other code change, and nothing in
this document is backend-specific. Absolute numbers in §7 would change; the
interface would not.

---

## 10. If you need to change any of this

Section 7 of the plan: stop and say so before changing the shape of `Region`,
`PointSelectionResult`, or `predict`. Adding a field is cheap; changing the
meaning of one is not. Tests in `tests/` pin the conventions above — if a change
is safe, they stay green.
