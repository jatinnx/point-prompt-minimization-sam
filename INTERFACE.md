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

No GPU and no weights? Replace two lines:

```python
from pointmin import load_dataset
regions_by_image = load_dataset("artifacts/step1_sam3_text")   # scored regions, from disk
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
| | | *— the document ends here; the rest are additive —* |
| `area_px` | `int` | `gt_mask.sum()`, precomputed |
| `bbox` | `(x, y, w, h)` | **x and y first**, matching the coordinate convention |
| `class_name` | `str` | e.g. `"buildings"` — for printing, never for logic |
| `class_region_index` | `int` | ordinal among blobs of the *same class* |
| `matched_mask_indices` | `tuple[int, ...]` | which automatic masks were matched |

Convenience: `region.key` is `f"{image_id}#{region_id}"`, and `region.summary()`
returns a JSON-safe dict with no arrays in it.

Want the document's dict and nothing else? `region.to_dict(spec_only=True)` returns
exactly the eight fields above that marker, in that order, with the additive five
dropped. The tuple is `pointmin.SPEC_FIELDS`, and it is the same tuple
`scripts/04_handoff_check.py` asserts against, so the two cannot drift apart.

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
    points=[(96, 67), (98, 62)],   # final list, AFTER pruning
    labels=[1, 0],                 # additive; omit entirely if all positive
    final_coverage=0.983,
    final_iou=0.624,
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
mask = prompt_sam3(image, [(96, 67)], region=region)   # region optional
```

A tested wrapper with the exact published signature. It re-encodes the image on
every call, which is what §4 is about — use it to get moving, not in a loop.

---

## 4. Prompting SAM — a session, not `prompt_sam3`

The document specifies `prompt_sam3(image, points)`, taking a raw image every
call. Measured here on SAM 3: the first prompt of a session costs **~1050 ms**, an
extra prompt inside that open session **~25 ms** — **40×**. An iterative loop pays
that gap on every iteration, so the image encode moved out of the call.

```python
with h.open(image_id) as sess:
    m1 = sess.predict([(120, 64)], region=region)             # ~1050 ms, encodes
    m2 = sess.predict([(120, 64), (130, 70)], region=region)  # ~25 ms
print(sess.num_sam_calls)                 # 2
```

SAM 3 encodes lazily inside the first `predict` rather than inside `open()`, so the
first call of a session is the expensive one, not `open()` — time your loop
accordingly. Building the model costs a further ~2.9 s, once per process, inside
the first thing that touches it.

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
with no weights present. Only `regions`, `automatic_masks` and `open` need them.

### Loading regions from disk instead

`artifacts/step1_sam3_text/` holds the pilot's scored regions, bit-packed, a few kB
each:

```python
from pointmin import load_dataset
regions_by_image = load_dataset("artifacts/step1_sam3_text")   # dict[str, list[Region]]
```

Identical `Region` objects, no weights, no GPU, no inference. Use this for
Steps 2–3 so we are both looking at the same numbers.

---

## 7. What the Step 1 pilot actually found — read this before designing

Ten images, one per land-use category, 179 regions, **SAM 3 prompted with the 17
class names**, thresholds coverage ≥ 0.90 and IoU ≥ 0.75. Full numbers in
`artifacts/step1_sam3_text/step1_report.json`, the run itself in
`step1_console.txt` beside it.

**154 of 179 regions (86%) are unrecognized. That is your workload.** Of those,
**71 (40% of all regions) have no matching mask at all** — SAM returned nothing
under that region's own class name. The remaining 83 are partial failures: a mask
exists, it is just wrong.

You therefore need both halves of the plan, in roughly equal measure. Steps 2–3
(choose a first point with nothing but the region to go on) carry the 40% where
there is no mask to improve on. Step 4 (place the next point in the remaining
missed area) carries the rest.

> Earlier versions of this section said only 2 regions (1.5%) had no mask at all,
> and told you an algorithm for that case addressed 1.5% of the problem. Those were
> numbers from the class-agnostic pass, where SAM is given the image and no names.
> Under SAM 3 + the 17 class names — which is the hand-off — that figure is 40%, and
> the advice inverts. If you designed against the old paragraph, this is the change
> that matters most.

Mean coverage 0.458, mean IoU 0.363 (`greedy_union`). Per class:

| class | regions | recognized | mean coverage | mean IoU |
| --- | --- | --- | --- | --- |
| field | 1 | 1 | 0.963 | 0.963 |
| ship | 15 | 12 | 0.945 | 0.850 |
| cars | 29 | 3 | 0.823 | 0.565 |
| pavement | 9 | 2 | 0.644 | 0.625 |
| water | 10 | 0 | 0.628 | 0.183 |
| dock | 1 | 0 | 0.565 | 0.543 |
| buildings | 35 | 3 | 0.372 | 0.333 |
| trees | 41 | 2 | 0.271 | 0.243 |
| grass | 23 | 2 | 0.250 | 0.211 |
| bare soil | 14 | 0 | 0.027 | 0.027 |
| sand | 1 | 0 | 0.000 | 0.000 |

Four things worth designing around:

**A name SAM does not return is a region you get nothing for.** Instances returned
per class name, against ground-truth regions of that class: `water` 3 for 10,
`bare soil` 4 for 14, `trees` 31 for 41, `sand` 1 for 1 but with zero overlap. The
five names absent from these ten images (airplane, chaparral, court, mobile home,
tanks) were never returned, which is correct, not a failure. `buildings` 40 for 35
and `field` 21 for 1 are the opposite problem — over-answering, which is where
spill comes from.

**Spill is still the `cars` story, and negative points still fix it — but placing
them is part of your Step 4.** `labels=0` (§4) is the only tool that can shrink an
over-large mask; more positive points cannot. Measured on SAM 3, one positive at
the region's deepest interior pixel, then one negative chosen two different ways:

| region | 1 positive | + negative at deepest spill pixel | + negative at spill pixel farthest from the region |
| --- | --- | --- | --- |
| `freew_301#5` cars, 59 px | cov 1.000 IoU 0.518 | 0.763 / 0.517 | **0.983 / 0.624** |
| `build_1601#6` cars, 73 px | cov 0.973 IoU 0.542 | 0.767 / 0.602 | **0.877 / 0.610** |
| `overp_1501#4` grass, 2440 px | cov 0.798 IoU 0.600 | **0.954 / 0.792** | 1.000 / 0.545 |
| `spars_101#13` trees, 1896 px | cov 0.974 IoU 0.593 | 0.694 / 0.678 | **0.823 / 0.681** |

Six of those eight negatives improved IoU, so the mechanism works — but neither
placement rule wins everywhere. "Farthest from the region" wins on the small car
regions and degenerates to an image corner on the large grass one. Record whichever
you use: `PointSelectionResult.labels` exists because a mixed point set replayed as
all-positive gives a different mask.

**Sprawling background classes are the hard cases, not the small objects.**
`bare soil`, `grass`, `trees` and `buildings` are worst, and they are worst for two
different reasons — SAM often does not return the name at all, and when it does,
one DLRSD component snaking between other objects is not a thing SAM has any reason
to return as one mask. `harbo_401#0` is a dock of 9907 px filling 15% of its
bounding box.

**Thresholds are provisional and the choice dominates the headline.** At
coverage ≥ 0.80 / IoU ≥ 0.50, 30.2% of regions are already recognized; at
0.90 / 0.75, 14.0%; at 0.95 / 0.80, 5.6%. The whole 4×5 grid is in the report
JSON. Calibrating this is Step 6, jointly, from your Step 4–5 numbers — don't tune
to it in the meantime.

### The other side of the toggle, for comparison only

The same model, the same 10 images and the same 179 regions, with the class names
withheld — SAM 3 given the image alone and asked to segment everything on a 16×16
point grid, matched class-agnostically:

| | SAM 3, image only | SAM 3, text (17 names) |
| --- | --- | --- |
| unrecognized | 149 / 179 (83%) | **154 / 179 (86%)** |
| no matching mask at all | 19 (10.6%) | **71 (40%)** |
| masks matched per region | 1.84 | 0.83 |
| mean coverage / IoU | 0.542 / 0.445 | 0.458 / 0.363 |

Text mode is harder on purpose: SAM has to *name* the region, not merely outline
something that overlaps it. The two columns are **not** interchangeable and should
never be averaged or quoted side by side as progress. `artifacts/step1_sam3/` holds
the image-only numbers if you want them; `artifacts/step1_sam3_text/` is the
hand-off.

---

## 8. Matching, and why there are two modes

A ground-truth region often corresponds to several masks, because SAM splits one
object into parts. Two modes, in `Config.match_mode`:

- `best_single` — highest-IoU single mask. Mean coverage 0.440, mean IoU 0.348.
- `greedy_union` — start there, keep adding whichever mask improves IoU most,
  stop when none does. **Mean coverage 0.458, mean IoU 0.363. This is the
  default.**

`best_single` would mark a correctly-but-piecewise segmented region as
unrecognized and send you to fix something that is not broken.

In text mode the choice barely moves the headline: both modes recognize the same
25 of 179 regions, because restricting candidates to one class name has already
done most of the narrowing — 0.83 masks matched per region, against 1.84 when the
names are withheld and the two modes differ 16.2% → 16.8%. Keep
`greedy_union` regardless: it is strictly better on the regions where several
same-name instances do tile one component, and Step 6 may move the thresholds to
where the difference matters again.

---

## 9. The recognition toggle — one model, asked two ways

The plan says SAM 3, and **SAM 3 is the only model here.** Everything in §7 was
produced with it. There is no second backend: `--recognition` decides what SAM 3 is
*told*, not which model runs. On `Config`, or as a flag on `scripts/01_inspect.py`
(Step 0) and `scripts/02_harness_report.py` (Step 1):

| | what SAM 3 is given | artifacts |
| --- | --- | --- |
| `--recognition text` | the image **and the 17 DLRSD class names** | `step0_sam3_text/` `step1_sam3_text/` — **the hand-off** |
| `--recognition automatic` | the image **alone**, segment-everything on a 16×16 point grid | `step0_sam3/` `step1_sam3/` |

`--recognition text` is the project's actual design: SAM gets the image and the 17
DLRSD class names, nothing else — no points, no boxes, no ground truth — and a
region counts as recognized **only if masks returned for its own class name** cover
it. `automatic` is class-agnostic: any blob that overlaps a region can credit it,
which asks whether SAM can outline the thing, not whether it can name it.

`--recognition auto` (the default) means text whenever the model can be handed a
phrase, so plain `Config()` gives you the hand-off configuration. A model that
cannot take a phrase — which in practice means a stub in `tests/` — is refused
rather than silently downgraded: `Harness.recognition` raises `ValueError` and
`concepts.concept_masks` raises `TypeError` if you reach it directly. The two
modes' numbers are far apart (§7), and quietly swapping one for the other would
corrupt every comparison built on top.

Both of SAM 3's prompt paths are exercised, and they are separate models behind one
id:

- **concept branch** — `Sam3Model` + `Sam3Processor`, for the 17 names. One image
  encode serves all 17 prompts.
- **tracker branch** — `Sam3TrackerModel` + `Sam3TrackerProcessor`, for the `(x, y)`
  points your Steps 2–5 send through `sess.predict`. Text never reaches it.

`scripts/04_handoff_check.py --with-sam` runs both and is how you confirm the point
path works on your machine.

The weights are in the project at **`SAM-modals/sam3`** (3.3 GB, gitignored,
revision in `REVISION.txt`), so a run needs neither the Hugging Face cache nor the
network — verified under `HF_HUB_OFFLINE=1`. `Config().sam3_model_id` resolves to it;
set `POINTMIN_SAM3_PATH` if your copy lives elsewhere, and with neither the code
falls back to the gated hub id `facebook/sam3`. See README, "Where the weights live".

Text-mode masks are cached under `artifacts/concept_masks/`, keyed by the prompt
template and score thresholds, so a Step 6 retune cannot be served stale masks.
Automatic masks cache the same way under `artifacts/auto_masks/`. Neither key
includes the model path, so moving the weights invalidates nothing. Neither cache
is committed; both regenerate.

---

## 10. If you need to change any of this

Section 7 of the plan: stop and say so before changing the shape of `Region`,
`PointSelectionResult`, or `predict`. Adding a field is cheap; changing the
meaning of one is not. Tests in `tests/` pin the conventions above — if a change
is safe, they stay green.
