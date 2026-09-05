# HANDOFF.md — Jatin → Raven

Steps 0 and 1 are done. This is the runbook for turning them into the input of
your Steps 2–5: what you get, how to load it, how to call SAM, what to hand back,
and what the pilot numbers say you should design around.

`INTERFACE.md` is the reference — every field, every convention, every deviation
from `point-plan.md`. This document is the procedure. Read this first, then that.

---

## 1. What you are getting

Everything below is committed, so `git clone` is the whole install as far as the
hand-off is concerned. No weights and no GPU needed to consume it.

| Path | What it is |
| --- | --- |
| `artifacts/step1_sam3_text/regions/*.npz` | **The hand-off.** 10 images, 179 scored `Region` objects, masks bit-packed. 146 kB total |
| `artifacts/step1_sam3_text/index.json` | What produced them: backend, recognition mode, match mode, thresholds, area floor |
| `artifacts/step1_sam3_text/step1_report.json` | Full Step 1 numbers: distributions, per class, per class *name*, threshold grid, per image |
| `artifacts/step1_sam3_text/step1_console.txt` | The Step 1 run, verbatim — the same numbers, readable |
| `artifacts/step1_sam3_text/handoff_check_console.txt` | `04_handoff_check.py --with-sam` passing, SAM 3's point path included |
| `artifacts/step1_sam3_text/coverage_vs_iou.png` | Where the 179 regions sit against the provisional thresholds |
| `artifacts/step1_sam3_text/example_results.json` | A valid `PointSelectionResult` file, so §6 is copyable not aspirational |
| `artifacts/step0_sam3_text/*.png` | Step 0's ten figures — image, GT, regions, the masks the 17 names returned, found/missed/spilled |
| `artifacts/step0_sam3_text/step0_summary.json` | Per-region shape stats and the minimum-area sweep |
| `artifacts/step0_sam3_text/step0_console.txt` | The Step 0 run, verbatim, including which names SAM stayed silent on |
| `pointmin/` | The harness. `Harness`, `Region`, sessions, metrics, concepts, the store |
| `scripts/04_handoff_check.py` | Verifies the hand-off is intact. Run it before you write anything |

Produced with **SAM 3 handed the image and the 17 DLRSD class names and nothing
else** — no points, no boxes, no ground truth — `greedy_union` matching,
`min_region_area=24`, and **provisional** thresholds coverage ≥ 0.90 / IoU ≥ 0.75.

`INTERFACE.md` §9 documents the toggle. There is one model; the other side of it is
the same SAM 3 given the image and **no** class names, and
`artifacts/step0_sam3/` + `artifacts/step1_sam3/` are what that produced on the same
ten images. Do not read one against the other as progress: §8 has both columns and
why they are not comparable.

---

## 2. First: prove the hand-off arrived intact

```bash
git clone <repo> && cd Point-project
pip install -r requirements.txt          # numpy, opencv, matplotlib, pytest
python -m pytest -q                      # 92 passed
python scripts/04_handoff_check.py       # no GPU, no weights
```

`04_handoff_check.py` re-derives `coverage` and `iou` from the stored masks and
compares them to the stored floats, checks `bbox` against the actual mask extent,
and asserts all eight Section 5 fields exist on all 179 regions. If it prints
`hand-off OK`, the two halves agree. If it fails, that is my bug — send me the
line it printed. Run it again after every pull. Verified from a bare clone: 92
tests pass and the check prints `hand-off OK` with no DLRSD tiles, no weights and
no GPU.

`--in` defaults to `artifacts/step1_sam3_text`, the hand-off. Point it at
`artifacts/step1_sam3` and you are checking the image-only side of the toggle
instead; the first two lines of its output tell you which one you got.

The 2100 DLRSD tiles are **not** in the repository — 264 MB of dataset that does
not belong in git. You do not need them to consume the hand-off, because every
region carries its own `gt_mask` and `matched_sam_mask`. You need them only to
prompt SAM for real (§4), and by then you need the weights too — 3.3 GB that are
also not in the repository, see §9. Tell me and I will get you both.

Its actual output on this machine, so you know what "intact" looks like — the full
run, with the 71 region keys spelled out, is in
`artifacts/step1_sam3_text/handoff_check_console.txt`:

```
hand-off: /home/cse-sdpl/Videos/Point-project/artifacts/step1_sam3_text
  backend=sam3 recognition=text match_mode=greedy_union
  10 images, 179 regions, 154 unrecognized
  Region contract holds for all 179 regions (8 required fields, scores recomputed
  from the masks, bbox checked against gt_mask extent)

workload per image
  image            regions  unrec  mean cov  mean IoU
  agric_1901             2      1     0.918     0.918
  baseb_1000            16     14     0.349     0.312
  build_1601            18     16     0.687     0.516
  dense_1401            32     30     0.250     0.239
  freew_301             21     21     0.467     0.302
  harbo_401             24     12     0.836     0.594
  mediu_1201            22     22     0.341     0.278
  overp_1501            11     10     0.262     0.203
  river_1301            12     11     0.206     0.189
  spars_101             21     17     0.544     0.476

71 regions have no matching mask at all (SAM returned nothing under this region's
own class name): baseb_1000#1, baseb_1000#2, ..., spars_101#18
the rest are partial failures -- a mask exists, it is just wrong

model-free check  FakeSession on agric_1901#0 at (134, 133): cov=0.503 iou=0.503
hand-off OK
```

Add `--with-sam` and it also opens a real session and prompts SAM 3's tracker
branch through `sess.predict` — the exact call your loop makes. That run prints a
`transformers` notice about instantiating `sam3_tracker` from a `sam3_video`
config; it is expected and harmless, and the masks that follow are real.

---

## 3. Load the regions — one function

```python
from pointmin import load_dataset

regions_by_image = load_dataset("artifacts/step1_sam3_text")   # dict[str, list[Region]]

targets = [r for rs in regions_by_image.values()
             for r in rs if r.status == "unrecognized"]
print(len(targets))                                  # 154
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
r.coverage, r.iou     # 1.0, 0.3806   what SAM managed unprompted
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

with h.open("freew_301") as sess:                # encodes the image once
    m1 = sess.predict([(96, 67)], region=region)         # ~1050 ms, pays the encode
    m2 = sess.predict([(96, 67), (98, 62)], labels=[1, 0], region=region)
    print(sess.num_sam_calls)                    # 2  -> PointSelectionResult
```

The first prompt of a session costs ~1050 ms and every extra prompt inside it
~25 ms — a 40× gap that an iterative loop would otherwise pay on every iteration.
That is the only reason the document's signature changed; `INTERFACE.md` §4 has the
full argument, including the detail that SAM 3 encodes inside the *first* `predict`
rather than inside `open()`, and that building the model costs a further ~2.9 s once
per process.

If you have already written code against `prompt_sam3(image, points)`, it runs
unchanged — the shim is real, tested, and exported:

```python
from pointmin import prompt_sam3

mask = prompt_sam3(h.load_image("freew_301"), [(96, 67)], region=region)
```

It re-encodes on every call. Fine for a one-off, wrong for a loop.

`region=` is optional and does the thing the document asked for: with it, the best
of SAM's candidate masks is chosen by IoU against `region.gt_mask`, so you never
compare candidates yourself. Without it, SAM's own confidence decides.

### You get negative points, and you will need them

`labels` follows SAM's convention: `1` positive, `0` negative, defaulting to all
positive. A positive-only interface cannot shrink a mask that is already too big,
and 26 of the pilot's 29 `cars` regions are exactly that — mean coverage 0.823
against mean IoU 0.565. Measured on `freew_301#5`:

| prompt | coverage | IoU |
| --- | --- | --- |
| SAM 3 unprompted, under the name `cars` | 1.000 | 0.381 |
| 1 positive at (96, 67), its deepest interior pixel | 1.000 | 0.518 |
| + 1 negative at (98, 62), the spill pixel farthest from the region | 0.983 | **0.624** |

Coverage was already perfect and had nothing to gain; only a negative point moved
IoU. **Where you put the negative decides whether it helps at all.** The same
positive with the negative on the *deepest* spill pixel instead gives 0.763 / 0.517
— no gain, because on a 59 px region that pixel is four pixels from the positive
and cuts into the region itself. `INTERFACE.md` §7 measures both rules across four
regions: six of eight negatives improve IoU, and neither rule wins everywhere —
farthest-from-region wins on the small `cars` regions and degenerates to an image
corner on a large `grass` one. Choosing this well is part of your Step 4.

If your result uses negatives, record them — `PointSelectionResult` has a `labels`
field for it, because a mixed point set replayed as all-positive gives a different
mask.

### No GPU? Use `FakeSession`

```python
with h.open_fake("freew_301") as sess:
    mask = sess.predict([(96, 67)], region=region)
```

Same contract, no model, fully deterministic: each positive point reveals a disk
of the region it lands in, negatives subtract one. Enough for a loop to make
progress and converge, so you can write and unit-test Steps 2–5 before touching a
weights. `open_fake(image_id, leak_px=3)` makes the mask spill past the region
so coverage hits 1.0 while IoU does not — use it to test the over-large branch.

`h.open_fake` reads the image file for its shape, so on a bare clone build the
session from the stored regions instead:

```python
import numpy as np
from pointmin.session import FakeSession

regions = regions_by_image["freew_301"]
stand_in = np.zeros((*regions[0].gt_mask.shape, 3), dtype=np.uint8)
with FakeSession(stand_in, "freew_301", regions) as sess:
    mask = sess.predict([(96, 67)], region=region)
```

Write the loop against `FakeSession`, then change one word to `open`.

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

Both functions work with no weights, so your stopping rule is testable without
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
        points=[(96, 67), (98, 62)],     # AFTER pruning
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
points_by_image(results)    # {'freew_301': [(96, 67)], ...}  <- Section 3's deliverable
```

`points_by_image` drops negative points and de-duplicates, because Section 3's
final output is "each image paired with a set of point locations, no class labels
attached". That is the shape Step 7 writes out.

`artifacts/step1_sam3_text/example_results.json` is a valid file if you want to
see one.

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

for image_id, regions in load_dataset("artifacts/step1_sam3_text").items():
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
that image's regions — reopening per region would pay the ~1.1 s encode 179 times
instead of 10. And `sess.num_sam_calls` counts calls since the session opened, so
if you share a session across regions like this, subtract the count you had before
the region started to get a per-region figure.

---

## 8. What the pilot found, and what it means for your design

Ten images, one per land-use category, 179 regions, **SAM 3 prompted with the 17
class names**, `greedy_union`, coverage ≥ 0.90 / IoU ≥ 0.75. Full numbers in
`artifacts/step1_sam3_text/step1_report.json`, the run itself in
`step1_console.txt` beside it.

**154 of 179 regions (86%) are unrecognized — that is your workload.** Of those,
**71 (40% of all regions) have no matching mask at all**: SAM returned nothing
under that region's own class name. The other 83 are partial failures — a mask
exists, it is just wrong.

You therefore need both halves of the plan in roughly equal measure. Steps 2–3 —
choosing a first point with nothing but the region to go on — carry the 40%. Step 4
— placing the next point in the remaining missed area — carries the rest.

> An earlier version of this section said 2 regions (1.5%) had no mask, and told
> you that an algorithm for that case addressed 1.5% of the problem. Those were
> numbers from the class-agnostic pass, where SAM is given the image and no names.
> Under SAM 3 + the 17 class names — which is the hand-off — it is 40% and the
> advice inverts. If you designed against that paragraph, this is the change that
> matters most.

**Placement matters more than count.** `04_handoff_check.py --with-sam` puts one
deliberately stupid point — the first interior pixel in raster order, i.e. a
boundary pixel — on the largest unrecognized region of each image:

```
  agric_1901#1   trees      area=  3754 point=(0, 0)     auto: cov=0.873 iou=0.873  ->  1 point: cov=0.823 iou=0.823
  baseb_1000#0   bare soil  area= 24276 point=(22, 0)    auto: cov=0.382 iou=0.382  ->  1 point: cov=0.002 iou=0.002
  build_1601#4   buildings  area= 20265 point=(247, 120) auto: cov=0.650 iou=0.650  ->  1 point: cov=0.693 iou=0.692
  dense_1401#11  buildings  area= 12055 point=(0, 17)    auto: cov=0.398 iou=0.398  ->  1 point: cov=0.102 iou=0.082
  freew_301#13   pavement   area= 50060 point=(0, 0)     auto: cov=0.801 iou=0.801  ->  1 point: cov=0.282 iou=0.282
  harbo_401#17   water      area= 19118 point=(86, 0)    auto: cov=0.883 iou=0.473  ->  1 point: cov=0.004 iou=0.004
  mediu_1201#13  grass      area= 34769 point=(0, 0)     auto: cov=0.174 iou=0.174  ->  1 point: cov=0.012 iou=0.012
  overp_1501#6   pavement   area= 43047 point=(78, 0)    auto: cov=0.000 iou=0.000  ->  1 point: cov=0.002 iou=0.002
  river_1301#9   trees      area= 38099 point=(195, 0)   auto: cov=0.000 iou=0.000  ->  1 point: cov=0.728 iou=0.666
  spars_101#4    grass      area= 16415 point=(0, 0)     auto: cov=0.656 iou=0.654  ->  1 point: cov=0.257 iou=0.256
```

Look at the `point=` column: every one of those is on an edge of the tile or of the
region, because "first interior pixel in raster order" is the worst rule that still
technically lands inside. Two rows improve (`river_1301#9` from 0.000 to 0.728
coverage, on a region SAM never named at all; `build_1601#4` slightly), and the
rest come out **worse than doing nothing**. A point on a boundary pixel lands on
whichever object owns the other side of the edge. `pointmin.interior_mask(mask)`
erodes a region so you can sample away from its border; use it, and consider
keeping the automatic mask when your point set scores worse than it did.

**Sprawling background classes are the hard cases, not the small objects:**

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

Two different failure modes hide in that table. `cars` — coverage 0.823, IoU 0.565,
3 of 29 recognized — is spill: the mask is already too big, and more positive
points cannot shrink one. That is §4's negative points. `bare soil` (0.027),
`sand` (0.000), `water` (0.183 IoU on 0.628 coverage) and `trees` (0.243) are
largely the opposite problem: SAM never returned the name, or returned instances
that do not overlap the region. Instances returned against ground-truth regions of
that class: `water` 3 for 10, `bare soil` 4 for 14, `trees` 31 for 41, `sand` 1 for
1 with zero overlap. `harbo_401#0` is a dock of 9907 px filling 15% of its bounding
box — one DLRSD component snaking between other objects is not something SAM has
any reason to return as one mask.

**Do not tune to the thresholds.** They are provisional and the choice dominates
the headline number: at coverage ≥ 0.80 / IoU ≥ 0.50, 30% of regions are already
recognized; at 0.90 / 0.75, 14%; at 0.95 / 0.80, 6%. The whole 4×5 grid is in the
report JSON. Step 6 calibrates it jointly, from your Step 4–5 numbers.

**Step 0 settled the two open questions you depend on.** Across all 179 regions,
**0 erode to nothing** under a 3×3 kernel — every region has an interior pixel to
put a point in, roads and docks included. So one connected component per class
holds up as the unit of work. And the 24 px area floor barely matters: 184 regions
at floor 0, 179 at 24, and 100.00% of labelled pixels retained either way.

### The other side of the toggle, for comparison only

Same model, same 10 images, same 179 regions, with the class names withheld — SAM 3
given the image alone and asked to segment everything on a 16×16 point grid, matched
class-agnostically:

| | SAM 3, image only | SAM 3, text (17 names) |
| --- | --- | --- |
| unrecognized | 149 / 179 (83%) | **154 / 179 (86%)** |
| no matching mask at all | 19 (10.6%) | **71 (40%)** |
| masks matched per region | 1.84 | 0.83 |
| mean coverage / IoU | 0.542 / 0.445 | 0.458 / 0.363 |

Text mode is harder on purpose: SAM has to *name* the region, not merely outline
something that overlaps it. The two columns are **not** interchangeable and must
never be averaged or quoted side by side as progress. `artifacts/step1_sam3/` holds
the image-only numbers; `artifacts/step1_sam3_text/` is the hand-off.

Note the shape of the difference: withholding the names costs SAM almost nothing in
*recognized* regions (30 against 25 of 179) but a great deal in regions it finds
nothing for at all (19 against 71). Naming is the hard part, outlining is not.

---

## 9. If you need more images, or different ones

The 10 pilot images are one per land-use category. To regenerate the hand-off with
different or more images — needs a GPU and the 3.3 GB of weights, so ask me if you
do not have that set up:

```bash
python scripts/02_harness_report.py --images harbo_401 beach_1101   # specific ones
python scripts/02_harness_report.py --n 21 --out artifacts/step1_full  # all categories
python scripts/04_handoff_check.py --in artifacts/step1_full
```

Then `load_dataset("artifacts/step1_full")`.

With no `--out`, the directory is named after the mode — `artifacts/step1_sam3_text`
for the hand-off, `artifacts/step1_sam3` for the image-only side. That is
deliberate: a bare run in one mode cannot overwrite the other's numbers under a name
that no longer says which mode they came from.

What SAM returned is cached on disk — text-mode masks under
`artifacts/concept_masks/`, automatic-mode masks under `artifacts/auto_masks/` —
keyed by the settings that change the output (the prompt template and score
thresholds for text mode, the point grid and filters for automatic), and not by
where the weights live. So re-running is cheap, moving the model invalidates
nothing, and changing a setting is never silently served a stale mask. Neither cache
is committed; both regenerate. Be warned about the image-only side specifically: a
*cold* automatic pass at 16 points/side is about **4.3 minutes per 256×256 tile** on
this GPU, so a fresh 10-image run is around 45 minutes and `03_auto_sweep.py` is
hours. Text mode is far cheaper.

The weights are in the project at **`SAM-modals/sam3`** — 3.3 GB, gitignored, so
they are *not* in the clone. `Config().sam3_model_id` resolves to that directory if
it is there, honours `POINTMIN_SAM3_PATH` ahead of it, and otherwise falls back to
the gated hub id `facebook/sam3`. To fetch your own copy:

```bash
hf auth login                                            # facebook/sam3 is gated
hf download facebook/sam3 --local-dir SAM-modals/sam3
```

With that copy present nothing touches the network or the shared Hugging Face
cache — verified here with `HF_HUB_OFFLINE=1` and a bogus `HF_HOME`.

Nothing that only reads the hand-off needs any of it — `load_dataset`, `h.score` and
`h.is_recognized` all work with no model present. Only `h.open`, `h.regions`,
`h.concept_masks` and `h.automatic_masks` do.

Live regions instead of stored ones, if you do have a GPU:

```python
h.regions("harbo_401")        # scored, needs the model
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

**`matched_sam_mask` can be `None`.** **71 of the 179 regions**, because SAM
returned nothing at all under that region's own class name. Guard it before using
it as an array; `coverage` and `iou` are both 0.0 in that case, and those 71 are
the regions where Steps 2–3 have nothing to improve on and must place a first point
from the region alone.

**A region's `coverage`/`iou` describe what SAM managed unprompted, not your
prompt.** They are the baseline you are trying to beat. Recompute with `h.score`
after every predict.

---

## 11. Sync checklist

Section 7 of the plan makes an interface change a "stop and tell the other person
immediately" moment, so:

- `pytest -q` → 92 passed, and `scripts/04_handoff_check.py` → `hand-off OK`. If
  either breaks after a pull, that is on me; send me the failing line.
- Adding a field to `Region` or `PointSelectionResult` is cheap. Changing what an
  existing one means is not — message me before, not after.
- Sync point 2 in the plan is "Raven's Steps 2–3 run against Jatin's real Step 1
  output". You can hit that now: `load_dataset("artifacts/step1_sam3_text")` is
  that output.
- Thresholds stay where they are until Step 6, when we set them together from your
  Step 4–5 numbers. Report distributions, not just recognized counts.

One thing to know before you quote any absolute number: **SAM 3 produced everything
here, in text mode, from the image and the 17 class names.** There is one model. The
other side of the toggle is the same SAM 3 with the names withheld, class-agnostic
(`artifacts/step0_sam3/`, `artifacts/step1_sam3/`), and `INTERFACE.md` §9 documents
it. The interface is identical either way; only the numbers move — §8 puts the two
side by side.






