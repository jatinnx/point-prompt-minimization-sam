# Point Prompt Minimization for SAM — data & evaluation harness

Jatin's half of the two-person project described in `point-plan.md`: **Section 6
Steps 0 and 1**, i.e. everything in Section 4 under "Jatin — Data & Evaluation
Infrastructure". Load DLRSD, run SAM's automatic mode, split the ground truth
into regions, match and score them, and decide which regions need point prompts.

Steps 2–5 — random-point baseline, centre point, iterative loop, backward
pruning — are Raven's and are deliberately **not** implemented here. Step 6
(threshold calibration) and Step 7 (full run) are joint and come later.

**Raven: start with [`HANDOFF.md`](HANDOFF.md), then [`INTERFACE.md`](INTERFACE.md).**
The first is the runbook — clone, verify, load the 179 scored regions, prompt SAM,
hand results back. The second is the field-by-field contract. Neither is this file.

---

## Running it

The interpreter is the pre-existing environment; there is no separate venv,
because a fresh torch install would not fit on this disk.

```bash
PY=/home/cse-sdpl/pytorch-env/bin/python

$PY -m pytest -q                      # 77 tests, ~3 s, no model needed
$PY scripts/00_extract_data.py        # DLRSD -> data/  (already done)
$PY scripts/01_inspect.py             # Step 0: joint manual inspection
$PY scripts/02_harness_report.py      # Step 1: harness measured on 10 images
$PY scripts/03_auto_sweep.py          # why the automatic-mode defaults are what they are
$PY scripts/04_handoff_check.py       # verify the hand-off; --with-sam to prompt too
```

Every script takes `--help`. Automatic masks are cached under
`artifacts/auto_masks/`, so a second run of anything is roughly free. Each of
`01`, `02` and `04` also has its console output saved beside its artifacts, so the
numbers are readable without a GPU.

## Layout

```
pointmin/
  class_map.py    DLRSD class names and palette — byte-identical to the shared copy
  config.py       every tunable in one dataclass; paths resolve to the project root
  dlrsd.py        image / label / colour-mask loading            (Section 4 item 1)
  sam_backend.py  Sam1Backend and Sam3Backend behind one protocol
  autoseg.py      automatic "segment everything" mode + disk cache (item 2)
  regions.py      ground truth -> connected-component Regions      (item 3)
  metrics.py      coverage, IoU, region-to-mask matching        (items 4 and 5)
  session.py      SamSession (encode once, prompt many) and FakeSession
  compat.py       the plan's literal prompt_sam3, for code already written to it
  results.py      PointSelectionResult — the return leg of the interface
  harness.py      the one object the point-selection half imports
  store.py        bit-packed Region persistence, so Raven needs no GPU
  viz.py          the Step 0 panel figures
scripts/          00 extract, 01 Step 0, 02 Step 1, 03 settings sweep, 04 hand-off check
tests/            77 tests: coordinates, metrics, regions, store, cache, results, compat, config
artifacts/        step0/, step1/, auto_masks/  (generated; step0+step1 are committed)
data/dlrsd/       images/ labels/ colour/  — 2100 each, 264 MB (not committed)
```

## Data

DLRSD, 2100 aerial tiles at 256×256, 17 classes, no background class and no
ignore index — every pixel is one of 1..17.

`scripts/00_extract_data.py` pulls three directories out of `dlrsd.zip` by name
rather than unpacking all 21 GB:

| source | destination | format |
| --- | --- | --- |
| `full_images/` | `data/dlrsd/images` | RGB PNG |
| `full_1cmasks/` | `data/dlrsd/labels` | mode **L**, pixel = class id 1..17 |
| `full_masks/` | `data/dlrsd/colour` | mode **P**, DLRSD palette, for figures only |

`labels` and `colour` are not interchangeable. `colour` is paletted, so
`cv2.IMREAD_GRAYSCALE` on it returns luminance, not class indices — a silent
corruption. `dlrsd.load_labels` cross-checks cv2 against PIL on every read and
raises if they disagree.

## The hand-off

What Section 4 says Jatin gives Raven is "a list of `Region` objects for a given
image". Live objects would mean Raven needs the checkpoint, a GPU and an inference
pass per image, so they are written to disk instead, bit-packed, and committed:

```python
from pointmin import load_dataset
regions_by_image = load_dataset("artifacts/step1")   # 10 images, 179 scored Regions
```

188 kB, no model, no GPU, and both halves look at byte-identical regions.
`scripts/04_handoff_check.py` re-derives every stored score from the stored masks
and asserts the Section 5 contract on all 179 — run it after any pull. Procedure
and pilot numbers: `HANDOFF.md`. Field reference: `INTERFACE.md`.

## Two things chosen from measurement, not defaults

**A session instead of `prompt_sam3(image, points)`.** The encoder pass is 349 ms
and a warm decode is 12 ms — 28×. The plan's signature re-encodes on every call,
which an iterative loop pays on every iteration. `harness.open(image_id)` encodes
once. Reasoning and the full revised contract are in `INTERFACE.md` §4.

**Automatic-mode settings.** The first version reused thresholds from unrelated
prior work on this machine (24 points/side, pred-IoU 0.84, stability 0.90) and
reported mean coverage 0.575, making the task look much harder than it is.
`scripts/03_auto_sweep.py` on the pilot:

| preset | points/side | pred-IoU | stability | masks/image | recognized | mean coverage |
| --- | --- | --- | --- | --- | --- | --- |
| strict | 16 | 0.88 | 0.92 | 26 | 7.3% | 0.434 |
| prior work | 24 | 0.84 | 0.90 | 51 | 16.2% | 0.575 |
| **moderate (default)** | **32** | **0.70** | **0.85** | **109** | **26.8%** | **0.765** |
| loose | 48 | 0.60 | 0.80 | 174 | 31.3% | 0.808 |

Moderate is the default: loose costs roughly twice the runtime for 0.043 more
coverage. The recognition thresholds (coverage 0.90, IoU 0.75) are **untouched
and provisional** — calibrating those is Step 6 and belongs to both of us.

## What Step 0 and Step 1 found

Step 0, six images picked to stress the region definition (dense residential,
harbour, freeway, agricultural, parking lot, beach) — figures in
`artifacts/step0/`. Step 1, ten images, one per land-use category, 179 regions —
`artifacts/step1/step1_report.json`. Answering Section 8's open questions:

**Is one connected component per class the right unit?** Mostly yes, with a
caveat. No region in the sample is thin enough to be erased by a 3×3 erosion, so
every region has an interior to place a point in. But sprawl is real:
`dense_1401#23` is 4955 px of pavement filling 8% of its bounding box, and
`harbo_401#0` is a 9907 px dock at 15%. These are the low-coverage regions. The
definition holds; the *scoring* is what strains on them.

**Is a 24 px floor sensible?** Yes, and it barely matters. Across the six
stress images: 124 regions at floor 0, 122 at 24, 109 at 96, and the floor of 24
discards 0.00% of labelled pixels. It removes specks without changing anything
that counts.

**What does a partial failure look like?** It is the normal case, not the
exception. 131 of 179 regions are unrecognized, but only **2** have no matching
automatic mask at all. The rest are found-but-wrong: right area, incomplete or
spilling. The last panel of each Step 0 figure splits pixels into found (green),
missed (red) and spilled (blue), which is what makes that visible.

One case worth keeping in mind, from the Step 0 table: region `freew_301#5`, 59
px of cars, has coverage **1.000** and IoU **0.376** — SAM found every pixel of it,
inside a mask nearly three times too big. Nine of the 122 Step 0 regions clear
coverage 0.95 and still fail on IoU. Coverage alone would have called all of them
successes; requiring both thresholds, as the plan does, is right.

`greedy_union` matching (add masks while IoU improves) beats `best_single` by
17.3% → 26.8% recognized and 0.668 → 0.765 mean coverage, because SAM routinely
returns one ground-truth object as several masks.

## Backend

The plan specifies SAM 3. `facebook/sam3` is gated on Hugging Face and this
machine has no approved token, so **SAM 1 ViT-B runs instead**, using the
checkpoint already on disk rather than a seventh copy of it. `Sam3Backend` is
written and `build_backend` prefers it; approval makes it active with no other
code change. Nothing above the backend protocol is model-specific.

`SAM1_CHECKPOINT_CANDIDATES` in `config.py` are this machine's paths. Elsewhere,
set `POINTMIN_SAM1_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth`. Nothing that only
reads the hand-off needs a checkpoint at all.

## Conventions worth knowing before editing

- `class_id` is native DLRSD **1..17**, not the 0..16 remap used elsewhere on this
  machine. Use `dlrsd.class_name(class_id)`; never index `CLASS_NAMES` directly.
- Points are `(x, y)` = (column, row); masks index as `mask[y, x]`. Tiles are
  square, so a transpose is invisible — coordinate tests deliberately use a
  non-square 40×90 image.
- `region_id` is unique per image, not per class. `INTERFACE.md` §2 explains why
  the plan's wording cannot be satisfied literally.
- The automatic-mask cache key includes the settings that produced the masks.
  Changing a threshold invalidates the cache instead of silently reusing it.
