# Point Prompt Minimization for SAM — data & evaluation harness

Jatin's half of the two-person project described in `point-plan.md`: **Section 6
Steps 0 and 1**, i.e. everything in Section 4 under "Jatin — Data & Evaluation
Infrastructure". Load DLRSD, ask SAM 3 what it can find given nothing but the
image and the 17 class names, split the ground truth into connected-component
regions, match and score them, and decide which regions need point prompts.

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

$PY -m pytest -q                      # 92 tests, ~4 s, no model needed
$PY scripts/00_extract_data.py        # DLRSD -> data/  (already done)
$PY scripts/01_inspect.py             # Step 0: joint manual inspection (6 stress images)
$PY scripts/02_harness_report.py      # Step 1: harness measured on 10 images
$PY scripts/03_auto_sweep.py          # Step 6 tool: sweeps the point grid. Hours, not minutes
$PY scripts/04_handoff_check.py       # verify the hand-off; --with-sam to prompt too
```

`01` and `02` carry the toggle. One model, SAM 3, asked two ways — and the default
is the hand-off, SAM 3 given the 17 class names:

```bash
$PY scripts/02_harness_report.py                                  # image + 17 names
$PY scripts/02_harness_report.py --recognition automatic          # the image alone
$PY scripts/04_handoff_check.py --in artifacts/step1_sam3_text    # this is the default
$PY scripts/04_handoff_check.py --in artifacts/step1_sam3         # the image-only side
```

With no `--out` the directory is named after the mode, so a run on one side of the
toggle cannot overwrite the other's numbers: `artifacts/step1_sam3_text` for the
hand-off, `artifacts/step1_sam3` for the image-only side.

The toggle picks the directory; it does not pick the images. A bare `01_inspect.py`
inspects six stress categories, while the committed Step 0 set is the same ten
pilot images `02` reports on — so reproducing what is in `artifacts/step0_sam3_text/`
means naming them, or the six stress figures land in there beside the ten:

```bash
$PY scripts/01_inspect.py --images agric_1901 baseb_1000 build_1601 dense_1401 \
      freew_301 harbo_401 mediu_1201 overp_1501 river_1301 spars_101
```

Every script takes `--help`. What SAM returned is cached on disk — text-mode
masks under `artifacts/concept_masks/`, automatic-mode masks under
`artifacts/auto_masks/` — so a second run of anything is roughly free. Each of
`01`, `02` and `04` also has its console output saved beside its artifacts, so the
numbers are readable without a GPU.

## Layout

```
pointmin/
  class_map.py    DLRSD class names and palette — byte-identical to the shared copy
  config.py       every tunable in one dataclass; paths resolve to the project root
  dlrsd.py        image / label / colour-mask loading            (Section 4 item 1)
  sam_backend.py  Sam3Backend: the concept branch and the tracker branch
  concepts.py     the 17 class names -> named masks + disk cache   (text mode)
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
tests/            92 tests: coordinates, metrics, regions, store, caches, concepts, results, compat, config
artifacts/        step0_sam3_text/ step1_sam3_text/ (the hand-off, image + 17 names),
                  step0_sam3/ step1_sam3/ (the image-only side of the toggle),
                  concept_masks/ auto_masks/ (caches, not committed)
SAM-modals/sam3/  the SAM 3 weights, 3.3 GB (not committed — see "Where the weights live")
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
image". Live objects would mean Raven needs the weights, a GPU and an inference
pass per image, so they are written to disk instead, bit-packed, and committed:

```python
from pointmin import load_dataset
regions_by_image = load_dataset("artifacts/step1_sam3_text")   # 10 images, 179 scored Regions
```

146 kB, no model, no GPU, and both halves look at byte-identical regions.
`scripts/04_handoff_check.py` re-derives every stored score from the stored masks
and asserts the Section 5 contract on all 179 — run it after any pull. Procedure
and pilot numbers: `HANDOFF.md`. Field reference: `INTERFACE.md`.

## Two things chosen from measurement, not defaults

**A session instead of `prompt_sam3(image, points)`.** The first `predict` of a
session costs ~1050 ms and every further prompt inside it ~25 ms — 40×. The plan's
signature re-encodes on every call, which an iterative loop pays on every
iteration; `harness.open(image_id)` encodes once. SAM 3 does that encode lazily
inside the first `predict` rather than inside `open()`, so it is the first call of
a session that is expensive, not `open()` — time a loop accordingly. Building the
model costs a further ~2.9 s once per process. Reasoning and the full revised
contract are in `INTERFACE.md` §4.

**16 points per side for `--recognition automatic`**, which governs the image-only
side only — the hand-off is text mode and never sees a point grid. It is what
produced `artifacts/step0_sam3/` and `artifacts/step1_sam3/`, so a bare image-only
run reproduces the committed numbers and hits the mask cache instead of recomputing.
Cost is quadratic in it and large in absolute terms: a cold automatic pass at
16/side is **~4.3 min per 256×256 tile** on this GPU (measured on `agric_1901` and
`dense_1401`), so 32/side would be roughly four times that and a four-preset sweep
is hours, not minutes. `scripts/03_auto_sweep.py` re-derives the point-grid /
threshold trade-off when Step 6 wants it; it has **not** been run on SAM 3, so
treat the grid settings as inherited rather than tuned. The recognition thresholds
(coverage 0.90, IoU 0.75) are likewise **untouched and provisional** — calibrating
those is Step 6 and belongs to both of us.

## What Step 0 and Step 1 found

Both steps were run with **SAM 3 given the image and the 17 DLRSD class names and
nothing else**. Step 0's figures for the ten pilot images are in
`artifacts/step0_sam3_text/`; Step 1's numbers are in
`artifacts/step1_sam3_text/step1_report.json`. The other side of the toggle — the
same ten images, the same 179 regions, SAM 3 given the image and nothing else — is
in `artifacts/step0_sam3/` and `artifacts/step1_sam3/`. Answering Section 8's open
questions:

**Is one connected component per class the right unit?** Mostly yes, with a
caveat. No region in the sample is thin enough to be erased by a 3×3 erosion — 0
of 179 — so every region has an interior to place a point in. But sprawl is real:
`dense_1401#23` is 4955 px of pavement filling 8% of its bounding box, and
`harbo_401#0` is a 9907 px dock at 15%. These are the low-coverage regions. The
definition holds; the *scoring* is what strains on them.

**Is a 24 px floor sensible?** Yes, and it barely matters. Across the ten pilot
images: 184 regions at floor 0, 179 at 24, 159 at 96, and the floor of 24 discards
0.00% of labelled pixels. It removes specks without changing anything that counts.

**What does a partial failure look like?** It is the common case, but under text
prompting it is no longer the *only* case. 154 of 179 regions are unrecognized, and
**71 of them (40%) have no matching mask at all** — SAM returned nothing under that
region's own class name. The other 83 are found-but-wrong: right area, incomplete
or spilling. The last panel of each Step 0 figure splits pixels into found (green),
missed (red) and spilled (blue), which is what makes that visible; panel 4 paints
each mask in the DLRSD colour of the name SAM filed it under, so a mislabel shows
up as a colour mismatch against panel 2.

Given the image alone, the same 179 regions give 149 unrecognized and only 19 with
no mask at all. Text mode is harder on purpose — SAM has to *name* the region, not
merely outline something overlapping it — and the two sets of numbers are not
comparable. `HANDOFF.md` §8 puts them side by side.

One case worth keeping in mind: region `freew_301#5`, 59 px of cars, has coverage
**1.000** and IoU **0.381** — SAM found every pixel of it, inside a mask nearly
three times too big. 15 of the 179 regions clear coverage 0.95 and still fail on
IoU. Coverage alone would have called all of them successes; requiring both
thresholds, as the plan does, is right. Only a negative point can shrink a mask
like that, which is why `predict` takes `labels`.

`greedy_union` matching (add masks while IoU improves) is kept as the default, but
in text mode it barely differs from `best_single` — both recognize the same 25 of
179 regions, because restricting candidates to one class name has already done most
of the narrowing (0.83 masks matched per region against 1.84 when the names are
withheld). Given the image alone the gap is real but small: 16.2% → 16.8%
recognized, 0.501 → 0.542 mean coverage.

## The toggle: one model, asked two ways

There is one model. `--recognition` decides what it is told, on `Config` and as a
flag on `01_inspect.py` and `02_harness_report.py`:

| | what SAM 3 gets | artifacts | recognized |
| --- | --- | --- | --- |
| `--recognition text` | the image **and the 17 DLRSD class names** | `step0_sam3_text/` `step1_sam3_text/` — **the hand-off** | 25 / 179 |
| `--recognition automatic` | the image **alone**, segment-everything on a 16×16 point grid | `step0_sam3/` `step1_sam3/` | 30 / 179 |

`text` is the project's actual design and the default: a region counts as
recognized only if masks returned **for its own class name** cover it. `automatic`
is class-agnostic — any blob that overlaps a region can credit it — which asks
whether SAM can outline the thing, not whether it can name it. The two columns are
not interchangeable and must never be averaged. `--recognition auto` means text
whenever the model can be handed a phrase, so plain `Config()` gives the hand-off.

SAM 3 is two models behind one id, and both paths are exercised: the **concept
branch** (`Sam3Model`) answers the 17 class names, one image encode serving all 17
prompts, and the **tracker branch** (`Sam3TrackerModel`) answers the `(x, y)` points
that `session.predict` sends. Text never reaches the tracker.

### Where the weights live

In the project, at **`SAM-modals/sam3`** — 3.3 GB, gitignored, revision recorded in
`SAM-modals/sam3/REVISION.txt`. `Config().sam3_model_id` resolves to it and both
branches load from there, so a run needs neither the Hugging Face cache (shared
with unrelated work on this machine, and holding the weights behind symlinked
blobs) nor the network: verified with `HF_HUB_OFFLINE=1` and a bogus `HF_HOME`.

Reproduce the copy on a fresh machine with

```bash
hf download facebook/sam3 --local-dir SAM-modals/sam3    # gated: `hf auth login` first
```

or keep it elsewhere and set `POINTMIN_SAM3_PATH=/path/to/sam3`; with neither, the
code falls back to the hub id `facebook/sam3`. Presence is decided by
`config.json`, so a half-finished copy falls through instead of failing deep inside
transformers. Nothing that only reads the hand-off needs the weights or a GPU.

## Conventions worth knowing before editing

- `class_id` is native DLRSD **1..17**, not the 0..16 remap used elsewhere on this
  machine. Use `dlrsd.class_name(class_id)`; never index `CLASS_NAMES` directly.
- Points are `(x, y)` = (column, row); masks index as `mask[y, x]`. Tiles are
  square, so a transpose is invisible — coordinate tests deliberately use a
  non-square 40×90 image.
- `region_id` is unique per image, not per class. `INTERFACE.md` §2 explains why
  the plan's wording cannot be satisfied literally.
- Both mask caches key on the settings that produced the masks — prompt template
  and score thresholds for text mode, point grid and filters for automatic — not on
  the model path. Changing a threshold invalidates the cache instead of silently
  serving a stale mask; moving the weights does not.
