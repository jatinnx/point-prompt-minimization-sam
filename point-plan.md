# Point Prompt Minimization for SAM — Team Project Plan

**Team:** Raven (Tushar Venkat Challa) + Jatin
**Purpose of this document:** so both of us — and anyone else who reads this later (TA, professor) — know exactly what we're building, why, how the work is split, and how we talk to each other about it. Written in plain language on purpose, so neither of us has to guess what the other meant.

---

## 1. What problem are we actually solving?

**Professor's official problem statement:**

> Given an image and one or more target objects, determine a minimal set of point prompts — both their count and their spatial placement — such that SAM produces a complete segmentation mask.

**In plain terms:** SAM (Segment Anything Model) is a tool that, if you click a point on an object in an image, draws a mask (outline) around that object. It's very good at this, but it doesn't always get everything right on its own. We want to figure out: for a given image, what is the *smallest number of clicks*, placed in the *smartest possible spots*, such that SAM ends up correctly segmenting the whole image?

Think of it like this: imagine you're teaching someone to trace a coloring book page by pointing at spots with your finger. You want to give them the fewest possible finger-points needed for them to color in the whole page correctly. That's the problem, translated to SAM and images.

**Why does this matter?** Because getting a human to manually draw a full outline (a dense mask) around every object in every image is expensive and slow. But getting a human to click a few points is fast and cheap. If we can figure out *which few points* are enough, we save a huge amount of annotation effort — while (later, in Part 2, not our job right now) still training a good segmentation model.

---

## 2. Key terms, explained simply

- **SAM3** — the specific version of the Segment Anything Model we're using. Give it an image and either points or run it in "automatic" mode (no input at all, it tries to find everything on its own).
- **DLRSD** — the dataset we're using. It's a set of aerial/satellite images (like from a drone or satellite) that already come with a "ground truth" — a hand-labeled map of exactly which pixels belong to which class (building, road, water, tree, etc.) for every image.
- **Ground truth (GT)** — the "correct answer." Since DLRSD already has this, we can use it to check how good SAM's guesses are, and to figure out where points should go, without needing a human to check every image by hand.
- **Mask** — a black-and-white (or True/False) image the same size as the original, where "True"/white pixels mean "this pixel belongs to the object," and "False"/black means "it doesn't."
- **Region** — one connected blob of a single class in the ground truth. E.g., if there are two separate buildings in an image, that's two regions (even though both are class "building").
- **Coverage** — of the pixels that *should* be part of a region (according to GT), what percentage did SAM actually find? (100% = SAM found the whole thing.)
- **IoU (Intersection over Union)** — a stricter way of comparing two masks: how much they overlap divided by how much space they take up combined. Punishes SAM more if it draws a mask that's the right general area but wrong shape (too big, too small, wrong boundary).
- **Recognized / Unrecognized region** — a region is "recognized" if SAM's automatic output already covers it well enough (both `coverage` and `IoU` clear some threshold we pick). Otherwise it's "unrecognized" — this is the part we need to fix with points.
- **Point prompt** — a single (x, y) pixel location we feed to SAM to say "there's something here, please segment it."
- **Pruning** — after we've found a set of points that works, checking whether we can remove any of them and *still* have it work. This is what makes our final point set small instead of just "big enough."

---

## 3. What is our actual deliverable?

**Confirmed with the TA:**

- The final output is a **folder of images, each paired with a set of point locations** — no class labels attached to the points. Just `(image, [(x1,y1), (x2,y2), ...])` for every image in DLRSD.
- We are **not** building a model that predicts points on brand-new images without ground truth. We're allowed to use ground truth (since we already have it for DLRSD) to figure out the best points — this is sometimes called an "oracle" approach, and it's the correct approach for what's been asked of us.
- "Unrecognized" means **both** (a) SAM produces nothing there at all, and (b) SAM produces a mask but it's incomplete or wrong. Both count as regions we need to fix.
- What happens after we deliver this (having someone manually label the chosen points, training a model on those labels, comparing against a fully-supervised model) is a **separate, later phase (Part 2)** that is *not* our concern right now. We are explicitly not building that yet.

---

## 4. How we're splitting the work

We're a team of two, and rather than one of us waiting on the other to finish before starting, we're splitting the pipeline **vertically** — by component, not by "who does the first half / second half in sequence." This means we can both start building today.

### Jatin — Data & Evaluation Infrastructure

**Job, in plain terms:** load the DLRSD images and their ground truth, run SAM3 on them, and build the scoring system that tells us "is this region okay, or does it need fixing?"

**Concretely, builds:**

1. Code to load a DLRSD image + its ground truth annotation.
2. A wrapper around SAM3's "automatic mode" (run it on an image with no prompts, get back a set of masks).
3. Code to break the ground truth into **regions** (connected components per class).
4. Code to match each GT region to the best-overlapping SAM mask.
5. Code to compute `coverage` and `IoU` for each region, and mark it recognized/unrecognized.

**Hands off to Raven:** a list of `Region` objects (exact format in Section 5 below) for a given image.

### Raven — Point Selection Algorithm

**Job, in plain terms:** given one "problem region" that SAM got wrong, figure out where to click (and how many times) to fix it, and then clean up that click-list so it's as small as possible.

**Concretely, builds:**

1. A way to generate the *first* candidate point for an unrecognized region (start simple: center of the region).
2. An iterative loop: place a point, ask SAM3 to try again with all points so far, check if it's good enough now, and if not, find the next point (in the *remaining missed area*, not randomly).
3. A stopping rule (region is good enough, OR we've hit a safety cap on points).
4. A **pruning** step: once we have a working point set, try removing points one at a time and see if it still works — throw away any point that wasn't actually necessary.

**Hands back to Jatin (well — to the final combined pipeline):** for each region, the final small list of points — see `PointSelectionResult` format in Section 5.

---

## 5. The interface — exactly what data passes between us

This is the most important section. If we don't agree on this exactly, our two halves won't fit together later. **Both of us should read this and confirm/edit it before writing any real code.**

### What Jatin hands to Raven — a `Region`

```python
Region = {
    "image_id": str,          # filename WITHOUT extension, e.g. "airplane_00"
    "class_id": int,          # DLRSD class index (e.g. 3 = "buildings")
    "region_id": int,         # which connected blob of that class, in this image (0, 1, 2, ...)
    "gt_mask": np.ndarray,    # full-image boolean array, shape (H, W). True = pixel belongs to this region
    "matched_sam_mask": np.ndarray | None,  # SAM's current best-guess mask for this region, or None
    "coverage": float,        # 0.0 to 1.0 — how much of gt_mask is covered by matched_sam_mask
    "iou": float,             # 0.0 to 1.0 — intersection-over-union of gt_mask and matched_sam_mask
    "status": str,            # "recognized" or "unrecognized"
}
```

### What Raven hands back — a `PointSelectionResult`

```python
PointSelectionResult = {
    "image_id": str,
    "region_id": int,
    "points": list[tuple[int, int]],   # final list of (x, y) points, AFTER pruning
    "final_coverage": float,
    "final_iou": float,
    "num_sam_calls": int,   # how many times SAM3 had to be called to get here (useful for our report)
}
```

### The function Raven will call constantly — `prompt_sam3`

Jatin provides this function; Raven's whole algorithm is built around calling it in a loop.

```python
def prompt_sam3(image: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    """
    image: numpy array, shape (H, W, 3), RGB, uint8 — the raw image, not a file path
    points: list of (x, y) pixel coordinates — all treated as "positive" prompts
             (i.e. "something is here", not "this is NOT here")
    returns: boolean numpy array, shape (H, W) — SAM3's single best mask for this prompt set
             (if SAM3 gives back multiple candidate masks, this function already
             picks the best one by comparing against ground truth — Raven doesn't
             need to do that comparison)
    """
```

### Small but important conventions — agree on these explicitly

| Thing | Our convention |
| --- | --- |
| Image ID | Filename without extension (e.g. `"airplane_00"`, not `"airplane_00.tif"`) |
| Coordinates | Pixel units (integers), NOT normalized 0–1 floats |
| Coordinate order | `(x, y)` = (column, row) — `x` is left-to-right, `y` is top-to-bottom, origin at top-left |
| Mapping a point to a mask array | `mask[y, x]` — note the order flips! numpy arrays are indexed (row, column) = (y, x) |
| Image format passed around | Raw numpy array (`H x W x 3`, uint8, RGB) — not a file path, so we don't risk loading/preprocessing the image slightly differently from each other |
| What counts as "region" | One connected component per class in the ground truth (we'll double check this makes sense by manually inspecting a real image first — see Section 6, Step 0) |

---

## 6. Our build plan — step by step, weakest version first

We're deliberately building the *simplest possible working version first*, then improving it — not trying to write the whole clever system in one go and debugging a giant pile of code at the end.

| Step | What happens | Who |
| --- | --- | --- |
| **0. Manual inspection** | Look at one real DLRSD image + its GT + one SAM3 run, by hand. Confirm what a "region" looks like in practice, and what a "partial" SAM mistake actually looks like. | **Both, together** |
| **1. Evaluation harness** | Build region matching + coverage/IoU scoring. Test on 5–10 images. | Jatin |
| **2. Random-point baseline** | For each unrecognized region, place one random point inside it, remeasure. Just to see if points help *at all*. | Raven (using stub/fake regions until Step 1 is ready) |
| **3. Center-point baseline** | Same as above, but the point is the region's center instead of random. | Raven |
| **4. Iterative / error-driven points** | Allow multiple points per region, each one placed in the *remaining* missed area, looping until the region is fixed or we hit a safety cap. | Raven |
| **5. Backward pruning** | Take Step 4's point sets and try removing points one at a time, keeping removals that don't break anything. | Raven |
| **6. Calibrate thresholds** | Look at real numbers from the small pilot (Steps 1–5) to properly choose our coverage/IoU thresholds and point cap — not guess them upfront. | Both |
| **7. Full run** | Run the finished pipeline (Jatin's harness + Raven's selector) across the entire DLRSD dataset. | Both |

**Important:** each step above should fully work before moving to the next. If we run out of time, we still have a complete, explainable result at whatever step we reached — not a half-broken complicated system.

---

## 7. How we communicate

- **Before starting real code:** both of us read this document, mark anything we'd change, and have one short call to resolve disagreements (should take ~20–30 minutes if we've each read it beforehand).
- **Sync points, tied to finished steps (not just "every Monday"):**1. After Step 0 (joint) and once Jatin's Step 1 works on a handful of real images → quick check-in, make sure Raven's stub-based work still matches the real `Region` format.2. Once Raven's Steps 2–3 (baselines) run against Jatin's real Step 1 output → second check-in.3. Once Steps 4–5 (iterative + pruning) are done → third check-in, look at real pilot numbers together, decide thresholds (Step 6).
- **If either of us needs to change the interface in Section 5** (e.g. discover we need an extra field), that's a "stop and tell the other person immediately" moment — don't quietly change your half's data format, since the other person's code depends on it exactly as written.
- **Shared place for notes/results:** (decide together — could be a shared folder, a shared doc, or a shared GitHub repo issue/README — whatever's easiest for both of us to check regularly).

---

## 8. Open questions / things to confirm with the TA later

- Exact coverage/IoU thresholds — we're calibrating from pilot data, but worth mentioning our chosen values when we report progress.
- Whether "region" should really be per-connected-component, or whether DLRSD has cases (like sprawling roads) where this breaks down — we'll know after Section 6, Step 0.
- Point cap — same, we'll set from data rather than guessing.
