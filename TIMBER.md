# Timber Instance Segmentation — Progress

Status of the Timber board segmentation work. Living document.
Technical reference (schema, thresholds, commands) lives in
`ann_pipeline/timber/README.md`; this file records what was tried and why.

Last updated: 2026-09-18.

## Goal

CT scans of **8 sawn timber boards** stacked in a 2×4 grid inside the scanner
bore. Segment each board and keep them **separate as instances** — semantic
segmentation is insufficient because all boards share one class.

Data: Supervisely project **382882 `Timber`**, dataset `01`, 150 frames of
778×778 PNG, 32 of them human-annotated. Local mirror at
`/mnt/D/datasets/ct_log/382882_timber/01/`.

## Status: solved on the annotated set

| metric | value |
|---|---|
| precision / recall / F1 | **1.000 / 1.000 / 1.000** |
| exact-8 frames | **32/32** |
| mean instance IoU | 0.958 |
| id agreement with GT | **256/256 (100%)** |
| frames with all 8 ids | 149/150 |

Deployed detector: `threshold_components_v2`. Purely classical — threshold,
connected components, morphological opening, shape gate. **No SAM, no trained
detector, no propagation model.** The boards are air-separated and move
sub-pixel between slices, so the problem does not need them.

Uploaded to Supervisely as dataset **`Phase 1` (id 1169571)**, 150 images /
1197 objects, human GT preserved bit-exact on the 32 anchor frames.

## What the data turned out to be

Three findings, each of which changed the approach:

1. **9 boards in frame, only 8 annotated.** A wide plank above the grid is
   never labelled. Its *area* overlaps the real boards (7974–9792 vs
   6839–8788px), so size cannot reject it — it needs a shape gate. Without one,
   precision is 0.90 and exact-8 accuracy 6%.

2. **Class titles are persistent board ids.** The numeric titles
   (`14 17 18 79 128 131 141 143`) identify the same physical board across
   slices. Instance identity was already in the labels — no tracking model
   needed, only propagation to unannotated frames.

3. **Boards are air-separated.** GT union splits into exactly 8 connected
   components on every annotated frame, with zero overlap. This is what makes
   the classical route viable at all.

## What was tried

| approach | outcome |
|---|---|
| threshold + top-8 components | precision 0.90 — grabs the unannotated plank every frame |
| \+ area band | no help; plank area overlaps real boards |
| \+ aspect gate (≤3.5) | F1 0.994, exact-8 0.94 |
| \+ morphological opening (r=3) | F1 0.998, exact-8 0.97 |
| watershed on distance transform | **worse** (F1 0.994, precision 0.992) — fragments the plank into board-sized pieces |
| opening + watershed combined | identical to opening alone; watershed never fires |
| **long-side gate only, no aspect** | **F1 1.000, exact-8 32/32** |

### The aspect-gate trap

Frame `033.png` resisted every fix, and the diagnosis was wrong twice. It looked
like two merged boards, so watershed seemed indicated — but tracing the pipeline
showed board `141` was already a *clean separate component*. It simply retained
a sliver of the plank after the opening, inflating its aspect to 3.62, just past
the 3.5 cutoff.

Measured post-opening across all 32 frames:

| gate | true boards | plank fragments | separates? |
|---|---|---|---|
| aspect | max **3.62** | min **3.29** | no — overlapping |
| long side | max **211.8** | min **245.8** | yes — 34px margin |

The opening had *changed which feature discriminates*. Aspect was load-bearing
before it and actively harmful after. Dropping it for a long-side gate at 228
(mid-gap) gave perfect detection.

**Lesson:** re-derive gate thresholds from the TP/FP feature split after any
change to upstream morphology, rather than tuning the existing gate.

## Known gaps

- **`182.png` is underexposed** — max intensity 146 vs the usual 250, mean 7.0.
  Three boards fall below the t=30 threshold; only 5 are detected and labelled.
  Uploaded as-is for annotator review. A per-frame adaptive threshold (Otsu, or
  a percentile of nonzero intensity) is the likely fix. **Not yet checked how
  many other frames are dim** — worth a scan of the intensity distribution
  across all 150 before deciding.
- **All thresholds are tuned on one scan** (`01`, 8 boards, this bore geometry).
  `max_long_side=228`, `min_px`/`max_px`, and t=30 are all scan-specific. A new
  scan with different board dimensions needs them re-derived, not reused.
- **Validation is on 32 frames from a single dataset.** The perfect scores mean
  the classical approach fits this scan, not that it generalizes. The 118
  unannotated frames were inspected visually but have no GT.
- **`labelerLogin` does not survive upload.** Supervisely overwrites it with the
  uploading account, so human-vs-auto provenance is not visible in the UI. An
  image-level tag (the `bumaska` pattern from `PHASES.md`) would carry it.
- **Mean IoU ~0.958 may be at the annotation floor**, not a detector limit — GT
  coverage of detected components runs 0.92–0.99, suggesting the human masks sit
  slightly inside the true board edges. Worth confirming before chasing the
  remaining 4%.

## If this needs to go further

In rough order of value:

1. Adaptive thresholding for dim frames (closes `182.png`, cheap).
2. Provenance tag on anchor frames, so reviewers see what is human-verified.
3. Validate on a second scan before trusting the thresholds as defaults.
4. Only if 1–3 expose real shape failures: SAM2 point-prompted per board, or a
   YOLO-OBB/seg model reusing the `ann_pipeline/knot` prep → train flow.
   Nothing measured so far justifies either.

## Code

All under `ann_pipeline/timber/`:

| file | purpose |
|---|---|
| `data.py` | load slices, per-instance mask composition from Supervisely JSON |
| `detectors.py` | detector variants; `threshold_components_v2` is deployed |
| `eval.py` | Hungarian instance matching, IoU / count / semantic metrics |
| `track.py` | propagate board ids from anchors to all 150 frames |
| `visualize.py` | plotly overlays |
| `render_dataset.py` | per-frame PNGs for inspection |
| `render_tracks.py` | per-frame PNGs with ids drawn |
| `upload_phase1.py` | build local export, upload to Supervisely |
