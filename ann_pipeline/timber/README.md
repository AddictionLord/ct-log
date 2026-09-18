# Timber instance segmentation

CT scans of 8 sawn timber boards stacked in a 2x4 grid inside the scanner bore.
Task: segment each board and keep them separate as instances.

Supervisely project **382882 `Timber`**, dataset `01` (150 images, 778x778 PNG).
Local root: `/mnt/D/datasets/ct_log/382882_timber/01/{img,ann}`.

## Annotation schema

32 of 150 frames are annotated. Each annotated frame has **exactly 8 bitmap
objects**, one per numeric class title (`14 17 18 79 127* 128 131 141 143`).
The class title is a **persistent timber id**, the same board across slices —
instance identity is already in the labels, no tracking needed.

Classes `timber` and `KNOT` are declared in the project meta but unused
(`127` is declared but never appears either). Only 8 ids occur in practice.

GT sanity: all 32 frames give exactly 8 instances and 8 connected components,
zero overlap except a 3px annotator artifact on `056.png`.

## Key data fact: 9 boards, 8 annotated

A wide plank sits above the 2x4 grid in every frame and is **never annotated**.
It cannot be rejected by area (7974-9792px, overlapping the true boards'
6839-8788px) but separates cleanly by shape:

| feature | true timbers | top plank |
|---|---|---|
| min-area-rect long side | 144.7-202.5 | 249.6-292.2 |
| aspect ratio | 2.0-3.3 | 3.9-7.1 |

Hence the `max_long_side=220` / `max_aspect=3.5` gates in `detectors.py`.
Without them precision is 0.90 and exact-8 accuracy 6%.

## Results (32 annotated frames, IoU thr 0.5)

| detector | exact-8 | mean inst IoU | precision | recall | F1 | semantic IoU |
|---|---|---|---|---|---|---|
| `threshold_components_split` | **0.969** | 0.959 | 1.000 | 0.996 | **0.998** | 0.955 |
| `threshold_components` | 0.938 | 0.956 | 1.000 | 0.988 | 0.994 | 0.944 |

Boards are air-separated, so intensity threshold + connected components solves
this; no SAM, no detector training needed. The `_split` variant adds a
morphological opening (r=3) that severs thin bridges between touching boards.

Remaining failure: `033.png` only — one board touches the top plank, the merged
blob is shape-rejected, so 7 of 8 are found. A per-component split (watershed
on the distance transform) would recover it if that frame matters.

## Run

```bash
python -m ann_pipeline.timber.eval --iou_thr 0.5
python -m ann_pipeline.timber.visualize --frames 033.png 091.png 110.png
```

Outputs `per_frame.csv` + `summary.csv` under `ann_pipeline/out/timber_eval/`.

## Full-dataset visual inspection

```bash
python -m ann_pipeline.timber.render_dataset
```

Writes one PNG per frame (all 150) to `ann_pipeline/out/timber_vis/`, plus
`index.csv`. GT is an orange fill (annotated frames only), predictions are cyan
outlines. Frames where `n_pred != 8` are prefixed `FLAG_` so they sort first.

Result: **148/150 frames yield exactly 8 instances.** Two flagged:

- `033.png` (annotated) — one board touches the top plank, merged blob is
  shape-rejected, 7 found.
- `182.png` (unannotated) — **underexposed scan**: max intensity 146 vs the
  usual 250, mean 7.0. Three boards fall below the t=30 threshold, 5 found.
  A data artifact near the end of the scan, not a detector flaw. A
  per-frame adaptive threshold (Otsu, or a percentile of nonzero intensity)
  would handle it if such frames need covering.
