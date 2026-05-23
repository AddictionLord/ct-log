# Subset 4 combined pipeline — version history

Each row = one Supervisely dataset under project 376641
(`SM_2025_automatic_annotations`). Newest at top.

## auto_4_combined_v3 — dataset id 1139713
**Date**: 2026-05-23. **Knot strategy**: YOLO-OBB + SAM2 with mask_input prior.

What changed vs v2:
- New `yolo11n-obb` model trained on full 45-anchor data (single-class knots).
  Val mAP50 = 0.931 vs 0.505 for the AABB-knot head — OBB matches the
  radially-elongated knot shape much better.
- SAM2 prompting: rasterise the YOLO OBB → 128×128 logits as `mask_input`
  prior (correct size for MedSAM2 t512; not 256). AABB-of-OBB used as the box
  prompt. Mask hard-clipped to AABB-of-OBB after SAM2.
- Pith pipeline unchanged from v2 (uses `yolo11n_v2_all45` AABB model).
- Wood union unchanged (threshold ∪ propagation_wood ∪ knot_mask ∪ pith).

Stats: 295 knots / 142 frames (1.01/frame). 5 frames flagged for pith review.

Files added/changed for v3:
- `ann_pipeline/knot/data_prep_obb.py` — single-class OBB label exporter
  (uses `cv2.minAreaRect` per knot CC).
- `ann_pipeline/out/knot_yolo_obb/` — generated OBB dataset.
- `ann_pipeline/out/knot_runs/yolo11n_obb_v1/` — trained OBB weights.
- `experiments/sm2025_subset4_propagate/build_combined_annotations.py` —
  added `yolo_obb_sam_knots()` and `--yolo_obb_weights` flag.
- `experiments/sm2025_subset4_propagate/fake_obb_test.py` — GT-derived OBB
  diagnostic that motivated v3. Shows mask_input prior wins by IoU ~0.70 vs
  ~0.56 for AABB baseline on 3 test frames.

How to regenerate:
```
conda run -n ct-log python -m experiments.sm2025_subset4_propagate.build_combined_annotations \
    --yolo_obb_weights /home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_obb_v1/weights/best.pt \
    --dataset_name auto_4_combined_v3 \
    --out_dir /tmp/sm2025_subset4_combined_v3 \
    --upload
```

Status: shipped. Waiting for Rostislav's visual review.

---

## auto_4_combined_v2 — dataset id 1139705
**Date**: 2026-05-23. **Knot strategy**: YOLO-AABB + SAM2 (box-only prompt).

What changed vs v1:
- Knots switched from propagation to YOLO+SAM2 (motivated by holdout eval
  showing YOLO+SAM2 wins detection F1 0.78 vs 0.55 with NMS iou=0.5).
- New YOLO weights `yolo11n_v2_all45` trained on **all 45** subset-4
  anchors + subsets 1, 2 (was 16-anchor `yolo11n_2cls_v1`).
- Wood union now includes the YOLO knot mask (so wood encloses every
  annotated knot).
- YOLO NMS iou=0.7 → 0.5 (PR sweep finding: removes 4/5 duplicate FPs).

Stats: 270 knots / 133 frames (0.92/frame). 5 flagged for pith review.

**Known problem**: SAM2 produces over-grown masks for many knots — it fills
whole radial wedges of wood grain that fall inside loose YOLO bboxes. Median
knot size 3254 px, but ~50 knots are 5000–15000 px (plausible knot range is
~500–3000 px). This is what v3 fixes.

Files added/changed for v2:
- `experiments/sm2025_subset4_propagate/build_combined_annotations.py` —
  refactored knot path to YOLO+SAM2, added NMS knob.
- `ann_pipeline/out/knot_runs/yolo11n_v2_all45/` — trained AABB weights
  (knot mAP50=0.50, pith mAP50=0.96).
- `experiments/sm2025_subset4_propagate/holdout_eval.py` — 80/20 holdout
  eval that motivated v2 (see PLAN.md).
- `experiments/sm2025_subset4_propagate/visualize_holdout_grid.py` — TP/FP/FN
  panels for eyeballing.
- `experiments/sm2025_subset4_propagate/compare_sam_prompts.py` — diagnostic
  for SAM2 prompt variants; revealed that point-based negatives plus bbox
  clipping help but don't fix the wedge-bleed.

Status: superseded by v3 because of mask-blob problem.

---

## auto_4_propagation_45anchors — dataset id 1139090
**Date**: 2026-05-22. **Pure propagation** with the refreshed 45 anchors.

What it is:
- Just the MedSAM2 video-predictor propagation across the 292-frame volume,
  seeded by 45 hand-annotated anchors (up from 16).
- Average anchor gap dropped from ~18 frames to ~6 frames.

No YOLO. No threshold wood. No combination logic. This is the propagation
output that all subsequent v2/v3 pipelines layer on top of.

Files: `run.py`, `upload.py`, `out/result.npz` (the cached 292-frame label
volume that v2 and v3 read).

Status: still active. Used as input to v2/v3.

---

## auto_4_combined_v1 — dataset id 1138743
**Date**: ~2026-05-21. **First combined pipeline**.

Recipe:
- Knot = propagation (each CC = one Knot object).
- Pith = YOLO bbox centre at conf=0.10 (production weights `yolo11n_2cls_v1`,
  16-anchor model) + propagation fallback + `[REVIEW: ...]` description flag.
- Wood = largest_cc(threshold ∪ propagation_wood).
- τ_flag = 7.98 px (mean+3σ of YOLO-vs-prop distance).

Stats: 318 knots / 292 frames (1.09/frame). 8 frames flagged.

What was already known to be wrong at v1 time:
- YOLO trained on only 16 anchors; subset-4 leakage in old holdout.
- Knots from propagation are clean shapes but recall drops sharply with
  anchor distance (>5 frames → recall ≈ 0; LOO sweep finding).

Status: superseded by v2 (and v2 by v3). Still on Supervisely for reference.

---

## auto_3_anchors_from_rosta_v2 — dataset id 1138708
**Date**: ~2026-05-20.

Wood-includes-knot fix applied to the initial propagation upload. Pith blob
is rendered as a centroid point (not a mask). Used to address Rostislav's
review comment that the original upload had wood-holes around knots.

Status: superseded. Pure-propagation work continued via
`auto_4_propagation_45anchors`.

---

## auto_3_anchors_from_rosta — dataset id 1138173
**Date**: ~2026-05-20.

First propagation upload of subset 4 (16-anchor version). Pith was a mask
blob; wood had holes where knots/pith were. Both fixed in v2.

Status: superseded.

---

# Quick reference

| dataset | knots | pith | wood |
|---|---|---|---|
| auto_3_anchors_from_rosta | propagation | propagation blob (mask) | propagation only |
| auto_3_anchors_from_rosta_v2 | propagation | propagation centroid | union of threshold + propagation |
| auto_4_propagation_45anchors | propagation (45 anchors) | propagation centroid | propagation only |
| auto_4_combined_v1 | propagation (45 anchors) | YOLO bbox (16-anchor model) | threshold ∪ propagation |
| auto_4_combined_v2 | YOLO+SAM2 (45-anchor model, AABB) | YOLO bbox (45-anchor model) | threshold ∪ propagation ∪ knot ∪ pith |
| **auto_4_combined_v3** | **YOLO-OBB+SAM2 (mask_input prior)** | YOLO bbox (45-anchor model) | threshold ∪ propagation ∪ knot ∪ pith |

# Key learnings (chronological)

1. **Leave-one-out sweep (sweep_distance.py)**: wood propagation is distance-
   invariant (Dice 0.95-0.99 at any anchor gap), but knot propagation falls
   off a cliff at ~5 frames from nearest anchor. Pith error is noisy and
   YOLO-better.

2. **Holdout eval (holdout_eval.py)**: the original 3-frame comparison was
   misleading because most of those frames were in YOLO's train set. With a
   proper 80/20 stratified split (9 holdouts), YOLO+SAM2 wins knot F1 0.67
   vs 0.55 (vs 0.78 vs 0.55 with NMS iou=0.5), and propagation wins matched-
   mask quality (Dice 0.84 vs 0.69).

3. **yolo11s = yolo11n on this dataset** (n=96 train images). Bigger backbone
   gives no gain. Dataset size is the bottleneck.

4. **YOLO NMS iou=0.5 is the single biggest win** for the AABB pipeline (F1
   0.67 → 0.78, removes duplicate detections SAM2 over-segments).

5. **SAM2 box-only prompting bleeds into wood grain** along the diagonals of
   loose axis-aligned bboxes. Negative-point prompts at corners/edges help
   ~15-42% in pixel count but produce visually weird shapes; not the right
   fix.

6. **OBB + mask_input prior (v3) is the right fix**: the AABB-to-knot shape
   mismatch was the root cause. Fake OBB test (using GT-derived OBBs) showed
   IoU 0.70 vs 0.56 vs the AABB baseline.

# Open questions

- Does v3 visually fix the mask-blob problem? Awaiting Rostislav.
- Small-knot recall: 0/n matched at IoU≥0.3 by either method on the holdout
  eval. Open whether to push for more small-knot annotations or change
  inference resolution.
- Should v3 also clip knots by wood (intersect with wood mask)? Cheap to add
  if any v3 knot extends outside log boundary.
- Worth evaluating v3's actual F1 on the same 9-page holdout? The OBB model
  was trained on all 45 anchors (no exclusion); for a fair comparison we'd
  rebuild with holdouts excluded.

# How to continue

Pure pipeline runs:
```
# v3 (current best)
conda run -n ct-log python -m experiments.sm2025_subset4_propagate.build_combined_annotations \
    --yolo_obb_weights /home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_obb_v1/weights/best.pt \
    --dataset_name auto_4_combined_v4 \
    --out_dir /tmp/sm2025_subset4_combined_v4 \
    --upload
```

Diagnostics (no upload, fast):
```
# SAM2 prompt variant sweep (for AABB pipeline)
conda run -n ct-log python -m experiments.sm2025_subset4_propagate.compare_sam_prompts \
    --pages 39 73 154 238 239 268 --corner_inset 0.1

# Fake OBB upper bound
conda run -n ct-log python -m experiments.sm2025_subset4_propagate.fake_obb_test \
    --pages 39 154 238

# Holdout eval (rebuild propagation if YOLO weights changed)
conda run -n ct-log python -m experiments.sm2025_subset4_propagate.holdout_eval \
    --yolo_weights <weights.pt> \
    --prop_npz experiments/sm2025_subset4_propagate/out/holdout_eval/prop_pred_holdout_free.npz
```

Training:
```
# AABB knots + pith (default)
conda run -n ct-log python -m ann_pipeline.knot.data_prep --subsets 1 2 4 \
    --out_dir /home/mary/code/ct-log/ann_pipeline/out/knot_yolo_vN
conda run -n ct-log python -m ann_pipeline.knot.train \
    --data /home/mary/code/ct-log/ann_pipeline/out/knot_yolo_vN/knots.yaml \
    --model yolo11n.pt --name yolo11n_vN

# OBB knots
conda run -n ct-log python -m ann_pipeline.knot.data_prep_obb --subsets 1 2 4
conda run -n ct-log python -m ann_pipeline.knot.train \
    --data /home/mary/code/ct-log/ann_pipeline/out/knot_yolo_obb/knots_obb.yaml \
    --model yolo11n-obb.pt --name yolo11n_obb_vN
```
