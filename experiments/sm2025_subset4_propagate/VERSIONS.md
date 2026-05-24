# Subset 4 combined pipeline — version history

Each row = one Supervisely dataset under project 376641
(`SM_2025_automatic_annotations`). Newest at top.

## auto_4_obb_augmented_propagation_v2 — dataset id 1139775
**Date**: 2026-05-24. **Knot strategy**: OBB-augmented propagation + anatomical filters.

What changed vs v1:
- **Anatomical filters** added to `build_combined_annotations.py`:
  - `--knot_min_px 150` (up from 50): noise floor. Real knots in this dataset
    are 500-1500 px; CCs <150 px are statistically junk.
  - `--pith_exclusion_px 40`: drop CCs whose centroid lies within 40 px of
    the pith point, or that contain the pith point. Catches "knot wraps
    pith" propagation drifts (e.g. image #4, #5 review feedback).
  - `--min_eccentricity 0.7`: drop CCs with regionprops eccentricity below
    0.7. Real knots are 0.83-0.99 (elongated radial blobs). The "cross-knot"
    propagation artifact on page 38 had ecc 0.327 — cleanly rejected.

Stats: 200 knots / 110 frames (0.68/frame, vs 262 / 139 in v1). 3 frames
flagged for pith review.

How to regenerate:
```
conda run -n ct-log python -m experiments.sm2025_subset4_propagate.build_combined_annotations \
    --npz experiments/sm2025_subset4_propagate/out/result_obb_aug_ellipse.npz \
    --knot_source prop_cc \
    --knot_min_px 150 \
    --pith_exclusion_px 40 \
    --min_eccentricity 0.7 \
    --dataset_name auto_4_obb_augmented_propagation_v2 \
    --out_dir /tmp/sm2025_subset4_obb_aug_v2 \
    --upload
```

Status: shipped. Likely the new baseline — awaiting Rostislav's review.

---

## auto_4_obb_augmented_propagation_v1 — dataset id 1139769
**Date**: 2026-05-24. **Knot strategy**: single-class MedSAM2 video propagation
seeded with anchor GT + YOLO-OBB ellipse rasters on non-anchor frames.

What changed vs v5 (paradigm shift):
- **Knots now come from propagation, not from the SAM2 image predictor.**
  We feed the MedSAM2 video predictor the same 45 anchor GT masks AND YOLO-OBB
  detections at every non-anchor frame where YOLO fires conf≥0.40. All knot
  seeds are merged into a single tracked object (single-class semantics);
  per-instance bitmaps are split via connected components only at the final
  Supervisely encoding stage.
- **Seed shape is an inscribed ellipse**, not the OBB rectangle. Rectangular
  seeds caused SAM2 to copy the rectangle into the output (rect-shaped masks).
  Ellipse aligns with natural knot shape — much cleaner output. Implemented
  in `run_obb_augmented.py` as `--seed_shape ellipse` (default).
- Pith and Wood pipelines unchanged from v5.

Visual verdict (compared to v5 baseline): tighter shapes on many frames, and
recovers some knots OBB+SAM2 image-predictor missed. Some artifacts remain
(blobs near pith, occasional cross/star shapes) — addressed in v2 filters.

Stats from the propagation output (`result_obb_aug_ellipse.npz`):
- 160 frames with knots (vs 82 in original prop, 162 in rect-seed variant)
- Mean 1446 px/frame-with-knot (vs 2584 in original prop)
- Total 231k knot pixels (vs 212k in original prop, 296k in rect variant)

Final encoded counts: 262 knots / 139 frames (0.90/frame).

Files added/changed:
- `experiments/sm2025_subset4_propagate/run_obb_augmented.py` — new
  propagation script. Doesn't modify `run.py` so we can A/B both.
- `experiments/sm2025_subset4_propagate/out/result_obb_augmented.npz` —
  rectangle-seed propagation output.
- `experiments/sm2025_subset4_propagate/out/result_obb_aug_ellipse.npz` —
  ellipse-seed propagation output (the one used for v1/v2).
- `experiments/sm2025_subset4_propagate/compare_obb_aug.py` — diagnostic
  side-by-side (original prop / aug prop / v5).
- `experiments/sm2025_subset4_propagate/build_combined_annotations.py` —
  added `--knot_source prop_cc` flag.

How to regenerate the propagation:
```
conda run -n ct-log python -m experiments.sm2025_subset4_propagate.run_obb_augmented \
    --seed_shape ellipse --out_name result_obb_aug_ellipse.npz
```

Status: superseded by v2 (same recipe + anatomical filters).

---

## auto_4_combined_v5 — dataset id 1139719
**Date**: 2026-05-23. **Knot strategy**: YOLO-OBB + SAM2 mask_input prior, **conf=0.40**.

What changed vs v4:
- Just bumped `--yolo_conf_knot` from 0.25 to 0.40. Holdout eval on the OBB
  path showed precision jumps from 0.60 (conf=0.25) to **1.00 (conf=0.40)**
  with recall only dropping 0.69 → 0.62. F1 effectively tied with AABB
  baseline (0.76 vs 0.78) but mask quality much better (IoU 0.62 vs 0.54).

Trade: fewer knots but every knot is real. Matches Rostislav's stated
preference ("missing object is faster to add than a wrong one is to fix").

Stats: 247 knots / 125 frames (0.85/frame, vs 1.01 in v3). 5 frames flagged
for pith review.

Caveat: holdout eval has n=13 GT knots — F1 diffs of 0.02 are within noise.
The visible quality difference (tighter masks, no FPs) is the more reliable
signal.

How to regenerate:
```
conda run -n ct-log python -m experiments.sm2025_subset4_propagate.build_combined_annotations \
    --yolo_obb_weights /home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_obb_v1/weights/best.pt \
    --yolo_conf_knot 0.40 \
    --dataset_name auto_4_combined_v5 \
    --out_dir /tmp/sm2025_subset4_combined_v5 \
    --upload
```

Status: shipped. Likely the new baseline.

---

## auto_4_combined_v4 — dataset id 1139718
**Date**: 2026-05-23. **No functional change vs v3** — added safety filters
that turned out to be no-ops on the current pipeline.

What changed vs v3 (in code, not in output):
- Knot masks now intersected with the final wood mask before keeping (drops
  any pixel outside the log boundary).
- Drop knots smaller than 50 px after intersection.
- Configurable via `--knot_min_px`.

Empirical result: 0/295 knots dropped (all OBB+mask_input knots were already
inside wood and ≥50 px). Filter retained as a safety net for future weight
changes that might regress.

Stats: 295 knots / 142 frames — bit-identical to v3.

Status: superseded by v5. Kept for the filter-code lineage.

---

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

Holdout F1 0.64 at conf=0.25 (precision 0.60, recall 0.69, IoU 0.62). The
conf=0.25 default carried over from AABB pipeline turned out too low for
the more-confident OBB head — bumping to 0.40 (v5) cleans up the FPs.

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

Status: superseded by v5 (same recipe + conf=0.40).

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

| dataset | knots | knot conf | pith | wood |
|---|---|---|---|---|
| auto_3_anchors_from_rosta | propagation | — | propagation blob (mask) | propagation only |
| auto_3_anchors_from_rosta_v2 | propagation | — | propagation centroid | threshold ∪ propagation |
| auto_4_propagation_45anchors | propagation (45 anchors) | — | propagation centroid | propagation only |
| auto_4_combined_v1 | propagation (45 anchors) | — | YOLO bbox (16-anchor model) | threshold ∪ propagation |
| auto_4_combined_v2 | YOLO+SAM2 (45-anchor model, AABB) | 0.25 | YOLO bbox (45-anchor model) | threshold ∪ propagation ∪ knot ∪ pith |
| auto_4_combined_v3 | YOLO-OBB+SAM2 (mask_input prior) | 0.25 | YOLO bbox (45-anchor model) | threshold ∪ propagation ∪ knot ∪ pith |
| auto_4_combined_v4 | (= v3 + wood-intersect + 50px filter, no-op) | 0.25 | YOLO bbox (45-anchor model) | as v3 |
| auto_4_combined_v5 | YOLO-OBB+SAM2 (mask_input prior) | 0.40 | YOLO bbox (45-anchor model) | as v3 |
| auto_4_obb_augmented_propagation_v1 | MedSAM2 video, anchor GT + OBB ellipse seeds → CCs | 0.40 (OBB conf for seeds) | YOLO bbox (45-anchor model) | as v3 |
| **auto_4_obb_augmented_propagation_v2** | **MedSAM2 video, anchor GT + OBB ellipse seeds → CCs + anatomical filters** | **0.40 (OBB conf for seeds)** | **YOLO bbox (45-anchor model)** | **as v3** |

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

7. **OBB needs higher conf threshold than AABB**. At conf=0.25 the OBB model
   produced more FPs than AABB (precision 0.60 vs 0.90) because OBB val mAP50
   is much higher (0.91 vs 0.50) — the model is more confident, so 0.25 lets
   through detections AABB wouldn't have made. At **conf=0.40** OBB hits
   precision=1.0 with recall=0.62, F1=0.76 (vs AABB F1=0.78). Effectively
   tied F1, but with better mask quality (IoU 0.62 vs 0.54). Shipped as v5.

8. **Wood-intersect + min-size knot filter is a no-op on the OBB pipeline**:
   all 295 v3 knots were already inside wood and ≥50 px. Filter kept as a
   safety net (v4).

9. **Holdout eval n=13 GT knots is statistically tiny**: F1 differences of
   ±0.02 between configurations are likely within noise. Trust visible mask
   quality and Rostislav's review over single-digit F1 differences.

10. **OBB-augmented propagation > v5 visually**. Injecting YOLO-OBB rasters
    as additional knot seeds at non-anchor frames doubles frame-with-knot
    coverage (82 → 160) and tightens average mask 29% (2584 → 1446 px).
    Single-class knot semantics in MedSAM2 video predictor avoids identity-
    tracking complexity. Shipped as augmented_propagation_v1/v2.

11. **OBB seed shape matters: ellipse beats rectangle**. Rect seeds get
    copied verbatim into SAM2 output ("rectangular masks" artifact, image
    #3). Inscribed ellipse matches natural knot shape — much cleaner.

12. **Anatomical filters are cheap, effective post-processors**. Three rules
    catch all the propagation-drift artifacts seen in v1:
    - drop CCs with eccentricity <0.7 (catches cross/star shapes)
    - drop CCs within 40 px of pith centroid (catches pith-blooms)
    - drop CCs <150 px (catches noise blobs)
    Page 38 cross artifact had ecc 0.327; legit knots are 0.83-0.99.

# Open questions

- Rostislav's verdict on augmented_propagation_v2 vs v5: which visual quality wins?
- Holdout eval on the OBB-augmented propagation pipeline (currently only v5
  has been holdout-evaluated). Need to retrain OBB without holdouts and
  rerun augmented propagation excluding holdout anchors.
- Small-knot recall: 0/n matched at IoU≥0.3 by either method on the holdout
  eval. Open whether to push for more small-knot annotations or change
  inference resolution.
- Holdout eval stability: rerun on a different 9-page split to see if OBB vs
  AABB difference holds — current single-split numbers are within noise.
- Pith keypoint head: would a `yolo11n-pose` model give sub-pixel pith
  localisation vs the current bbox-centroid approach?
- Wood UNet: replace threshold (fails on dark slices) with a small UNet
  trained on the 45 wood annotations.
- Seed-every-N-frames experiment: does sparser OBB seeding (every 2-3 non-
  anchor frames) improve or hurt augmented propagation? Not tested.

# How to continue

Pure pipeline runs:
```
# Current best — OBB-augmented propagation v2 (anatomical filters)
# Requires result_obb_aug_ellipse.npz from run_obb_augmented.py first.
conda run -n ct-log python -m experiments.sm2025_subset4_propagate.build_combined_annotations \
    --npz experiments/sm2025_subset4_propagate/out/result_obb_aug_ellipse.npz \
    --knot_source prop_cc \
    --knot_min_px 150 \
    --pith_exclusion_px 40 \
    --min_eccentricity 0.7 \
    --dataset_name auto_4_obb_augmented_propagation_vN \
    --out_dir /tmp/sm2025_subset4_obb_aug_vN \
    --upload

# Regenerate the augmented propagation npz (only needed if YOLO-OBB weights changed):
conda run -n ct-log python -m experiments.sm2025_subset4_propagate.run_obb_augmented \
    --seed_shape ellipse --out_name result_obb_aug_ellipse.npz

# Previous baseline — v5 (pure YOLO-OBB+SAM2 image predictor)
conda run -n ct-log python -m experiments.sm2025_subset4_propagate.build_combined_annotations \
    --yolo_obb_weights ann_pipeline/out/knot_runs/yolo11n_obb_v1/weights/best.pt \
    --yolo_conf_knot 0.40 \
    --dataset_name auto_4_combined_vN \
    --out_dir /tmp/sm2025_subset4_combined_vN \
    --upload
```

Diagnostics (no upload, fast):
```
# OBB-augmented vs original prop vs v5 — non-anchor frames
conda run -n ct-log python -m experiments.sm2025_subset4_propagate.compare_obb_aug \
    --pages 36 37 42 53 89 95 128 200 --conf 0.40

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
