# Subset 4 — combined pipeline plan

Captures the state after Rostislav's review of the propagation upload
(`auto_3_anchors_from_rosta_v2`, dataset id 1138173) and our YOLO-vs-propagation
head-to-head on the 3 held-out val frames.

## State of play

### What works

- **Sparse-anchor + MedSAM2 video propagation** produces good Wood/Knot/Pith
  outlines as long as the gap between anchors is ≤ ~10 frames (Rostislav's
  observation, confirmed by visual inspection on Supervisely).
- **YOLO+SAM (production pipeline, `yolo11n_2cls_v1`)** is extremely accurate
  on Pith (mean centroid 0.65 px on the 3 held-out frames) and clean on Knot
  *when it detects them* (mean 2.5 px when matched), but misses obvious knots
  on subset 3 — annotators do significant manual rework on knot fix-up.
- **Threshold-based Wood** has Dice 0.994 on the 96 GT slices it was tuned on
  but visibly collapses on low-contrast slices (see image #4 in the review
  thread).

### What's broken

- Propagation degrades sharply at >30 frames from the nearest anchor (page 201
  in the holdout: 0/3 knots detected at 31 frames out).
- Threshold Wood can produce ragged or fragmented masks on dark slices.
- Smart-tool in Supervisely **cannot refine existing masks** — confirmed by web
  search of the official docs. Only creates new ones. Workflow for fixing a
  propagated mask is: delete the wrong object, then smart-tool-create a new
  one, or use brush.

### Comparison (3 val frames, leave-3-out for propagation)

| metric | YOLO+SAM | propagation |
|---|---|---|
| pith mean dist (px) | 0.65 | 3.98 |
| knot recall | 100% (10/10) | 20% (2/10) |
| knot matched dist (px) | 2.5 | 49 |

Caveat: the 3 frames sit 4–35 frames from the nearest non-holdout anchor, so
the propagation numbers represent a mix of "close to anchor" and "far from
anchor" regimes. The leave-one-out sweep below characterises this properly.

### Leave-one-out sweep (16 pages, full volume re-propagation each)

See `out/sweep/loo_sweep.png` and `loo_sweep.csv`.

| metric | dist ≤ 4 (n=7) | dist ≥ 7 (n=9) |
|---|---|---|
| pith mean err (px) | 3.1 | 2.8 |
| knot recall | 43% | 0% |
| wood Dice | 0.975 | 0.974 |

Findings:

1. **Wood is distance-invariant**: Dice 0.95–0.99 across the full 2–52 frame
   range. Video propagation is excellent at carrying a large contiguous mask.
2. **Knot recall is a cliff at ~5 frames**: 7/16 pages with dist ≤ 4 had
   matched knots, 0/9 with dist ≥ 7 did. Knot propagation collapses fast.
3. **Pith error is dominated by per-frame variance, not distance**: even at
   the closest neighbours (2 frames) error can be 2.5 px. YOLO at 1.0 px (val)
   is unambiguously better; the distance signal here is weak.

Implication for anchor strategy: **wood/pith are tolerant of large gaps;
knots need anchors every ≤5 frames** to propagate reliably.

## Plan

Three orthogonal refinements proposed and discussed; ordered by value-to-effort.

### 3a. YOLO-led pith with propagation fallback + disagreement flag

The data unambiguously favours YOLO for pith. The propagation pith blob is
useful as a recall fallback and as an anomaly signal.

Per-frame decision rule:

```
yolo_pith = YOLO predict at conf >= τ_yolo  (e.g. 0.5)
prop_pith = centroid of propagated pith blob

if yolo_pith and prop_pith:
    d = dist(yolo_pith, prop_pith)
    emit yolo_pith
    if d >= τ_flag:
        add Supervisely tag "pith_disagreement" to the annotation
elif yolo_pith:
    emit yolo_pith
elif prop_pith:
    emit prop_pith
    add tag "pith_propagation_only"
else:
    emit nothing
    add tag "pith_missing"
```

- `τ_yolo`: **0.10** (lower than production's 0.25). Empirically on the val
  set there are *zero* pith false positives at any threshold 0.10–0.55; the
  only effect of raising the threshold is to lose recall (94% → 89% → 50%).
  Counter-intuitive but the model is already perfectly precise on pith.
- `τ_flag`: pick from the leave-one-out sweep — mean + 3σ of the YOLO-vs-prop
  distance on frames where both fire. Initial guess: 10–20 px.

Annotator workflow benefit: the tag lets reviewers prioritise. Frames without
tags are presumed-correct; tagged frames get human attention.

### 3b. High-precision YOLO knots as synthetic propagation anchors (deferred)

Idea: at conf=0.7, YOLO knot detections become extra seeds for SAM video
propagation between human anchors.

**Status:** technically equivalent to the
`scripts/yolo_to_sam_video.py` experiment archived in the README, which got
Dice 0.503 vs 0.575 for per-slice 2D — i.e. it made things *worse*. SAM2 video
memory doesn't seem to gain from extra seeds on this dataset.

**Plan:** defer. Revisit only if (a) we have denser human anchors (every ~10
frames per Rostislav's plan), and (b) we can isolate whether the old failure
was due to low-quality seeds (conf=0.25) rather than the architecture. If we
ever retest, run with conf=0.7+ and compare to anchor-only on the same volume.

### 3c. Wood = union of threshold and propagation

Image #3 (propagation Wood, smooth outline) and image #4 (threshold collapsing
on a dark slice) show the two methods fail in different regimes. Union strategy:

```
wood_union = threshold_largest_cc(img, t=30) | (propagated_pred == WOOD)
wood = largest_connected_component(wood_union)
```

- Both methods are precision-strong, recall-weak; union ≈ recall boost with
  small precision cost.
- Final largest-CC filter drops spurious blobs (e.g. if threshold accidentally
  catches non-wood metal artefacts and propagation didn't).
- Cost: ~20 lines of code, no extra compute (we already have both).
- Risk: if both methods are wrong in the same region, union won't fix it.
  Acceptable — human review catches it.

### Wood-includes-knot fix (already shipped)

Rostislav noted Wood was missing knot-shaped holes. Fixed in `upload.py`:

```python
wood_mask = (pred == WOOD) | (pred == KNOT) | (pred == PITH)
```

Already re-uploaded as `auto_3_anchors_from_rosta_v2` (dataset id 1138708).

## Build order

1. ✅ Leave-one-out sweep — done. Wood distance-invariant; knot recall cliff at
   5 frames; pith error noisy and YOLO-better.
2. ✅ **3a + 3c implemented** as `build_combined_annotations.py`. Pipeline:
   YOLO pith at conf=0.10 (precision 1.0 on val) with propagation fallback +
   description-field flag; Wood = largest_cc(threshold ∪ propagation_wood);
   Knot = propagation (CC per object).
3. ✅ Uploaded as `auto_4_combined_v1` (project 376641, dataset id 1138743).
   - τ_flag = 7.98 px (auto from mean+3σ of YOLO-vs-prop distance: μ=2.42,
     σ=1.85, max=18.66 px across 289 frames where both fire).
   - 292 Wood, 292 Pith (all from YOLO), 318 Knot instances.
   - 8 frames tagged `[REVIEW: ...]` (5 pith disagreements, 3 propagation
     missing). YOLO fired on every frame.
4. ⏸ **3b parked** — needs denser anchors before retest is worthwhile.
5. ✅ Subset 4 refreshed (16 → 45 anchors).
6. ✅ Propagation re-run uploaded as `auto_4_propagation_45anchors`
   (dataset 1139090).
7. ✅ Proper holdout eval (see below) — replaces the n=3 leakage-tainted
   comparison from step 3.
8. ✅ **v2 pipeline shipped** as `auto_4_combined_v2` (dataset 1139705).
   See "Combined v2 recipe" below.

## Holdout eval (proper comparison, leakage-free)

The original 3-frame comparison had two problems: (a) tiny n, (b) 13/16
subset-4 anchors were in YOLO's training set, so YOLO got partial credit for
memorisation. With 45 anchors available we can afford a real split.

**Protocol**: 80/20 stratified split of the 45 annotated pages, every 5th.
- holdout (9): 7, 49, 103, 152, 175, 214, 236, 247, 285
- train (36): the rest

Both methods retrained/repropagated without holdouts. Anchor distance to
nearest training anchor ranges 1–11 frames per holdout, mixing the "near" (≤5)
and "far" (>5) regimes characterised by the LOO sweep.

Knot matching is IoU-Hungarian at IoU≥0.30 between per-instance masks.

### Results (yolo11n + holdout-free weights)

| metric | YOLO+SAM2 (default NMS) | YOLO+SAM2 (iou=0.5) | propagation |
|---|---|---|---|
| F1 (all 9 frames) | 0.67 | **0.78** | 0.55 |
| F1 (near, ≤5fr) | 0.78 | **0.90** | 0.60 |
| F1 (far, >5fr) | 0.00 | 0.00 | 0.00 |
| precision (all) | 0.64 | **0.90** | 0.67 |
| recall (all) | 0.69 | 0.69 | 0.46 |
| matched mean IoU | 0.58 | 0.54 | **0.75** |
| matched mean Dice | 0.73 | 0.69 | **0.84** |
| pith mean dist (px) | 1.18 | 1.18 | 1.85 |

### Findings

1. **YOLO NMS iou=0.7 → 0.5 is the single biggest win**: F1 0.67 → 0.78 with
   zero recall change. Removes 4 of 5 duplicate-detection FPs. Mask-IoU NMS
   gives the same numbers — duplicates are the same problem at both layers.
2. **YOLO+SAM2 wins on detection**, propagation wins on mask quality when it
   does match. They complement: YOLO+SAM2 for what's there, propagation for
   shape refinement (if we ever want it; v2 just uses YOLO+SAM2 masks).
3. **yolo11s did not improve over yolo11n** at this training-set size
   (96 train images). Bigger backbone gave identical F1, slightly worse IoU.
   Dataset is the bottleneck, not capacity.
4. **Far regime (>5 fr from anchor) is broken for both methods.** Only 2 GT
   knots in 5 far frames so the signal is weak, but propagation's earlier
   distance cliff is confirmed and YOLO doesn't pick up the slack.
5. **Small knots (<200 px)**: 0 matched by either method. Open question
   whether to invest in more small-knot annotations or change architecture.

## Combined v2 recipe (shipped as `auto_4_combined_v2`, dataset 1139705)

Inputs:
- YOLO weights: `yolo11n_v2_all45/weights/best.pt` (trained on all 45
  subset-4 anchors + subsets 1, 2; no holdout exclusion since v2 is for
  production, not evaluation).
- Propagation: `out/result.npz` (45-anchor full-volume propagation).
- SAM2 image predictor for box→mask.

Per-frame rule:
- **Knot** = YOLO+SAM2 with conf=0.25, **NMS iou=0.5**. One Supervisely Knot
  object per detection. No propagation fallback (eval showed it hurts more
  than helps).
- **Pith** = YOLO highest-conf bbox centre at conf=0.10; falls back to
  propagation pith centroid if YOLO misses. `[REVIEW: ...]` description tag
  if YOLO/prop disagree by ≥τ_flag or either is missing.
- **Wood** = largest_cc(threshold(t=30) ∪ propagation_wood ∪ knot_mask ∪
  pith). Knot mask in the union ensures wood encloses every annotated knot.
- **τ_flag** auto = mean + 3σ of YOLO-vs-prop pith distance (here 10.51 px;
  μ=2.70, σ=2.60, max=25.18 across 289 frames where both fire).

Stats: 270 knots across 133/292 frames (mean 0.92/frame), 292 pith (all YOLO),
5 frames flagged for pith review.

## Open question: smart-tool workflow

Web search confirmed Supervisely Smart Tool only *creates* segments — it can't
refine an existing one. Rostislav's mental model assumed otherwise. Practical
implication for the annotator workflow:

- A wrong propagated mask → delete + smart-tool-create new (two clicks + a
  prompt box).
- A small fix → brush tool with shortcuts (his preferred path for knot
  cleanup).

This argues for **higher precision over higher recall** in our generated
masks: a missing object is faster to add (smart-tool box) than a wrong one is
to fix (delete + recreate). Especially for knots, where false positives are
the more visible failure mode.

## Files

- `run.py` — sparse-anchor propagation across the volume.
- `render_overlays.py` — pith-as-centroid visualisation from `result.npz`.
- `upload.py` — encode + upload to Supervisely (wood-includes-knot fix applied).
- `compare_methods.py` — *deprecated*. The n=3 leaky comparison; superseded by
  `holdout_eval.py`.
- `sweep_distance.py` — leave-one-out distance sweep across all anchors.
- `holdout_eval.py` — 80/20 holdout eval, IoU-Hungarian matching, P/R/F1,
  size stratification, YOLO PR sweep. Supports `--yolo_nms_iou` and
  `--mask_nms_iou`. Caches propagation to `prop_pred_holdout_free.npz`.
- `visualize_holdout_grid.py` — TP/FP/FN visual grid for the 9 holdout frames.
- `build_combined_annotations.py` — v2 pipeline (YOLO+SAM2 knots, YOLO pith,
  union wood). Currently configured for `yolo11n_v2_all45` + `result.npz`.
- `out/result.npz` — propagated label volume for subset 4 (45 anchors).
- `out/holdout_eval/` — eval outputs (per_page.csv, summary.csv, visual grid).
- `out/holdout_eval_*` — NMS-knob sweep variants.
