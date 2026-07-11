# CT Log

## Semi-automatic annotation options

Three pipelines, trading off human effort vs. quality:

### 1. Anchor-augmented propagation (best quality, ~20 anchors/log)

Human annotates ~20 knot-only anchor frames → MedSAM2 video propagator fills
the rest, seeded with YOLO-OBB detections at non-anchor frames. Two seed
variants:

- **Ellipse seeds (v2.2)**: slightly crisper shapes on softwood, fewer
  detections.
- **SAM point seeds (v2.3)**: natural shapes on both hardwood and softwood,
  +17% recall. Recommended universal recipe.

Post-filters clean up artifacts (size, pith exclusion, eccentricity, solidity,
fill_holes). Wood and pith are fully automatic.

### 2. Anchor-free OBB propagation (zero human effort)

YOLO-OBB detections → ellipse rasters → MedSAM2 video propagation in
overlapping windows. No human annotations at all. Works well on softwood.
Degrades to ellipse-shaped outputs on hardwood.

### 3. Per-frame YOLO-OBB + SAM2 (no propagation, no anchors)

YOLO-OBB detection → SAM2 image predictor per slice. Simplest pipeline, no
temporal propagation, no anchors. Lower frame coverage than propagation-based
options but every detection is high precision (conf=0.40 → precision 1.0).

### Recommendation

Option 1 (point seeds + 20 anchors) gives the best results on any wood type
and is the recommended deployment path. Option 2 is the zero-effort fallback
for softwood. Option 3 is the simplest code path but covers fewer frames.

## Detector: single 2-class OBB (knot + pith)

The recommended detector is one **2-class OBB model** (class 0 = knot, class 1 =
pith), replacing the older split of a single-class OBB knot model plus a
separate axis-aligned knot+pith model. Pith is emitted as a small square OBB
around each pith point.

Prep with `ann_pipeline/knot/data_prep_obb_2cls.py`, train `yolo11n-obb.pt`. On
a log-level holdout (val = whole held-out logs, e.g. 2+10) it beats the
two-model split on every metric:

| | knot mAP50 | pith mAP50 | pith median px err |
|---|---|---|---|
| two models (OBB knot + axis-aligned pith) | 0.906 | 0.942 | 1.52 |
| single 2-class OBB | 0.913 | 0.990 | 0.97 |

Pith localizes at the annotation floor (median 0.97px, 87% within 2px). Knot
AP degrades past IoU 0.75 (AP90≈0.06) because OBB boxes are fit to the knot
mask — fine for propagation seeding, which needs the seed on the knot, not a
pixel-tight trace. The old axis-aligned knot head was weak/unstable (0.35
mAP50) and is dropped.

**Retrain flow** (detectors read a local disk root, not Supervisely directly —
see `ann_pipeline/PHASES.md`): download reviewed frames → `data_prep_obb_2cls`
→ `knot/train`.

## Two-phase annotation workflow

Two Supervisely projects split the workflow:

- **Phase1** — all frames, ~10-15% annotated by humans (knots only). These
  serve as anchors for propagation.
- **Phase2** — all frames have annotations. Anchor frames copied from Phase1
  (tagged `bumaska`), rest auto-generated. Annotators correct auto output and
  tag corrected frames with `bumaska`.

The `bumaska` image-level tag tracks review progress: tagged = human-verified.

Setup script: `python -m ann_pipeline.scripts.setup_phases --subset 4`

See `ann_pipeline/PHASES.md` for full details.

## Wood mask: iterative boundary peel

The default wood detector (`threshold_largest_cc`, t=30) overshoots into the
bark ring by ~1200px on average. `threshold_peel` (in
`ann_pipeline/wood/detectors.py`) fixes this by iteratively removing dark
boundary pixels (intensity < 60) — reduces bark overshoot to ~374px with
minimal wood loss. No model needed, ~11 it/s.

Phase2 auto annotations use the peel detector for wood.
