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
