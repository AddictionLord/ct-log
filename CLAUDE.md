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

## Knot segmentation default: YOLO11n-seg at 800 px

Per-frame knot masks for auto annotation come from `yolo11n_seg_v5_val10_800` (YOLO11n-seg,
`imgsz=800`, conf 0.25, union of instance masks clipped to the peel wood mask, no post-filters). This
is the default in `ann_pipeline/scripts/build_kwp_dataset.py`. On held-out log 10 its pooled knot
IoU is 0.750, and it finds 93 % of knots. It beats:
- YOLO11s/m-seg (0.738 / 0.734);
- the DINOv3 student (0.672);
- the propagation pipeline, which loses thin and straight knots. Straight-knot Dice on log 2 was
  0.31 for propagation vs 0.69 for YOLO-seg.

Provenance and the sweep are in `ann_pipeline/DETECTOR_PROVENANCE.md`. Phase2 pre-labels on
unreviewed logs still come from the older propagation pipeline until replaced.

**Combined default (decided 2026-10-07, interim until further U-Net experiments report):**
- **Knots:** YOLO11n-seg @800.
- **Pith:** the segmentation session's smp U-Net (ResNet-50, 800 px), no snap. On log 10 it gets
  68 % of pith points on the exact pixel and 96 % within 1 px, vs 61 % / 96 % for the snapped
  2-class OBB.
- **Wood:** `threshold_peel`.

U-Net pith enters `build_kwp_dataset.py` through `--pith_root` (per-log JSON exported by the
segmentation session). Without `--pith_root`, the builder falls back to the snapped OBB pith.

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

## Training runs and experiment tracking

Agents that launch, monitor or report training (locally or on euler.mendelu.cz) must follow
`TRAINING_AGENTS.md`. MLflow tracking is on DagsHub
(`https://dagshub.com/AddictionLord/ct-log.mlflow`, public); credentials live in `.env`.

### Monitoring long runs cheaply

Every wake-up of the main session re-sends its whole context, so do not poll with the model or
use periodic heartbeats. Watch a run with `scripts/euler_watch.sh` started as a background
command of the session that should react:

```bash
scripts/euler_watch.sh <job> <remote_run_script> <remote_train_info_yaml>
```

- Polling is a bash probe every 5 min (zero model tokens). The script exits, and so wakes the
  session, only when a decision is needed: FINISHED, FAILED (supervisor gave up), BUG, MLFLOW,
  STALL, UNREACHABLE, RECOVERY limit. Events are appended to `logs/euler_watch_<job>.log`.
- It recovers on its own from a container restart/recreation (re-stages the MLflow credentials
  from `.env` and relaunches the job, which resumes from `~/work`).
- A new traceback is triaged by a small model (`claude -p --model sonnet --tools ""`, headless,
  no tools, ~$0.02 and 4 s per call). TRANSIENT (OOM from another GPU user, network, signal) is
  left to the on-euler supervisor; BUG wakes the main session.
- Status questions ("jak je na tom trénink?") are answered with one probe on demand, not by
  keeping a heartbeat.
- Start each new experiment in a fresh session: context is the dominant token cost.
  `TRAINING_AGENTS.md` and `ann_pipeline/DETECTOR_PROVENANCE.md` carry the needed background.
