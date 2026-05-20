# ann_pipeline — automatic CT log annotation

End-to-end pipeline that takes raw CT slices and produces Supervisely-ready
annotations for **Wood**, **Knot** (multi-instance), and **Pith** (single point per slice).

The pipeline is deployed by uploading new datasets into the Supervisely project
[SM_2025_automatic_annotations (376641)](https://app.supervisely.com/projects/376641/datasets),
where human annotators review/correct rather than annotate from scratch.

---

## Architecture (current production)

For every slice we run three independent detectors in parallel:

```
                    ┌─────────────────────────────────────┐
                    │  raw CT slice (778x778, RGBA TIFF)  │
                    └─────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼──────────────────────────┐
        ▼                         ▼                          ▼
  ┌───────────┐           ┌────────────┐            ┌────────────────┐
  │ threshold │           │ YOLO11n    │            │ YOLO11n        │
  │ > 30 +    │           │ class=Knot │            │ class=Pith     │
  │ largest CC│           │ bboxes     │            │ bbox -> centre │
  └───────────┘           └─────┬──────┘            └─────┬──────────┘
        │                       ▼                         ▼
        │              ┌─────────────────┐       ┌──────────────────┐
        │              │ SAM2 image      │       │ point (no SAM)   │
        │              │ predictor with  │       │ — emitted as a   │
        │              │ box+point       │       │ Supervisely      │
        │              │ prompts         │       │ point object     │
        │              └──────┬──────────┘       └─────┬────────────┘
        ▼                     ▼                        ▼
  Wood mask          Knot bitmap (per knot)        Pith point
        │                     │                        │
        └─────────────────────┴────────────────────────┘
                              │
                              ▼
              Supervisely-format JSON per slice
              (image-as-tiff + ann-as-json)
                              │
                              ▼
                     Supervisely upload
                     (sly SDK image.upload_paths +
                      annotation.upload_paths)
```

### Why this exact stack

- **Wood = intensity threshold (no model).** Tested against 96 GT slices: mean Dice **0.994**, 27 ms/slice. SAM with corner-negative prompts collapsed to Dice 0.02, YOLO would be overkill.
- **Knot = YOLO bbox + SAM mask.** YOLO11n on 51 training slices gives mAP50 = 0.84, knot-centroid recall 97%. The bbox is fed to SAM2's image predictor with the bbox centroid as an additional positive point — the combined prompt gives **mean Dice 0.695** vs 0.674 for bbox-only.
- **Pith = YOLO centroid as a point object.** Reasoning: YOLO 2-class detector (knot + pith) gives **mean pith centroid distance 1.02 px** (vs 10 px for the best classical detector — Hough/wood-centroid/etc.). We emit the centroid as a Supervisely Point — the existing project's Pith class shape is `point`, not bitmap.

### What we tried and rejected

| approach | result | why dropped |
|---|---|---|
| Classical pith (wood centroid, Hough, gradient-radial vote) | mean err 10 px | YOLO is 10× more accurate |
| Adding Wood as a 3rd YOLO class | n/a | intensity threshold already at Dice 0.994 |
| SAM2 video predictor (`add_new_mask` propagation) | identical Dice to per-slice 2D | video memory doesn't help static cross-sections |
| SAM2 video predictor (`add_new_points_or_box` propagation) | 0.532 mean Dice vs 0.531 for 2D | same — temporal context adds nothing for radial knots |
| Per-instance mask propagation | 0.580 mean Dice vs 0.575 single-mask | marginal, not worth complexity |

See [the experiments dir](../experiments/) for the failed-but-instructive prior runs.

---

## Repository layout

```
ann_pipeline/
├── data.py                     # Supervisely loader: PithSlice dataclass + iter
├── wood/
│   ├── detectors.py            # threshold_largest_cc, threshold_morphology, SAM variants
│   └── eval.py                 # wood eval on all 96 GT slices (CLI: -m ann_pipeline.wood.eval)
├── pith/
│   ├── detectors.py            # classical pith detectors (kept for baseline comparisons)
│   └── eval.py                 # SliceResult / DetectorReport types
├── knot/
│   ├── data_prep.py            # Supervisely -> YOLO format (CLI: -m ann_pipeline.knot.data_prep)
│   ├── visualize_bboxes.py     # sanity-check the bbox extraction
│   ├── train.py                # YOLO11n training (CLI: -m ann_pipeline.knot.train)
│   ├── predict.py              # render val predictions for a trained model
│   └── centroid_eval.py        # Hungarian-matched centroid distance per class
├── scripts/
│   ├── eval_pith.py            # all 5 classical pith detectors on every annotated slice
│   ├── yolo_to_sam_eval.py     # YOLO -> SAM2-image per-slice eval (the prod inference path)
│   ├── yolo_to_sam_video.py    # legacy video experiment (mask seeds)
│   ├── yolo_to_sam_video_boxpoint.py  # legacy video experiment (box+point seeds)
│   └── build_supervisely_annotations.py  # the PRODUCTION script — generate + upload
└── out/                        # all eval outputs (gitignored)
    ├── pith_eval/              # classical pith eval
    ├── wood_eval/              # wood detector comparison
    ├── knot_yolo/              # generated YOLO training dataset
    ├── knot_runs/              # YOLO training runs
    ├── knot_centroid_eval/     # centroid-distance per class
    └── yolo_to_sam_eval_*/     # YOLO->SAM eval at various confidence + prompt modes
```

---

## Environment

Use the **`ct-log`** conda env (Python 3.11). MedSAM2 + DINOv3 are vendored under
`thirdparty/`, both installed via `pip install -e /home/mary/code/ct-log`.

```bash
conda activate ct-log
# Optional sanity check:
python -c "from sam2.build_sam import build_sam2; from ultralytics import YOLO; import supervisely; print('OK')"
```

Models and checkpoints (NOT in the repo, must exist on disk):

| asset | path |
|---|---|
| MedSAM2 checkpoint | `/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt` |
| Trained YOLO11n (knot+pith) | `ann_pipeline/out/knot_runs/yolo11n_2cls_v1/weights/best.pt` |
| YOLO11n pretrained | downloaded on first run by `ultralytics` |
| Source CT data | `/mnt/D/datasets/ct_log/375492_SM_2025/{1,2,3,4}/{img,ann}/` |

---

## Workflows

All commands run from the repo root (`/home/mary/code/ct-log`).

### 1. Prepare YOLO training data from Supervisely

Walks the chosen Supervisely subsets, decodes bitmaps, extracts knot bboxes
(from connected components in the decoded mask) and pith bboxes (16x16 box
around each annotated point), writes the YOLO dataset.

```bash
python -m ann_pipeline.knot.data_prep \
    --subsets 1 2 4 \
    --out_dir ann_pipeline/out/knot_yolo \
    --val_frac 0.2
```

Output structure:

```
ann_pipeline/out/knot_yolo/
├── images/{train,val}/dsN_page_NNN.jpg
├── labels/{train,val}/dsN_page_NNN.txt   # YOLO format: cls cx cy w h
└── knots.yaml                            # nc: 2, names: [knot, pith]
```

**Sanity-check the bbox extraction** before training:

```bash
python -m ann_pipeline.knot.visualize_bboxes
# -> ann_pipeline/out/knot_bboxes/montage.png  (grid of all annotated slices with bboxes overlaid)
```

### 2. Train YOLO11n (2-class)

```bash
python -m ann_pipeline.knot.train \
    --epochs 200 \
    --name yolo11n_knots_v3
# -> ann_pipeline/out/knot_runs/yolo11n_knots_v3/weights/best.pt
```

Training auto-stops via `patience=50`. Heavy rotation/flip augmentation is on
by default (knots are radially symmetric around the pith). ~2–3 minutes on a
GTX 1650 for ~50-slice dataset.

Watch out for the **MLflow auth gotcha** — the script disables ultralytics' MLflow
integration via `SETTINGS["mlflow"] = False` because the ct-log env's MLflow URL
requires auth. If you re-enable MLflow you'll get a 401 error.

### 3. Evaluate the trained detector

Two complementary evals:

**YOLO mAP50 (Ultralytics default):**
```bash
python -m ann_pipeline.knot.predict \
    --weights ann_pipeline/out/knot_runs/yolo11n_knots_v3/weights/best.pt
# -> ann_pipeline/out/knot_predictions/val_predictions.png
```

**Centroid-distance per class (the metric we actually care about):**
```bash
python -m ann_pipeline.knot.centroid_eval \
    --weights ann_pipeline/out/knot_runs/yolo11n_knots_v3/weights/best.pt
# -> ann_pipeline/out/knot_centroid_eval/val_centroids.png
```

Current production weights (`yolo11n_2cls_v1`) on the 18-slice val:

| class | recall | mean centroid dist | median | p90 |
|---|---|---|---|---|
| knot | 97% (33/34) | 5.9 px | 3.1 px | 12.3 px |
| pith | 89% (16/18) | 1.0 px | 1.2 px | 1.6 px |

### 4. Evaluate the full YOLO → SAM pipeline (2D, per-slice)

This is the headline number for the production pipeline.

```bash
python -m ann_pipeline.scripts.yolo_to_sam_eval \
    --conf 0.25 \
    --knot_prompt box+point \
    --out_dir ann_pipeline/out/yolo_to_sam_eval_prod
```

Outputs:
- `val_yolo_to_sam.png` — overlays for each val slice (GT=orange, pred knot=cyan, pred pith=orange outline, YOLO bbox=red dashed)
- `val_per_slice.csv` — per-slice Dice + pith centroid distance
- Per-slice NPZ files with `gt_knot`, `pred_knot`, `pred_pith`

Current production prompt = **`box+point`** (combined): mean knot Dice 0.695 vs
0.674 for bbox-only. See the `out/yolo_to_sam_eval_{box,point,box+point}/` dirs
for the ablation that established this.

### 5. Standalone classical pith eval (baseline reference)

For comparison with the YOLO-based pith detector. Kept around in case you ever
want a model-free path.

```bash
python -m ann_pipeline.scripts.eval_pith
# -> ann_pipeline/out/pith_eval/{summary.csv, montage/<detector>.png}
```

Best classical detector: `wood_centroid` — 10.4 px mean error, 10 ms/slice.
**Worse than YOLO** (1.0 px) — included for documentation, not production use.

### 6. Standalone wood eval

```bash
python -m ann_pipeline.wood.eval
# -> ann_pipeline/out/wood_eval/{summary.csv, threshold_largest_cc.png}
```

Best: `threshold_largest_cc` — 0.994 mean Dice across 96 slices.

### 7. Generate annotations for a new subset (LOCAL, dry-run)

The **production script**. Builds Supervisely-format annotations for an entire
subset and writes them to disk in the layout Supervisely expects.

```bash
python -m ann_pipeline.scripts.build_supervisely_annotations \
    --subset 3 \
    --out_dir /tmp/ct-log-subset3 \
    --conf 0.25 \
    --wood_thresh 30
```

This writes:
```
/tmp/ct-log-subset3/
├── meta.json                     # project-level class definitions
└── 3/                            # dataset
    ├── img/page_NNN.tiff         # copied from source
    └── ann/page_NNN.tiff.json    # generated annotations
```

Speed: ~5 it/s on GTX 1650 → 298 slices ≈ 1 minute.

**Always do this locally first and visually inspect** before uploading:

```bash
python /tmp/visualize_export.py /tmp/ct-log-subset3/3 /tmp/ct-log-subset3_overlays.png
```

(`/tmp/visualize_export.py` decodes the generated JSONs back to masks and
overlays them; if it doesn't exist anymore, the code is reproducible from
`ann_pipeline/scripts/yolo_to_sam_eval.py`'s rendering block.)

### 8. Upload to Supervisely

Set the token in `/home/mary/code/ct-log/.env` (gitignored):

```
SUPERVISELY_TOKEN=<your token from app.supervisely.com -> Account -> API Token>
```

Then run the same script with `--upload`:

```bash
set -a && source /home/mary/code/ct-log/.env && set +a
python -m ann_pipeline.scripts.build_supervisely_annotations \
    --subset 3 \
    --out_dir /tmp/ct-log-subset3 \
    --upload \
    --project_id 376641 \
    --dataset_name auto_3
```

The upload path:
1. Connects via `sly.Api(server, token)`
2. Resolves project 376641 (`SM_2025_automatic_annotations`) — workspace 122325
3. Verifies the project's classes match `CLASS_REGISTRY` (Knot/Wood/Pith). Fails loudly if not.
4. Refuses to overwrite an existing dataset with the same name — delete via GUI first
5. Creates the new dataset, uploads TIFFs via `image.upload_paths`, attaches annotations via `annotation.upload_paths`

**Smoke-test workflow** (recommended before any new full upload):

```bash
# tiny upload — 5 mid-volume pages to verify round-trip
python -m ann_pipeline.scripts.build_supervisely_annotations \
    --subset 3 --page_min 100 --page_max 104 \
    --out_dir /tmp/ct-log-smoke \
    --upload --project_id 376641 --dataset_name _smoke_test

# eyeball in the GUI -> if good, delete _smoke_test, then do the full subset
```

### 9. Sweep confidence threshold (for FN debugging)

If you suspect YOLO is missing knots on a deployment volume, sweep conf:

```bash
for c in 0.10 0.15 0.20 0.25; do
    python -m ann_pipeline.scripts.yolo_to_sam_eval \
        --conf "$c" --knot_prompt box+point \
        --out_dir "ann_pipeline/out/yolo_to_sam_eval_conf${c}"
done
```

As of `yolo11n_2cls_v1`, the **best val Dice is at conf=0.25** (lowering threshold
adds more false positives than true positives on val). But for a deployment
volume where the model has lower recall, conf=0.10 may surface more true
positives even though val Dice doesn't show the benefit. Tested via uploading
both `auto_3` (conf=0.25) and `auto_3_conf010` (conf=0.10) as siblings — the
latter has +44 pages with knot detections and +37 pith points across 298 slices.

---

## Failed approaches archive

For posterity / so we don't re-derive them:

- **Propagation from a single annotated anchor slice** (`experiments/sm2025_slice14/`) — mean Dice 0.47 across the 20-frame window. Falls off rapidly with distance from the anchor.
- **Multi-anchor propagation** (`experiments/sm2025_multi_anchor/`) — Dice 0.58. Empty anchors actively suppress the class, which is good for bounding but hurts neighbouring frames.
- **Per-instance propagation** (`experiments/sm2025_per_instance/`) — 0.58. Per-instance vs merged-mask is a wash.
- **Combined: per-instance + multi-anchor** (`experiments/sm2025_combined/`) — 0.580. Marginal.
- **Hybrid 2D→video with `add_new_mask` on every detected frame** (`scripts/yolo_to_sam_video.py`) — 0.503, no improvement over 2D. Video predictor passes through seeded masks.
- **Hybrid 2D→video with `add_new_points_or_box` on every detected frame** (`scripts/yolo_to_sam_video_boxpoint.py`) — 0.532. Same as 2D box+point within noise.

The lesson: **SAM2's video memory was designed for temporal motion**. Cross-sections of static structures don't benefit from cross-frame attention. Per-slice 2D is the right architecture for this dataset.

---

## Known limitations

1. **YOLO recall on novel slices.** Val recall is 97% knot / 89% pith, but on
   subset 3 (entirely unseen) the GUI shows visible FNs. The path forward is
   more training annotations, not architecture changes.
2. **Knot false positives at bark rim.** SAM2 produces a mask wherever YOLO
   says "knot here", including spurious detections at the wood/air boundary.
   Postprocessing filter (reject boxes whose centre is in the bark ring) is a
   plausible future improvement but not implemented.
3. **Pith point ≠ pith mask.** We emit a single point per slice for Pith,
   matching the existing project's Pith shape. If the annotators ever want a
   pith *mask*, we'd swap in SAM2's point→mask call (already used internally
   during eval, just not currently emitted to the export).
4. **4GB VRAM is tight for any video-predictor experiment** on full 300-slice
   volumes. Per-slice inference fits comfortably.

---

## Re-training cadence

When your colleagues annotate more slices, rebuild the training set and retrain:

```bash
# 1. Pull the latest Supervisely export (manual; or via the SDK in a future enhancement)
# 2. Rebuild YOLO dataset
python -m ann_pipeline.knot.data_prep --subsets 1 2 4

# 3. Retrain
python -m ann_pipeline.knot.train --name yolo11n_knots_vN

# 4. Verify the new weights beat the old on the centroid eval
python -m ann_pipeline.knot.centroid_eval \
    --weights ann_pipeline/out/knot_runs/yolo11n_knots_vN/weights/best.pt

# 5. Update WEIGHTS in scripts/build_supervisely_annotations.py to the new run
# 6. Regenerate + upload the auto_N datasets
```
