# Two-phase annotation workflow

## Overview

```
Phase1 (human anchors)          Phase2 (auto + human review)
┌──────────────────────┐        ┌──────────────────────────────┐
│ All frames, ~10-15%  │        │ All frames have annotations  │
│ annotated by humans  │───────>│ Anchors: copied from Phase1  │
│ (knots only needed)  │        │ Rest: auto-generated         │
│                      │        │ Tag 'bumaska' = human-verified│
└──────────────────────┘        └──────────────────────────────┘
```

## Phase1 — anchor collection

- One dataset per log (e.g. `Phase1/4` for subset 4)
- Annotators create knot masks on ~every 10-15th frame
- Wood and pith are NOT needed — computed automatically
- Target: ~20 anchors per log, denser in knot-heavy regions

## Phase2 — auto annotations for review

- Same images as Phase1, but every frame has annotations
- **Anchor frames**: copied verbatim from Phase1, tagged `bumaska`
- **Non-anchor frames**: OBB-augmented ellipse propagation + filters
- Annotators correct auto annotations, then tag the frame `bumaska`
- Untagged frames = untouched auto output

## The `bumaska` tag

Image-level tag. Meaning: "a human has verified this frame."

- Anchor frames arrive pre-tagged (they're already human-verified)
- Annotators add the tag after correcting an auto frame
- Lets us track coverage: % of frames with `bumaska` = review progress

## Auto annotation pipeline (Phase2)

Recipe: OBB-augmented anchor propagation v2.2 (ellipse seeds).

1. MedSAM2 video propagation seeded with anchor GT + YOLO-OBB ellipses
2. Per-frame filters: min 150px, pith exclusion 25px, eccentricity >0.7,
   solidity >0.85, fill_holes
3. Pith: YOLO bbox centroid (conf=0.10)
4. Wood: iterative peel (t=30, peel<60, n=5) ∪ propagated wood

## Supervisely projects

| project | ID | description |
|---|---|---|
| Phase1 | 377327 | Human anchors |
| Phase2 | 377328 | Auto + human review |
| Human-collection | 378193 | Accumulated `bumaska` (human-reviewed) frames from Phase2 |

Datasets: `4` (subset 4, 292 frames, 45 anchors).

## Human-collection: accumulating reviewed frames

`Human-collection` (378193) mirrors Phase2's dataset layout but holds only the
`bumaska`-tagged (human-reviewed) frames. It is the clean training pool.

- **Initial build** (create-only, errors if it exists):
  `python -m ann_pipeline.utils.build_human_collection`
- **Incremental sync** (append-only; never touches/overwrites existing frames,
  matches by image name, creates missing datasets, idempotent):
  `python -m ann_pipeline.utils.update_human_collection [--dry-run]`

## Retrain workflow (knot detectors)

The knot data-prep scripts read a **local** project root
(`/mnt/D/datasets/ct_log/375492_SM_2025/{subset}/{ann,img}/`), not Supervisely
directly. New reviewed frames must first be pulled to disk:

```bash
set -a && source .env && set +a

# 1. Pull bumaska frames from Phase2 to the local root (skips existing files;
#    --overwrite to refresh a subset whose reviewed set grew, e.g. subset 4).
python -m ann_pipeline.utils.download_all_tagged_images_from_supervisely \
    --subsets 08 10

# 2. Build datasets with a log-level holdout (whole logs -> val via
#    --val_subsets; all others -> train). Both detectors share the split.
python -m ann_pipeline.knot.data_prep_obb \
    --subsets 1 2 3 4 08 10 --val_subsets 2 10 \
    --out_dir ann_pipeline/out/knot_yolo_obb_v3
python -m ann_pipeline.knot.data_prep \
    --subsets 1 2 3 4 08 10 --val_subsets 2 10 \
    --out_dir ann_pipeline/out/knot_yolo_v3

# 3. Train (OBB needs an -obb pretrained model; task auto-detected).
conda run -n ct-log python -m ann_pipeline.knot.train \
    --data ann_pipeline/out/knot_yolo_obb_v3/knots_obb.yaml \
    --model yolo11n-obb.pt --name yolo11n_obb_holdout_v3
conda run -n ct-log python -m ann_pipeline.knot.train \
    --data ann_pipeline/out/knot_yolo_v3/knots.yaml \
    --model yolo11n.pt --name yolo11n_2cls_holdout_v3
```

**Recommended detector — single 2-class OBB (knot + pith):** one model replaces
the OBB-knot + axis-aligned-pith split and beats it on every metric (see
`CLAUDE.md`). Build with the 2-class OBB prep instead of the two preps above:

```bash
python -m ann_pipeline.knot.data_prep_obb_2cls \
    --subsets 1 2 3 4 08 10 --val_subsets 2 10 \
    --out_dir ann_pipeline/out/knot_yolo_obb_2cls_v3
conda run -n ct-log python -m ann_pipeline.knot.train \
    --data ann_pipeline/out/knot_yolo_obb_2cls_v3/knots_obb_2cls.yaml \
    --model yolo11n-obb.pt --name yolo11n_obb_2cls_holdout_v3
```

Log-level holdout (val = held-out logs, e.g. 2 + 10) gives an honest
generalization estimate to unseen logs, unlike the default per-frame random
split where correlated slices from one log leak across train/val.

## Running

```bash
# Build locally (dry run):
set -a && source .env && set +a
conda run -n ct-log python -m ann_pipeline.scripts.setup_phases \
    --subset 4 --out_dir /tmp/phases --dry_run

# Build + upload:
conda run -n ct-log python -m ann_pipeline.scripts.setup_phases \
    --subset 4 --out_dir /tmp/phases --upload
```
