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

Datasets: `4` (subset 4, 292 frames, 45 anchors).

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
