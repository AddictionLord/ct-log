# Project Context for ct-log

Read this file at the start of every session. It contains all decisions and background needed to work on this project without re-deriving them.

The full research document with model comparison tables, performance numbers, and per-class prompting strategy is in `context/research.md`.

---

## What This Project Is

**ct-log** — semi-automatic annotation pipeline for CT scans of wooden logs. Part of the Digiwood research initiative / LignoSilva Center of Excellence, Zvolen, Slovakia.

Goal: minimise annotator effort by having humans refine model-generated masks rather than annotate from scratch.

## This Repo

**MedSAM2** (`bowang-lab/MedSAM2`) — the foundation model for the pipeline. Based on SAM2.1-Tiny. Accepts bbox/point/mask prompts and propagates masks bidirectionally through a 3D volume treated as a video sequence. This is the correct repo; MedSAM v1 was cloned by mistake earlier.

## Dataset

- ~24 fully annotated seed CT slices (stored in Supervisely)
- ~300 greyscale single-channel CT slices per log (1 slice/cm) — coherent 3D volume when stacked
- Classes: `sound_knot, crack, pith, wood, bark, moisture, moisture_real, rot, insects, compression_wood, background`
- Hard classes: crack (thin linear voids), rot, moisture (low contrast, variable across slices)
- Knots span ~20–30 slices; bark/pith stable across all 300

## Pipeline Architecture

```
Seed dataset (24 images)
        ↓
   Train v1 model
        ↓
v1 predicts mask on key slice(s)
        ↓
Derive bbox from mask  ←─────────────────────────────┐
        ↓                                             │
 MedSAM2 refines + propagates across 300 slices      │
        ↓                                             │
Human annotator reviews and corrects (Supervisely)    │
        ↓                                             │
  Corrected masks → train v2 model ──────────────────┘
```

**Phase 1 (cold start):** No trained model yet. Feed primitive annotations from a prior project (points, bboxes, rough masks) directly as prompts to MedSAM2 → propagate → human review.

**MedSAM2's role:** Purely a boundary refiner. It does not classify — class knowledge lives in the domain model (v1, v2, ...).

## Model Selection Rationale

MedSAM2 selected over plain SAM2, MedSAM3, plain SAM3. Key reasons:
- Accepts geometric prompts (bbox, point, mask) — needed to bridge domain model output to MedSAM2
- Fine-tuned on 455k+ 3D medical CT/MRI/PET pairs; smallest domain gap to wood CT
- Treats sequential slices as video → bidirectional propagation through 300 slices
- ~6–10 Dice pts better than plain SAM2 on CT; biggest gains on low-contrast structures (rot, moisture, cracks) — exactly the hard classes

**MedSAM3 ruled out:** Text-prompt only in v1; no geometric input exposed. Monitor GitHub issues #8 and #10 — do not revisit until authors add bbox/mask support.

## Hardware (Development Machine)

```
GPU:    NVIDIA GeForce GTX 1650 (Turing, compute 7.5)
VRAM:   4GB
CUDA:   12.9
OS:     Linux (dual-boot with Windows 11 — always use Linux for ML work)
```

VRAM constraints:
- Full 300-slice propagation: fits on 4GB with default settings (tested, ~38s, no OOM)
- `max_num_maskmem: 3` mitigation not needed

## Environment

```bash
conda activate medsam   # Python 3.10, PyTorch 2.5.1+cu121 (compatible with CUDA 12.9)
```

The env is named `medsam` (v1 env repurposed — v1 package was uninstalled, MedSAM2 installed in its place). Do not use `conda run`.

## Inference Scripts

### `medsam2_infer_CT_lesion_npz_recist.py` — **our base script**
- **Input:** directory of `.npz` files with keys `imgs (D,H,W) uint8`, `gts`, `recist`, `spacing`
- **Prompt:** RECIST line on middle slice → bbox or N sampled points
- **Output:** `.npz` with `segs` + `spacing`, optional PNG overlays
- **Anchor:** middle slice (z_mid), bidirectional propagation
- **For us:** Replace RECIST → prompt logic with domain model mask/bbox. Everything else stays.

### `medsam2_infer_3D_CT.py` — NIfTI + DeepLesion CSV, not reusable
- **Input:** directory of `.nii.gz` files + hardcoded `CT_DeepLesion/DeepLesion_Dataset_Info.csv`
- **Prompt:** bbox read from the CSV (tightly coupled to DeepLesion dataset)
- **Output:** `.nii.gz` masks
- **For us:** Not reusable — CSV dependency is hardcoded, NIfTI format differs from our NPZ pipeline.

### `medsam2_infer_video.py` — JPEG frames + PNG masks, original SAM2 format
- **Input:** directory of subdirs with JPEG frames; separate dir with DAVIS-format palettised PNG masks
- **Prompt:** full mask on frame 0 (or all frames with `--use_all_masks`; or per-object with `--track_object_appearing_later_in_video`)
- **Output:** PNG masks per frame
- **For us:** Richest prompt (full mask), but requires converting volumes to JPEG on disk. Anchors on frame 0 by default — wrong for knots. More work to adapt than extending the NPZ script.

**Decision:** Build our pipeline entry point on top of `medsam2_infer_CT_lesion_npz_recist.py`.

Multi-class workflow: MedSAM2 propagates one class per run — loop over classes.

Key prompting principle: do not anchor all classes from frame 1. Use the structurally representative key slice per class (e.g. knot midpoint, not first appearance). See `context/research.md` §6 for the full per-class prompting table.

## Checkpoints

`/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt` — 149MB, the only checkpoint needed.

Fine-tuning is ruled out: SAM2.1-Tiny training requires 8–16GB VRAM minimum (batch size 2, multi-GPU). GTX 1650 (4GB) cannot do it. Inference only.

Weights also on HuggingFace: https://huggingface.co/wanglab/MedSAM2

## Open Questions

- GTX 1650 OOM threshold: resolved — 300 slices fits fine with default settings
- Supervisely export format: confirm PNG-per-class vs JSON polygons before building ingestion script
- MedSAM3 geometric prompts: monitor issues #8 and #10
