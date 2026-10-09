# Segmentation dataset versions

Record of every dataset version the DINOv3 knot/wood/pith segmentation model (`src/train_kwp.py`)
is trained on. Check this before comparing runs: a metric difference between runs on different
dataset versions is not attributable to the model.

Each version gets:
- an ID (`kwp-ds-vN`), used in run notes and MLflow tags;
- the exact source of every label (model, weights sha256, thresholds, post-processing);
- logs and splits;
- a per-file sha256 manifest in `ann_pipeline/dataset_manifests/<id>.sha256`;
- locations (local and euler).

On-disk layout (what `CTLogKwpDataset` reads): `<root>/<log>/{img,ann}`, one Supervisely JSON per
annotated slice (`Wood` bitmap, `Knot` bitmaps, `Pith` point). `img/` must contain **every** slice
of the log, not only annotated ones: with `window: 1` the model's input is `[z-1, z, z+1]`, and the
dataset looks up neighbours among the files in `img/`. Missing slices silently give the model a
wrong neighbour.

Related: detector versions and volume aliases (`CT/01` == log 1, `CT/04` == log 4, `CT/10` == log 10)
are in `ann_pipeline/DETECTOR_PROVENANCE.md`.

## kwp-ds-v1 (retroactive record, frozen 2026-09-30)

Used by every segmentation run up to and including `kwp-v1-capped-nopith-784` (16 ep) and
`kwp-v1-capped-nopith-784-40ep`.

Location: euler `~/work/ctlog-eval/data/{377328_phase2,378193_human_collection,generated}`.
Manifest: `dataset_manifests/kwp-ds-v1.sha256`, 6856 annotation files, sha256 of the manifest
`3fc85cee826202e03f61d98c1b88ab512da4f08444b37ef9f2839539cab9c483` (computed on euler
2026-09-30, same file in the repo).

| part | logs | frames | label source |
|---|---|---|---|
| `377328_phase2/` (train) | 2, 3, 05, 06, 08, 09 | 297 + 298 + 299 + 299 + 298 + 300 | Supervisely Phase2 as downloaded 2026-09-20: mix of human-reviewed (`bumaska`) frames and auto pre-annotations. Pre-annotations came from different pipelines per log: logs 2 (and 1, 4) from `setup_phases` (anchor-augmented propagation, old axis-aligned pith model `yolo11n_v2_all45`); 3, 05, 06, 09 from anchor-free OBB propagation with detector v3 (regenerated 2026-07-12/13); 08 anchor-free (June) with 15-27 human frames. Log 2 had only 34 reviewed frames at download time. |
| `generated/` (train) | 13-21, 23-27 | ~299 each, 4185 | Anchor-free propagation `run_obb_only.py`, detector `yolo11n_obb_2cls_v4`, conf 0.40, then `scripts/npz_to_dataset.py` (`build_auto_annotation` post-filters: area < 150, pith exclusion 25 px, eccentricity < 0.7, solidity < 0.85). Pith model: `npz_to_dataset.py` default `yolo11n_obb_2cls_holdout_v3` unless overridden (not verified which was passed). Generated 2026-09-21. |
| `378193_human_collection/1`, `/4` (val) | 1, 4 | 291, 291 | Human (Human-collection project), downloaded 2026-09-19. |
| `378193_human_collection/10` (test) | 10 | 293 | Human, downloaded 2026-09-19. |

Known defects, found 2026-09-29/30:
- **Stale validation labels.** Human-collection was out of date since July because
  `update_human_collection` never overwrote edited frames: 55 frames of log 1 and 28 of log 4 were
  older versions than Phase2. Log 10 (test) was identical to Phase2, so **test metrics stand**; val
  metrics are measured against partly outdated labels.
- **Wrong neighbour slices for human logs.** `378193_human_collection/<log>/img` holds only the
  reviewed frames (img = ann counts), so with `window: 1` a frame next to an unreviewed slice gets
  the next reviewed slice as its neighbour instead of the true adjacent one.
- **Auto knot labels miss most thin and straight knots.** Measured against human labels
  (`eval/overnight_2026-09-29.local.md`): the post-filters reject 97% of thin (< 8 px) knot
  instances; on log 2, straight radial knots are detected in 48% of cases (Dice 0.307), per-frame
  knot Dice 0.453. A model trained on these labels inherits it: DINOv3 epoch 15 finds straight
  knots on log 2 in 24% of cases.
- Log 2 was in training with mostly auto labels; it is now fully human-reviewed.

## kwp-ds-v2 (built 2026-09-30)

Locations:
- local `/mnt/D/datasets/ct_log/kwp_ds_v2/<log>/{img,ann}` (img/ are symlinks to the raw slices);
- euler `~/work/ctlog-eval/data/kwp_ds_v2/<log>/{img,ann}` (img/ are symlinks into the existing
  `377328_phase2/`, `378193_human_collection/` and `generated/` dirs, plus 9 slices uploaded to
  complete the human stacks: log 1 p041, log 4 p013, log 10 p000/001/002/100/291/298, log 08 p000).

Manifest: `dataset_manifests/kwp-ds-v2.sha256`, 6850 annotation files, sha256 of the manifest
`05abd00072db038abcb08bd30b4330ae99d9a928fe7d1c77f8e24fcaa0b26e83`. Verified identical on euler
(`sha256sum -c` over all 6850 files) and loadable with `CTLogKwpDataset` there. Full generator
record: `dataset_manifests/kwp-ds-v2.DATASET.json` (built from commit `69498d7` with the builder
and these docs not yet committed, `git_dirty: true`; they were committed right after).

Generator weights sha256: knot `yolo11n_seg_v5_val10` `296e6f21ff965533…`, pith
`yolo11n_obb_2cls_v5_n640` `092e9030052db721…` (full hashes in the DATASET.json).

| log | kind | annotated | slices in img/ | frames with knot |
|---|---|---|---|---|
| 3 | auto | 298 | 298 | 181 |
| 05 | auto | 299 | 299 | 170 |
| 06 | auto | 299 | 299 | 178 |
| 08 | auto (27 human frames) | 299 | 299 | 178 |
| 09 | auto | 300 | 300 | 144 |
| 13-21, 23-27 | auto | 4185 | 4185 | 119-199 per log |
| 1 | human | 291 | 292 | 170 |
| 2 | human | 290 | 297 | 192 |
| 4 | human | 291 | 292 | 201 |
| 10 | human | 293 | 299 | 177 |

QA, v1 vs v2 knot labels on the same 14 generated logs (components ≥ 50 px;
`logs/overnight_20260929/compare_ds_v1_v2.txt`):

| | frames with knot | thin knots < 8 px | median thickness | elongated ≥ 3.5 |
|---|---|---|---|---|
| v1 | 67-187 per log | 0.0-4.5 % | 13.4-22.4 px | 37-80 % |
| v2 | 119-199 per log | 14.6-46.6 % | 8.0-17.9 px | 54-93 % |
| human logs 1, 2, 4, 10 | 170-201 | 10.3-47.4 % | 8.2-14.4 px | 48-86 % |

v2's auto labels are close to the human distribution. v1 had almost no thin knots.

Change vs v1: knot labels on auto logs come from a segmentation detector instead of the
propagation pipeline; human labels are fresh and corrected; human logs get the full slice stack.

Builder: `python -m ann_pipeline.scripts.build_kwp_dataset` (writes `DATASET.json` with generator
sha256s and per-log stats, and `MANIFEST.sha256`).

| part | logs | label source |
|---|---|---|
| auto | 3, 05, 06, 08, 09, 13-21, 23-27 | Knot: `yolo11n_seg_v5_val10` (YOLO11n-seg, trained on human logs 1, 2, 4, 08; never saw log 10), conf 0.25, union of instance masks, clipped to wood, one object per 8-connected component, no other filters. Wood: `largest_cc_fill(threshold_peel(gray, 30, 60, 5) \| knot)`. Pith: `yolo11n_obb_2cls_v5_n640`, highest-confidence pith box centre, conf 0.10. Log 08: the 27 human-reviewed frames use human labels. |
| human | 1, 2, 4, 10 | Phase2 `bumaska` frames, snapshot `human_reviewed_20260929` (1192 frames, verified identical to Human-collection). `img/` links the full raw slice stack. |

Evidence for the knot-label change (held-out logs, per-frame knot Dice vs human labels):

| label source | log 2 | log 10 |
|---|---|---|
| v1-style pipeline (propagation + post-filters) | 0.453 | 0.650 |
| YOLO-seg as in v2 | 0.738 | 0.804 |

Caveat: the knot generator was trained on human logs 1 and 4, which are the usual segmentation
val logs. Log 10 is clean for both the generator and the segmentation model.

## kwp-ds-v3 (built 2026-10-02)

v2 plus 31 previously unused CT volumes, which roughly doubles the number of distinct training logs
(19 auto logs to 50). Requested by the segmentation trainer: tuning had saturated on v2 and the
error was dominated by the gap between train logs and unseen logs.

Locations:
- local `/mnt/D/datasets/ct_log/kwp_ds_v3/<log>/{img,ann}`; new raw slices are symlinked as
  `/mnt/D/datasets/ct_log/generated/<log>/img` -> `CT/<log>/page_*.tiff`;
- euler `~/work/ctlog-eval/data/kwp_ds_v3/<log>/{img,ann}` (img/ links built by `setup_links.sh`,
  same sources as v2; new raw slices in `data/generated/<log>/img`; extra human slices copied from
  `kwp_ds_v2/extra_img`). Annotation tarball `data/kwp_ds_v3_ann.tgz`
  (sha256 `8a9c1f6430b7697db47df4a99baefa38fc1bfbe88e3e3052115a0dc2a75cdf67`).

Manifest: `dataset_manifests/kwp-ds-v3.sha256`, 16140 annotation files, sha256 of the manifest
`fe7220fe85bf0641c9d303ce64457fbdb28bdaeb35f043e06d9b6c386e1d65a0`. Verified on euler
(`sha256sum -c` over all files), and `CTLogKwpDataset(54 dirs, window=1)` loads 16140 samples. Full
record: `dataset_manifests/kwp-ds-v3.DATASET.json` (built from commit `04ad41f`, with the pith snap
not yet committed, `git_dirty: true`).

| part | logs | label source |
|---|---|---|
| auto (as v2) | 3, 05, 06, 08, 09, 13-21, 23-27 | Same generator and post-processing as v2. |
| auto (new) | 11, 28-31, 33, 34, 36-44, 46-60 | Same generator and post-processing as v2. Raw volumes `CT/<log>`, ~300 slices each, 778x778. All 54 CT volumes are distinct (page_150 hashes). Volume numbers 12, 22, 32, 35 and 45 do not exist in `CT/`. |
| human | 1, 2, 4, 10 | Unchanged from v2 (snapshot `human_reviewed_20260929`). |

The only recipe change vs v2 is pith on auto frames: the highest-confidence pith box centre is now
snapped to the darkest pixel in a 5x5 window (`ann_pipeline.pith.detectors.snap_to_darkest`). On
held-out log 10 against human clicks this gives 61 % exact and 96 % within 1 px, vs 65 % within 1 px
unsnapped.

QA against v2 on the 23 shared logs (`logs/kwp_ds_v3/qa_v2_vs_v3.txt`): Knot and Wood objects are
identical on every frame. Pith moved on auto frames only, mean 0.9-1.6 px per log, max 4.24 px.

New auto logs: 298-301 frames each, 92-208 frames with knot (v2 auto logs: 119-199). Log 50 (92)
is the low outlier and has not been inspected visually. Per-log numbers are in the DATASET.json.

Caveat as v2: the knot generator was trained on human logs 1, 2, 4 and 08. Log 10 is the only
test log that neither the generator nor the segmentation model saw.

QA of the auto logs (2026-10-03, `logs/kwp_ds_v3/qa_auto_logs.txt`, generator re-run on every 2nd
slice). The new logs match the v2 auto logs on every proxy:
- mean confidence of kept knot instances: new 0.559 (0.476-0.628 across logs) vs v2 0.561 (0.494-0.618);
- share of kept instances with confidence 0.25-0.35: 0.20 vs 0.19;
- share of knot pixels within 12 px of the wood boundary: 0.048 vs 0.045;
- wood-area CV along the log: 0.090 vs 0.086.

Two flags:
- **Log 50** is out of distribution: a dark, irregular heartwood with intrusions into the sapwood
  ring, and the lowest confidence (0.476). Candidate for exclusion.
- **Logs 41, 42 and 53** have mild excess ring spill (~0.10).

v3 trained worse than v2 with uniform sampling and recovered with 5x human oversampling. This points
to the human/auto mixture share (human frames ~17 % of v2, ~7 % of v3), not to bad individual logs.
Caveat (trainer, 2026-10-04): seed noise between two identical fine-tuned 784 runs is 0.009 pooled
knot IoU (0.018 on log 10). The v3-vs-v2 and weighting differences above (≤0.01) are therefore not
conclusive from single runs.

## kwp-ds-v4 (built 2026-10-07)

The combined auto-annotation decided on 2026-10-07: YOLO11n-seg @800 knots, U-Net pith and peel wood.
Same 54 logs and human frames as v3.

Locations:
- local `/mnt/D/datasets/ct_log/kwp_ds_v4/<log>/{img,ann}`;
- euler `~/work/ctlog-eval/data/kwp_ds_v4/<log>/{img,ann}` (img/ links built by `setup_links.sh`,
  same sources as v3). Annotation tarball `data/kwp_ds_v4_ann.tgz`
  (sha256 `088998d8bf0bc90f05a8366946d82c17fcb23dab2ee9a91ceeb340d5f03ccb52`).

Manifest: `dataset_manifests/kwp-ds-v4.sha256`, 16140 annotation files, sha256 of the manifest
`65b388d2e27cebc0e18916754de03e31bb07cf2061f9013573378918a5dd34f6`. Verified on euler
(`sha256sum -c`), and `CTLogKwpDataset(54 dirs, window=1)` loads 16140 samples. Full record:
`dataset_manifests/kwp-ds-v4.DATASET.json`.

| part | logs | label source |
|---|---|---|
| auto | 3, 05, 06, 08, 09, 11, 13-21, 23-31, 33, 34, 36-44, 46-60 | Knot: `yolo11n_seg_v5_val10_800` (sha256 `47e00848…`), imgsz 800, conf 0.25, union of instance masks clipped to wood, one object per 8-connected component. Wood: `largest_cc_fill(threshold_peel(gray, 30, 60, 5) \| knot)`. Pith: `unet_pith_v1` export from the segmentation session (smp U-Net ResNet-50 800 px, `kwp-v3-unet-r50-800-humanw5-ema-teachersplit` epoch 11, EMA, checkpoint sha256 `54a6ce72…`; 3-slice clamped stack; `rint(p*778)`; no snap; tar sha256 `37f62e97…`). Log 08: the 27 human frames use human labels. |
| human | 1, 2, 4, 10 | Unchanged from v2/v3 (snapshot `human_reviewed_20260929`). |

Log-10 evidence for the choice (`ann_pipeline/DETECTOR_PROVENANCE.md`):
- knot IoU: YOLO n@800 0.750 vs U-Net 0.720;
- pith exact / <=1 px: U-Net 68 % / 96 % (mean 0.38 px) vs snapped OBB 61 % / 96 % (mean 0.43 px).

Differences vs v3 on the auto frames:
- **Knots:** frames with knot 8194 vs 8091, knot pixels per log x0.90-1.08 (median x1.03). Human
  logs are identical.
- **Pith agreement:** the U-Net point matches v3's snapped OBB on 63 % of frames, 91 % within 1 px,
  95 % within 2 px. 185 frames differ by more than 5 px (max 74.7 px); which model is right on
  those is not checked.
- **Pith presence:** the U-Net always predicts a point. 110 auto frames had no OBB pith in v3 and now
  have one.

Caveats:
- The U-Net was trained on the v3 auto logs, whose pith labels were snapped OBB, plus human logs
  1/2/4/08. Its pith on the auto logs is therefore partly in-sample.
- The U-Net checkpoint was selected on log 10.

## kwp-ds-v5 (built 2026-10-09)

kwp-ds-v4 with the knot labels on auto frames replaced by a U-Net ensemble (distillation step).
Everything else is identical to v4: the 54 logs, the human frames (including the 27 human frames of
08), pith `unet_pith_v1`, and the wood rule. Wood is recomputed from the new knots:
`largest_cc_fill(threshold_peel | knot)`.

Locations:
- local `/mnt/D/datasets/ct_log/kwp_ds_v5/<log>/{img,ann}`;
- euler `~/work/ctlog-eval/data/kwp_ds_v5/<log>/{img,ann}` (`setup_links.sh`, same sources as v4).
  Annotation tarball `data/kwp_ds_v5_ann.tgz`
  (sha256 `8404af93209e719c28550c476e497f121890f7a0bbee30f06b415b312b256d3e`).

Manifest: `dataset_manifests/kwp-ds-v5.sha256`, 16140 annotation files, sha256 of the manifest
`f43822ba841c8188be7063beb5626a5f6a1204f1069ee04db68cafea4f1b7f05`. Verified on euler
(`sha256sum -c`), and `CTLogKwpDataset(54 dirs, window=1)` loads 16140 samples. Full record:
`dataset_manifests/kwp-ds-v5.DATASET.json`. Each log has `knot_inputs_sha256`: the sha256 of the
sorted per-PNG sha256 lines actually read.

Knot source on auto frames (`build_kwp_dataset.py --knot_root`):
- Probabilities: `/mnt/D/datasets/ct_log/knot_probs_ens_v4_tta/<log>/page_XXX.png`
  (uint8 round(p*255), native 778).
- Ensemble: the arithmetic mean of two U-Nets trained on kwp-ds-v4 (teacher split), EffV2-S
  (`0a7f6dee…`) and ConvNeXt-S (`0a99076c…`), epoch 11 each. Each member uses 8-way TTA
  (rot90 x hflip), averaged before the ensemble mean.
- Threshold: p >= 128/255, clipped to wood, one object per 8-connected component, no other filters.
- Records: the segmentation session's `META.json` and the per-log tar `SHA256SUMS` are copied to
  `dataset_manifests/kwp-ds-v5.knot_probs_{META.json,SHA256SUMS}`.

Log-10 evidence (segmentation session, native-778 metric): ensemble knot IoU 0.765 (precision
0.863, recall 0.871). Single v4 U-Nets score 0.752-0.754; YOLO11n-seg @800 (the v4 knot source)
scores 0.750.

Differences vs v4 on the auto frames:
- frames with knot 7997 vs 8194;
- knot pixels per log x0.91-1.01, median x0.96. The ensemble is tighter: fewer, slimmer knots.
- Human annotation files are byte-identical. Pith is identical. 7772 of 16140 files are identical
  overall.

Leak caveat: the ensemble members were trained on human logs 1, 2, 4 and 08 plus the v4 auto logs
(teacher split). Logs 1 and 4 are therefore not clean for models trained on v5. Log 10 is clean.
The members' checkpoints were selected on log 10.
