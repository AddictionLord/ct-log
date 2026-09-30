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
