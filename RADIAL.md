# Thin knots: radial and knot-axis views (log 10)

Status on 2026-10-17: **closed on the model side.** Radial views, knot-axis planes, a z-consistency
post-step and a context ablation were tried on log 10. None of them gives a model-side gain on thin knots.
The only real gain was M4 → M5, and M5 is already the Phase2 default (see
[ann_pipeline/PHASES.md](ann_pipeline/PHASES.md)). What is left is a labelling-convention question for the
annotators.

History: phase 1 ran on Precision on 2026-10-15 (session "Axial experiments"). On 2026-10-16 it moved to
Vostro: the scripts were rebuilt from that session's transcript and reproduce its numbers exactly.

## Where things are

- **Code:** [scripts/radial/](scripts/radial/). Run everything from the repo root with the `ct-log` conda env.
  - [scripts/radial/README.md](scripts/radial/README.md) describes the data layout and geometry.
  - [scripts/radial/FINDINGS_PHASE1.md](scripts/radial/FINDINGS_PHASE1.md) holds the qualitative gallery notes. Its links are relative to the data folder.
- **Data (Vostro only, 288 MB, regenerable):** `/mnt/D/datasets/ct_log/radial/log10/`.
  - **Inputs:** `kwp_ds_v3/10`, `knot_probs_ens_v4/10` (M4), `report_20261010/pred_m8`, `knot_probs_m5/10` and `model_outputs_m5/10` (M5).
  - **M5 checkpoint:** `/mnt/D/models/ct_log/m5/seg_head_epoch_47.pth` (sha256 `228ed105…88b0`). Log 10 is held out.

Regenerate:

```bash
python -m scripts.radial.build_layout      # images/, annotations/, predictions/m4, meta.json
python -m scripts.radial.knots             # knots.json (GT + M4 instances, thin flag, Hungarian IoU, z±1 neighbours)
python -m scripts.radial.make_radial       # radial/ planes + geometry assertion
python -m scripts.radial.gallery           # gallery/index.html
python -m scripts.radial.z_consistency     # z_consistency.json
python -m scripts.radial.branch_planes     # branch_planes/ (oracle knot-axis planes, contrast)
python -m scripts.radial.context_ablation  # predictions/m5_nocontext/, context_ablation.json (GPU, ~5 min on a GTX 1650)
```

Still only on Precision (`/home/mary/dev/ct-log`, uncommitted), not needed for anything above:
- `report_2026-10/radial_prompt.md`: the original phase-1 task.
- `scripts/export_slicer_log.py` and `scripts/unwrap_slice.py`.
- `/mnt/D/datasets/ct_log/slicer/log10/`: NRRD volumes, `open_log10.py`. These are needed only for `slicer_check.py`, which runs inside 3D Slicer; Slicer is not installed on Vostro.

## Results

The cut-off between thin and thick knots is minor axis < 8 px. A knot counts as found when its per-knot
Hungarian IoU is ≥ 0.3. Log 10 has 272 GT knots, 39 of them thin.

### 1. Radial planes (phase 1)

- **Thin knots are rays almost in the axial plane:** median radial extent 82 px, width about 5 px.
- **They span very few slices:** in a radial plane a thin knot is a band 1–2 rows tall.
- **Pairs of slices:** 35 of 39 thin GT knots are labelled in exactly two adjacent slices.
- **What M4 misses:** 11 thin knots, all at IoU 0. 10 of them are the fainter second slice of a branch that M4 finds in the neighbouring slice.
- **Not visible in the model at all:** in 9 of those misses, M4's probability is ≤ 1/255 everywhere inside the knot.
- **Geometry verified:** row z of each knot-aligned plane equals the axial label along the ray, for all 272 GT knots. A 3D Slicer reslice matches the numpy sampler: MAE 0.25 grey levels, correlation 0.99998.

### 2. Z-consistency post-step: rejected

The rule lowers the threshold to t_low wherever M4 has a knot at the same place in z±1. The results come
from `z_consistency.json`; t_low was swept on the test log, so even the best row is optimistic.

| | per-knot IoU | thin found | pixel IoU | precision | recall |
|---|---|---|---|---|---|
| M4 baseline | 0.705 | 28/39 | 0.763 | 0.865 | 0.866 |
| t_low 0.40 | 0.708 | 29/39 | 0.756 | 0.836 | 0.888 |
| t_low 0.05 | 0.672 | 30/39 | 0.717 | 0.759 | 0.928 |

### 3. Label continuity on the human logs

Share of thin knots that are isolated, have one labelled neighbour slice, or have both, computed from
`report_20261010/thin_width_records.json`:

| log | isolated | one neighbour | both |
|---|---|---|---|
| 1 | 9 % | 79 % | 12 % |
| 2 | 10 % | 78 % | 12 % |
| 4 | 19 % | 72 % | 9 % |
| 10 | 8 % | 90 % | 3 % |

The labels show no systematic omission of second slices; log 4 is the outlier. This proxy cannot count slices
that are visible but unlabelled.

### 4. Planes along the knot axis: dropped

`branch_planes.py` fits each thin knot's rise from its GT pixels in both slices, which is an oracle, and
samples a plane tilted along it.
- **The tilt is tiny:** the rise is 0.002–0.035 slice/px, at most about half a slice over the knot, so the plane is effectively axial.
- **No new evidence:** the tilted plane's higher contrast is the partner slice's line blended in, not the missed piece becoming visible. A refinement step on these planes would have nothing to work with.
- **Contrast of the missed pieces:** in the axial slice, on each knot's own radius range, the misses have median contrast 16 grey levels against 23.5 for found knots. 4 of the 11 misses are at 21–29, so low contrast is only part of the story.

### 5. M5 and z±1 context ablation

`context_ablation.py` runs M5 with the input stack [z, z, z] instead of [z−1, z, z+1]. It reproduces the
stored M5 probabilities to within 3/255.

| log 10 | pixel IoU | precision | recall | per-knot IoU | thin found |
|---|---|---|---|---|---|
| M4 | 0.763 | 0.865 | 0.866 | 0.705 | 28/39 |
| M5 | 0.766 | 0.852 | 0.884 | 0.730 | 33/39 |
| M5, no z±1 context | 0.737 | 0.904 | 0.800 | 0.663 | 25/39 |

- **M5 already fixes part of it:** M5 finds 5 of M4's 11 misses (004_01, 079_00, 122_00, 175_00, 215_02).
- **Context helps:** without z±1, M5 loses thin knots and finds none of the misses. The idea that "the model suppresses second slices because of the neighbour" is refuted.

### 6. The 6 knots M5 still misses

- **4 faint thin knots,** contrast 6–12: 203_00 (isolated, plausibly the leading edge of a big knot two slices later), 093_01, 021_00 and 026_00.
- **2 small stubs next to the pith:** 069_00 and 227_00, at r < 22 px.
  - Each is the faint tail of a knot that is clearly visible in z−1, where M5 segments it, so they look like partial-volume effects.
  - Inside both, M5 predicts class wood with p_knot = 0, and its pith point is correct (0 and 1 px off).
  - Crops: `branch_planes/near_pith_069_227.png`.
- **No failure in the pith region.** M5 knot recall by distance from the pith:
  - 0.41 at 0–5 px and 0.64 at 5–10 px, but that is only about 1,300 of roughly 233k GT knot pixels;
  - ≥ 0.82 from 10 px out;
  - 0.77 beyond 120 px, near the bark.

## Open follow-ups

1. **Labelling convention for thin knots,** to agree with Rosťa and the annotators. Should faint second slices and partial-volume tails (like 069_00 and 227_00) be labelled at all? The answer decides whether the 6 remaining misses are errors or label noise. This needs annotator time, so it was parked on 2026-10-17.
2. **Annotation QA tooling, if wanted later.** Knot-aligned radial strips of raw voxels show z-continuity at a glance: missing second slices, single-slice leading edges, unlabelled branches such as M4's 066_01. Two candidate forms:
   - a Slicer script that steps through knots and sets the radial and branch-aligned planes;
   - per-knot PNG strips.
   Nothing has been uploaded to Supervisely.
3. **Phase2** already holds M5 pre-labels (all classes from the model, 2026-10-16) on every frame without human annotation: logs 3, 05, 06, 09 and the untagged frames of 08. Logs 50 and 52 have M5 preview builds (`preview_m5_model_5052_build`) but are not in Phase2.
