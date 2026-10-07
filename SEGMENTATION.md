# Segmentation Roadmap

Status of the DINOv3-based defect segmentation pipeline and the planned
direction. Living document.

## Current pipeline (baseline)

- **Entry point**: `src/train_dino.py` (the real one). `src/train.py` is a
  throwaway single-batch overfit sandbox — ignore it.
- **Model**: frozen DINOv3 ViT-L/16 backbone + `SimpleSegmentationHead`
  (4× `ConvTranspose2d` decoder, ×16 upsample). Linear-probe setup — only the
  head trains, backbone runs under `torch.no_grad()` in `.eval()`.
- **Features**: `get_intermediate_layers(n=1)` — last layer's patch tokens only.
- **Loss**: `0.4 · focal + 0.6 · tversky` (α=0.3, β=0.7 → recall-favoring).
  Tversky currently includes background and averages over all classes.
- **Metrics**: single macro `MeanIoU` (incl. background). No per-class IoU.
- **Logging**: `ILogger` → `LocalLogger` + `MlflowLogger` via `CombinedLogger`.
  MLflow points at `http://localhost:5000`, experiment `ct-log`.
- **Input**: grayscale CT slice loaded as RGB (`convert("RGB")`) — the same
  slice replicated across all 3 channels. **2 of 3 channels are wasted.**

### Known gaps in the baseline (prerequisites for trusting any number)

- `evaluate: false` in config → val/test/checkpoint branches never run; no
  checkpoint is actually saved.
- All three splits point at the same dir (`data/processed/set_24`) — no real
  train/val/test separation.
- No per-class IoU/Dice logged (rare defect classes invisible behind macro mean).
- Tversky includes background, diluting the rare-class signal.
- `import segmentation_head` is a bare import (not `src.`), only works from `src/`.
- `num_classes (+1)` bookkeeping is muddled across files (TODO comment exists).

## Decision: Option A — channel-stacked 2.5D (next step)

**What**: instead of replicating one grayscale slice across R/G/B, load a
3-slice axial window `[z−1, z, z+1]` and place each slice in one channel.
Predict the center slice's mask.

**Why this first**:
- **Zero architecture change.** DINOv3 still receives a 3-channel input; the
  patch embed, normalization, and head are untouched. Only the dataset
  `__getitem__` changes.
- We are currently wasting 2 of 3 channels (same slice replicated). This costs
  nothing and feeds real axial context for free.
- CT data is volumetric — adjacent slices are highly correlated (knots span
  many slices, pith is axially continuous, cracks propagate). A purely 2D model
  discards all of that. Channel-stacking captures local 3D continuity at no
  added compute.
- Highest ROI move available. ~an afternoon of dataset code.

**Limitations**:
- Window fixed at 3 (the channel budget). Captures only local axial context,
  not long-range structure.
- Per-slice ImageNet normalization still applies; the 3 channels now carry
  genuinely different content, which is the intent.

## Experiment plan

Compare **A (3-slice stack)** against the **original DINOv3 baseline**
(single slice replicated 3×), everything else held fixed.

Before either run is meaningful, close the baseline gaps:
1. Real train/val/test splits (not all `set_24`).
2. Turn `evaluate` on so checkpoints + val/test IoU actually run.
3. Log per-class IoU (populate `EpochMetrics.extra`).
4. Exclude background from Tversky.

Then run baseline vs. Option A under identical config; track in MLflow under
experiment `ct-log` with distinct run names. Compare per-class IoU on defect
classes (knot, crack, pith), not just macro mean.

## Results: A vs. baseline (2026-09-19)

Both arms ran 30 epochs, identical config, **only `window` differs**. 3-class
knot/wood/pith on full logs; log-level holdout (train = log 4, val = log 10).
MLflow experiment `ct-log-kwp-2.5d`.

At each arm's best epoch by `mean_iou_fg`:

| | epoch | wood | knot | pith | **fg mean IoU** |
|---|---|---|---|---|---|
| baseline (`window=0`, slice replicated 3×) | 23 | 0.939 | 0.269 | 0.040 | **0.4158** |
| Option A (`window=1`, `[z−1, z, z+1]`) | 28 | 0.941 | 0.266 | 0.035 | **0.4139** |
| delta | | +0.002 | −0.003 | −0.005 | **−0.0019** |

Best-across-all-epochs per class: knot 0.287 (baseline) vs 0.269 (A); pith
0.040 (baseline) vs 0.035 (A). Mean fg over the last 5 epochs: 0.4074
(baseline) vs 0.4078 (A).

**Verdict: null result.** Channel-stacked 2.5D gives no measurable gain. The
−0.0019 gap is far inside epoch-to-epoch noise (fg swings ~0.02 between
adjacent epochs in both arms), so the two arms are statistically
indistinguishable — this is "no effect", not "A is worse".

One real difference: **pith activates much earlier under A** (nonzero from
epoch 7, vs ~epoch 23 for the baseline), even though it converges to the same
place. Weak evidence that axial context helps the sparsest class learn faster,
but it does not improve the final result.

Caveats before over-reading this:
- Only **one training log** (log 4) and one val log (log 10) — a single
  holdout pair, so generalization estimates are noisy.
- Extreme class imbalance (knot ~0.09%, pith ~0.01% of pixels) — pith IoU is
  near the floor in both arms and may be dominated by annotation granularity,
  not model capacity.
- The frozen ViT-L probe with a single-layer feature + scratch decoder may
  simply be the bottleneck, masking any input-side gain.

**Implication for sequencing:** the cheap win did not materialize, so the
"A → B → C" ladder's premise (that axial context is worth exploiting) is not
yet supported. Before investing in Option B, it is probably worth attacking
the bottleneck instead — multi-layer features, a stronger head, or unfreezing
part of the backbone — and getting more annotated logs so the holdout is not a
single pair.

## Stage 1: multi-layer features (2026-09-19)

Hypothesis from the 2.5D null: the frozen ViT-L probe with **single-layer**
features and a scratch decoder is the bottleneck, masking any input-side gain.
Test: concatenate the last `n` intermediate layers as head input
(`n_layers`, 1024·n dims), everything else fixed. `window=0` throughout.

| run | best fg | **last-5 mean** | wood | knot | pith | max knot | max pith |
|---|---|---|---|---|---|---|---|
| baseline (n=1, w=0) | 0.4160 | 0.4074 | 0.939 | 0.269 | 0.040 | 0.287 | 0.040 |
| 2.5D (n=1, w=1) | 0.4140 | 0.4078 | 0.941 | 0.266 | 0.035 | 0.269 | 0.035 |
| n=2 (w=0) | 0.4260 | 0.4156 | 0.943 | 0.284 | 0.050 | 0.284 | 0.050 |
| **n=4 (w=0)** | 0.4230 | **0.4186** | 0.948 | 0.297 | 0.025 | 0.298 | 0.041 |

**Use the last-5-epoch mean, not best-epoch.** Pith IoU swings between 0.001
and 0.050 on *adjacent* epochs, so a lucky epoch flatters any run — n=2's
0.4260 "best" is a single spike while its last-5 mean sits below n=4.

**Multi-layer features help — the head was part of the bottleneck.** By last-5
mean: n=4 (0.4186) > n=2 (0.4156) > n=1 (0.4074), a **+0.011** gain for n=4.
Unlike the 2.5D null this shows up consistently across epochs and in knot IoU
(0.297 vs 0.269). It also converges much faster: pith fires around epoch 8
under n=4 vs epoch 23 for the baseline.

But the gain is **sub-linear in depth** — going 1→2 layers buys most of it,
2→4 adds little (+0.003), and n=4 costs 3.3× the head parameters (11M → 36M).
So depth is not the remaining lever.

**+0.011 is real but modest**, well short of closing the gap to usable knot
segmentation (knot IoU still ~0.30). That points at the other hypothesis:
a single training log is the data floor.

## Stage 2: more training data (2026-09-19)

Stage 1 showed depth is sub-linear, so the remaining hypothesis was the
**single-training-log data floor**. Human-collection log 1 turned out to be
fully reviewed as of September (291 frames, up from 69 in July), so training
moved to **logs 4 + 1** (582 samples, knot pixel fraction doubled to 0.00188)
with **log 10 still held out** — so every number stays comparable.

| run | best fg | **last-5 mean** | knot | pith | max knot | max pith |
|---|---|---|---|---|---|---|
| baseline (n=1, 1 log, w=0) | 0.4160 | 0.4074 | 0.269 | 0.040 | 0.287 | 0.040 |
| 2.5D (n=1, 1 log, w=1) | 0.4140 | 0.4078 | 0.266 | 0.035 | 0.269 | 0.035 |
| n=4 (1 log, w=0) | 0.4230 | 0.4186 | 0.297 | 0.025 | 0.298 | 0.041 |
| **n=4 (2 logs, w=0)** | **0.4450** | 0.4382 | **0.342** | 0.039 | **0.346** | 0.050 |
| n=4 (2 logs, w=1) 2.5D | 0.4410 | 0.4384 | 0.328 | 0.043 | 0.328 | **0.059** |

**Data is the dominant lever.** Doubling the training logs adds **+0.020**
last-5 mean on top of the multi-layer head (0.4186 → 0.4382) — roughly twice
what multi-layer itself bought (+0.011), and it lifts knot IoU far more
(0.298 → 0.346 max, vs 0.287 → 0.298 from depth alone).

Cumulative from the original baseline: fg **0.4074 → 0.4382 (+0.031)**,
knot **0.287 → 0.346 (+0.059)**.

Two qualitative changes worth noting:
- **Both rare classes are strong simultaneously** for the first time (knot
  0.333 with pith 0.050 at ep21). In the single-log runs pith only spiked on
  epochs where knot dipped, which is what made best-epoch numbers so
  unreliable.
- Knot converges far faster: 0.279 at epoch 1, versus 0.040 for the single-log
  n=4 run at the same epoch.

ValLoss rose briefly around epoch 24 (0.2547 → 0.2679) while TrainLoss kept
falling, but it recovered to 0.2581 by epoch 27, so this was a blip rather
than sustained overfitting at 30 epochs.

### 2.5D retest: the null is robust

The last row is the clean 2.5D test — **same head (n=4), same data (2 logs),
same val log, only `window` differs**. Paired over all 30 matched epochs:

```
mean diff  -0.0003      stdev 0.0102      2.5D wins 13/30 epochs
```

Last-5 means are 0.4384 (2.5D) vs 0.4382 (2D) — identical to three decimals.
The per-epoch difference is a coin flip whose spread (±0.010) is an order of
magnitude larger than its mean. **2.5D has now returned a null in two
independent settings** (single-log n=1, and two-log n=4), so the original
result was not a masking effect: axial context via channel-stacking simply
does not help this task.

The one reproducible 2.5D effect is on **pith**, the sparsest class: it
reaches nonzero IoU around epoch 7 under `window=1` versus ~epoch 23 under
`window=0`, and its max pith is the highest of any run (0.059). Axial context
reliably makes the rarest class *learn faster* and tolerate a slightly higher
ceiling, but this does not survive into the foreground mean.

## Conclusion: what actually moves the needle

Ranked by last-5 mean fg IoU gain over the original baseline (0.4074):

| lever | gain | cost |
|---|---|---|
| **more training data** (1 → 2 logs) | **+0.020** | annotation effort |
| multi-layer features (n=1 → 4) | +0.011 | 11M → 36M head params |
| 2.5D axial context (w=0 → 1) | **±0.000** | 3 slice loads per sample |

Combined, head + data take fg **0.4074 → 0.4382** and knot **0.287 → 0.346**.

**Data is the dominant lever and should be the default next investment.** The
gain from one extra log exceeded the gain from quadrupling feature depth, and
it was the only change that made knot and pith strong *simultaneously* rather
than trading off.

### Next steps

1. **Pull more reviewed logs.** Human-collection log 1 only became fully
   reviewed between July and September; check for newly completed logs (2, 08
   and 3 were partial) and retrain. This is the highest-expected-value move.
2. **Revisit schedule/augmentation** now that the data is larger — 30 epochs
   was still improving at the end of the two-log runs, so the budget may be
   short, and augmentation is untested.
3. **Do not pursue Option B (feature mid-fusion) or Option C (true 3D) on the
   axial-context rationale.** Two independent nulls say channel-level axial
   context buys nothing here; a more elaborate fusion scheme would need a
   different justification than "CT is volumetric".

## v1 model (2026-09-20)

First deliverable. Frozen DINOv3 ViT-L/16, 4 fused intermediate layers,
2.5D input (`window=1`), 320x320, trained 30 epochs at constant LR on
6 auto-annotated Phase2 logs + human log 4 (2082 samples). Model selected by
best 5-epoch-smoothed val fg IoU (epoch ~23).

Weights: `/mnt/D/models/ct-log/kwp_v1_seg_head.pth` (seg head),
`kwp_v1_full_state.pth` (full training state incl. pith head).

**Held-out test (log 10, never used for training or selection):**

| class | IoU |
|---|---|
| background | 0.997 |
| **wood** | **0.958** |
| knot | 0.356 |
| pith (segmentation) | 0.158 |
| **foreground mean** | **0.490** |

Pith regression head: **12.1px median** error (test), 17.5px (val).

**What is usable.** Wood segmentation at 0.958 is production-quality and can
replace or cross-check the `threshold_peel` detector. Knot at 0.356 finds
large knots reliably but misses thin ones, so it is useful for seeding and
review assistance, not as a final annotation. Pith at 12px is far worse than
the YOLO-OBB detector's 0.97px median - **keep using the detector for pith**.

### Known limitation: thin-knot blindness

Predictions on val show the model outputting *no knot at all* on frames whose
knots are thin faint streaks, while handling thick lobed knots well. Cause is a
training-label size bias:

| source | median knot thickness | knots/sample |
|---|---|---|
| auto logs 2,3,05,06,08 | 20-37px | 8-25 |
| human logs 1,4,10 | 7-13px | 33-48 |

Propagation merges adjacent knots into single fat blobs and misses smaller
ones, so the model learns a fat-knot prior. This is **not** fixable via the
post-processing filters (min 150px, eccentricity, solidity are all rejection
filters; `fill_holes` only fills interiors) - it originates in MedSAM2
propagation. Point seeds do not fix it either (17.9px vs ellipse 16.1px,
against human 7.2px on the same log).

### Measured diagnostics behind v1

- **Capacity is not the limit.** Train fg reached 0.792 (knot 0.675) at epoch
  30 and was still climbing, while val peaked at 0.525 - a +0.29 gap. This is
  a variance problem, not a capacity one, so higher resolution / a bigger
  decoder is *not* the priority.
- **Synthetic data helps despite its bias.** Human-only ablation (291 frames,
  log 4) reached val fg 0.427 vs 0.525 for the mixed 2082-frame set, and
  overfit *harder* (+0.329 vs +0.291). Keep the auto data.
- **Data available for scaling:** 55 raw CT volumes on disk (~16,500 frames),
  only 9 annotated. Supervisely holds no further annotated logs.

### Next steps

1. **Augmentation** - variance is confirmed as the binding constraint. Include
   scale/erosion on knot masks specifically, to counter the fat-knot prior.
2. **More auto logs** - the ablation shows volume helps; 46 unannotated raw
   logs are available. Generate in a small batch first and re-measure the
   thickness bias before scaling.
3. **Use v1 to improve propagation** - the wood head (0.958) is already good
   enough to constrain propagation, closing the loop.

## v1-extended (2026-09-21) — trained on 20 auto logs

Same architecture as v1 (frozen DINOv3 ViT-L/16, 4 fused layers, 2.5D
`window=1`, 320x320, constant LR, pith regression head). Changes: **20 auto
training logs** (6 existing Phase2 + 14 newly generated with retrained
detector v4) and **val moved to human logs 1+4**, test still human log 10.

**Held-out test (log 10), best checkpoint (epoch ~12 by smoothed val fg):**

| class | v1 (7 logs) | **v1-extended (20 logs)** |
|---|---|---|
| wood | 0.958 | **0.966** |
| knot | 0.356 | **0.390** |
| pith (seg) | 0.158 | **0.236** |
| **fg mean** | 0.490 | **0.531** |
| pith regression (median px) | 12.1 | **5.2** |

**The extended dataset worked: test fg 0.490 -> 0.531 (+0.041).** Pith
regression more than halved its error (12.1px -> 5.2px), and knot gained
+0.034. Every class improved.

Val peaked at fg 0.557 (epoch 12); train-val gap at epoch 10 was +0.137,
roughly half the +0.204 seen at the same epoch with 7 logs - more data reduced
overfitting as expected.

**Attribution warning.** Three variables moved at once (detector v3->v4,
6->20 auto logs, val composition 1 -> 1+4), and the run reached only ~16
epochs where v1 peaked at 23. The +0.041 is a floor, not a converged result,
and cannot be attributed to any single change.

**Detector v4** (by the Annotations session): mAP50 0.966 / mAP50-95 0.7395 at
epoch 149 (v3: 0.952 / 0.695), trained on logs 1/4/2/08/3, val log 10. Box
width on held-out log 10: v3 16.1px -> v4 14.6px, human GT 12.9px.

### Run did not complete

Crashed three times and was not resumed: rc=132 (SIGILL) after 6.4h, then
rc=1, then a third start that died without the supervisor recording it - the
supervisor process itself died, so auto-resume never fired. Training stopped
at 2026-09-21 10:02 mid-epoch 16. Full state is in
`kwp_v1_extended.resume.pth`, so the run is resumable from epoch ~15.

### kwp-v1-baseline-overnight: real converged baseline (2026-09-25)

Reran v1-extended's split (20 logs, val=1+4, test=10) cleanly to completion on
euler: 40 epochs, batch 8, cosine LR (the earlier run used constant LR and
never converged — see "Run did not complete" above). Code at `729eda9`.

**Test: fg 0.536, knot 0.419, wood 0.967, pith median 5.75px.** Best smoothed
val fg 0.5057 at epoch ~39 (raw peak 0.5526). This is a genuine converged
number, not a floor — first clean finish this dataset has had.

Train-eval gap grew from +0.069 (epoch 0) to +0.330 (epoch 30): train fg
reached ~0.87 while val plateaued at 0.53-0.55 from epoch ~12 onward. Capacity
is not the constraint; the model overfits without augmentation (none is used
yet — see "Later options" below).

### Class-balanced loss weighting: negative result (2026-09-25)

Hypothesis: focal+Tversky with no class weighting under-serves knot (0.14% of
train pixels) and pith (0.0059%) relative to wood (10.7%) and background
(89.1%). Tried inverse-frequency weighting (`power=1.0`) on the same split/
schedule as the baseline above, for a same-conditions comparison.

`effective_number_weights` (Cui et al. 2019) was tried first and rejected
before launch: at pixel-count scale (1e5-1e9), the paper's typical beta
(0.9-0.9999, calibrated for instance/image counts) makes `beta**n` underflow
to 0 for every class, silently collapsing to uniform weights. A correctly
scaled beta (~1-1e-7) exists but only separates knot/pith from wood/
background, not wood from background (both still underflow) — see
`src/utils/class_balance.py` and `scripts/compute_class_pixel_counts.py`.

| metric | baseline | class-weighted | delta |
|---|---|---|---|
| test fg | **0.536** | 0.447 | -0.089 |
| test knot | **0.419** | 0.318 | -0.101 |
| test wood | 0.967 | 0.799 | -0.168 |
| test background | 0.998 | 0.978 | -0.020 |
| test pith seg IoU | 0.222 | 0.223 | tied |
| test pith error (median px) | 8.29 | **5.59** | better |
| best smoothed val fg | 0.5057 | 0.4644 | -0.041 |

**Made everything worse, including knot** — the class it targeted. Weights
used: background 0.00026, wood 0.0021, knot 0.158, pith 3.84 (~15,000:1 pith:
background ratio). Val fg started at 0.076 (vs baseline's 0.42 at epoch 0),
took ~16 epochs to reach where the baseline started, then plateaued ~0.46.
Train-eval gap stayed smaller throughout (+0.222 vs +0.330 at epoch 30) — not
because it generalized better, but because it was still underfitting the easy
classes at that point, never catching up to the baseline's optimum.

The one plausible positive: pith localization error dropped from 8.3px to
5.6px median despite flat pith seg IoU. Pith is a separate coordinate-
regression head not touched by these seg weights, so this is likely an
indirect effect via shared backbone-adjacent feature use, not a mechanism the
experiment was testing. Uncertain, not re-verified.

**Conclusion: reject `power=1.0` inverse-frequency weighting.** If revisited,
try a much gentler weighting (e.g. `power=0.3-0.5`, or effective-number with
beta scaled to touch only knot/pith, leaving wood/background near 1.0) rather
than assuming stronger reweighting helps more — this run is evidence it
doesn't. Not attempted this session; the mechanism (`class_weighting: none|
effective_number|inverse_frequency` in `KwpTrainingConfig`) is in place and
tested (22 unit tests), so a follow-up is a config change, not new code.

### Thin-knot filter analysis (for v2)

The annotation filters reject essentially all thin knots. Ablation over
generated logs 13-16 (543 instances entering filters, 274 thin <8px):

| filter | rejected | thin rejected | share of thin losses |
|---|---|---|---|
| area<150 | 227 | 224 | 84% |
| solidity<0.85 | 63 | 43 | 16% |
| eccentricity<0.7 | 34 | **0** | **0%** |
| kept | 219 | **7** | — |

**97% of thin knots are rejected.** Propagation produces thin knots in
quantity; the filters discard them. `eccentricity<0.7` contributes nothing -
it rejects values *below* 0.7, i.e. round blobs, so thin streaks pass
trivially. Note the chain is ordered with area first, so these are
first-responsible shares, not independent contributions.

Proposed v2 experiment: 2x2 over area (150/40) x solidity (0.85/0.6), leaving
eccentricity alone, plus a real-vs-noise check on recovered instances (score
them against human labels on log 10, which is out of both detectors' training).

## kwp-ds-v2 runs: labels, resolution, fine-tuning, augmentation (2026-09-30 to 2026-10-02)

All runs on euler (RTX 5070 Ti), DINOv3 ViT-L/16, capped class weighting, no pith segmentation,
pith regression head, cosine LR. Train = 19 auto logs + human log 2, val = human logs 1 and 4,
test = human log 10. v1 and v2 val numbers are not comparable (v2 corrected 83 val frames);
test log 10 is identical in both. The superseded filter sweep above was replaced by kwp-ds-v2
(plain YOLO-seg labels, no filter chain; see `ann_pipeline/DATASET_VERSIONS.md`).

Test log 10:

| run | fg | knot | wood | pith median |
|---|---|---|---|---|
| v1 320 frozen, 40 ep | 0.699 | 0.436 | 0.961 | 5.61 px |
| v2 320 frozen, 40 ep (repeat) | 0.728 (0.726) | 0.493 (0.492) | 0.963 | 5.58 (5.27) px |
| v1 784 frozen, 40 ep | 0.778 | 0.585 | 0.972 | 4.27 px |
| v2 784 frozen, 20 ep | 0.802 | 0.633 | 0.971 | 4.71 px |
| v2 784 frozen, 40 ep, bf16 | 0.808 | 0.643 | 0.973 | 4.72 px |
| v2 784 last 4 blocks, lr 1e-5, 20 ep | 0.822 | 0.672 | 0.973 | 4.90 px |
| v2 784 last 4 blocks, lr 3e-5, 20 ep | 0.828 | 0.680 | 0.975 | 4.82 px |
| v2 784 last 4 blocks, lr 3e-5, aug, 20 ep | 0.823 | 0.674 | 0.973 | 6.66 px* |
| v2 784 last 4 blocks, lr 3e-5, aug, 20+20 ep (warm restart) | 0.825 | 0.677 | 0.974 | 4.85 px |

\* pith head taken from the final epoch, not the best one (bug fixed afterwards).

Findings:

- **Labels**: kwp-ds-v2 adds ~+0.05 test knot at both 320 and 784. Seed noise at 320 frozen is
  ~0.001 knot (two identical runs), so this is far above noise.
- **bf16 backbone** (head, loss, optimizer in fp32): 5.5x backbone throughput, feature cosine
  0.9994 vs fp32, 99.99% identical predicted pixels. 784 epoch 29 -> ~7.5 min.
- **Bias/variance via the log-2 probe** (`train_probe_logs`: the human-labelled train log
  evaluated separately). At 320 the gap probe -> val is 0.24 knot (variance); at 784 frozen it is
  small (~0.07) with val flattening at ~0.67 (capacity limit); fine-tuning the last 4 blocks
  breaks that ceiling but reopens the gap (0.10 at ep 15); augmentation (rotation 0-360, flip,
  intensity jitter) closes it to 0.03 and gives the lowest val loss of all runs.
- **Per-log paired block bootstrap** (`scripts/eval_per_log.py`, blocks of 20 slices, 2000
  resamples; logs 1/4 were used for checkpoint selection) over logs 1+4+10, knot difference
  vs last4 lr 3e-5: aug 20+20 +0.0017 [-0.0030, +0.0064] (tie), lr 1e-5 -0.0047 [-0.0086,
  -0.0012], frozen 40 ep -0.0162 [-0.0227, -0.0097]. The fine-tuning gain is log-dependent
  (+0.037 on log 10, not significant on log 1). Single-log 95% intervals are about +-0.04, so
  differences between models are now smaller than differences between logs.

Conclusion: tuning on 20 training logs has saturated (models within ~0.005 knot). The next
lever is more distinct training logs; with augmentation in place, more backbone capacity is
the second candidate.

## kwp-ds-v3 runs: more auto logs and human-frame weighting (2026-10-03)

kwp-ds-v3 = v2 + 31 new auto-labelled logs (same YOLO-seg teacher, auto pith snapped to the
darkest pixel in 5x5): 51 train logs, ~15.3k frames (2.6x v2); val/test unchanged. Recipe as the
best v2 run (784, last 4 blocks lr 3e-5, head 1e-4, bf16, augmentation), 12 epochs (~1.5x the v2
step budget). `human_frame_weight` samples `bumaska`-tagged frames (all of log 2 + 27 frames of log
08, 317 of 15265) w times as often.

Knot IoU, paired block bootstrap over human logs 1+4+10 (`scripts/eval_per_log.py`):

| model | knot pooled | knot log 10 | pith median pooled | pith log 10 |
|---|---|---|---|---|
| v2 aug 20+20 (reference) | 0.685 | 0.677 | 6.04 px | 4.85 px |
| v3 uniform | 0.676 | 0.668 | 5.09 px | 5.20 px |
| v3 human 5x | 0.682 | 0.674 | 5.99 px | 5.03 px |
| v3 human 10x | 0.680 | 0.670 | 5.71 px | 5.19 px |

Paired differences (pooled knot): v3 uniform - v2 = -0.0096 [-0.0135, -0.0053]; human 5x - v3
uniform = +0.0060 [+0.0028, +0.0092]; human 10x - human 5x = -0.0012 [-0.0050, +0.0022];
v2 - human 5x = +0.0036 [-0.0008, +0.0075] (significant only on log 4, +0.012).

Findings:

- **More teacher-labelled logs alone made knots worse** (human share diluted from ~4.9% (v2) to
  ~2.1% of draws). *Correction 2026-10-06:* an earlier version of this note said the student "moved
  back to the teacher's level", converting the teacher's per-frame Dice 0.804 to IoU ~0.672. That
  conversion is only valid for pooled Dice; the teacher's pooled knot IoU on log 10 is **0.745**,
  well above every student (see "Student vs teacher" below). The dilution effect itself is
  measured between our own runs and stands.
- **Human-frame weighting recovers most of it and saturates at ~5x.** 10x is on par with 5x.
  Human labels are worth several times more than teacher labels; further gains need more
  distinct human-reviewed logs rather than heavier weighting of the one we have.
- **Pith moves the other way**: uniform v3 has the best pith (5.1 px pooled); weighting human
  frames pulls it back to ~6 px, likely because human pith points and the snapped auto pith
  points are not consistent. `human_weight_applies_to_pith: false` keeps the 5x sampling for
  segmentation but weights human frames 1/w in the pith loss.

### Warm restart of human 5x with decoupled pith (2026-10-04)

12 more epochs from the final 5x weights (`init_from`), fresh cosine at half the peak LR, 5x human
sampling, human frames weighted 1/5 in the pith loss.

| model | knot pooled | knot log 1 / 4 / 10 | pith median pooled | pith log 10 |
|---|---|---|---|---|
| v3 human 5x | 0.682 | 0.686 / 0.682 / 0.674 | 5.99 px | 5.03 px |
| **v3 human 5x, 12+12 ep, pith decoupled** | **0.684** | 0.689 / 0.687 / 0.670 | **5.13 px** | 5.01 px |
| v2 aug 20+20 | 0.685 | 0.683 / 0.694 / 0.677 | 6.04 px | 4.85 px |
| v3 uniform | 0.676 | 0.679 / 0.677 / 0.668 | 5.09 px | 5.20 px |

Paired vs v3 human 5x (pooled): warm restart knot +0.0021 [-0.0006, +0.0049], fg +0.0015
[+0.0001, +0.0029]; v2 +0.0036 [-0.0008, +0.0075].

- **Undertraining is at most a small effect**: 12 more epochs add ~+0.002 knot (not significant
  pooled; +0.005 on log 4, -0.004 on log 10, both within noise). Val knot peaked at 0.694 (ep 9),
  the highest of all runs, but test log 10 did not move.
- **The pith decoupling works**: pooled pith 5.99 -> 5.13 px (log 4: 6.14 -> 4.41 px), back to the
  level of uniform v3, while keeping the knot gain of human weighting. (The pith change is
  confounded with the continuation, but the continuation alone did not move pith in v2.)
- **Best overall model so far**: knot on par with v2 aug (pooled 0.684 vs 0.685, v2 still ahead on
  log 4 and log 10 by ~0.007, within noise) and the best pith of all knot-competitive models.
  Checkpoint: `ckpt/kwp_v3_784_last4_aug_humanw5_cont_bf16.*` (head, `.backbone.pth`,
  `.pith.pth`) on euler.

### Student vs teacher on clean log 10 (2026-10-06)

Measured by the Annotations agent from exported student probabilities (`scripts/export_knot_probs.py`,
consistency check: student pooled IoU 0.675 at p>=0.5, native 778, vs 0.671 in our eval) and the
teacher's masks, same 293 slices, pooled IoU (`logs/autoann_20261006/REPORT.local.md`):

| model | pooled knot IoU | Dice / frame | straight-knot Dice |
|---|---|---|---|
| YOLO11n-seg teacher `yolo11n_seg_v5_val10` (conf >= 0.25) | **0.745** | 0.804 | 0.817 |
| best DINOv3 student (p >= 0.5) | 0.676 | 0.737 | 0.751 |
| mean of both (>= 0.5) | 0.741 | 0.776 | 0.780 |

The 6M-parameter YOLO11n-seg beats our best student by ~0.07 IoU, and fusing them does not beat the
teacher. Candidate causes: (a) resolution: the DINOv3 head decodes from a 49 x 49 token grid (16 px
patches) without high-resolution skips, while thin knots are ~8 px wide; YOLO-seg masks come from a
multi-scale FPN at native resolution; (b) distillation: the student learns mostly from the teacher's
labels; (c) data, a confound: the teacher trained on ~900 human frames of logs 1, 2, 4, 08, the
student on 317 human frames (logs 1 and 4 are its validation set), so the teacher had ~3x the human
data.

Knot-width breakdown on log 10 (`fusion_log10.npz` teacher masks vs exported student probabilities):
recall is equal (teacher 0.883, student 0.879) and equal per width bucket for knots > 10 px; the student
misses more knots < 10 px (pixel recall 0.68-0.71 vs 0.76, only 7% of knot pixels) and is better on round
knots. The gap is precision (0.745 vs 0.827): of the student's 70k false-positive pixels, 49k sit within
5 px of a real knot (teacher: 29k), so the student's knots are too wide. Raising the student threshold
only reaches 0.692 (tuned on log 10), so it is boundary localization, not calibration.

**Data confound ruled out (2026-10-07)**: the same best recipe trained on the teacher's human split
(human logs 1, 2, 4 and the 27 frames of 08 in training, 899 human frames; log 10 for validation and
test; `train_kwp_v3_784_last4_aug_humanw5_ema_pithhm_teachersplit_bf16.yaml`) reaches test knot IoU
0.672 on log 10, the same as with 317 human frames (0.671); pith 0.86 px, wood 0.974. Three times the
human data does not close the gap, so it lies in the model (coarse 49 x 49 token decoder, recall-biased
loss), not in the data. Next: YOLO11 s/m/n-seg at 800 px and a U-Net (ResNet-50, 800 px) on the same
split.

### Single-model comparison on the teacher's split (2026-10-07)

All trained on human logs 1, 2, 4, 08 (899 frames) + auto logs, best checkpoint selected on log 10,
evaluated on log 10. Knot IoU pooled over the 293 annotated slices at native 778 (YOLO numbers: the
Annotations agent's function, wood-clipped; U-Net/DINO: exported probabilities, unclipped; clipping
changes the teacher by +0.001).

| model | knot IoU (p >= 0.5 / conf 0.25) | precision | recall | pith median | wood IoU |
|---|---|---|---|---|---|
| YOLO11n-seg @640 (teacher) | 0.745 | 0.827 | 0.883 | (OBB detector: 0.81 px) | – |
| YOLO11n-seg @800 | **0.750** | | | | – |
| YOLO11s / m-seg @800 | 0.738 / 0.734 | | | | – |
| DINOv3 ViT-L, last 4 blocks, 784 | 0.676 (training eval 0.672) | 0.745 | 0.879 | 0.86 px | 0.974 |
| **U-Net ResNet-50, 800** | 0.720 (training eval 0.708) | 0.772 | **0.914** | **0.41 px** | **0.983** |

- **Bigger YOLO does not help** (n > s > m at 899 frames): data-limited, not capacity-limited.
- **U-Net vs DINOv3**: +0.044 knot IoU, half the pith error, higher wood IoU, at a fraction of the
  compute. The full-resolution decoder fixes most of DINO's boundary problem.
- **U-Net vs YOLO**: the U-Net finds more knots (recall 0.914 vs 0.883; thin knots < 10 px pixel
  recall 0.81-0.84 vs 0.76; elongated 0.89 vs 0.84) but draws them wider (false-positive pixels at
  knot edges 43k vs 29k). At threshold 0.8 its recall equals the teacher's (0.883) at precision 0.818
  (IoU 0.738; threshold tuned on log 10, optimistic). The remaining gap is boundary width, likely from
  the recall-biased Tversky loss (beta 0.7).
- **Pith** from the U-Net heatmap channel (0.41 px median, p90 0.80 px) is the best of all models,
  better than the YOLO OBB detector (0.81 px).

### Seed noise of fine-tuned runs: the caveat on everything above (2026-10-04)

An identical repeat of the v3 human 5x run (no fixed seed) gives, paired vs the original:
log 1 -0.009 [-0.014, -0.003], log 4 -0.002 [-0.006, +0.003], log 10 -0.018 [-0.028, -0.009],
pooled **-0.009 [-0.013, -0.006]** knot (test log 10: 0.674 vs 0.655). The bootstrap intervals
exclude 0 because they only capture evaluation noise; training noise of a fine-tuned 784 run is
~0.01 knot pooled and ~0.02 on a single log, about 10x the frozen-backbone value (0.001 at 320).

Consequence: single-run differences below ~0.01 (v3 uniform vs v2 -0.010, human 5x vs uniform
+0.006, warm restart +0.002, 10x vs 5x) are not conclusive. Robust so far: kwp-ds-v2 labels over
v1 (~+0.05), fine-tuning the last 4 blocks over frozen (+0.016 pooled, +0.04 on log 10), 784 over
320. Future config comparisons need 2-3 seeds per arm (or variance reduction: SWA/EMA weights,
ensembles).

### EMA of the trainable weights (2026-10-04)

`ema_decay: 0.9995` (~1 epoch horizon), v3 human 5x with decoupled pith, two identical seeds.
Paired over logs 1+4+10:

| pair | knot pooled (each) | seed-pair knot diff pooled | pith median pooled (each) |
|---|---|---|---|
| no EMA (5x, 5x repeat; pith not decoupled) | 0.682 / 0.673 | -0.0090 [-0.0125, -0.0057] | 5.99 / 5.99 px |
| EMA (s1, s2; pith decoupled) | 0.683 / 0.678 | -0.0051 [-0.0084, -0.0019] | 4.98 / 4.76 px |

- Seed spread roughly halves (0.009 -> 0.005 pooled; log 10 0.018 -> 0.003), mean knot unchanged
  (0.6805 vs 0.6775, within noise). One pair per arm, so this is indicative, not proven.
- Pith improves by ~1 px pooled and is consistent across the two EMA seeds (log 10: 4.38 / 4.38 px
  vs 5.03 / 5.44 px). Confounded with the decoupled pith loss (decoupling alone, in the warm
  restart, gave 5.13 px).
- EMA costs nothing at train time; keep it on (`ema_decay: 0.9995`) for future runs.

### Pith: heatmap / soft-argmax head (2026-10-05)

The original pith head (`PithRegressionHead`) mean-pools all patch tokens and regresses (x, y)
with an MLP, so it can only recover position indirectly from the pooled vector. Diagnosed on the
best fine-tuned model: correlation with the true pith 0.85 on log 10, but worse than a constant
(the log's mean pith) on train log 2; on a synthetic task where position is only encoded in the
patch grid it cannot localize at all (246 px vs 8.6 px for a heatmap head).

`pith_head_type: heatmap` (`PithHeatmapHead`) keeps the 49x49 patch grid, decodes a 196x196 logit
map and predicts the soft-argmax coordinate (DSNT-style); trained with `pith_loss_type: l1`,
weight 10. Same run as the frozen v2 784 40-epoch bf16 baseline except the pith head:

| pith head | test log 10 pith median | mean | p90 | val pith median (ep 39) | test knot |
|---|---|---|---|---|---|
| pooled MLP (baseline) | 4.72 px (4.44 in per-log eval) | – | – | 7.0 px | 0.643 |
| **heatmap / soft-argmax** | **1.05 px** | 1.19 px | 2.16 px | 0.87 px | 0.644 |

Pith reaches the level of the YOLO 2-class OBB detector (0.81 px median on log 10) already after
one epoch (val 1.2 px), with no effect on segmentation. The heatmap head is the default choice for
pith from now on.

**Best model (2026-10-05)**: the best fine-tuned v3 recipe (784, last 4 blocks lr 3e-5, aug, human
5x with decoupled pith loss, EMA 0.9995, 12 epochs) with the heatmap pith head
(`src/configs/train_kwp_v3_784_last4_aug_humanw5_ema_pithhm_bf16.yaml`). Paired over logs 1+4+10:

| model | knot pooled | knot log 10 | pith median pooled | pith log 1 / 4 / 10 |
|---|---|---|---|---|
| **EMA + heatmap pith** | 0.683 | 0.671 | **0.78 px** | 0.78 / 0.68 / 0.85 px |
| EMA s1 (pooled-MLP pith) | 0.683 | 0.672 | 4.98 px | 6.83 / 4.29 / 4.38 px |
| v2 aug 20+20 | 0.685 | 0.677 | 6.04 px | 6.57 / 6.30 / 4.85 px |

Knot vs EMA s1: +0.0000 [-0.0038, +0.0034]; v2 aug vs it: +0.0020 [-0.0019, +0.0054] (both ties).
Test log 10: knot 0.671, fg 0.823, pith median 0.85 px, mean 0.98 px, p90 1.64 px. Checkpoints:
`ckpt/kwp_v3_784_last4_aug_humanw5_ema_pithhm_bf16.{pth,backbone.pth,pith.pth}` on euler (EMA
weights), best copy `ckpt_local/kwp_v3_784_last4_aug_humanw5_ema_pithhm_bf16/.../seg_head_epoch_11.pth`.

Student-vs-teacher disagreement (best model vs v3 auto labels, knot IoU; logs were in training, so
only the ranking matters): 50 0.596, 09 0.685, 52 0.698, 06 0.701, 42 0.701, 53 0.705, 41 0.730,
3 0.763, 08 0.779, 05 0.784. Used to rank logs for human review.

QA of the 31 new auto logs (Annotations agent, `logs/kwp_ds_v3/qa_auto_logs.{txt,json}`): teacher
confidence, near-threshold share, outer-ring spill and wood-mask stability match the v2 auto logs
(conf mean 0.559 vs 0.561; spill 0.048 vs 0.045). Only log 50 is out of distribution (dark,
irregular heartwood; lowest confidence 0.476, likely missed knots); logs 41/42/53 have mildly more
ring spill (~0.10 vs v2 max 0.08). Per-log label quality does not explain v3 < v2; the mixture
share of human vs teacher labels does.

## Later options — superseded

The original plan was a ladder: **A** (channel-stacked 2.5D) → **B**
(mid-fusion of per-slice features) → **C** (true 3D), on the premise that
axial context is worth exploiting because CT is volumetric.

**That premise did not survive the experiments.** A returned a null twice, in
two independent settings. B and C are more elaborate ways of feeding the model
the same axial information that A showed it cannot use here, so the ladder's
rationale is gone. They are kept below only for reference:

- **Option B — mid-fusion of per-slice features**: run frozen DINOv3 on N
  slices, fuse patch features (axial attention / 3D conv) before the decoder.
  Larger axial receptive field than channel-stacking. Would need a new
  justification, not the volumetric-prior argument.
- **Option C — true 3D** (3D U-Net / 3D-patch transformer on sub-volumes):
  abandons DINOv3 pretraining and needs far more labeled volume than is
  available. Heaviest lift.

See **Conclusion: what actually moves the needle** above for the replacement
plan — more annotated logs first, then schedule/augmentation.
