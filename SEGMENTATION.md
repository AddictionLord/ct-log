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

## Later options (decide after A vs. baseline)

- **Option B — mid-fusion of per-slice features**: run frozen DINOv3 on N
  slices, fuse patch features (axial attention / 3D conv) before the decoder.
  Keeps backbone frozen; cache features to disk to amortize N× forward passes.
  Larger axial receptive field than channel-stacking. Moderate effort.
- **Option C — true 3D** (3D U-Net / 3D-patch transformer on sub-volumes):
  highest accuracy ceiling and proper 3D consistency, but abandons DINOv3
  pretraining and needs far more labeled volume than currently available.
  Realistic only once the semi-automatic annotation pipeline has filled out
  several full logs. Heaviest lift — do not start here.

**Sequence**: A (now) → measure → B if axial context pays off → C only if A/B
plateau and annotation volume justifies dropping the DINOv3 prior.
