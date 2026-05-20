# CT Log Segmentation — Semi-Automatic Annotation Pipeline
### Full Technical Reference for Claude Code
 
**Project:** CT Log Detection / Digiwood / LignoSilva Center of Excellence, Zvolen, Slovakia  
**Date:** April 2026  
**Status:** Architecture decided, ready for implementation
 
---
 
## 1. Project Context
 
The goal is to develop a semantic segmentation model for CT scans of wooden logs. CT scanning allows non-destructive inspection of internal log structure, enabling optimised sawing plans and quality grading. The project is part of the **Digiwood** research initiative under the **LignoSilva** Center of Excellence.
 
The immediate task is building a **semi-automatic annotation pipeline** that minimises annotator effort by having humans refine model-generated masks rather than annotate from scratch.
 
---
 
## 2. Dataset
 
### Annotated Seed Set
- ~24 fully annotated CT slice images
- Annotated in **Supervisely**
- Source: primitive annotations (points, bounding boxes, rough masks) from a prior related project
### Raw Unannotated Data
- ~300 CT slice images per log, one slice per centimetre
- Slices form a coherent 3D volume when stacked
- Greyscale, single-channel images (standard CT output)
- This sequential structure is critical — it enables slice propagation
### Segmentation Classes
```
sound_knot, crack, pith, wood (heartwood), bark, moisture, moisture_real,
rot, insects, compression_wood, background
```
 
Semantic segmentation task (not instance). Multiple individual knots of the same class can be separated post-hoc using connected component analysis / contour extraction.
 
### Key Domain Properties
- Knots appear lighter (higher density) than surrounding wood on CT — density contrast drives segmentation
- Knot extent: typically ~20–30 slices (appears, grows, peaks, disappears)
- Bark and pith are structurally stable across all 300 slices
- Cracks are thin linear voids — hardest class to segment reliably
- Rot and moisture have variable appearance — low contrast, hardest to propagate
---
 
## 3. Annotation Pipeline Architecture
 
### Active Learning Loop
 
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
Human annotator reviews and corrects                  │
        ↓                                             │
  Corrected masks → train v2 model ──────────────────┘
        ↓
   Repeat (v2 → v3 → ...)
```
 
### Phase 1 — Cold Start (No trained model yet)
 
**Input:** Primitive annotations from prior project (points, bboxes, rough masks on individual slices)  
**Process:** Feed directly as prompts to MedSAM2 → propagate across 300 slices per class  
**Output:** Full-volume masks for human review  
**Human role:** Correct propagation failures; do not annotate from scratch
 
### Phase 2+ — Active Learning Loop
 
**Input:** Domain-trained model predicts mask on key slice  
**Process:** Extract bbox from predicted mask → MedSAM2 refines → propagates across volume  
**Output:** Higher quality full-volume masks  
**Human role:** Decreasing correction effort with each model iteration
 
### Annotation Platform
**Supervisely** — used for both storing annotations and human review/correction. Natively supports SAM-based smart annotation tools (no custom UI integration needed for the human-facing part).
 
---
 
## 4. Foundation Model Selection
 
### Models Evaluated
 
| Model | Base | Geometric prompts | CT 3D support | Maturity | Decision |
|---|---|---|---|---|---|
| **MedSAM2** | SAM2.1-Tiny | ✅ bbox, points, masks | ✅ Native | ✅ Production-ready | **Selected** |
| Plain SAM2 | SAM2.1 | ✅ bbox, points, masks | ✅ Via video mode | ✅ Production-ready | Inferior on CT |
| MedSAM3 v1 | SAM3 + LoRA | ❌ Text only (v1) | ❌ Not exposed | ⚠️ Immature | Ruled out |
| Plain SAM3 | SAM3 | ❌ Text only | ❌ Not exposed | ✅ Ready | Wrong modality |
 
### Decision: MedSAM2
 
**Repository:** https://github.com/bowang-lab/MedSAM2  
**Weights:** https://huggingface.co/wanglab/MedSAM2  
**Paper:** https://arxiv.org/abs/2504.03600
 
Reasons:
 
1. **Prompt compatibility:** Accepts bounding boxes, points, and masks — all available from phase 1 primitives
2. **CT greyscale domain:** Fine-tuned on 455,000+ 3D medical CT/MRI/PET image-mask pairs; smallest domain gap to wood CT among options
3. **3D volume support:** Treats sequential CT slices as a video, matching the 300-slice log format exactly
4. **Slice propagation:** Prompt one key slice → bidirectional propagation through full volume via streaming memory bank
5. **Performance:** ~6–10 Dice points better than plain SAM2 on CT data; largest gains on low-contrast/ambiguous structures (rot, moisture, cracks) — exactly the hard classes
6. **Human-in-the-loop:** Documented workflow with validated 85%+ annotation cost reduction in CT lesion annotation studies
7. **Hardware:** Built on SAM2.1-Tiny specifically to reduce VRAM requirements; no A100 needed
8. **Maturity:** Weights available, inference scripts work, actively maintained
### Why MedSAM3 Was Ruled Out (Thoroughly Investigated)
 
MedSAM3 (https://github.com/Joey-S-Liu/MedSAM3) was inspected in detail:
 
- `infer_sam.py` signature: `predict(image_path: str, text_prompts: List[str])` — no geometric input
- `inference_lora.py` — same, text only
- README explicitly states: "MedSAM3-v1 is a pure text-guided model"
- SAM3's Tracker (which does support mask input) is **not exposed** in MedSAM3
- GitHub issue #8: asks for bbox/mask support — unanswered by authors
- GitHub issue #10: asks for point/box prompts — unanswered
- GitHub issue #11: missing YAML config for LoRA weights — blocks out-of-box inference
- Supported task list not yet published (mentioned in README itself)
**Conclusion:** MedSAM3 cannot serve as a mask-prompt refiner in v1. Do not revisit until issues #8/#10 are addressed by authors.
 
### Performance Tables
 
**MedSAM2 on 3D CT (Dice score):**
 
| Model | CT Organs | CT Lesions | MRI Organs | MRI Lesions |
|---|---|---|---|---|
| SAM2.1-Tiny | ~80% | ~72% | ~82% | ~80% |
| SAM2.1-Large | ~80% | ~72% | ~83% | ~81% |
| EfficientMedSAM | 83.6% | 78.0% | — | — |
| **MedSAM2** | **88.8%** | **86.7%** | **87.1%** | **88.4%** |
 
**MedSAM3 paper (2D benchmarks, Dice score):**
 
| Method | BUSI | RIM-ONE | ISIC2018 | Kvasir-SEG |
|---|---|---|---|---|
| SAM3 text-only | 0.000 | 0.000 | 0.219 | 0.000 |
| SAM3 text+box | 0.711 | 0.830 | 0.818 | 0.767 |
| MedSAM3 text-only | 0.267 | 0.083 | 0.569 | 0.144 |
| MedSAM3 text+box | 0.777 | 0.898 | 0.906 | 0.883 |
 
Note: plain SAM3 text-only scores 0.000 on most medical benchmarks — not usable for CT segmentation without geometric prompts.
 
### Transfer to Wood CT
 
MedSAM2's medical CT pretraining transfers **partially** to wood CT:
 
- **Transfers well:** Greyscale CT modality; density-contrast feature representations; encoder weights
- **Does not transfer:** Human anatomy shape priors (smooth round organs) do not match wood structures (radial fibres, concentric rings, thin cracks)
- **Practical implication:** MedSAM2 advantage is largest for ambiguous low-contrast structures. For clear-boundary classes (bark, pith), plain SAM2 would be nearly equivalent. MedSAM2 is still the right choice because the hard classes are where annotation effort concentrates.
- The domain-trained segmentation model (v1, v2, ...) carries all class-specific knowledge; MedSAM2 acts purely as a **boundary refiner**, not a classifier.
---
 
## 5. Local Development Setup
 
### Hardware (Development Machine)
 
```
GPU:    NVIDIA GeForce GTX 1650 (Turing, compute capability 7.5)
VRAM:   4GB
CUDA:   12.9
Driver: 576.83
OS:     Windows 11 dual-boot with Linux (use Linux for all ML work)
```
 
The Windows side should not be used for MedSAM2 — the repo is Linux-first (bash scripts, Linux paths). Always boot into Linux for development.
 
### VRAM Constraints
 
4GB is tight. Realistic expectations:
 
- **Single-slice inference:** Likely fits
- **Short sequence (20–30 slices):** Probably fits with reduced memory bank — test empirically
- **Full 300-slice propagation:** Will likely OOM — use only for pipeline logic, not full annotation runs
### VRAM Optimisation for GTX 1650
 
Reduce the memory bank size before testing:
 
```python
# In SAM2 config or passed at initialisation
max_num_maskmem: 3  # default is 7; drop to 3 for 4GB VRAM
```
 
### Installation (Linux)
 
```bash
# 1. Create environment
conda create -n medsam2 python=3.12 -y
conda activate medsam2
 
# 2. Install PyTorch (cu124 wheels are compatible with CUDA 12.9)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
 
# 3. Clone and install MedSAM2
git clone https://github.com/bowang-lab/MedSAM2.git
cd MedSAM2
pip install -e ".[dev]"
 
# 4. Download weights
bash download.sh
 
# 5. Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce GTX 1650
```
 
### Full-Scale Inference
 
For full 300-slice volume runs, use a machine with 8–16GB+ VRAM (RTX 3080/4080 or cloud GPU). The pipeline developed locally transfers directly — no code changes needed.
 
---
 
## 6. MedSAM2 Inference Reference
 
### Key Inference Modes
 
**3D CT volume with bounding box prompt:**
```bash
python medsam2_infer_3D_CT.py -i CT_images/ -o segmentation/
```
 
**Slice propagation from annotated mask:**
```bash
python medsam2_infer_video.py -i input_frames/ -m input_mask_path -o output_masks/
```
 
The `-m` flag accepts a binary mask from any key frame and propagates bidirectionally through the sequence.
 
### Multi-Class Workflow
 
MedSAM2 propagates one class per run. Script a loop:
 
```python
classes = [
    "sound_knot", "crack", "pith", "bark", "wood", "rot",
    "moisture", "compression_wood", "insects", "background"
]
 
for cls in classes:
    mask_path = f"key_frame_masks/{cls}.png"
    output_path = f"propagated/{cls}/"
    run_medsam2_propagation(input_frames_dir, mask_path, output_path)
```
 
### Per-Class Prompting Strategy
 
| Class | Key frame to use | Notes |
|---|---|---|
| Bark | First or last slice | Stable ring, propagates reliably |
| Wood / heartwood | Any central slice | Large stable region |
| Background | First or last slice | Stable |
| Pith | Middle slice | Small central point, very stable |
| Sound knot | Middle slice of knot extent | Spans ~20–30 slices; anchor from midpoint, not first appearance |
| Compression wood | Most visible slice | Often follows knot location |
| Crack | Most visible slice | Thin void; may need 2–3 prompt frames |
| Rot | Most representative slice | Variable appearance across slices |
| Moisture | Most representative slice | Low contrast, variable |
| Insects | Most visible slice | Small, localised |
 
**Key principle:** Do not anchor all classes from frame 1. Use structurally representative key frames per class. MedSAM2 supports multiple prompt frames.
 
### Deriving a Bounding Box from a Predicted Mask (Phase 2+)
 
```python
import numpy as np
 
def mask_to_bbox(mask: np.ndarray, padding: int = 8) -> tuple:
    """Convert binary mask to (x_min, y_min, x_max, y_max) bounding box."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    h, w = mask.shape
    return (
        max(0, x_min - padding),
        max(0, y_min - padding),
        min(w, x_max + padding),
        min(h, y_max + padding)
    )
```
 
---
 
## 7. Open Questions / Future Considerations
 
- **MedSAM3 geometric prompts:** Monitor GitHub issues #8 and #10; re-evaluate if authors respond and add bbox/mask support
- **MedSAM3 text cold-start:** Text prompting with `"knot"`, `"crack"` etc. could theoretically give a zero-shot starting point before a v1 model exists — not reliable enough for the pipeline yet
- **GTX 1650 OOM threshold:** Test empirically with `max_num_maskmem: 3`; find the longest sequence that fits
- **Supervisely export format:** Confirm mask export format (PNG per class vs JSON polygons) before building the propagation ingestion script
---
 
## 8. References
 
| Resource | URL |
|---|---|
| MedSAM2 GitHub | https://github.com/bowang-lab/MedSAM2 |
| MedSAM2 weights (HuggingFace) | https://huggingface.co/wanglab/MedSAM2 |
| MedSAM2 paper | https://arxiv.org/abs/2504.03600 |
| MedSAM3 GitHub | https://github.com/Joey-S-Liu/MedSAM3 |
| MedSAM3 weights | https://huggingface.co/lal-Joey/MedSAM3_v1 |
| MedSAM3 paper | https://arxiv.org/abs/2511.19046 |
| CT log defect segmentation CNN comparison (2024) | https://www.sciencedirect.com/science/article/abs/pii/S0168169924006355 |
| SAM2 zero-shot 3D CT study (2025) | https://arxiv.org/html/2603.23116 |

