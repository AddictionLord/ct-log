# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation

```bash
conda create -n medsam2 python=3.12 -y && conda activate medsam2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
bash download.sh  # downloads checkpoints from HuggingFace
```

## Key Commands

**Inference — MedSAM2 (GPU):**
```bash
# 3D CT segmentation (generic)
python medsam2_infer_3D_CT.py -i CT_DeepLesion/images -o CT_DeepLesion/segmentation

# RECIST-prompted CT lesion segmentation (FLARE25)
python medsam2_infer_CT_lesion_npz_recist.py -i ./data/validation_public_npz -o ./data/RECIST_pred

# Medical video segmentation
python medsam2_infer_video.py -i input_video_path -m input_mask_path -o output_video_path
```

**Inference — Efficient MedSAM2 (CPU-compatible):**
```bash
python eff_medsam2_infer_CT_lesion_npz_recist.py -i ./data/validation_public_npz -o ./data/segs_test
```

**Training:**
```bash
# Single-node MedSAM2 (edit config first to set dataset path)
sh single_node_train_medsam2.sh

# Single-node Efficient MedSAM2 (downloads EfficientTAM checkpoint automatically)
sh single_node_train_eff_medsam2_FLARE25.sh

# Multi-node (SLURM)
sbatch multi_node_train.sh
```

Training entry point is `training/train.py`, launched with `-c <config_yaml>`.

**Gradio demo:**
```bash
python app.py
```

## Architecture

The repo contains two parallel model families sharing the same training pipeline:

### MedSAM2 (`sam2/`)
Adapts Meta's SAM 2.1 (Hiera backbone, `sam2_hiera_tiny`) for 3D medical volume segmentation. Core classes:
- `sam2/modeling/sam2_base.py` — base model (image encoder + memory attention + mask decoder)
- `sam2/sam2_video_predictor_npz.py` — `SAM2VideoPredictorNPZ`: extends the video predictor to accept pre-loaded tensors (shape `[D, 3, H, W]`) instead of a video path; this is the key adaptation for volumetric CT data
- `sam2/build_sam.py` — factory functions (`build_sam2_video_predictor_npz`, etc.) using Hydra `compose` + `instantiate`

### Efficient MedSAM2 (`efficient_track_anything/`)
Adapts EfficientTAM (ViT-based, lightweight) for CPU-viable inference. Mirrors the `sam2/` structure:
- `efficient_track_anything/build_efficienttam.py` — factory functions
- `efficient_track_anything/efficienttam_video_predictor_npz.py` — same NPZ-tensor interface as above

### Training pipeline (`training/`)
Shared across both model families. Key components:
- `training/train.py` — entry point; uses `submitit` for SLURM, `torch.multiprocessing` for single-node
- `training/trainer.py` — `Trainer` class orchestrating the train loop
- `training/dataset/vos_raw_dataset.py` — `NPZRawDataset` reads `.npz` files (keys: `imgs`, `gts`, `recist`); treated as "videos" where depth slices are frames
- `training/dataset/vos_dataset.py` — `VOSDataset` wraps raw datasets with transforms and frame sampling
- `training/loss_fns.py` — `MultiStepMultiMasksAndIous` loss (mask + dice + IoU + class)

### Data format
NPZ files have keys:
- `imgs`: `(D, H, W)` uint8, range `[0, 255]`
- `gts`: `(D, H, W)` uint8, segmentation labels
- `recist`: `(D, H, W)` binary, RECIST diameter line on the tumor's middle slice
- `spacing`: voxel spacing `(z, y, x)` in mm

At inference, slices are expanded to RGB (`[D, 3, H, W]`) and normalized with ImageNet stats before being passed to the predictor.

### Config system
All model and training configs are Hydra YAML files under `sam2/configs/` and `efficient_track_anything/configs/`. Training configs must have the dataset `folder` key set to an absolute path. Config is loaded via `hydra.compose(config_name=..., overrides=[...])`.

### Inference flow (3D CT)
1. Load NPZ, resize each slice to 512×512 RGB, normalize → tensor `[D, 3, 512, 512]`
2. `predictor.init_state(images, height, width)` — encodes all frames
3. Add prompt on middle slice (`z_mid`) via `add_new_points_or_box` (box derived from RECIST line diameter, or sampled points)
4. `add_new_mask` to seed the mask memory
5. `propagate_in_video` forward from `z_mid`, then reset + propagate reverse
6. Save predictions as compressed NPZ with keys `segs`, `spacing`

Efficient MedSAM2 additionally crops the volume to `[z_mid ± diameter/2]` before propagation, using physical spacing to convert the RECIST diameter to slice count.
