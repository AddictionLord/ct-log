"""Side-by-side: original propagation vs OBB-augmented propagation.

Per page, render 4 panels:
    [GT] [original prop] [OBB-augmented prop] [v5 (OBB+SAM2 image predictor)]
to see whether OBB-augmented prop ≥ v5 on knot quality.

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.compare_obb_aug \\
        --pages 7 39 154 175 201 234 241 297
"""

import argparse
import json
import pathlib
from typing import Optional

from ann_pipeline.knot.data_prep import knot_mask_from_ann
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from ultralytics import YOLO

from experiments.sm2025_subset4_propagate.run import CLASS_IDS, page_ann_path

ORIG_NPZ = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/result.npz"
AUG_NPZ = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/result_obb_aug_ellipse.npz"
YOLO_OBB_WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_obb_v1/weights/best.pt"
SAM_CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
SAM_MODEL_CFG = "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"
SOURCE_IMG_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025/4/img"
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/compare_obb_aug"


def load_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1).astype(np.uint8)
    if arr.shape[-1] == 4:
        return arr[..., :3].astype(np.uint8)
    return arr.astype(np.uint8)


def low_res_logits(binary_mask: np.ndarray, size: int = 128) -> np.ndarray:
    low = cv2.resize(binary_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return np.where(low > 0, 10.0, -10.0).astype(np.float32)[None]


def v5_knot_mask(yolo: YOLO, sam_pred: SAM2ImagePredictor, rgb: np.ndarray, conf: float, nms_iou: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), dtype=bool)
    res = yolo.predict(rgb, conf=conf, iou=nms_iou, verbose=False)[0]
    if res.obb is None or len(res.obb) == 0:
        return out
    xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_pred.set_image(rgb)
        for corners in xyxyxyxy:
            obb_raster = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(obb_raster, [corners.astype(np.int32)], 1)
            x1 = max(0, int(np.floor(corners[:, 0].min())))
            y1 = max(0, int(np.floor(corners[:, 1].min())))
            x2 = min(w, int(np.ceil(corners[:, 0].max())))
            y2 = min(h, int(np.ceil(corners[:, 1].max())))
            aabb = np.array([x1, y1, x2, y2], dtype=np.float32)
            m, _, _ = sam_pred.predict(
                box=aabb, mask_input=low_res_logits(obb_raster.astype(bool)), multimask_output=False
            )
            mask = m[0].astype(bool)
            clipped = np.zeros_like(mask)
            clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            out |= clipped
    return out


def load_gt_knot_mask(page: int) -> Optional[np.ndarray]:
    try:
        with open(page_ann_path(page)) as f:
            ann = json.load(f)
    except FileNotFoundError:
        return None
    if not any(obj.get("classTitle") == "Knot" for obj in ann.get("objects", [])):
        return None
    return knot_mask_from_ann(ann).astype(bool)


def render_panel(ax, rgb, mask, title, gt_mask=None) -> None:
    gray = rgb.mean(axis=-1).astype(np.uint8)
    ax.imshow(gray, cmap="gray")
    overlay = np.zeros((*gray.shape, 4), dtype=np.float32)
    overlay[mask] = (1.0, 0.55, 0.0, 0.45)
    ax.imshow(overlay)
    if gt_mask is not None and gt_mask.any():
        cs, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if len(c) > 2:
                cc = c.squeeze(1)
                ax.plot(cc[:, 0], cc[:, 1], color="lime", linewidth=1.6)
    ax.set_title("%s  (%d px)" % (title, int(mask.sum())), fontsize=10)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, nargs="+", required=True)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    orig = np.load(ORIG_NPZ, allow_pickle=False)
    aug = np.load(AUG_NPZ, allow_pickle=False)
    orig_pages = orig["pages"].tolist()
    aug_pages = aug["pages"].tolist()

    yolo = YOLO(YOLO_OBB_WEIGHTS)
    print("loading SAM2 ...")
    sam = build_sam2("//" + SAM_MODEL_CFG, SAM_CHECKPOINT)
    sam_pred = SAM2ImagePredictor(sam)

    for page in args.pages:
        img_path = "%s/page_%03d.tiff" % (SOURCE_IMG_DIR, page)
        rgb = load_rgb(img_path)
        h, w = rgb.shape[:2]
        gt_mask = load_gt_knot_mask(page)

        orig_knot = orig["pred"][orig_pages.index(page)] == CLASS_IDS["Knot"]
        aug_knot = aug["pred"][aug_pages.index(page)] == CLASS_IDS["Knot"]
        v5_knot = v5_knot_mask(yolo, sam_pred, rgb, args.conf, args.nms_iou)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
        gt_display = gt_mask if gt_mask is not None else np.zeros((h, w), dtype=bool)
        render_panel(axes[0], rgb, gt_display, "GT" + ("" if gt_mask is not None else " — no GT"))
        render_panel(axes[1], rgb, orig_knot, "original propagation", gt_mask=gt_mask)
        render_panel(axes[2], rgb, aug_knot, "OBB-aug propagation (ellipse seed)", gt_mask=gt_mask)
        render_panel(axes[3], rgb, v5_knot, "v5 (OBB+SAM2 image)", gt_mask=gt_mask)
        fig.suptitle("page %d  (conf=%.2f)" % (page, args.conf), fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = "%s/page_%03d.png" % (args.out_dir, page)
        plt.savefig(out_path, dpi=120)
        plt.close()
        print("page %d wrote %s" % (page, out_path))


if __name__ == "__main__":
    main()
