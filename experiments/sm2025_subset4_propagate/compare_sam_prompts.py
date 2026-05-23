"""Diagnostic: compare SAM2 prompt variants for knot mask generation.

For each frame and YOLO knot bbox, render three SAM2 outputs:
    1. box only                       (current v2 baseline)
    2. box + positive point at centre (classic SAM2 prompt)
    3. box + 4 negative points near box corners (constrain to box interior)

Saves a montage per frame so we can pick the variant with the tightest masks.

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.compare_sam_prompts \\
        --pages 39 73 238 239 40 154
"""

import argparse
import pathlib
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from ultralytics import YOLO

YOLO_WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_v2_all45/weights/best.pt"
SAM_CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
SAM_MODEL_CFG = "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"
SOURCE_IMG_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025/4/img"
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/sam_prompt_compare"


def load_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1).astype(np.uint8)
    if arr.shape[-1] == 4:
        return arr[..., :3].astype(np.uint8)
    return arr.astype(np.uint8)


def predict_with_prompt(
    sam: SAM2ImagePredictor,
    box: np.ndarray,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
) -> np.ndarray:
    kwargs = {"box": box.astype(np.float32), "multimask_output": False}
    if point_coords is not None:
        kwargs["point_coords"] = point_coords.astype(np.float32)
        kwargs["point_labels"] = point_labels.astype(np.int32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, _, _ = sam.predict(**kwargs)
    return masks[0].astype(bool)


def corner_points(box: np.ndarray, inset_frac: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Return 4 points placed slightly inside each box corner, labelled negative (0)."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    dx = w * inset_frac
    dy = h * inset_frac
    pts = np.array(
        [
            [x1 + dx, y1 + dy],
            [x2 - dx, y1 + dy],
            [x1 + dx, y2 - dy],
            [x2 - dx, y2 - dy],
        ]
    )
    labels = np.array([0, 0, 0, 0])
    return pts, labels


def centre_point(box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return np.array([[cx, cy]]), np.array([1])


def edge_midpoints(box: np.ndarray, inset_frac: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Return 4 negative points at the midpoint of each box edge (top/bottom/left/right),
    slightly inset toward the interior so they sit just inside the bbox."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = w * inset_frac
    dy = h * inset_frac
    pts = np.array(
        [
            [cx, y1 + dy],
            [cx, y2 - dy],
            [x1 + dx, cy],
            [x2 - dx, cy],
        ]
    )
    labels = np.array([0, 0, 0, 0])
    return pts, labels


def clip_mask_to_bbox(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Zero out mask pixels outside the YOLO bbox."""
    h, w = mask.shape
    x1 = max(0, int(np.floor(box[0])))
    y1 = max(0, int(np.floor(box[1])))
    x2 = min(w, int(np.ceil(box[2])))
    y2 = min(h, int(np.ceil(box[3])))
    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def render_panel(
    ax, rgb: np.ndarray, boxes: List[np.ndarray], masks: List[np.ndarray], title: str, extra_pts=None
) -> None:
    gray = rgb.mean(axis=-1).astype(np.uint8)
    ax.imshow(gray, cmap="gray")
    overlay = np.zeros((*gray.shape, 4), dtype=np.float32)
    total_px = 0
    for m in masks:
        overlay[m] = (1.0, 0.55, 0.0, 0.45)
        total_px += int(m.sum())
    ax.imshow(overlay)
    for b in boxes:
        ax.add_patch(
            mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor="red", linewidth=1.5)
        )
    if extra_pts is not None:
        for (x, y), lab in extra_pts:
            color = "lime" if lab == 1 else "magenta"
            ax.plot(x, y, marker="o", color=color, markersize=6, mec="white", mew=1.0)
    ax.set_title("%s  (total %d px)" % (title, total_px), fontsize=10)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, nargs="+", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--corner_inset", type=float, default=0.05)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    yolo = YOLO(YOLO_WEIGHTS)
    print("loading SAM2 ...")
    sam = build_sam2("//" + SAM_MODEL_CFG, SAM_CHECKPOINT)
    sam_pred = SAM2ImagePredictor(sam)

    for page in args.pages:
        img_path = "%s/page_%03d.tiff" % (SOURCE_IMG_DIR, page)
        rgb = load_rgb(img_path)
        res = yolo.predict(rgb, conf=args.conf, iou=args.nms_iou, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            print("page %d: no detections" % page)
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        boxes = [xyxy[i] for i in range(len(cls)) if cls[i] == 0]
        if not boxes:
            print("page %d: no knot detections" % page)
            continue

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam_pred.set_image(rgb)
            masks_box, masks_combined, masks_full = [], [], []
            combined_pts_for_render, full_pts_for_render = [], []
            for b in boxes:
                m1 = predict_with_prompt(sam_pred, b, None, None)
                cpts, clab = centre_point(b)
                kpts, klab = corner_points(b, args.corner_inset)
                epts, elab = edge_midpoints(b, args.corner_inset)
                comb_pts = np.concatenate([cpts, kpts], axis=0)
                comb_labs = np.concatenate([clab, klab], axis=0)
                m2 = predict_with_prompt(sam_pred, b, comb_pts, comb_labs)
                full_pts = np.concatenate([cpts, kpts, epts], axis=0)
                full_labs = np.concatenate([clab, klab, elab], axis=0)
                m3 = predict_with_prompt(sam_pred, b, full_pts, full_labs)
                masks_box.append(clip_mask_to_bbox(m1, b))
                masks_combined.append(clip_mask_to_bbox(m2, b))
                masks_full.append(clip_mask_to_bbox(m3, b))
                combined_pts_for_render.extend([(tuple(p), int(l)) for p, l in zip(comb_pts, comb_labs)])
                full_pts_for_render.extend([(tuple(p), int(l)) for p, l in zip(full_pts, full_labs)])

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        render_panel(axes[0], rgb, boxes, masks_box, "box only (clipped)")
        render_panel(
            axes[1], rgb, boxes, masks_combined, "centre(+) + 4 corners(−) (clipped)", extra_pts=combined_pts_for_render
        )
        render_panel(
            axes[2],
            rgb,
            boxes,
            masks_full,
            "centre(+) + 4 corners(−) + 4 edges(−) (clipped)",
            extra_pts=full_pts_for_render,
        )
        fig.suptitle(
            "page %d  (yolo conf>=%.2f, nms_iou=%.2f)  all masks bbox-clipped" % (page, args.conf, args.nms_iou),
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = "%s/page_%03d.png" % (args.out_dir, page)
        plt.savefig(out_path, dpi=120)
        plt.close()
        per_box = [
            (
                i,
                int(masks_box[i].sum()),
                int(masks_combined[i].sum()),
                int(masks_full[i].sum()),
            )
            for i in range(len(boxes))
        ]
        print("page %d wrote %s" % (page, out_path))
        for i, b1, b2, b3 in per_box:
            print("    box %d: clipped_box=%d  combined=%d  full=%d" % (i, b1, b2, b3))


if __name__ == "__main__":
    main()
