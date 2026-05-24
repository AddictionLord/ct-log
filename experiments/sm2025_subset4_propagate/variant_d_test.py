"""Variant D diagnostic: per-knot multi-prompt SAM2 call.

For each YOLO-OBB detection:
    1. Find overlapping propagation CC (if any).
    2. Build combined prompt:
       - mask_input  = rasterise(OBB ∪ prop_CC)  → 128×128 logits
       - box         = AABB of (OBB ∪ prop_CC)
       - points      = [centre_OBB(+), centre_prop(+), 4 corners(−)]
    3. Run SAM2 once with all of the above.
    4. Clip the result to the joint AABB.

Renders: GT | propagation | v5 baseline | Variant D
on a chosen set of frames.
"""

import argparse
import json
import pathlib
from typing import List, Optional

from ann_pipeline.knot.data_prep import knot_mask_from_ann
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy import ndimage as ndi
import torch
from ultralytics import YOLO

from experiments.sm2025_subset4_propagate.run import CLASS_IDS, page_ann_path

YOLO_OBB_WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_obb_v1/weights/best.pt"
SAM_CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
SAM_MODEL_CFG = "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"
SOURCE_IMG_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025/4/img"
NPZ = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/result.npz"
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/variant_d"


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


def clip_to_aabb(mask: np.ndarray, aabb: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    x1, y1, x2, y2 = aabb.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def overlapping_prop_cc(prop_knot_mask: np.ndarray, aabb: np.ndarray, min_overlap_px: int = 8) -> np.ndarray:
    h, w = prop_knot_mask.shape
    out = np.zeros_like(prop_knot_mask)
    x1, y1, x2, y2 = aabb.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    lab, n = ndi.label(prop_knot_mask, structure=np.ones((3, 3), dtype=np.uint8))
    for k in range(1, n + 1):
        comp = lab == k
        overlap = comp[y1:y2, x1:x2].sum()
        if overlap >= min_overlap_px:
            out |= comp
    return out


def joint_aabb(obb_corners: np.ndarray, prop_cc: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    xs = list(obb_corners[:, 0])
    ys = list(obb_corners[:, 1])
    if prop_cc.any():
        ys_p, xs_p = np.nonzero(prop_cc)
        xs.extend([xs_p.min(), xs_p.max()])
        ys.extend([ys_p.min(), ys_p.max()])
    x1 = max(0, int(np.floor(min(xs))))
    y1 = max(0, int(np.floor(min(ys))))
    x2 = min(w, int(np.ceil(max(xs)) + 1))
    y2 = min(h, int(np.ceil(max(ys)) + 1))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def corner_points(aabb: np.ndarray, inset_frac: float = 0.10) -> List[tuple]:
    x1, y1, x2, y2 = aabb
    dx = (x2 - x1) * inset_frac
    dy = (y2 - y1) * inset_frac
    return [(x1 + dx, y1 + dy), (x2 - dx, y1 + dy), (x1 + dx, y2 - dy), (x2 - dx, y2 - dy)]


def variant_d_predict(
    sam_pred: SAM2ImagePredictor,
    obb_corners: np.ndarray,
    prop_cc: np.ndarray,
    shape,
) -> tuple:
    h, w = shape
    obb_raster = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(obb_raster, [obb_corners.astype(np.int32)], 1)
    obb_raster = obb_raster.astype(bool)

    union_shape = obb_raster | prop_cc
    aabb = joint_aabb(obb_corners, prop_cc, shape)

    obb_cx, obb_cy = obb_corners[:, 0].mean(), obb_corners[:, 1].mean()
    positives = [(float(obb_cx), float(obb_cy))]
    if prop_cc.any():
        ys, xs = np.nonzero(prop_cc)
        positives.append((float(xs.mean()), float(ys.mean())))
    negatives = corner_points(aabb)
    pts = np.array(positives + negatives, dtype=np.float32)
    labs = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int32)

    mask_input = low_res_logits(union_shape)
    m, _, _ = sam_pred.predict(
        box=aabb,
        mask_input=mask_input,
        point_coords=pts,
        point_labels=labs,
        multimask_output=False,
    )
    return m[0].astype(bool), aabb, pts, labs


def baseline_predict(sam_pred: SAM2ImagePredictor, obb_corners: np.ndarray, shape) -> tuple:
    h, w = shape
    obb_raster = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(obb_raster, [obb_corners.astype(np.int32)], 1)
    x1 = max(0, int(np.floor(obb_corners[:, 0].min())))
    y1 = max(0, int(np.floor(obb_corners[:, 1].min())))
    x2 = min(w, int(np.ceil(obb_corners[:, 0].max())))
    y2 = min(h, int(np.ceil(obb_corners[:, 1].max())))
    aabb = np.array([x1, y1, x2, y2], dtype=np.float32)
    mask_input = low_res_logits(obb_raster.astype(bool))
    m, _, _ = sam_pred.predict(box=aabb, mask_input=mask_input, multimask_output=False)
    return m[0].astype(bool), aabb


def render_panel(ax, rgb, masks, title, gt_mask=None, points=None) -> None:
    gray = rgb.mean(axis=-1).astype(np.uint8)
    ax.imshow(gray, cmap="gray")
    overlay = np.zeros((*gray.shape, 4), dtype=np.float32)
    total = 0
    for m in masks:
        overlay[m] = (1.0, 0.55, 0.0, 0.45)
        total += int(m.sum())
    ax.imshow(overlay)
    if gt_mask is not None and gt_mask.any():
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) > 2:
                cc = c.squeeze(1)
                ax.plot(cc[:, 0], cc[:, 1], color="lime", linewidth=1.8)
    if points is not None:
        for (x, y), lab in points:
            color = "lime" if lab == 1 else "magenta"
            ax.plot(x, y, "o", color=color, markersize=5, mec="white", mew=0.8)
    ax.set_title("%s  (%d px, %d knots)" % (title, total, len(masks)), fontsize=10)
    ax.axis("off")


def load_gt_knot_mask(page: int) -> Optional[np.ndarray]:
    try:
        with open(page_ann_path(page)) as f:
            ann = json.load(f)
    except FileNotFoundError:
        return None
    if not any(obj.get("classTitle") == "Knot" for obj in ann.get("objects", [])):
        return None
    return knot_mask_from_ann(ann).astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, nargs="+", required=True)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    data = np.load(NPZ, allow_pickle=False)
    npz_pages = data["pages"].tolist()
    pred_vol = data["pred"]

    yolo = YOLO(YOLO_OBB_WEIGHTS)
    print("loading SAM2 ...")
    sam = build_sam2("//" + SAM_MODEL_CFG, SAM_CHECKPOINT)
    sam_pred = SAM2ImagePredictor(sam)

    for page in args.pages:
        img_path = "%s/page_%03d.tiff" % (SOURCE_IMG_DIR, page)
        rgb = load_rgb(img_path)
        h, w = rgb.shape[:2]
        prop_knot = pred_vol[npz_pages.index(page)] == CLASS_IDS["Knot"]
        gt_mask = load_gt_knot_mask(page)

        res = yolo.predict(rgb, conf=args.conf, iou=args.nms_iou, verbose=False)[0]
        if res.obb is None or len(res.obb) == 0:
            xyxyxyxy = np.empty((0, 4, 2), dtype=np.float32)
        else:
            xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()

        baseline_masks: List[np.ndarray] = []
        d_masks: List[np.ndarray] = []
        all_points: List[tuple] = []

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam_pred.set_image(rgb)
            for corners in xyxyxyxy:
                m_base, aabb_base = baseline_predict(sam_pred, corners, (h, w))
                baseline_masks.append(clip_to_aabb(m_base, aabb_base))

                prop_cc = overlapping_prop_cc(prop_knot, aabb_base)
                m_d, aabb_d, pts, labs = variant_d_predict(sam_pred, corners, prop_cc, (h, w))
                d_masks.append(clip_to_aabb(m_d, aabb_d))
                for p, lab in zip(pts, labs):
                    all_points.append(((float(p[0]), float(p[1])), int(lab)))

        any_d = np.zeros((h, w), dtype=bool)
        for m in d_masks:
            any_d |= m
        lab_p, n_cc = ndi.label(prop_knot, structure=np.ones((3, 3), dtype=np.uint8))
        n_added = 0
        for k in range(1, n_cc + 1):
            comp = lab_p == k
            if comp.sum() < 50:
                continue
            if np.logical_and(comp, any_d).sum() == 0:
                d_masks.append(comp)
                n_added += 1
        print("page %d: added %d propagation-only knots (Case 3)" % (page, n_added))

        fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
        render_panel(
            axes[0],
            rgb,
            [gt_mask] if gt_mask is not None else [],
            "GT knots" + ("" if gt_mask is not None else " — no GT"),
            gt_mask=None,
        )
        render_panel(axes[1], rgb, [prop_knot], "propagation", gt_mask=gt_mask)
        render_panel(axes[2], rgb, baseline_masks, "v5 baseline", gt_mask=gt_mask)
        render_panel(axes[3], rgb, d_masks, "Variant D + Case 3", gt_mask=gt_mask, points=all_points)
        fig.suptitle(
            "page %d  (conf=%.2f, n_OBB=%d). Lime=GT contour, magenta dot=negative, lime dot=positive."
            % (page, args.conf, len(xyxyxyxy)),
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = "%s/page_%03d.png" % (args.out_dir, page)
        plt.savefig(out_path, dpi=120)
        plt.close()
        per_knot = [(i, int(baseline_masks[i].sum()), int(d_masks[i].sum())) for i in range(len(baseline_masks))]
        print("page %d wrote %s" % (page, out_path))
        for i, b, d in per_knot:
            print("    knot %d: baseline=%d  variant_d=%d (diff %+d)" % (i, b, d, d - b))


if __name__ == "__main__":
    main()
