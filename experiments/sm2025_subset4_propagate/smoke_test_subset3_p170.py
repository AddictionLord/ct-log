"""Smoke test on subset-3 page 170 (hardwood-like log, OOD for our models).

For each YOLO-OBB detection at conf >= --conf, render 4 panels:

  1. seed ellipse alone (what we fed into the propagator)
  2. obb-only propagation output mask (what came out)
  3. obb-augmented propagation output (with anchors)
  4. SAM2 image-predictor: AABB-of-OBB + N positive points along the long
     axis of the OBB (a la Supervisely smart-tool style)

Plus a 5th panel for context: gray image + OBB polygon overlay.

Goal: see whether panel (4) recovers the real knot shape on a frame where
the propagation pipelines fall back to ellipse-like outputs. Tests the
hypothesis that pure SAM image-predictor prompts work better than mask
seeds on hardwood textures.
"""

import argparse
import pathlib

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from ultralytics import YOLO

from experiments.sm2025_subset4_propagate.run import CLASS_IDS

YOLO_OBB_WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_obb_v1/weights/best.pt"
SAM_CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
SAM_MODEL_CFG = "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"
SUBSET3_IMG_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025/3/img"
NPZ_OBB_ONLY = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/result_subset3_obb_only.npz"
NPZ_OBB_AUG = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/result_subset3_obb_aug.npz"
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/smoke_subset3"


def load_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1).astype(np.uint8)
    if arr.shape[-1] == 4:
        return arr[..., :3].astype(np.uint8)
    return arr.astype(np.uint8)


def fit_obb_params(corners: np.ndarray):
    """Returns ((cx, cy), (long_len, short_len), angle_deg_of_long_axis)."""
    (cx, cy), (w, h), angle = cv2.minAreaRect(corners.astype(np.float32))
    if h > w:
        long_len, short_len = h, w
        angle = angle + 90.0
    else:
        long_len, short_len = w, h
    return (cx, cy), (long_len, short_len), angle


def ellipse_mask(corners: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    (cx, cy), (long_len, short_len), angle = fit_obb_params(corners)
    out = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        out,
        center=(int(round(cx)), int(round(cy))),
        axes=(max(1, int(long_len / 2)), max(1, int(short_len / 2))),
        angle=angle,
        startAngle=0,
        endAngle=360,
        color=1,
        thickness=-1,
    )
    return out.astype(bool)


def axis_points(corners: np.ndarray, n_points: int, inset_frac: float = 0.1):
    """N positive points evenly spaced along the long axis of the OBB."""
    (cx, cy), (long_len, _), angle = fit_obb_params(corners)
    theta = np.deg2rad(angle)
    direction = np.array([np.cos(theta), np.sin(theta)])
    half = (long_len / 2.0) * (1.0 - inset_frac)
    if n_points == 1:
        offsets = [0.0]
    else:
        offsets = list(np.linspace(-half, half, n_points))
    pts = np.array([[cx + d * direction[0], cy + d * direction[1]] for d in offsets])
    labels = np.ones(len(pts), dtype=np.int32)
    return pts, labels


def obb_negatives(corners: np.ndarray, outset_frac: float = 0.10, include_long_ends: bool = True):
    """Negative points placed just outside the OBB.

    - 2 short-axis negatives: perpendicular to the long axis, at mid-length,
      one above and one below the OBB.
    - 2 long-end negatives (optional): along the long axis, beyond the tips.
    `outset_frac` pushes them this fraction of the relevant dimension outside.
    """
    (cx, cy), (long_len, short_len), angle = fit_obb_params(corners)
    theta = np.deg2rad(angle)
    long_dir = np.array([np.cos(theta), np.sin(theta)])
    short_dir = np.array([-np.sin(theta), np.cos(theta)])
    pts = []
    push_short = (short_len / 2.0) * (1.0 + outset_frac)
    pts.append([cx + push_short * short_dir[0], cy + push_short * short_dir[1]])
    pts.append([cx - push_short * short_dir[0], cy - push_short * short_dir[1]])
    if include_long_ends:
        push_long = (long_len / 2.0) * (1.0 + outset_frac)
        pts.append([cx + push_long * long_dir[0], cy + push_long * long_dir[1]])
        pts.append([cx - push_long * long_dir[0], cy - push_long * long_dir[1]])
    pts = np.array(pts)
    labels = np.zeros(len(pts), dtype=np.int32)
    return pts, labels


def aabb_of_obb(corners: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    x1 = max(0, int(np.floor(corners[:, 0].min())))
    y1 = max(0, int(np.floor(corners[:, 1].min())))
    x2 = min(w, int(np.ceil(corners[:, 0].max())))
    y2 = min(h, int(np.ceil(corners[:, 1].max())))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def obb_polygon_mask(corners: np.ndarray, shape, inflate_frac: float = 0.10) -> np.ndarray:
    """Rasterise OBB polygon, optionally inflated outward by inflate_frac of mean side length."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    if inflate_frac > 0:
        (cx, cy), (long_len, short_len), angle = fit_obb_params(corners)
        mean_side = (long_len + short_len) / 2.0
        push = mean_side * inflate_frac / 2.0
        centre = np.array([cx, cy])
        inflated = corners + (corners - centre) / np.linalg.norm(corners - centre, axis=1, keepdims=True) * push
        cv2.fillPoly(out, [inflated.astype(np.int32)], 1)
    else:
        cv2.fillPoly(out, [corners.astype(np.int32)], 1)
    return out.astype(bool)


def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as _ndi

    lab, n = _ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n <= 1:
        return mask
    sizes = _ndi.sum(mask, lab, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return lab == largest


def render_mask(ax, gray, mask, title, color=(1.0, 0.55, 0.0, 0.45), obb=None, aabb=None, points=None):
    ax.imshow(gray, cmap="gray")
    overlay = np.zeros((*gray.shape, 4), dtype=np.float32)
    if mask is not None and mask.any():
        overlay[mask] = color
    ax.imshow(overlay)
    if obb is not None:
        ax.add_patch(mpatches.Polygon(obb, fill=False, edgecolor="lime", linewidth=1.4))
    if aabb is not None:
        x1, y1, x2, y2 = aabb
        ax.add_patch(
            mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1.0, linestyle="--")
        )
    if points is not None:
        for (x, y), lab in points:
            c = "lime" if lab == 1 else "magenta"
            ax.plot(x, y, "o", color=c, markersize=6, mec="white", mew=1.0)
    npx = int(mask.sum()) if mask is not None else 0
    ax.set_title("%s (%d px)" % (title, npx), fontsize=10)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=int, default=170)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--n_points", type=int, default=5)
    parser.add_argument(
        "--obb_inflate",
        type=float,
        default=0.10,
        help="Fraction of mean side to inflate OBB before clipping (0 = exact OBB).",
    )
    parser.add_argument(
        "--largest_cc",
        action="store_true",
        default=True,
        help="Keep only the largest connected component after clipping.",
    )
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    img_path = "%s/page_%03d.tiff" % (SUBSET3_IMG_DIR, args.page)
    rgb = load_rgb(img_path)
    gray = rgb.mean(axis=-1).astype(np.uint8)
    h, w = gray.shape

    print("loading YOLO-OBB ...")
    yolo = YOLO(YOLO_OBB_WEIGHTS)
    res = yolo.predict(rgb, conf=args.conf, iou=args.nms_iou, verbose=False)[0]
    if res.obb is None or len(res.obb) == 0:
        print("no OBB detections at conf=%.2f, abort" % args.conf)
        return
    xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()
    print("detected %d OBBs" % len(xyxyxyxy))

    print("loading SAM2 image predictor ...")
    sam = build_sam2("//" + SAM_MODEL_CFG, SAM_CHECKPOINT)
    sam_pred = SAM2ImagePredictor(sam)

    obb_only = np.load(NPZ_OBB_ONLY, allow_pickle=False)
    obb_aug = np.load(NPZ_OBB_AUG, allow_pickle=False)
    pages = obb_only["pages"].tolist()
    idx = pages.index(args.page)
    obb_only_mask = obb_only["pred"][idx] == CLASS_IDS["Knot"]
    obb_aug_mask = obb_aug["pred"][idx] == CLASS_IDS["Knot"]

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_pred.set_image(rgb)
        for i, corners in enumerate(xyxyxyxy):
            seed = ellipse_mask(corners, (h, w))
            pos_pts, pos_labs = axis_points(corners, args.n_points)
            aabb = aabb_of_obb(corners, (h, w))
            x1, y1, x2, y2 = aabb.astype(int)

            def run_sam(pts, labs):
                m, _, _ = sam_pred.predict(
                    box=aabb,
                    point_coords=pts.astype(np.float32),
                    point_labels=labs.astype(np.int32),
                    multimask_output=False,
                )
                mask = m[0].astype(bool)
                # AABB clip only (no OBB polygon)
                aabb_clipped = np.zeros_like(mask)
                aabb_clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                # Then largest CC
                return keep_largest_cc(aabb_clipped)

            mask_pos = run_sam(pos_pts, pos_labs)

            neg_short_pts, neg_short_labs = obb_negatives(corners, include_long_ends=False)
            pts_v2 = np.concatenate([pos_pts, neg_short_pts], axis=0)
            labs_v2 = np.concatenate([pos_labs, neg_short_labs], axis=0)
            mask_pos_short = run_sam(pts_v2, labs_v2)

            neg_all_pts, neg_all_labs = obb_negatives(corners, include_long_ends=True)
            pts_v3 = np.concatenate([pos_pts, neg_all_pts], axis=0)
            labs_v3 = np.concatenate([pos_labs, neg_all_labs], axis=0)
            mask_pos_all = run_sam(pts_v3, labs_v3)

            fig, axes = plt.subplots(1, 5, figsize=(25, 5.4))
            render_mask(axes[0], gray, None, "image + OBB", obb=corners)
            render_mask(axes[1], gray, seed, "seed ellipse")
            render_mask(
                axes[2],
                gray,
                mask_pos,
                "SAM: box + %d pos" % args.n_points,
                points=[(tuple(p), int(l)) for p, l in zip(pos_pts, pos_labs)],
            )
            render_mask(
                axes[3],
                gray,
                mask_pos_short,
                "SAM: box + %d pos + 2 short neg" % args.n_points,
                points=[(tuple(p), int(l)) for p, l in zip(pts_v2, labs_v2)],
            )
            render_mask(
                axes[4],
                gray,
                mask_pos_all,
                "SAM: box + %d pos + 4 neg" % args.n_points,
                points=[(tuple(p), int(l)) for p, l in zip(pts_v3, labs_v3)],
            )

            fig.suptitle(
                "subset 3 page %d  knot %d/%d  (conf=%.2f, n_pos=%d)"
                % (args.page, i, len(xyxyxyxy), args.conf, args.n_points),
                fontsize=12,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            out_path = "%s/page_%03d_knot_%d.png" % (args.out_dir, args.page, i)
            plt.savefig(out_path, dpi=120)
            plt.close()
            print(
                "  wrote %s (pos=%d, pos+short_neg=%d, pos+all_neg=%d, seed=%d)"
                % (out_path, int(mask_pos.sum()), int(mask_pos_short.sum()), int(mask_pos_all.sum()), int(seed.sum()))
            )


if __name__ == "__main__":
    main()
