"""Fake OBB test for SAM2 knot prompting.

For each annotated subset-4 page in the test list, decode GT knot bitmaps,
fit a minimum-area rotated rectangle to each connected component, then
prompt SAM2 three ways:

  1. Axis-aligned bbox of the OBB        (baseline; what YOLO would give)
  2. Mask_input prior from rasterised OBB (256x256 downsampled rect mask)
  3. Point-sampling along OBB axes:
       - 3 positives along the long axis (centre + 1/4 + 3/4)
       - 2 negatives just outside the OBB on the short axis (above + below)
       - 2 negatives just outside the OBB on the long axis (left + right ends)

All three SAM2 outputs are hard-clipped to the axis-aligned bbox of the OBB.

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.fake_obb_test \\
        --pages 39 73 154 238 239 268
"""

import argparse
import base64
import json
import pathlib
from typing import List, Optional, Tuple
import zlib

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch

SAM_CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
SAM_MODEL_CFG = "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"
SOURCE_IMG_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025/4/img"
SOURCE_ANN_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025/4/ann"
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/fake_obb"


def load_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1).astype(np.uint8)
    if arr.shape[-1] == 4:
        return arr[..., :3].astype(np.uint8)
    return arr.astype(np.uint8)


def decode_bitmap(b64: str) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(b64))
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., 3] > 0
    return arr > 0


def gt_knot_components(ann_path: str, h: int, w: int) -> List[np.ndarray]:
    with open(ann_path) as f:
        ann = json.load(f)
    out: List[np.ndarray] = []
    for obj in ann.get("objects", []):
        if obj.get("classTitle") != "Knot":
            continue
        bmp = obj["bitmap"]
        ox, oy = bmp["origin"]
        patch = decode_bitmap(bmp["data"])
        ph, pw = patch.shape
        mask = np.zeros((h, w), dtype=bool)
        mask[oy : oy + ph, ox : ox + pw][patch] = True
        if mask.sum() >= 8:
            out.append(mask)
    return out


def fit_obb(mask: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float], float]:
    """Returns (4-corner array (4,2), centre, (long_len, short_len), angle_deg).

    cv2.minAreaRect returns angle for the `width` side, not the long axis. If
    height > width we rotate the angle 90° so `angle_deg` describes the long
    axis direction.
    """
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=-1).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    if h > w:
        long_len, short_len = h, w
        angle += 90.0
    else:
        long_len, short_len = w, h
    corners = cv2.boxPoints(rect)
    return corners, (cx, cy), (long_len, short_len), float(angle)


def rasterise_obb(corners: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 1)
    return mask.astype(bool)


def aabb_of_obb(corners: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    x1 = max(0, int(np.floor(corners[:, 0].min())))
    y1 = max(0, int(np.floor(corners[:, 1].min())))
    x2 = min(w, int(np.ceil(corners[:, 0].max())))
    y2 = min(h, int(np.ceil(corners[:, 1].max())))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def sample_axis_points(
    centre: Tuple[float, float],
    long_len: float,
    short_len: float,
    angle_deg: float,
    inset_frac: float = 0.10,
    outset_frac: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """3 positives along the long axis (centre, 1/4, 3/4 from centre) +
    2 negatives off short-axis (above/below the OBB) +
    2 negatives off long-axis (beyond the OBB ends).
    inset_frac shrinks the positive sampling so points stay inside the rect
    even if the rect is loose; outset_frac pushes negatives just outside."""
    cx, cy = centre
    theta = np.deg2rad(angle_deg)
    long_dir = np.array([np.cos(theta), np.sin(theta)])
    short_dir = np.array([-np.sin(theta), np.cos(theta)])
    half_long = long_len / 2.0
    half_short = short_len / 2.0
    pos_long_offsets = [0.0, half_long * (1.0 - inset_frac) / 2.0, -half_long * (1.0 - inset_frac) / 2.0]
    positives = [np.array([cx, cy]) + d * long_dir for d in pos_long_offsets]
    neg_short = [
        np.array([cx, cy]) + (half_short * (1.0 + outset_frac)) * short_dir,
        np.array([cx, cy]) - (half_short * (1.0 + outset_frac)) * short_dir,
    ]
    neg_long = [
        np.array([cx, cy]) + (half_long * (1.0 + outset_frac)) * long_dir,
        np.array([cx, cy]) - (half_long * (1.0 + outset_frac)) * long_dir,
    ]
    pts = np.array(positives + neg_short + neg_long)
    labels = np.array([1, 1, 1, 0, 0, 0, 0])
    return pts, labels


def sam_predict(
    sam: SAM2ImagePredictor,
    box: Optional[np.ndarray] = None,
    point_coords: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    mask_input: Optional[np.ndarray] = None,
) -> np.ndarray:
    kwargs = {"multimask_output": False}
    if box is not None:
        kwargs["box"] = box.astype(np.float32)
    if point_coords is not None:
        kwargs["point_coords"] = point_coords.astype(np.float32)
        kwargs["point_labels"] = point_labels.astype(np.int32)
    if mask_input is not None:
        kwargs["mask_input"] = mask_input.astype(np.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, _, _ = sam.predict(**kwargs)
    return masks[0].astype(bool)


def low_res_mask(binary_mask: np.ndarray, size: int = 128) -> np.ndarray:
    """Resize a full-res binary mask to size×size logits-style input for SAM2.
    MedSAM2 t512 uses 128 (= 4 × 32 embedding dim)."""
    low = cv2.resize(binary_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    logits = np.where(low > 0, 10.0, -10.0).astype(np.float32)
    return logits[None]


def clip(mask: np.ndarray, aabb: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    x1, y1, x2, y2 = aabb.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2.0 * inter / denom) if denom > 0 else 0.0


def render_panel(ax, rgb: np.ndarray, masks: List[np.ndarray], title: str, overlays=None) -> None:
    gray = rgb.mean(axis=-1).astype(np.uint8)
    ax.imshow(gray, cmap="gray")
    overlay = np.zeros((*gray.shape, 4), dtype=np.float32)
    for m in masks:
        overlay[m] = (1.0, 0.55, 0.0, 0.45)
    ax.imshow(overlay)
    if overlays is not None:
        for kind, data in overlays:
            if kind == "obb":
                ax.add_patch(mpatches.Polygon(data, fill=False, edgecolor="lime", linewidth=1.5))
            elif kind == "aabb":
                x1, y1, x2, y2 = data
                ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1.2))
            elif kind == "pt":
                (x, y), lab = data
                ax.plot(x, y, marker="o", color="lime" if lab == 1 else "magenta", markersize=5, mec="white", mew=0.8)
    total = sum(int(m.sum()) for m in masks)
    ax.set_title("%s (%d px)" % (title, total), fontsize=10)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, nargs="+", required=True)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("loading SAM2 ...")
    sam = build_sam2("//" + SAM_MODEL_CFG, SAM_CHECKPOINT)
    sam_pred = SAM2ImagePredictor(sam)

    rows = []
    for page in args.pages:
        img_path = "%s/page_%03d.tiff" % (SOURCE_IMG_DIR, page)
        ann_path = "%s/page_%03d.tiff.json" % (SOURCE_ANN_DIR, page)
        rgb = load_rgb(img_path)
        h, w = rgb.shape[:2]
        gt_components = gt_knot_components(ann_path, h, w)
        if not gt_components:
            print("page %d: no GT knots, skip" % page)
            continue

        masks_aabb, masks_prior, masks_points = [], [], []
        overlays_aabb, overlays_prior, overlays_points = [], [], []
        ious_aabb, ious_prior, ious_points = [], [], []

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam_pred.set_image(rgb)
            for gt in gt_components:
                corners, centre, (long_len, short_len), angle = fit_obb(gt)
                aabb = aabb_of_obb(corners, (h, w))
                obb_raster = rasterise_obb(corners, (h, w))
                low = low_res_mask(obb_raster)
                pts, labs = sample_axis_points(centre, long_len, short_len, angle)

                m1 = sam_predict(sam_pred, box=aabb)
                m2 = sam_predict(sam_pred, box=aabb, mask_input=low)
                m3 = sam_predict(sam_pred, box=aabb, point_coords=pts, point_labels=labs)
                m1, m2, m3 = clip(m1, aabb), clip(m2, aabb), clip(m3, aabb)

                masks_aabb.append(m1)
                masks_prior.append(m2)
                masks_points.append(m3)
                ious_aabb.append(iou(m1, gt))
                ious_prior.append(iou(m2, gt))
                ious_points.append(iou(m3, gt))

                overlays_aabb.append(("aabb", aabb.tolist()))
                overlays_aabb.append(("obb", corners.tolist()))
                overlays_prior.append(("obb", corners.tolist()))
                overlays_points.append(("obb", corners.tolist()))
                for p, lab in zip(pts, labs):
                    overlays_points.append(("pt", ((float(p[0]), float(p[1])), int(lab))))

        rows.append(
            {
                "page": page,
                "n_knots": len(gt_components),
                "mean_iou_aabb": float(np.mean(ious_aabb)),
                "mean_iou_prior": float(np.mean(ious_prior)),
                "mean_iou_points": float(np.mean(ious_points)),
                "mean_dice_aabb": float(
                    np.mean([dice(masks_aabb[i], gt_components[i]) for i in range(len(gt_components))])
                ),
                "mean_dice_prior": float(
                    np.mean([dice(masks_prior[i], gt_components[i]) for i in range(len(gt_components))])
                ),
                "mean_dice_points": float(
                    np.mean([dice(masks_points[i], gt_components[i]) for i in range(len(gt_components))])
                ),
            }
        )

        fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
        render_panel(
            axes[0],
            rgb,
            gt_components,
            "GT knots",
            overlays=[("obb", c.tolist()) for c, *_ in [fit_obb(g) for g in gt_components]],
        )
        render_panel(axes[1], rgb, masks_aabb, "AABB-of-OBB (baseline)", overlays=overlays_aabb)
        render_panel(axes[2], rgb, masks_prior, "AABB + mask_input prior", overlays=overlays_prior)
        render_panel(axes[3], rgb, masks_points, "AABB + axis points", overlays=overlays_points)
        fig.suptitle(
            "page %d  (GT IoU AABB=%.2f  prior=%.2f  points=%.2f)"
            % (page, rows[-1]["mean_iou_aabb"], rows[-1]["mean_iou_prior"], rows[-1]["mean_iou_points"]),
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = "%s/page_%03d.png" % (args.out_dir, page)
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(
            "page %d  IoU AABB=%.2f prior=%.2f points=%.2f  wrote %s"
            % (page, rows[-1]["mean_iou_aabb"], rows[-1]["mean_iou_prior"], rows[-1]["mean_iou_points"], out_path)
        )

    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(pathlib.Path(args.out_dir) / "summary.csv", index=False)
        print("\nsummary across %d pages:" % len(rows))
        for col_iou, col_dice, name in [
            ("mean_iou_aabb", "mean_dice_aabb", "AABB-of-OBB"),
            ("mean_iou_prior", "mean_dice_prior", "AABB + mask_input"),
            ("mean_iou_points", "mean_dice_points", "AABB + axis points"),
        ]:
            print("  %-22s mean IoU=%.3f  mean Dice=%.3f" % (name, df[col_iou].mean(), df[col_dice].mean()))


if __name__ == "__main__":
    main()
