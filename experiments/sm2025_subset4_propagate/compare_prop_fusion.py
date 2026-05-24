"""Compare three knot-mask strategies on the worst v3 frames:

  1. baseline       — OBB → rasterise → SAM2 mask_input prior + AABB-of-OBB
                      box prompt + AABB clip.  (this is v3/v5)
  2. fused_prior    — same as baseline but mask_input = rasterised OBB UNION
                      overlapping propagation knot CC (variant A).
  3. intersected    — baseline mask AND propagation knot mask within the AABB
                      (variant B).

For each frame: 4-panel figure
    [image+OBB] [baseline] [fused_prior] [intersected]
with per-knot pixel counts so we can quantify the shift.

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.compare_prop_fusion \\
        --pages 39 73 154 238 239 268
"""

import argparse
import json
import pathlib
from typing import List, Optional

from ann_pipeline.knot.data_prep import knot_mask_from_ann
import cv2
import matplotlib.patches as mpatches
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
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/prop_fusion"


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
    """Return the union of propagation knot CCs whose intersection with the
    AABB has at least min_overlap_px pixels. Empty mask if none."""
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


def render_panel(
    ax,
    rgb: np.ndarray,
    masks: List[np.ndarray],
    title: str,
    aabbs=None,
    obbs=None,
    overlay_color=(1.0, 0.55, 0.0, 0.45),
    gt_mask: Optional[np.ndarray] = None,
) -> None:
    gray = rgb.mean(axis=-1).astype(np.uint8)
    ax.imshow(gray, cmap="gray")
    overlay = np.zeros((*gray.shape, 4), dtype=np.float32)
    total = 0
    for m in masks:
        overlay[m] = overlay_color
        total += int(m.sum())
    ax.imshow(overlay)
    if gt_mask is not None and gt_mask.any():
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) > 2:
                cc = c.squeeze(1)
                ax.plot(cc[:, 0], cc[:, 1], color="lime", linewidth=1.8)
    if obbs is not None:
        for corners in obbs:
            ax.add_patch(mpatches.Polygon(corners, fill=False, edgecolor="cyan", linewidth=1.0, linestyle="-"))
    if aabbs is not None:
        for b in aabbs:
            ax.add_patch(
                mpatches.Rectangle(
                    (b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor="red", linewidth=0.8, linestyle="--"
                )
            )
    ax.set_title("%s  (%d px, %d knots)" % (title, total, len(masks)), fontsize=10)
    ax.axis("off")


def load_gt_knot_mask(page: int) -> Optional[np.ndarray]:
    """Return full-canvas binary GT knot mask if the page has annotations, else None."""
    ann_path = page_ann_path(page)
    try:
        with open(ann_path) as f:
            ann = json.load(f)
    except FileNotFoundError:
        return None
    if not any(obj.get("classTitle") == "Knot" for obj in ann.get("objects", [])):
        return None
    return knot_mask_from_ann(ann).astype(bool)


ALL_ANNOTATED = [
    7,
    18,
    28,
    35,
    39,
    49,
    64,
    74,
    84,
    101,
    103,
    111,
    120,
    136,
    145,
    152,
    154,
    157,
    166,
    167,
    175,
    188,
    193,
    196,
    201,
    214,
    223,
    232,
    234,
    235,
    236,
    238,
    241,
    243,
    245,
    247,
    254,
    257,
    265,
    276,
    285,
    291,
    293,
    295,
    297,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pages",
        type=int,
        nargs="+",
        default=None,
        help="Explicit page list. If unset, uses --annotated or --auto_n.",
    )
    parser.add_argument(
        "--annotated",
        action="store_true",
        help="Use all 45 annotated subset-4 pages so we always have GT overlays.",
    )
    parser.add_argument(
        "--auto_n",
        type=int,
        default=0,
        help="If set, pick this many pages evenly spread across all annotated pages.",
    )
    parser.add_argument(
        "--nonanchor_distances",
        type=int,
        nargs="+",
        default=None,
        help="If set, pick one non-anchor page per requested distance to nearest anchor. "
        "e.g. --nonanchor_distances 1 2 3 4 5 8 12",
    )
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    if args.pages:
        pages_to_run = args.pages
    elif args.annotated:
        pages_to_run = ALL_ANNOTATED
    elif args.auto_n > 0:
        idx = np.linspace(0, len(ALL_ANNOTATED) - 1, args.auto_n).astype(int)
        pages_to_run = [ALL_ANNOTATED[i] for i in idx]
    elif args.nonanchor_distances:
        anchor_set = set(ALL_ANNOTATED)
        chosen: List[int] = []
        for d in args.nonanchor_distances:
            for p in range(min(ALL_ANNOTATED), max(ALL_ANNOTATED) + 1):
                if p in anchor_set or p in chosen:
                    continue
                if min(abs(p - a) for a in ALL_ANNOTATED) == d:
                    chosen.append(p)
                    break
        pages_to_run = chosen
        print("picked non-anchor pages: %s" % list(zip(args.nonanchor_distances, chosen)))
    else:
        msg = "Provide --pages, --annotated, --auto_n, or --nonanchor_distances"
        raise ValueError(msg)
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    data = np.load(NPZ, allow_pickle=False)
    npz_pages = data["pages"].tolist()
    pred_vol = data["pred"]

    yolo = YOLO(YOLO_OBB_WEIGHTS)
    print("loading SAM2 ...")
    sam = build_sam2("//" + SAM_MODEL_CFG, SAM_CHECKPOINT)
    sam_pred = SAM2ImagePredictor(sam)

    for page in pages_to_run:
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
        fused_masks: List[np.ndarray] = []
        intersect_masks: List[np.ndarray] = []
        aabbs: List[np.ndarray] = []
        fused_priors_rendered: List[np.ndarray] = []

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
                aabbs.append(aabb)

                prior_baseline = low_res_logits(obb_raster.astype(bool))
                m_base, _, _ = sam_pred.predict(box=aabb, mask_input=prior_baseline, multimask_output=False)
                baseline_masks.append(clip_to_aabb(m_base[0].astype(bool), aabb))

                prop_cc = overlapping_prop_cc(prop_knot, aabb)
                fused_raster = obb_raster.astype(bool) | prop_cc
                fused_priors_rendered.append(fused_raster)
                prior_fused = low_res_logits(fused_raster)
                m_fused, _, _ = sam_pred.predict(box=aabb, mask_input=prior_fused, multimask_output=False)
                fused_masks.append(clip_to_aabb(m_fused[0].astype(bool), aabb))

                inter = baseline_masks[-1] & prop_cc
                intersect_masks.append(clip_to_aabb(inter, aabb))

        union_masks = list(baseline_masks)
        n_prop_added = 0
        lab, n_cc = ndi.label(prop_knot, structure=np.ones((3, 3), dtype=np.uint8))
        any_baseline = np.zeros((h, w), dtype=bool)
        for m in baseline_masks:
            any_baseline |= m
        for k in range(1, n_cc + 1):
            comp = lab == k
            if comp.sum() < 50:
                continue
            if np.logical_and(comp, any_baseline).sum() == 0:
                union_masks.append(comp)
                n_prop_added += 1

        fig, axes = plt.subplots(1, 6, figsize=(30, 5.4))
        render_panel(
            axes[0],
            rgb,
            [gt_mask] if gt_mask is not None else [],
            "GT knots" + ("" if gt_mask is not None else " — none on this page"),
            overlay_color=(0.2, 1.0, 0.4, 0.45),
        )
        render_panel(axes[1], rgb, [prop_knot], "propagation knots", obbs=[c for c in xyxyxyxy], gt_mask=gt_mask)
        render_panel(
            axes[2],
            rgb,
            baseline_masks,
            "baseline (OBB→SAM2)",
            aabbs=aabbs,
            obbs=[c for c in xyxyxyxy],
            gt_mask=gt_mask,
        )
        render_panel(
            axes[3],
            rgb,
            fused_masks,
            "fused prior (OBB ∪ prop)",
            aabbs=aabbs,
            obbs=[c for c in xyxyxyxy],
            gt_mask=gt_mask,
        )
        render_panel(
            axes[4],
            rgb,
            intersect_masks,
            "intersected (baseline ∩ prop)",
            aabbs=aabbs,
            obbs=[c for c in xyxyxyxy],
            gt_mask=gt_mask,
        )
        render_panel(
            axes[5],
            rgb,
            union_masks,
            "C: union (v5 + non-overlapping prop CCs, +%d)" % n_prop_added,
            obbs=[c for c in xyxyxyxy],
            gt_mask=gt_mask,
        )
        annotated_tag = " GT-annotated" if gt_mask is not None else ""
        fig.suptitle(
            "page %d%s  (conf=%.2f, nms_iou=%.2f, n_OBB=%d). Lime contour=GT, cyan polygon=OBB det, red dashed=AABB."
            % (page, annotated_tag, args.conf, args.nms_iou, len(xyxyxyxy)),
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = "%s/page_%03d.png" % (args.out_dir, page)
        plt.savefig(out_path, dpi=120)
        plt.close()

        per_knot = [
            (i, int(baseline_masks[i].sum()), int(fused_masks[i].sum()), int(intersect_masks[i].sum()))
            for i in range(len(baseline_masks))
        ]
        print("page %d wrote %s (union added %d prop CCs)" % (page, out_path, n_prop_added))
        for i, b, f, x in per_knot:
            print("    knot %d: baseline=%d  fused=%d  intersect=%d" % (i, b, f, x))


if __name__ == "__main__":
    main()
