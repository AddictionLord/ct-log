"""Evaluate wood-mask detectors on every Wood-annotated slice across all subsets.

Approaches compared:
  - threshold_largest_cc      (pure intensity, no model)
  - threshold_morphology      (threshold + closing + hole fill)
  - sam_centre_plus_corners   (SAM2 with image centre + 4 corner negatives)
  - sam_pith_plus_corners     (SAM2 with YOLO-predicted pith point + 4 corners)

Per slice: report Dice against GT Wood mask. Print summary, save CSV + montage.

Run from repo root:
    conda run -n ct-log python -m ann_pipeline.wood.eval
"""

import argparse
import base64
import json
import os
from os.path import join
import pathlib
import time
from typing import Optional, Tuple
import zlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ann_pipeline.wood.detectors import (
    sam_centre_plus_corners,
    sam_pith_plus_corners,
    threshold_largest_cc,
    threshold_morphology,
)

PROJECT_ROOT = "/mnt/D/datasets/ct_log/375492_SM_2025"
YOLO_WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_2cls_v1/weights/best.pt"
CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
MODEL_CFG = "//" + "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"


def decode_bitmap(b64: str) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(b64))
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., 3] > 0
    return arr > 0


def wood_mask_from_ann(ann: dict) -> np.ndarray:
    h, w = ann["size"]["height"], ann["size"]["width"]
    mask = np.zeros((h, w), dtype=np.uint8)
    for obj in ann.get("objects", []):
        if obj.get("classTitle") != "Wood":
            continue
        bmp = obj["bitmap"]
        ox, oy = bmp["origin"]
        patch = decode_bitmap(bmp["data"])
        ph, pw = patch.shape
        mask[oy : oy + ph, ox : ox + pw][patch] = 1
    return mask


def load_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3]
    else:
        arr = np.stack([arr] * 3, axis=-1)
    return arr.astype(np.uint8)


def load_gray(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    return arr.astype(np.uint8)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


def find_wood_slices(project_root: str, subsets) -> list:
    """Return list of (subset_id, ann_filename) for slices that have a Wood object."""
    out = []
    for sub in subsets:
        ann_dir = join(project_root, sub, "ann")
        for f in sorted(os.listdir(ann_dir)):
            with open(join(ann_dir, f)) as fh:
                d = json.load(fh)
            if any(o.get("classTitle") == "Wood" for o in d.get("objects", [])):
                out.append((sub, f))
    return out


def yolo_pith_point(yolo, rgb: np.ndarray, conf: float = 0.25) -> Optional[Tuple[float, float]]:
    res = yolo.predict(rgb, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    cls = res.boxes.cls.cpu().numpy().astype(int)
    xyxy = res.boxes.xyxy.cpu().numpy()
    pith_mask = cls == 1
    if not pith_mask.any():
        return None
    b = xyxy[pith_mask][0]
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=PROJECT_ROOT)
    parser.add_argument("--subsets", nargs="+", default=["1", "2", "4"])
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/wood_eval")
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--montage_n", type=int, default=24, help="max slices in montage")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slices = find_wood_slices(args.project_root, args.subsets)
    print(f"{len(slices)} wood-annotated slices")

    yolo = YOLO(YOLO_WEIGHTS)
    sam = build_sam2(MODEL_CFG, CHECKPOINT)
    predictor = SAM2ImagePredictor(sam)

    rows = []
    examples = {}  # detector_name -> list of (img, gt, pred, page, dice) for montage

    for sub, ann_fname in tqdm(slices, desc="evaluating"):
        sm_dir = join(args.project_root, sub)
        with open(join(sm_dir, "ann", ann_fname)) as f:
            ann = json.load(f)
        img_fname = ann_fname[: -len(".json")]
        page = int(img_fname.replace("page_", "").replace(".tiff", ""))
        rgb = load_rgb(join(sm_dir, "img", img_fname))
        gray = load_gray(join(sm_dir, "img", img_fname))
        gt = wood_mask_from_ann(ann)

        # 1. threshold + largest CC
        t0 = time.perf_counter()
        pred_thresh = threshold_largest_cc(gray, thresh=args.threshold)
        ms_thresh = (time.perf_counter() - t0) * 1000.0
        d_thresh = dice(pred_thresh, gt)

        # 2. threshold + morphology
        t0 = time.perf_counter()
        pred_morph = threshold_morphology(gray, thresh=args.threshold)
        ms_morph = (time.perf_counter() - t0) * 1000.0
        d_morph = dice(pred_morph, gt)

        # 3. SAM with image centre + 4 corner negatives
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_sam_centre = sam_centre_plus_corners(rgb, predictor)
        ms_sam_centre = (time.perf_counter() - t0) * 1000.0
        d_sam_centre = dice(pred_sam_centre, gt)

        # 4. SAM with YOLO pith point as positive + 4 corner negatives
        pith_pt = yolo_pith_point(yolo, rgb)
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_sam_pith = sam_pith_plus_corners(rgb, predictor, pith_xy=pith_pt)
        ms_sam_pith = (time.perf_counter() - t0) * 1000.0
        d_sam_pith = dice(pred_sam_pith, gt)

        rows.append(
            {
                "subset": sub,
                "page": page,
                "gt_px": int(gt.sum()),
                "thresh_dice": d_thresh,
                "thresh_ms": ms_thresh,
                "morph_dice": d_morph,
                "morph_ms": ms_morph,
                "sam_centre_dice": d_sam_centre,
                "sam_centre_ms": ms_sam_centre,
                "sam_pith_dice": d_sam_pith,
                "sam_pith_ms": ms_sam_pith,
                "yolo_found_pith": pith_pt is not None,
            }
        )

        # cache a few examples for visualisation
        for name, pred, d in [
            ("threshold_largest_cc", pred_thresh, d_thresh),
            ("threshold_morphology", pred_morph, d_morph),
            ("sam_centre_plus_corners", pred_sam_centre, d_sam_centre),
            ("sam_pith_plus_corners", pred_sam_pith, d_sam_pith),
        ]:
            examples.setdefault(name, []).append((gray, gt, pred, page, sub, d))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_slice.csv", index=False)

    summary = []
    for name, key in [
        ("threshold_largest_cc", "thresh"),
        ("threshold_morphology", "morph"),
        ("sam_centre_plus_corners", "sam_centre"),
        ("sam_pith_plus_corners", "sam_pith"),
    ]:
        d = df[f"{key}_dice"]
        ms = df[f"{key}_ms"]
        summary.append(
            {
                "detector": name,
                "n": len(d),
                "mean_dice": d.mean(),
                "median_dice": d.median(),
                "p10_dice": d.quantile(0.1),
                "max_dice": d.max(),
                "mean_ms": ms.mean(),
            }
        )
    sdf = pd.DataFrame(summary).sort_values("mean_dice", ascending=False)
    sdf.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== summary (sorted by mean Dice) ===")
    print(sdf.to_string(index=False))

    # render a montage per detector (sub-sampled to montage_n)
    for name, items in examples.items():
        items = items[: args.montage_n]
        n = len(items)
        cols = min(4, n) if n else 1
        rows_n = (n + cols - 1) // cols if n else 1
        fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n))
        axes = np.array(axes).reshape(-1)
        for ax, (gray, gt, pred, page, sub, d_val) in zip(axes, items):
            ax.imshow(gray, cmap="gray")
            ax.imshow(np.ma.masked_where(~gt.astype(bool), gt), alpha=0.3, cmap="autumn", vmin=0, vmax=1)
            for c in _mask_contours(pred):
                ax.plot(c[:, 0], c[:, 1], color="cyan", linewidth=1.2)
            ax.set_title(f"ds{sub} p{page} dice={d_val:.3f}", fontsize=9)
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"{name}.  GT: orange overlay.  Pred: cyan outline.", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}.png", dpi=100)
        plt.close()

    print(f"\nresults: {out_dir}")


def _mask_contours(mask: np.ndarray) -> list:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.squeeze(1) for c in contours if len(c) > 2]


if __name__ == "__main__":
    main()
