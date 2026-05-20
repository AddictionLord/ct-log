"""Use the trained YOLO 2-class detector to prompt MedSAM2 on each val slice,
then compare the resulting masks against the Supervisely GT.

Per slice:
  - YOLO predicts knot bboxes (cls=0) and pith bboxes (cls=1)
  - For each knot bbox: SAM2.predict(box=...) -> binary mask, OR'd into the
    semantic knot mask
  - For each pith bbox: SAM2.predict(point=bbox_centroid, label=1) -> mask
  - GT knot mask is decoded from Supervisely; Dice is computed only on Knot
    (Pith GT is a 3-px disk so its Dice is not informative — visualised only)

Run from repo root:
    conda run -n ct-log python -m ann_pipeline.scripts.yolo_to_sam_eval
"""

import argparse
import json
from os.path import join
import pathlib
from typing import List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from ultralytics import YOLO

from ann_pipeline.knot.data_prep import knot_mask_from_ann, pith_bboxes_from_ann

PROJECT_ROOT = "/mnt/D/datasets/ct_log/375492_SM_2025"
DATA_DIR = "/home/mary/code/ct-log/ann_pipeline/out/knot_yolo"
WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_2cls_v1/weights/best.pt"
CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
MODEL_CFG_PATH = "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"
OUT_DIR_DEFAULT = "/home/mary/code/ct-log/ann_pipeline/out/yolo_to_sam_eval"


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


def load_ann_for_image(image_stem: str) -> Tuple[dict, str, str]:
    """Resolve `ds{N}_page_NNN` back to the Supervisely ann + img paths."""
    subset_id, _, page_part = image_stem.partition("_")
    subset_id = subset_id.replace("ds", "")
    fname = page_part + ".tiff"
    ann_path = join(PROJECT_ROOT, subset_id, "ann", fname + ".json")
    img_path = join(PROJECT_ROOT, subset_id, "img", fname)
    with open(ann_path) as f:
        ann = json.load(f)
    return ann, ann_path, img_path


def extract_pith_point(ann: dict) -> Optional[Tuple[float, float]]:
    """Return (x, y) of the first Pith point in the annotation, or None."""
    for obj in ann.get("objects", []):
        if obj.get("classTitle") == "Pith":
            pts = obj.get("points", {}).get("exterior", [])
            if pts:
                return float(pts[0][0]), float(pts[0][1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--pith_bbox_half", type=int, default=8)
    parser.add_argument(
        "--knot_prompt",
        choices=["box", "point", "box+point"],
        default="box",
        help="How to prompt SAM2 for each YOLO knot detection.",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_slice").mkdir(exist_ok=True)

    img_dir = pathlib.Path(args.data_dir) / "images" / args.split
    image_paths = sorted(img_dir.glob("*.jpg"))
    print(f"loaded {len(image_paths)} {args.split} images")

    yolo = YOLO(args.weights)

    print("building SAM2 image predictor ...")
    cfg = "//" + MODEL_CFG_PATH
    sam = build_sam2(cfg, CHECKPOINT)
    predictor = SAM2ImagePredictor(sam)

    rows = []
    n = len(image_paths)
    cols = min(4, n)
    gridrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(gridrows, cols, figsize=(5 * cols, 5 * gridrows))
    axes = np.array(axes).reshape(-1)

    for ax, img_path in zip(axes, image_paths):
        ann, _, _ = load_ann_for_image(img_path.stem)
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        h, w = img_rgb.shape[:2]

        gt_knot = knot_mask_from_ann(ann)
        gt_pith_boxes = pith_bboxes_from_ann(ann, half_size=args.pith_bbox_half)

        # YOLO predictions
        result = yolo.predict(img_path, conf=args.conf, verbose=False)[0]
        pred_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        pred_cls = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.empty((0,), int)
        knot_bboxes = pred_xyxy[pred_cls == 0]
        pith_bboxes = pred_xyxy[pred_cls == 1]

        # SAM2 inference
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(img_rgb)

            knot_mask = np.zeros((h, w), dtype=bool)
            for box in knot_bboxes:
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                kwargs = {"multimask_output": False}
                if args.knot_prompt in ("box", "box+point"):
                    kwargs["box"] = np.array(box, dtype=np.float32)
                if args.knot_prompt in ("point", "box+point"):
                    kwargs["point_coords"] = np.array([[cx, cy]], dtype=np.float32)
                    kwargs["point_labels"] = np.array([1], dtype=np.int32)
                masks, scores, _ = predictor.predict(**kwargs)
                knot_mask |= masks[0].astype(bool)

            pith_mask = np.zeros((h, w), dtype=bool)
            for box in pith_bboxes:
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[cx, cy]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=False,
                )
                pith_mask |= masks[0].astype(bool)

        d_knot = dice(knot_mask.astype(np.uint8), gt_knot)

        # Pith centroid distance: nearest YOLO pith bbox centroid to the GT point
        gt_pith_pt = extract_pith_point(ann)
        pith_dist: Optional[float] = None
        if gt_pith_pt is not None and len(pith_bboxes) > 0:
            cxs = (pith_bboxes[:, 0] + pith_bboxes[:, 2]) / 2.0
            cys = (pith_bboxes[:, 1] + pith_bboxes[:, 3]) / 2.0
            dists = np.hypot(cxs - gt_pith_pt[0], cys - gt_pith_pt[1])
            pith_dist = float(dists.min())

        rows.append(
            {
                "image": img_path.name,
                "n_pred_knot": int((pred_cls == 0).sum()),
                "n_pred_pith": int((pred_cls == 1).sum()),
                "gt_knot_px": int(gt_knot.sum()),
                "pred_knot_px": int(knot_mask.sum()),
                "pred_pith_px": int(pith_mask.sum()),
                "dice_knot": d_knot,
                "pith_centroid_dist_px": pith_dist,
                "gt_has_pith": gt_pith_pt is not None,
            }
        )

        # visualization: image + overlays
        ax.imshow(img_rgb)
        # GT knot mask (solid lime overlay)
        ax.imshow(np.ma.masked_where(~gt_knot.astype(bool), gt_knot), alpha=0.35, cmap="autumn", vmin=0, vmax=1)
        # predicted knot mask (cyan outline)
        for contour in _mask_contours(knot_mask):
            ax.plot(contour[:, 0], contour[:, 1], color="cyan", linewidth=1.5)
        # predicted pith mask (orange outline)
        for contour in _mask_contours(pith_mask):
            ax.plot(contour[:, 0], contour[:, 1], color="orange", linewidth=1.5)
        # GT pith bboxes (small cyan rectangles for reference)
        for x_min, y_min, x_max, y_max in gt_pith_boxes:
            ax.add_patch(
                mpatches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor="dodgerblue",
                    linewidth=1.0,
                )
            )
        # YOLO knot bboxes (dashed red)
        for x_min, y_min, x_max, y_max in knot_bboxes:
            ax.add_patch(
                mpatches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.0,
                    linestyle="--",
                )
            )
        ax.set_title(f"{img_path.stem}  knot Dice={d_knot:.2f}", fontsize=9)
        ax.axis("off")

        # also save the per-slice mask
        np.savez_compressed(
            out_dir / "per_slice" / f"{img_path.stem}.npz",
            gt_knot=gt_knot.astype(np.uint8),
            pred_knot=knot_mask.astype(np.uint8),
            pred_pith=pith_mask.astype(np.uint8),
        )

    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        f"YOLO -> SAM2 (image predictor) on val, conf>={args.conf}, knot_prompt={args.knot_prompt}.  "
        "GT knot: orange overlay.  Pred knot: cyan outline.  Pred pith: orange outline.  "
        "Red dashed: YOLO knot bboxes (prompts).  Blue: GT pith bbox.",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.split}_yolo_to_sam.png", dpi=100)
    plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{args.split}_per_slice.csv", index=False)

    valid = df[df.dice_knot.notna()]
    print("\n=== knot Dice (SAM2 fed YOLO bboxes) ===")
    print(f"  slices with GT knots: {len(valid)}")
    print(f"  mean Dice:   {valid.dice_knot.mean():.3f}")
    print(f"  median Dice: {valid.dice_knot.median():.3f}")
    print(f"  p10 Dice:    {valid.dice_knot.quantile(0.1):.3f}")
    print(f"  max Dice:    {valid.dice_knot.max():.3f}")

    pith_df = df[df.gt_has_pith & df.pith_centroid_dist_px.notna()]
    n_with_gt_pith = int(df.gt_has_pith.sum())
    n_matched_pith = len(pith_df)
    print("\n=== pith centroid distance (YOLO pred vs GT point) ===")
    print(f"  slices with GT pith: {n_with_gt_pith}, matched by YOLO: {n_matched_pith}")
    if len(pith_df):
        print(f"  mean dist:   {pith_df.pith_centroid_dist_px.mean():.2f} px")
        print(f"  median dist: {pith_df.pith_centroid_dist_px.median():.2f} px")
        print(f"  p90 dist:    {pith_df.pith_centroid_dist_px.quantile(0.9):.2f} px")
        print(f"  max dist:    {pith_df.pith_centroid_dist_px.max():.2f} px")

    print(f"\nvis: {out_dir / f'{args.split}_yolo_to_sam.png'}")
    print(f"csv: {out_dir / f'{args.split}_per_slice.csv'}")


def _mask_contours(mask: np.ndarray) -> List[np.ndarray]:
    """Return list of (N,2) contours from a binary mask."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.squeeze(1) for c in contours if len(c) > 2]


if __name__ == "__main__":
    main()
