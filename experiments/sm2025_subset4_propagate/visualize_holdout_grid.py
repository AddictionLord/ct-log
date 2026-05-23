"""TP/FP/FN visual grid for the 9 holdout pages.

For each holdout page render a row:
    [grayscale + GT outlines]  [YOLO+SAM2 TP/FP/FN]  [propagation TP/FP/FN]

Colour code (matched at IoU >= --iou_thr):
    TP outline = lime
    FP outline = red
    FN outline = orange (GT-side, unmatched)
    Pith GT    = blue cross
    Pith pred  = magenta cross (YOLO bbox centre OR prop blob centroid)

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.visualize_holdout_grid \\
        --yolo_weights .../yolo11n_holdout_v1/weights/best.pt
"""

import argparse
import json
import pathlib
from typing import Dict, List, Optional, Tuple

from ann_pipeline.knot.data_prep import knot_mask_from_ann
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.optimize import linear_sum_assignment
import torch
from ultralytics import YOLO

from experiments.sm2025_subset4_propagate.holdout_eval import (
    HOLDOUT_PAGES,
    iou,
    split_components,
)
from experiments.sm2025_subset4_propagate.run import (
    CLASS_IDS,
    page_ann_path,
    page_img_path,
)

CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
MODEL_CFG_PATH = "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"
OUT_PATH = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/holdout_eval/visual_grid.png"


def match(preds: List[np.ndarray], gts: List[np.ndarray], iou_thr: float):
    if not preds or not gts:
        return [], list(range(len(preds))), list(range(len(gts)))
    mat = np.zeros((len(preds), len(gts)), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            mat[i, j] = iou(p, g)
    ri, ci = linear_sum_assignment(1.0 - mat)
    tp, mp, mg = [], set(), set()
    for r, c in zip(ri, ci):
        if mat[r, c] >= iou_thr:
            tp.append((int(r), int(c), float(mat[r, c])))
            mp.add(int(r))
            mg.add(int(c))
    fp = [i for i in range(len(preds)) if i not in mp]
    fn = [j for j in range(len(gts)) if j not in mg]
    return tp, fp, fn


def contours(mask: np.ndarray) -> List[np.ndarray]:
    m = mask.astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.squeeze(1) for c in cs if len(c) > 2]


def draw_outlines(ax, masks: List[np.ndarray], color: str, lw: float = 1.5, label: Optional[str] = None) -> None:
    for i, m in enumerate(masks):
        for c in contours(m):
            ax.plot(c[:, 0], c[:, 1], color=color, linewidth=lw, label=label if i == 0 and label else None)


def gt_pith(ann_path: str) -> Optional[Tuple[float, float]]:
    with open(ann_path) as f:
        ann = json.load(f)
    for obj in ann.get("objects", []):
        if obj.get("classTitle") == "Pith":
            pts = obj.get("points", {}).get("exterior", [])
            if pts:
                return float(pts[0][0]), float(pts[0][1])
    return None


def yolo_sam_inference(
    yolo: YOLO, sam_pred: SAM2ImagePredictor, img_path: str, conf: float
) -> Tuple[List[np.ndarray], Optional[Tuple[float, float]]]:
    arr = np.array(Image.open(img_path))
    if arr.ndim == 2:
        rgb = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        rgb = arr[..., :3]
    else:
        rgb = arr
    rgb = rgb.astype(np.uint8)

    res = yolo.predict(rgb, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return [], None
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    knot_boxes = [b for b, c in zip(xyxy, cls) if c == 0]
    pith_pt: Optional[Tuple[float, float]] = None
    best_pith_conf = -1.0
    for b, c, cf in zip(xyxy, cls, confs):
        if c == 1 and cf > best_pith_conf:
            best_pith_conf = float(cf)
            pith_pt = ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)

    knot_masks: List[np.ndarray] = []
    if knot_boxes:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam_pred.set_image(rgb)
            for box in knot_boxes:
                masks, _, _ = sam_pred.predict(box=np.array(box, dtype=np.float32), multimask_output=False)
                knot_masks.append(masks[0].astype(bool))
    return knot_masks, pith_pt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo_weights",
        default="/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_holdout_v1/weights/best.pt",
    )
    parser.add_argument(
        "--prop_npz",
        default="/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/holdout_eval/prop_pred_holdout_free.npz",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou_thr", type=float, default=0.30)
    parser.add_argument("--out_path", default=OUT_PATH)
    args = parser.parse_args()

    yolo = YOLO(args.yolo_weights)
    print("loading SAM2 image predictor ...")
    sam = build_sam2("//" + MODEL_CFG_PATH, CHECKPOINT)
    sam_pred = SAM2ImagePredictor(sam)

    data = np.load(args.prop_npz, allow_pickle=False)
    pages_arr = data["pages"].tolist()
    pred_vol = data["pred"]
    prop_by_page: Dict[int, np.ndarray] = {p: pred_vol[pages_arr.index(p)] for p in HOLDOUT_PAGES}

    n_rows = len(HOLDOUT_PAGES)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.5 * n_rows))

    for row, page in enumerate(HOLDOUT_PAGES):
        ann_path = page_ann_path(page)
        img_path = page_img_path(page)
        rgb = np.array(Image.open(img_path))
        if rgb.ndim == 3:
            gray = rgb[..., :3].mean(axis=-1).astype(np.uint8)
        else:
            gray = rgb

        with open(ann_path) as f:
            ann = json.load(f)
        gt_mask = knot_mask_from_ann(ann)
        gts = split_components(gt_mask.astype(bool))
        gt_p = gt_pith(ann_path)

        yolo_masks, yolo_p = yolo_sam_inference(yolo, sam_pred, img_path, args.conf)
        prop_pred = prop_by_page[page]
        prop_masks = split_components(prop_pred == CLASS_IDS["Knot"])
        ys, xs = np.nonzero(prop_pred == CLASS_IDS["Pith"])
        prop_p = (float(xs.mean()), float(ys.mean())) if len(xs) > 0 else None

        y_tp, y_fp, y_fn = match(yolo_masks, gts, args.iou_thr)
        p_tp, p_fp, p_fn = match(prop_masks, gts, args.iou_thr)

        from experiments.sm2025_subset4_propagate.holdout_eval import nearest_anchor_distance

        all_annotated = [
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
        train_anchors = [p for p in all_annotated if p not in HOLDOUT_PAGES]
        dist = nearest_anchor_distance(page, train_anchors)

        ax0 = axes[row, 0]
        ax0.imshow(gray, cmap="gray")
        draw_outlines(ax0, gts, "lime", lw=2.0, label="GT knots")
        if gt_p is not None:
            ax0.plot(gt_p[0], gt_p[1], marker="+", color="cyan", markersize=14, mew=2.0, label="GT pith")
        ax0.set_title("page %d  (dist to anchor: %d)\n%d GT knots" % (page, dist, len(gts)))
        ax0.axis("off")

        for ax_idx, (name, masks, tp, fp, fn, pith) in enumerate(
            [("YOLO+SAM2", yolo_masks, y_tp, y_fp, y_fn, yolo_p), ("Propagation", prop_masks, p_tp, p_fp, p_fn, prop_p)]
        ):
            ax = axes[row, 1 + ax_idx]
            ax.imshow(gray, cmap="gray")
            for c in contours(gt_mask):
                ax.plot(c[:, 0], c[:, 1], color="lime", linewidth=0.8, alpha=0.4)
            tp_masks = [masks[i] for i, _, _ in tp]
            fp_masks = [masks[i] for i in fp]
            fn_masks = [gts[j] for j in fn]
            draw_outlines(ax, tp_masks, "lime", lw=2.0)
            draw_outlines(ax, fp_masks, "red", lw=2.0)
            draw_outlines(ax, fn_masks, "orange", lw=2.0)
            if pith is not None and gt_p is not None:
                ax.plot(pith[0], pith[1], marker="x", color="magenta", markersize=12, mew=2.0)
                ax.plot(gt_p[0], gt_p[1], marker="+", color="cyan", markersize=14, mew=2.0)
                d = float(np.hypot(pith[0] - gt_p[0], pith[1] - gt_p[1]))
                ax.text(
                    5,
                    gray.shape[0] - 10,
                    "pith=%.2f px" % d,
                    color="white",
                    fontsize=9,
                    bbox=dict(facecolor="black", alpha=0.5, pad=2),
                )
            ax.set_title("%s  TP=%d FP=%d FN=%d" % (name, len(tp), len(fp), len(fn)))
            ax.axis("off")

    legend_handles = [
        mpatches.Patch(color="lime", label="TP / GT outline"),
        mpatches.Patch(color="red", label="FP (pred unmatched)"),
        mpatches.Patch(color="orange", label="FN (GT unmatched)"),
        mpatches.Patch(color="cyan", label="GT pith +"),
        mpatches.Patch(color="magenta", label="pred pith x"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5, fontsize=11)
    fig.suptitle(
        "Holdout TP/FP/FN grid (IoU>=%.2f, conf=%.2f). yolo weights: %s"
        % (args.iou_thr, args.conf, pathlib.Path(args.yolo_weights).parent.parent.name),
        fontsize=13,
        y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    pathlib.Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_path, dpi=110)
    plt.close()
    print("wrote", args.out_path)


if __name__ == "__main__":
    main()
