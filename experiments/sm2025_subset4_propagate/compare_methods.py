"""Head-to-head comparison: YOLO+SAM vs anchor-propagation on the 3 subset-4
val pages (39, 201, 243).

These three pages are in YOLO's val set (so YOLO weights never trained on them)
and we re-run propagation with those 3 anchors HELD OUT so propagation never
sees them either.

For each of the 3 frames we report:
  * pith centroid distance (px) for YOLO bbox centroid and for propagation
    pith-blob centroid, vs GT pith point.
  * knot mAP-style centroid matching for both methods (Hungarian).

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.compare_methods
"""

import argparse
import base64
import json
import os
from os.path import join
import pathlib
from typing import Dict, List, Tuple
import zlib

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor_npz
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from experiments.sm2025_subset4_propagate.run import (
    CLASS_IDS,
    find_anchor_pages,
    list_pages,
    load_tiff_gray,
    page_ann_path,
    page_img_path,
    propagate_segment,
    render_annotation,
)

HOLDOUT_PAGES = [39, 201, 243]
YOLO_WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_2cls_v1/weights/best.pt"
CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/compare"


def knot_components_centroids(mask: np.ndarray, min_px: int = 8) -> List[Tuple[float, float]]:
    labelled, n = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    out = []
    for k in range(1, n + 1):
        comp = labelled == k
        if comp.sum() < min_px:
            continue
        ys, xs = np.nonzero(comp)
        out.append((float(xs.mean()), float(ys.mean())))
    return out


def hungarian_match(
    preds: List[Tuple[float, float]], gts: List[Tuple[float, float]]
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    if not preds and not gts:
        return [], [], []
    if not preds:
        return [], [], list(range(len(gts)))
    if not gts:
        return [], list(range(len(preds))), []
    p = np.array(preds, np.float32)
    g = np.array(gts, np.float32)
    d = np.linalg.norm(p[:, None, :] - g[None, :, :], axis=-1)
    ri, ci = linear_sum_assignment(d)
    matched = [(int(r), int(c), float(d[r, c])) for r, c in zip(ri, ci)]
    matched_p = {r for r, _, _ in matched}
    matched_g = {c for _, c, _ in matched}
    return (
        matched,
        [i for i in range(len(preds)) if i not in matched_p],
        [i for i in range(len(gts)) if i not in matched_g],
    )


def render_gt_knot_centroids(ann_path: str, h: int, w: int) -> List[Tuple[float, float]]:
    """Each Knot annotation in the JSON becomes one GT centroid (the centroid
    of its decoded bitmap)."""
    with open(ann_path) as f:
        ann = json.load(f)
    out = []
    for obj in ann.get("objects", []):
        if obj["classTitle"] != "Knot":
            continue
        bmp = obj["bitmap"]
        ox, oy = bmp["origin"]
        raw = zlib.decompress(base64.b64decode(bmp["data"]))
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr.ndim == 3 and arr.shape[2] == 4:
            patch = arr[..., 3] > 0
        else:
            patch = arr > 0
        ys, xs = np.nonzero(patch)
        out.append((float(xs.mean()) + ox, float(ys.mean()) + oy))
    return out


def run_propagation_holdout(holdouts: List[int], size: int = 512) -> Dict[int, np.ndarray]:
    """Re-run propagation on the full subset 4 volume WITHOUT the holdout anchors.
    Returns a dict {page -> prediction array} only for the holdout pages.
    """
    pages = list_pages()
    all_anchors = find_anchor_pages(pages)
    anchors = [a for a in all_anchors if a not in holdouts]
    page_to_idx = {p: i for i, p in enumerate(pages)}
    anchor_idx = [page_to_idx[p] for p in anchors]
    sample = load_tiff_gray(page_img_path(pages[0]))
    h, w = sample.shape
    print("frames: %d, kept anchors (%d): %s; holdouts: %s" % (len(pages), len(anchors), anchors, holdouts))

    vol = np.stack([load_tiff_gray(page_img_path(p)) for p in pages])
    anchor_masks, anchor_pith = {}, {}
    for p in anchors:
        m, pith = render_annotation(page_ann_path(p), h, w)
        anchor_masks[p] = m
        anchor_pith[p] = pith

    repo_root = os.path.abspath(join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty", "MedSAM2"))
    model_cfg = "//" + join(repo_root, "sam2/configs/sam2.1_hiera_t512.yaml")
    predictor = build_sam2_video_predictor_npz(model_cfg, CHECKPOINT)

    pred = np.zeros_like(vol, dtype=np.uint8)
    first_a = anchor_idx[0]
    if first_a > 0:
        sub = vol[: first_a + 1]
        pa = anchors[0]
        seg = propagate_segment(
            predictor,
            sub,
            seed_masks_a={c: np.zeros((h, w), np.uint8) for c in CLASS_IDS},
            seed_pith_a=[],
            seed_masks_b=anchor_masks[pa],
            seed_pith_b=anchor_pith[pa],
            size=size,
        )
        pred[: first_a + 1] = seg
    for s in range(len(anchor_idx) - 1):
        a, b = anchor_idx[s], anchor_idx[s + 1]
        sub = vol[a : b + 1]
        pa, pb = anchors[s], anchors[s + 1]
        seg = propagate_segment(
            predictor,
            sub,
            seed_masks_a=anchor_masks[pa],
            seed_pith_a=anchor_pith[pa],
            seed_masks_b=anchor_masks[pb],
            seed_pith_b=anchor_pith[pb],
            size=size,
        )
        pred[a : b + 1] = seg
        print("segment %d..%d done" % (pa, pb))
    last_a = anchor_idx[-1]
    if last_a < len(pages) - 1:
        sub = vol[last_a:]
        pa = anchors[-1]
        seg = propagate_segment(
            predictor,
            sub,
            seed_masks_a=anchor_masks[pa],
            seed_pith_a=anchor_pith[pa],
            seed_masks_b=None,
            seed_pith_b=None,
            size=size,
        )
        pred[last_a:] = seg

    return {p: pred[page_to_idx[p]] for p in holdouts}


def yolo_predict(
    model: YOLO, img_path: str, conf: float = 0.25
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Returns (knot_centroids, pith_centroids) — YOLO bbox centroids per class."""
    arr = np.array(Image.open(img_path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    res = model.predict(arr.astype(np.uint8), conf=conf, verbose=False)[0]
    knots, piths = [], []
    if res.boxes is None or len(res.boxes) == 0:
        return knots, piths
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    for b, c in zip(xyxy, cls):
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        if c == 0:
            knots.append((float(cx), float(cy)))
        elif c == 1:
            piths.append((float(cx), float(cy)))
    return knots, piths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    prop_pred = run_propagation_holdout(HOLDOUT_PAGES)

    model = YOLO(YOLO_WEIGHTS)
    rows = []
    sample = load_tiff_gray(page_img_path(HOLDOUT_PAGES[0]))
    h, w = sample.shape

    for p in HOLDOUT_PAGES:
        ann_path = page_ann_path(p)
        gt_masks, gt_pith_pts = render_annotation(ann_path, h, w)
        gt_knots = render_gt_knot_centroids(ann_path, h, w)

        yolo_knots, yolo_piths = yolo_predict(model, page_img_path(p), conf=args.conf)

        pred = prop_pred[p]
        prop_knots = knot_components_centroids(pred == CLASS_IDS["Knot"])
        ys, xs = np.nonzero(pred == CLASS_IDS["Pith"])
        prop_pith = (float(xs.mean()), float(ys.mean())) if len(xs) > 0 else None

        gt_pith = gt_pith_pts[0] if gt_pith_pts else None
        yolo_pith = yolo_piths[0] if yolo_piths else None

        def dist(a, b):
            if a is None or b is None:
                return float("nan")
            return float(np.hypot(a[0] - b[0], a[1] - b[1]))

        y_knots_matched, y_knots_fp, y_knots_fn = hungarian_match(yolo_knots, gt_knots)
        p_knots_matched, p_knots_fp, p_knots_fn = hungarian_match(prop_knots, gt_knots)

        rows.append(
            {
                "page": p,
                "n_gt_knots": len(gt_knots),
                "yolo_pith_dist_px": dist(yolo_pith, gt_pith),
                "prop_pith_dist_px": dist(prop_pith, gt_pith),
                "yolo_knots_pred": len(yolo_knots),
                "yolo_knots_matched": len(y_knots_matched),
                "yolo_knots_mean_dist": float(np.mean([d for *_, d in y_knots_matched]))
                if y_knots_matched
                else float("nan"),
                "yolo_knots_fp": len(y_knots_fp),
                "yolo_knots_fn": len(y_knots_fn),
                "prop_knots_pred": len(prop_knots),
                "prop_knots_matched": len(p_knots_matched),
                "prop_knots_mean_dist": float(np.mean([d for *_, d in p_knots_matched]))
                if p_knots_matched
                else float("nan"),
                "prop_knots_fp": len(p_knots_fp),
                "prop_knots_fn": len(p_knots_fn),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = join(OUT_DIR, "compare.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print("wrote", csv_path)

    summary = pd.DataFrame(
        {
            "method": ["yolo", "propagation"],
            "pith_mean_dist_px": [df["yolo_pith_dist_px"].mean(), df["prop_pith_dist_px"].mean()],
            "knot_recall": [
                df["yolo_knots_matched"].sum() / df["n_gt_knots"].sum(),
                df["prop_knots_matched"].sum() / df["n_gt_knots"].sum(),
            ],
            "knot_mean_dist_px": [df["yolo_knots_mean_dist"].mean(), df["prop_knots_mean_dist"].mean()],
            "knot_fp_total": [df["yolo_knots_fp"].sum(), df["prop_knots_fp"].sum()],
            "knot_fn_total": [df["yolo_knots_fn"].sum(), df["prop_knots_fn"].sum()],
        }
    )
    print("\nsummary across the 3 holdout pages:")
    print(summary.to_string(index=False))
    summary.to_csv(join(OUT_DIR, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
