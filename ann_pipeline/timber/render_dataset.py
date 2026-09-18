"""Render every frame in the Timber dataset to PNG for visual inspection.

Annotated frames get GT fill + prediction outlines; unannotated frames get
prediction outlines only. Filenames are prefixed so problems sort to the top.
"""

import argparse
import json
import os
from os.path import join
import pathlib
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ann_pipeline.timber.data import (
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SUBSET,
    load_gray,
    timber_instances_from_ann,
)
from ann_pipeline.timber.detectors import threshold_components_v2
from ann_pipeline.timber.eval import iou, match_instances

GT_COLOR = (255, 140, 0)
PRED_COLOR = (0, 255, 255)
BAD_COLOR = (0, 0, 255)


def render_frame(
    gray: np.ndarray,
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    title: str,
    flagged: bool,
) -> np.ndarray:
    rgb = np.stack([gray] * 3, axis=-1).astype(np.float32)
    for mask in gt_masks:
        rgb[mask] = 0.6 * rgb[mask] + 0.4 * np.array(GT_COLOR, dtype=np.float32)
    vis = rgb.astype(np.uint8).copy()
    for mask in pred_masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, PRED_COLOR, 2)
    color = BAD_COLOR if flagged else (255, 255, 255)
    cv2.putText(vis, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/timber_vis")
    parser.add_argument("--expected", type=int, default=8)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    args = parser.parse_args()

    subset_dir = join(args.project_root, args.subset)
    img_dir = join(subset_dir, "img")
    ann_dir = join(subset_dir, "ann")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_fname in tqdm(sorted(os.listdir(img_dir)), desc="rendering"):
        gray = load_gray(join(img_dir, img_fname))
        ann_path = join(ann_dir, img_fname + ".json")
        gt_masks = []
        if os.path.exists(ann_path):
            with open(ann_path) as fh:
                ann = json.load(fh)
            gt_masks = [ann["objects"] and m for m in timber_instances_from_ann(ann).values()]
        pred_masks = threshold_components_v2(gray)

        annotated = len(gt_masks) > 0
        mean_iou = float("nan")
        if annotated:
            matched, _, _ = match_instances(pred_masks, gt_masks, args.iou_thr)
            if matched:
                mean_iou = float(np.mean([m[2] for m in matched]))
        flagged = len(pred_masks) != args.expected

        stem = img_fname.rsplit(".", 1)[0]
        tag = "GT" if annotated else "noGT"
        title = f"{stem}  {tag}  n_pred={len(pred_masks)}"
        if annotated:
            title += f"  meanIoU={mean_iou:.3f}"
        prefix = "FLAG_" if flagged else ""
        vis = render_frame(gray, gt_masks, pred_masks, title, flagged)
        cv2.imwrite(str(out_dir / f"{prefix}{stem}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        rows.append(
            {
                "frame": img_fname,
                "annotated": annotated,
                "n_pred": len(pred_masks),
                "n_gt": len(gt_masks),
                "mean_inst_iou": mean_iou,
                "flagged": flagged,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "index.csv", index=False)
    print(f"\nwrote {len(df)} PNGs to {out_dir}")
    print(f"annotated: {int(df.annotated.sum())}   unannotated: {int((~df.annotated).sum())}")
    print(f"flagged (n_pred != {args.expected}): {int(df.flagged.sum())}")
    counts = df.n_pred.value_counts().sort_index()
    print("\nn_pred distribution:")
    print(counts.to_string())
    if df.flagged.any():
        print("\nflagged frames:")
        print(df[df.flagged][["frame", "annotated", "n_pred", "mean_inst_iou"]].to_string(index=False))


if __name__ == "__main__":
    main()
