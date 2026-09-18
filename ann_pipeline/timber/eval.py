"""Evaluate timber instance detectors against the annotated frames.

Metrics:
  - per-instance IoU over Hungarian-matched (pred, gt) pairs
  - precision / recall / F1 at an IoU threshold
  - count accuracy: fraction of frames recovering exactly 8 instances
  - semantic IoU / Dice on the union, separating boundary from splitting quality
"""

import argparse
from os.path import join
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ann_pipeline.timber.data import (
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SUBSET,
    find_annotated_slices,
    load_slice,
)
from ann_pipeline.timber.detectors import (
    threshold_components,
    threshold_components_split,
    threshold_components_watershed,
    threshold_components_full,
    threshold_components_v2,
)

DETECTORS = {
    "threshold_components": threshold_components,
    "threshold_components_split": threshold_components_split,
    "threshold_components_watershed": threshold_components_watershed,
    "threshold_components_full": threshold_components_full,
    "threshold_components_v2": threshold_components_v2,
}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def match_instances(
    preds: List[np.ndarray], gts: List[np.ndarray], iou_thr: float
) -> Tuple[List[Tuple[int, int, float]], int, int]:
    if not preds or not gts:
        return [], len(preds), len(gts)
    iou_mat = np.zeros((len(preds), len(gts)), dtype=np.float64)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_mat[i, j] = iou(p, g)
    rows, cols = linear_sum_assignment(1.0 - iou_mat)
    matched = [(int(r), int(c), float(iou_mat[r, c])) for r, c in zip(rows, cols) if iou_mat[r, c] >= iou_thr]
    return matched, len(preds) - len(matched), len(gts) - len(matched)


def evaluate(subset_dir: str, iou_thr: float) -> pd.DataFrame:
    rows = []
    for ann_fname in find_annotated_slices(subset_dir):
        gray, instances = load_slice(subset_dir, ann_fname)
        gts = list(instances.values())
        gt_union = np.zeros(gray.shape, dtype=bool)
        for g in gts:
            gt_union |= g
        for name, detector in DETECTORS.items():
            preds = detector(gray)
            matched, n_fp, n_fn = match_instances(preds, gts, iou_thr)
            pred_union = np.zeros(gray.shape, dtype=bool)
            for p in preds:
                pred_union |= p
            ious = [m[2] for m in matched]
            rows.append(
                {
                    "frame": ann_fname[: -len(".json")],
                    "detector": name,
                    "n_pred": len(preds),
                    "n_gt": len(gts),
                    "tp": len(matched),
                    "fp": n_fp,
                    "fn": n_fn,
                    "mean_inst_iou": float(np.mean(ious)) if ious else 0.0,
                    "min_inst_iou": float(np.min(ious)) if ious else 0.0,
                    "semantic_iou": iou(pred_union, gt_union),
                    "semantic_dice": dice(pred_union, gt_union),
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, sub in df.groupby("detector"):
        tp, fp, fn = sub["tp"].sum(), sub["fp"].sum(), sub["fn"].sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append(
            {
                "detector": name,
                "frames": len(sub),
                "exact_8": float((sub["n_pred"] == 8).mean()),
                "mean_inst_iou": sub["mean_inst_iou"].mean(),
                "min_inst_iou": sub["min_inst_iou"].min(),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "semantic_iou": sub["semantic_iou"].mean(),
            }
        )
    return pd.DataFrame(out).sort_values("f1", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/timber_eval")
    args = parser.parse_args()

    subset_dir = join(args.project_root, args.subset)
    df = evaluate(subset_dir, args.iou_thr)
    summary = summarize(df)

    import pathlib

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_frame.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print("=== summary (IoU thr %.2f) ===" % args.iou_thr)
    print(summary.to_string(index=False))
    print("\nworst frames by mean instance IoU:")
    print(df.sort_values("mean_inst_iou").head(8).to_string(index=False))


if __name__ == "__main__":
    main()
