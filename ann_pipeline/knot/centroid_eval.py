"""Centroid-distance evaluation for the 2-class YOLO detector.

For each val image, compares predicted bbox centroids against GT bbox
centroids per class via Hungarian matching. Reports per-class:
  - mean matched centroid distance
  - count of GT instances unmatched (false negatives / missed)
  - count of predictions unmatched (false positives / extra)

Run from repo root:
    conda run -n ct-log python -m ann_pipeline.knot.centroid_eval
"""

import argparse
import pathlib
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

CLASS_NAMES = {0: "knot", 1: "pith"}


def parse_yolo_label(label_path: pathlib.Path, w: int, h: int) -> List[Tuple[int, float, float, float, float]]:
    """Returns [(cls, x_min, y_min, x_max, y_max), ...] in pixel coords."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().strip().splitlines():
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        cx, cy, bw, bh = [float(x) for x in parts[1:]]
        x_min = (cx - bw / 2) * w
        y_min = (cy - bh / 2) * h
        x_max = (cx + bw / 2) * w
        y_max = (cy + bh / 2) * h
        out.append((cls, x_min, y_min, x_max, y_max))
    return out


def bbox_centroid(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def match_centroids(
    pred_cs: List[Tuple[float, float]], gt_cs: List[Tuple[float, float]]
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Hungarian-match predictions to GTs by Euclidean distance.
    Returns (matched_pairs[(pred_idx, gt_idx, dist)], unmatched_pred_idx, unmatched_gt_idx).
    """
    if not pred_cs and not gt_cs:
        return [], [], []
    if not pred_cs:
        return [], [], list(range(len(gt_cs)))
    if not gt_cs:
        return [], list(range(len(pred_cs))), []
    cost = np.zeros((len(pred_cs), len(gt_cs)))
    for i, p in enumerate(pred_cs):
        for j, g in enumerate(gt_cs):
            cost[i, j] = np.hypot(p[0] - g[0], p[1] - g[1])
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = [(int(r), int(c), float(cost[r, c])) for r, c in zip(row_ind, col_ind)]
    matched_preds = {m[0] for m in matched}
    matched_gts = {m[1] for m in matched}
    unmatched_pred = [i for i in range(len(pred_cs)) if i not in matched_preds]
    unmatched_gt = [j for j in range(len(gt_cs)) if j not in matched_gts]
    return matched, unmatched_pred, unmatched_gt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_2cls_v1/weights/best.pt",
    )
    parser.add_argument("--data_dir", default="/home/mary/code/ct-log/ann_pipeline/out/knot_yolo")
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/knot_centroid_eval")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_slice").mkdir(exist_ok=True)

    img_dir = pathlib.Path(args.data_dir) / "images" / args.split
    label_dir = pathlib.Path(args.data_dir) / "labels" / args.split

    model = YOLO(args.weights)
    image_paths = sorted(img_dir.glob("*.jpg"))

    rows = []
    per_class_dists = {0: [], 1: []}
    per_class_missed = {0: 0, 1: 0}
    per_class_extra = {0: 0, 1: 0}
    per_class_gt_total = {0: 0, 1: 0}

    n = len(image_paths)
    cols = min(4, n)
    gridrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(gridrows, cols, figsize=(5 * cols, 5 * gridrows))
    axes = np.array(axes).reshape(-1)

    for ax, img_path in zip(axes, image_paths):
        img = np.array(Image.open(img_path))
        img_gray = img[..., 0] if img.ndim == 3 else img
        h, w = img_gray.shape[:2]

        result = model.predict(img_path, conf=args.conf, verbose=False)[0]
        pred_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        pred_cls = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.empty((0,), int)

        gt_items = parse_yolo_label(label_dir / f"{img_path.stem}.txt", w, h)

        ax.imshow(img_gray, cmap="gray")
        for cls in (0, 1):
            gt_boxes = [b[1:] for b in gt_items if b[0] == cls]
            pred_mask = pred_cls == cls
            pred_boxes = pred_xyxy[pred_mask]
            gt_cs = [bbox_centroid(b) for b in gt_boxes]
            pred_cs = [bbox_centroid(b) for b in pred_boxes]
            matched, un_pred, un_gt = match_centroids(pred_cs, gt_cs)
            per_class_gt_total[cls] += len(gt_cs)
            for _, _, d in matched:
                per_class_dists[cls].append(d)
            per_class_missed[cls] += len(un_gt)
            per_class_extra[cls] += len(un_pred)
            rows.append(
                {
                    "image": img_path.name,
                    "class": CLASS_NAMES[cls],
                    "n_gt": len(gt_cs),
                    "n_pred": len(pred_cs),
                    "n_matched": len(matched),
                    "mean_dist": float(np.mean([d for _, _, d in matched])) if matched else None,
                    "n_missed": len(un_gt),
                    "n_extra": len(un_pred),
                }
            )
            color_gt = "lime" if cls == 0 else "cyan"
            color_pred = "red" if cls == 0 else "orange"
            for x_min, y_min, x_max, y_max in gt_boxes:
                ax.add_patch(
                    mpatches.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        fill=False,
                        edgecolor=color_gt,
                        linewidth=1.5,
                    )
                )
            for x_min, y_min, x_max, y_max in pred_boxes:
                ax.add_patch(
                    mpatches.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        fill=False,
                        edgecolor=color_pred,
                        linewidth=1.5,
                        linestyle="--",
                    )
                )
            for gx, gy in gt_cs:
                ax.plot(gx, gy, "o", mfc="none", mec=color_gt, ms=10, mew=2)
            for px, py in pred_cs:
                ax.plot(px, py, "x", color=color_pred, ms=10, mew=2)
        ax.set_title(img_path.stem, fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        "2-class YOLO predictions.  "
        "GT: solid rect + circle (knot=lime, pith=cyan).  "
        "Pred: dashed rect + X (knot=red, pith=orange).",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.split}_centroids.png", dpi=100)
    plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{args.split}_per_slice.csv", index=False)

    print("\n=== centroid eval summary ===")
    for cls in (0, 1):
        d = per_class_dists[cls]
        name = CLASS_NAMES[cls]
        n_gt = per_class_gt_total[cls]
        n_matched = len(d)
        recall = n_matched / n_gt if n_gt else float("nan")
        print(
            f"  {name}: gt={n_gt}, matched={n_matched} ({recall * 100:.1f}% recall), "
            f"missed={per_class_missed[cls]}, extra={per_class_extra[cls]}, "
            f"mean_dist={np.mean(d):.2f}px, median={np.median(d) if d else float('nan'):.2f}px, "
            f"p90={np.percentile(d, 90) if d else float('nan'):.2f}px"
            if d
            else f"  {name}: gt={n_gt}, matched=0"
        )

    print(f"\nvis: {out_dir / f'{args.split}_centroids.png'}")
    print(f"csv: {out_dir / f'{args.split}_per_slice.csv'}")


if __name__ == "__main__":
    main()
