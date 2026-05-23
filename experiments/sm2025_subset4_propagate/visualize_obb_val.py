"""Visualise YOLO-OBB predictions vs GT on the OBB val split.

For each val image render a panel: image + GT OBBs (lime) + predicted OBBs
(red dashed) + confidence labels. Saves a single montage.

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.visualize_obb_val
"""

import argparse
import pathlib
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_obb_v1/weights/best.pt"
DATA_DIR = "/home/mary/code/ct-log/ann_pipeline/out/knot_yolo_obb"
OUT_PATH = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/obb_val.png"


def parse_obb_labels(label_path: pathlib.Path, w: int, h: int) -> List[np.ndarray]:
    if not label_path.exists():
        return []
    out: List[np.ndarray] = []
    for line in label_path.read_text().strip().splitlines():
        if not line:
            continue
        parts = line.split()
        coords = np.array([float(x) for x in parts[1:]]).reshape(4, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        out.append(coords)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out_path", default=OUT_PATH)
    args = parser.parse_args()

    pathlib.Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    img_dir = pathlib.Path(args.data_dir) / "images" / "val"
    label_dir = pathlib.Path(args.data_dir) / "labels" / "val"
    image_paths = sorted(img_dir.glob("*.jpg"))

    model = YOLO(args.weights)

    n = len(image_paths)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, img_path in zip(axes, image_paths):
        img = np.array(Image.open(img_path))
        gray = img[..., 0] if img.ndim == 3 else img
        h, w = gray.shape[:2]

        result = model.predict(img_path, conf=args.conf, verbose=False)[0]
        pred_corners: List[np.ndarray] = []
        pred_confs: List[float] = []
        if result.obb is not None and len(result.obb) > 0:
            xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
            confs = result.obb.conf.cpu().numpy()
            for c, cf in zip(xyxyxyxy, confs):
                pred_corners.append(c)
                pred_confs.append(float(cf))
        gt_obbs = parse_obb_labels(label_dir / f"{img_path.stem}.txt", w, h)

        ax.imshow(gray, cmap="gray")
        for corners in gt_obbs:
            ax.add_patch(mpatches.Polygon(corners, fill=False, edgecolor="lime", linewidth=2))
        for corners, cf in zip(pred_corners, pred_confs):
            ax.add_patch(mpatches.Polygon(corners, fill=False, edgecolor="red", linewidth=1.6, linestyle="--"))
            cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
            ax.text(cx, cy, "%.2f" % cf, color="red", fontsize=8, weight="bold", ha="center", va="center")
        ax.set_title("%s  gt=%d pred=%d" % (img_path.name, len(gt_obbs), len(pred_corners)), fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("YOLO11n-OBB knot predictions on val split (lime=GT, red=pred, conf>=%.2f)" % args.conf, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(args.out_path, dpi=110)
    plt.close()
    print("wrote", args.out_path)


if __name__ == "__main__":
    main()
