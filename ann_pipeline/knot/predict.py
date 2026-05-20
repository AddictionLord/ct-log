"""Run the trained YOLO knot detector on validation slices and produce
a montage showing predictions (red) vs GT (lime green) bboxes.
"""

import argparse
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO


def parse_yolo_label(label_path: pathlib.Path, w: int, h: int) -> list:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        if not line:
            continue
        _, cx, cy, bw, bh = [float(x) for x in line.split()]
        x_min = int((cx - bw / 2) * w)
        y_min = int((cy - bh / 2) * h)
        x_max = int((cx + bw / 2) * w)
        y_max = int((cy + bh / 2) * h)
        boxes.append((x_min, y_min, x_max, y_max))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_knots_v1/weights/best.pt",
    )
    parser.add_argument("--data_dir", default="/home/mary/code/ct-log/ann_pipeline/out/knot_yolo")
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/knot_predictions")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = pathlib.Path(args.data_dir) / "images" / args.split
    label_dir = pathlib.Path(args.data_dir) / "labels" / args.split

    model = YOLO(args.weights)

    image_paths = sorted(img_dir.glob("*.jpg"))
    n = len(image_paths)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, img_path in zip(axes, image_paths):
        img = np.array(Image.open(img_path))
        if img.ndim == 3:
            img_gray = img[..., 0]
        else:
            img_gray = img
        h, w = img_gray.shape[:2]

        result = model.predict(img_path, conf=args.conf, verbose=False)[0]
        pred_boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        pred_confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

        gt_boxes = parse_yolo_label(label_dir / f"{img_path.stem}.txt", w, h)

        ax.imshow(img_gray, cmap="gray")
        for x_min, y_min, x_max, y_max in gt_boxes:
            ax.add_patch(
                mpatches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                    label="GT",
                )
            )
        for (x_min, y_min, x_max, y_max), c in zip(pred_boxes, pred_confs):
            ax.add_patch(
                mpatches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                    linestyle="--",
                )
            )
            ax.text(x_min, y_min - 2, f"{c:.2f}", color="red", fontsize=9, weight="bold")
        ax.set_title(f"{img_path.name}  gt={len(gt_boxes)} pred={len(pred_boxes)}", fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"YOLO11n knot predictions ({args.split} split, conf>={args.conf})", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.split}_predictions.png", dpi=100)
    plt.close()
    print(f"wrote {out_dir / f'{args.split}_predictions.png'}")


if __name__ == "__main__":
    main()
