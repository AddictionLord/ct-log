"""Visualize YOLO knot detections at high confidence alongside the propagated
knot masks for subset 4.

For each chosen page, render a row of side-by-side panels:
    [grayscale]  [propagation only]  [YOLO conf=0.25]  [YOLO conf=0.40]  [YOLO conf=0.55]

  * propagation: filled cyan masks (per connected component)
  * YOLO: red bounding boxes + confidence labels overlaid on the prop mask

The idea is to see at a glance which knots YOLO catches that propagation misses
(and vice versa), and how the conf threshold trades precision vs recall.

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.visualize_yolo_high_conf
"""

import argparse
from os.path import join
import pathlib
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

from experiments.sm2025_subset4_propagate.run import CLASS_IDS

YOLO_WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_2cls_v1/weights/best.pt"
SOURCE_IMG_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025/4/img"
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/yolo_vs_prop"

CONF_LEVELS = [0.25, 0.40, 0.55]


def load_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def yolo_knots_at_conf(model: YOLO, img_rgb: np.ndarray, conf: float) -> List[Tuple[float, float, float, float, float]]:
    """Returns [(x1, y1, x2, y2, conf), ...] for knot detections."""
    res = model.predict(img_rgb, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    out = []
    for b, c, cf in zip(xyxy, cls, confs):
        if c == 0:
            out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(cf)))
    return out


def overlay_prop_mask(ax, gray: np.ndarray, prop_knot_mask: np.ndarray) -> None:
    ax.imshow(gray, cmap="gray")
    overlay = np.zeros((*gray.shape, 4), dtype=np.float32)
    overlay[prop_knot_mask] = (0.0, 0.8, 1.0, 0.45)
    ax.imshow(overlay)


def draw_yolo_boxes(ax, boxes: List[Tuple[float, float, float, float, float]]) -> None:
    for x1, y1, x2, y2, cf in boxes:
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1.5))
        ax.text(x1, y1 - 4, "%.2f" % cf, color="red", fontsize=7, weight="bold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/result.npz")
    parser.add_argument("--n_vis", type=int, default=16)
    parser.add_argument(
        "--pages",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit list of pages to render (overrides --n_vis spread).",
    )
    args = parser.parse_args()

    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz, allow_pickle=False)
    pages = data["pages"]
    pred = data["pred"]
    is_anchor = data["is_anchor"]
    anchors = data["anchors"]

    model = YOLO(YOLO_WEIGHTS)

    if args.pages is not None:
        chosen = []
        for p in args.pages:
            idx = int(np.where(pages == p)[0][0])
            chosen.append(idx)
    else:
        non_anchor = [i for i in range(len(pages)) if not is_anchor[i]]
        step = max(1, len(non_anchor) // args.n_vis)
        anchor_idx = [int(i) for i in range(len(pages)) if is_anchor[i]]
        chosen = sorted(set(non_anchor[::step][: args.n_vis]) | set(anchor_idx))

    print("rendering %d frames; anchors=%s" % (len(chosen), list(anchors)))

    for i in chosen:
        p = int(pages[i])
        img_path = join(SOURCE_IMG_DIR, "page_%03d.tiff" % p)
        rgb = load_rgb(img_path)
        gray = rgb.mean(axis=-1).astype(np.uint8)
        prop_knot = pred[i] == CLASS_IDS["Knot"]

        boxes_by_conf = {c: yolo_knots_at_conf(model, rgb, c) for c in CONF_LEVELS}

        n_cols = 2 + len(CONF_LEVELS)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.2))
        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title("page_%03d %s" % (p, "(ANCHOR)" if is_anchor[i] else ""))
        axes[0].axis("off")

        overlay_prop_mask(axes[1], gray, prop_knot)
        axes[1].set_title("propagated knots (cyan)")
        axes[1].axis("off")

        for ax, c in zip(axes[2:], CONF_LEVELS):
            overlay_prop_mask(ax, gray, prop_knot)
            draw_yolo_boxes(ax, boxes_by_conf[c])
            ax.set_title("YOLO conf>=%.2f (n=%d) + prop" % (c, len(boxes_by_conf[c])))
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(join(OUT_DIR, "page_%03d.png" % p), dpi=110)
        plt.close()

    print("wrote %d images to %s" % (len(chosen), OUT_DIR))


if __name__ == "__main__":
    main()
