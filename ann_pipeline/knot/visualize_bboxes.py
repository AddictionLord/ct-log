"""Render every knot-annotated slice with the bboxes derived from connected
components overlaid. Used to sanity-check the Supervisely -> YOLO conversion
before training. Output: one PNG per slice plus a grid montage.
"""

import argparse
import json
from os.path import join
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ann_pipeline.knot.data_prep import (
    DEFAULT_SM_DIR,
    bboxes_from_mask,
    find_knot_slices,
    knot_mask_from_ann,
    load_tiff_gray,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm_dir", default=DEFAULT_SM_DIR)
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/knot_bboxes")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_slice").mkdir(exist_ok=True)

    ann_files = find_knot_slices(args.sm_dir)
    n = len(ann_files)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, ann_fname in zip(axes, ann_files):
        with open(join(args.sm_dir, "ann", ann_fname)) as fh:
            ann = json.load(fh)
        img_fname = ann_fname[: -len(".json")]
        img = load_tiff_gray(join(args.sm_dir, "img", img_fname))
        mask = knot_mask_from_ann(ann)
        boxes = bboxes_from_mask(mask)

        ax.imshow(img, cmap="gray")
        ax.imshow(np.ma.masked_where(~mask.astype(bool), mask), alpha=0.3, cmap="autumn")
        for x_min, y_min, x_max, y_max in boxes:
            ax.add_patch(
                mpatches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.5,
                )
            )
        page = int(img_fname.replace("page_", "").replace(".tiff", ""))
        ax.set_title(f"p{page} k={len(boxes)}", fontsize=9)
        ax.axis("off")

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.imshow(img, cmap="gray")
        ax2.imshow(np.ma.masked_where(~mask.astype(bool), mask), alpha=0.3, cmap="autumn")
        for x_min, y_min, x_max, y_max in boxes:
            ax2.add_patch(
                mpatches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
            )
        ax2.set_title(f"page_{page:03d}  ({len(boxes)} knot bboxes)")
        ax2.axis("off")
        plt.tight_layout()
        fig2.savefig(out_dir / "per_slice" / f"page_{page:03d}.png", dpi=100)
        plt.close(fig2)

    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Knot bboxes derived from Supervisely masks ({n} slices)")
    plt.tight_layout()
    plt.savefig(out_dir / "montage.png", dpi=100)
    plt.close()
    print(f"wrote per-slice + montage to {out_dir}")


if __name__ == "__main__":
    main()
