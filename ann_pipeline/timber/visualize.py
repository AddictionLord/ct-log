"""Render timber GT and predictions as plotly figures."""

import argparse
import pathlib
from os.path import join
from typing import List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ann_pipeline.timber.data import (
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SUBSET,
    find_annotated_slices,
    load_slice,
)
from ann_pipeline.timber.detectors import threshold_components_split

PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
    ],
    dtype=np.uint8,
)


def colorize(gray: np.ndarray, masks: List[np.ndarray], alpha: float = 0.55) -> np.ndarray:
    rgb = np.stack([gray] * 3, axis=-1).astype(np.float32)
    for idx, mask in enumerate(masks):
        color = PALETTE[idx % len(PALETTE)]
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
    return rgb.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--frames", nargs="+", default=None, help="image names, e.g. 033.png")
    parser.add_argument(
        "--out_html",
        default="/home/mary/code/ct-log/ann_pipeline/out/timber_eval",
        help="directory for self-contained HTML figures; empty string opens a browser instead",
    )
    args = parser.parse_args()

    subset_dir = join(args.project_root, args.subset)
    available = find_annotated_slices(subset_dir)
    if args.frames:
        wanted = {f if f.endswith(".json") else f + ".json" for f in args.frames}
        targets = [f for f in available if f in wanted]
    else:
        targets = available[:3]
    if not targets:
        msg = "no matching annotated frames"
        raise ValueError(msg)

    out_dir = pathlib.Path(args.out_html) if args.out_html else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for ann_fname in targets:
        gray, instances = load_slice(subset_dir, ann_fname)
        gt_masks = [instances[k] for k in sorted(instances)]
        pred_masks = threshold_components_split(gray)
        name = ann_fname[: -len(".json")]
        figures = [
            ("source", px.imshow(gray, color_continuous_scale="gray", title=f"{name} - source")),
            ("gt", px.imshow(colorize(gray, gt_masks), title=f"{name} - GT ({len(gt_masks)} instances)")),
            ("pred", px.imshow(colorize(gray, pred_masks), title=f"{name} - pred ({len(pred_masks)} instances)")),
        ]
        for kind, fig in figures:
            if out_dir is None:
                fig.show()
                continue
            path = out_dir / f"{name.replace('.png', '')}_{kind}.html"
            fig.write_html(str(path), include_plotlyjs="cdn")
            print(path)


if __name__ == "__main__":
    main()
