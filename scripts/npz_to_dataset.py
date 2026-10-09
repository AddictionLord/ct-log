"""Convert anchor-free propagation .npz output into the on-disk dataset layout.

The segmentation dataset reads Supervisely-style annotations from
<out>/<log>/{img,ann}. Propagation emits only a knot prediction array, so wood
and pith are derived here with the same logic the Supervisely uploader uses,
keeping generated logs consistent with the existing Phase2 training logs.
"""

import argparse
import json
import pathlib
import shutil

from ann_pipeline.scripts.upload_ct_logs_to_phases import build_auto_annotation
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

DEFAULT_YOLO = "ann_pipeline/out/knot_runs/yolo11n_obb_2cls_holdout_v3/weights/best.pt"


def to_gray_rgb(frame: np.ndarray):
    """Split a stored frame into grayscale and RGB views.

    Args:
        frame: [H, W] or [H, W, C] uint8 array.

    Returns:
        tuple: (gray [H, W], rgb [H, W, 3]) uint8 arrays.
    """
    if frame.ndim == 3:
        rgb = frame[..., :3].astype(np.uint8)
        gray = rgb.mean(axis=-1).astype(np.uint8)
    else:
        gray = frame.astype(np.uint8)
        rgb = np.stack([gray] * 3, axis=-1)
    return gray, rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="Propagation output .npz")
    parser.add_argument("--log", type=str, required=True, help="Log id, used as output dir name")
    parser.add_argument("--src_img_dir", type=str, required=True, help="Raw page_NNN.tiff dir")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--yolo_weights", type=str, default=DEFAULT_YOLO)
    args = parser.parse_args()

    data = np.load(args.npz)
    pred, pages = data["pred"], data["pages"]
    imgs = data["imgs"] if "imgs" in data else None

    out_dir = pathlib.Path(args.out_root) / args.log
    img_dir, ann_dir = out_dir / "img", out_dir / "ann"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.yolo_weights)
    src_dir = pathlib.Path(args.src_img_dir)

    written = 0
    for index, page in enumerate(tqdm(pages, desc=f"log {args.log}")):
        name = f"page_{int(page):03d}.tiff"
        src = src_dir / name
        if not src.exists():
            continue

        if imgs is not None:
            gray, rgb = to_gray_rgb(imgs[index])
        else:
            from PIL import Image

            gray, rgb = to_gray_rgb(np.array(Image.open(src)))

        annotation = build_auto_annotation(gray, rgb, pred[index], yolo)
        with (ann_dir / f"{name}.json").open("w") as f:
            json.dump(annotation, f)

        target = img_dir / name
        if not target.exists():
            shutil.copy2(src, target)
        written += 1

    print(f"log {args.log}: wrote {written} frames to {out_dir}")


if __name__ == "__main__":
    main()
