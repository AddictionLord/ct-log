"""Convert Supervisely Knot annotations to YOLO OBB format (single-class).

Walks one or more sub-datasets under a Supervisely project root and emits
oriented bounding box labels for the Knot class. Pith is not exported by this
script — the existing axis-aligned model in `data_prep.py` handles pith.

YOLO OBB label format (one line per instance):
    cls x1 y1 x2 y2 x3 y3 x4 y4
all coords normalized to [0, 1] over image width/height.

Output layout:
    out_dir/
      images/train/ds{N}_page_NNN.jpg
      images/val/ds{N}_page_NNN.jpg
      labels/train/ds{N}_page_NNN.txt
      labels/val/ds{N}_page_NNN.txt
      knots_obb.yaml

Run:
    conda run -n ct-log python -m ann_pipeline.knot.data_prep_obb \\
        --subsets 1 2 4 \\
        --out_dir /home/mary/code/ct-log/ann_pipeline/out/knot_yolo_obb
"""

import argparse
import json
import os
from os.path import join
import pathlib
import random
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage import measure

from ann_pipeline.knot.data_prep import (
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SUBSETS,
    find_annotated_slices,
    knot_mask_from_ann,
    load_tiff_gray,
)


def obb_corners_from_mask(mask: np.ndarray, min_px: int = 8) -> List[np.ndarray]:
    """Per connected component, fit min-area rect; return 4×2 corner arrays."""
    labels = measure.label(mask, connectivity=2)
    out: List[np.ndarray] = []
    for region in measure.regionprops(labels):
        if region.area < min_px:
            continue
        coords = region.coords[:, ::-1].astype(np.float32)
        rect = cv2.minAreaRect(coords)
        corners = cv2.boxPoints(rect)
        out.append(corners.astype(np.float32))
    return out


def corners_to_yolo_obb_line(corners: np.ndarray, img_w: int, img_h: int, cls: int = 0) -> str:
    norm = corners.copy()
    norm[:, 0] /= img_w
    norm[:, 1] /= img_h
    norm = np.clip(norm, 0.0, 1.0)
    flat = norm.reshape(-1)
    return "%d " % cls + " ".join("%.6f" % v for v in flat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/knot_yolo_obb")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--exclude_pages",
        type=str,
        default="",
        help="Comma-separated 'subset:page' pairs to drop (e.g. '4:7,4:49').",
    )
    args = parser.parse_args()

    excluded = set()
    if args.exclude_pages:
        for tok in args.exclude_pages.split(","):
            sub, pg = tok.strip().split(":")
            excluded.add((sub.strip(), int(pg)))

    out_dir = pathlib.Path(args.out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    train_items: List[Tuple[str, str]] = []
    val_items: List[Tuple[str, str]] = []
    for subset_id in args.subsets:
        sm_dir = join(args.project_root, subset_id)
        if not os.path.isdir(join(sm_dir, "ann")):
            print(f"  skipping subset {subset_id}: no ann/")
            continue
        ann_files = find_annotated_slices(sm_dir)
        if excluded:
            kept = []
            for f in ann_files:
                page_num = int(f.replace("page_", "").split(".")[0])
                if (subset_id, page_num) not in excluded:
                    kept.append(f)
            dropped = len(ann_files) - len(kept)
            if dropped:
                print(f"  subset {subset_id}: excluded {dropped} pages")
            ann_files = kept
        rng.shuffle(ann_files)
        n_val = max(1, int(len(ann_files) * args.val_frac)) if ann_files else 0
        for f in ann_files[:n_val]:
            val_items.append((subset_id, f))
        for f in ann_files[n_val:]:
            train_items.append((subset_id, f))
        print(f"  subset {subset_id}: {len(ann_files)} annotated slices ({n_val} val / {len(ann_files) - n_val} train)")

    print(f"\ntotal: {len(train_items)} train + {len(val_items)} val slices")

    total_obb = 0
    n_with_knots = 0
    for split, items in [("train", train_items), ("val", val_items)]:
        for subset_id, ann_fname in items:
            sm_dir = join(args.project_root, subset_id)
            with open(join(sm_dir, "ann", ann_fname)) as fh:
                ann = json.load(fh)
            img_fname = ann_fname[: -len(".json")]
            img = load_tiff_gray(join(sm_dir, "img", img_fname))
            h, w = img.shape

            obbs = obb_corners_from_mask(knot_mask_from_ann(ann))
            total_obb += len(obbs)
            if obbs:
                n_with_knots += 1

            base_stem = img_fname.replace(".tiff", "")
            stem = f"ds{subset_id}_{base_stem}"
            Image.fromarray(img).convert("RGB").save(out_dir / "images" / split / f"{stem}.jpg", quality=95)
            lines = [corners_to_yolo_obb_line(c, w, h, cls=0) for c in obbs]
            (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    yaml_path = out_dir / "knots_obb.yaml"
    yaml_path.write_text(f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [knot]\n")
    print(f"\nwrote {total_obb} OBBs across {n_with_knots}/{len(train_items) + len(val_items)} slices")
    print(f"YOLO-OBB dataset at {out_dir}")
    print(f"config: {yaml_path}")


if __name__ == "__main__":
    main()
