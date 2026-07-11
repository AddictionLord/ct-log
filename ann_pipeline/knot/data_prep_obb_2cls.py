"""Convert Supervisely annotations to YOLO OBB format with two classes: knot + pith.

Extends the single-class OBB prep (data_prep_obb.py) so a single OBB model can
detect both knots (oriented boxes fit to the knot mask) and pith (a small
axis-aligned square emitted as an OBB quad around each pith point). This lets one
model replace the previous split of OBB-for-knots + axis-aligned-for-pith.

Class ids:  0 = knot, 1 = pith

Output layout mirrors data_prep_obb.py (images/{train,val}, labels/{train,val},
knots_obb_2cls.yaml).

Run:
    conda run -n ct-log python -m ann_pipeline.knot.data_prep_obb_2cls \\
        --subsets 1 2 3 4 08 10 --val_subsets 2 10 \\
        --out_dir ann_pipeline/out/knot_yolo_obb_2cls_v3
"""

import argparse
import json
import os
from os.path import join
import pathlib
import random
from typing import List, Tuple

import numpy as np
from PIL import Image

from ann_pipeline.knot.data_prep import (
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SUBSETS,
    find_annotated_slices,
    knot_mask_from_ann,
    load_tiff_gray,
    pith_bboxes_from_ann,
)
from ann_pipeline.knot.data_prep_obb import corners_to_yolo_obb_line, obb_corners_from_mask


def bbox_to_corners(box: Tuple[int, int, int, int]) -> np.ndarray:
    """Axis-aligned (x_min, y_min, x_max, y_max) -> 4x2 clockwise corners."""
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/knot_yolo_obb_2cls")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--val_subsets",
        nargs="+",
        default=None,
        help="Log-level holdout: subset ids whose every slice goes to val; all others go entirely to train.",
    )
    parser.add_argument("--pith_bbox_half", type=int, default=8)
    args = parser.parse_args()

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
        if args.val_subsets is not None:
            target = val_items if subset_id in args.val_subsets else train_items
            for f in ann_files:
                target.append((subset_id, f))
            tag = "all val, holdout" if subset_id in args.val_subsets else "all train"
            print(f"  subset {subset_id}: {len(ann_files)} annotated slices ({tag})")
            continue
        rng.shuffle(ann_files)
        n_val = max(1, int(len(ann_files) * args.val_frac)) if ann_files else 0
        for f in ann_files[:n_val]:
            val_items.append((subset_id, f))
        for f in ann_files[n_val:]:
            train_items.append((subset_id, f))
        print(f"  subset {subset_id}: {len(ann_files)} annotated slices ({n_val} val / {len(ann_files) - n_val} train)")

    print(f"\ntotal: {len(train_items)} train + {len(val_items)} val slices")

    total_knot = 0
    total_pith = 0
    for split, items in [("train", train_items), ("val", val_items)]:
        for subset_id, ann_fname in items:
            sm_dir = join(args.project_root, subset_id)
            with open(join(sm_dir, "ann", ann_fname)) as fh:
                ann = json.load(fh)
            img_fname = ann_fname[: -len(".json")]
            img = load_tiff_gray(join(sm_dir, "img", img_fname))
            h, w = img.shape

            knot_obbs = obb_corners_from_mask(knot_mask_from_ann(ann))
            pith_boxes = pith_bboxes_from_ann(ann, half_size=args.pith_bbox_half)
            total_knot += len(knot_obbs)
            total_pith += len(pith_boxes)

            base_stem = img_fname.replace(".tiff", "")
            stem = f"ds{subset_id}_{base_stem}"
            Image.fromarray(img).convert("RGB").save(out_dir / "images" / split / f"{stem}.jpg", quality=95)
            lines = [corners_to_yolo_obb_line(c, w, h, cls=0) for c in knot_obbs]
            lines += [corners_to_yolo_obb_line(bbox_to_corners(b), w, h, cls=1) for b in pith_boxes]
            (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    yaml_path = out_dir / "knots_obb_2cls.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnc: 2\nnames: [knot, pith]\n"
    )
    print(f"\nwrote {total_knot} knot OBBs + {total_pith} pith OBBs across {len(train_items) + len(val_items)} slices")
    print(f"config: {yaml_path}")


if __name__ == "__main__":
    main()
