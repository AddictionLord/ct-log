"""Convert Supervisely Knot annotations to YOLO bbox format.

Walks one or more sub-datasets under a Supervisely project root (e.g.
375492_SM_2025/{1,2,3,4}) and merges them into a single YOLO dataset.
Filenames are prefixed with the sub-dataset ID (e.g. ds1_page_014.jpg) to
avoid collisions across sub-datasets that share page numbers.

Pipeline per slice:
  1. decode every Knot bitmap (zlib -> PNG -> alpha channel)
  2. composite onto a full-canvas binary mask at the bitmap's `origin`
  3. extract one bbox per connected component
  4. write the slice as JPEG + a YOLO .txt label file

Output layout:
    out_dir/
      images/train/ds{N}_page_NNN.jpg
      images/val/ds{N}_page_NNN.jpg
      labels/train/ds{N}_page_NNN.txt
      labels/val/ds{N}_page_NNN.txt
      knots.yaml
"""

import argparse
import base64
import json
import os
from os.path import join
import pathlib
import random
from typing import List, Tuple
import zlib

import cv2
import numpy as np
from PIL import Image
from skimage import measure

DEFAULT_PROJECT_ROOT = "/mnt/D/datasets/ct_log/375492_SM_2025"
DEFAULT_SUBSETS = ["1", "2", "4"]


def decode_bitmap(b64: str) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(b64))
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., 3] > 0
    return arr > 0


def load_tiff_gray(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    return arr.astype(np.uint8)


def knot_mask_from_ann(ann: dict) -> np.ndarray:
    h, w = ann["size"]["height"], ann["size"]["width"]
    mask = np.zeros((h, w), dtype=np.uint8)
    for obj in ann.get("objects", []):
        if obj.get("classTitle") != "Knot":
            continue
        bmp = obj["bitmap"]
        ox, oy = bmp["origin"]
        patch = decode_bitmap(bmp["data"])
        ph, pw = patch.shape
        mask[oy : oy + ph, ox : ox + pw][patch] = 1
    return mask


def bboxes_from_mask(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    labels = measure.label(mask, connectivity=2)
    boxes = []
    for region in measure.regionprops(labels):
        y_min, x_min, y_max, x_max = region.bbox
        boxes.append((x_min, y_min, x_max - 1, y_max - 1))
    return boxes


def to_yolo_line(box: Tuple[int, int, int, int], img_w: int, img_h: int, cls: int = 0) -> str:
    x_min, y_min, x_max, y_max = box
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def pith_bboxes_from_ann(ann: dict, half_size: int) -> List[Tuple[int, int, int, int]]:
    """Each Pith point becomes a small square bbox of side 2*half_size, clamped to image."""
    h, w = ann["size"]["height"], ann["size"]["width"]
    out = []
    for obj in ann.get("objects", []):
        if obj.get("classTitle") != "Pith":
            continue
        for x, y in obj.get("points", {}).get("exterior", []):
            x, y = int(x), int(y)
            x_min = max(0, x - half_size)
            y_min = max(0, y - half_size)
            x_max = min(w - 1, x + half_size)
            y_max = min(h - 1, y + half_size)
            out.append((x_min, y_min, x_max, y_max))
    return out


def find_annotated_slices(sm_dir: str) -> List[str]:
    """Return ann filenames in `sm_dir/ann` that contain at least one Knot OR Pith object."""
    ann_dir = join(sm_dir, "ann")
    out = []
    for f in sorted(os.listdir(ann_dir)):
        with open(join(ann_dir, f)) as fh:
            d = json.load(fh)
        if any(o["classTitle"] in ("Knot", "Pith") for o in d.get("objects", [])):
            out.append(f)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=DEFAULT_SUBSETS,
        help="Sub-dataset IDs to include (e.g. 1 2 4). Subset 3 has no annotations.",
    )
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/knot_yolo")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--pith_bbox_half",
        type=int,
        default=8,
        help="Half side of the square bbox placed around each Pith point (so total side = 2*half).",
    )
    parser.add_argument(
        "--exclude_pages",
        type=str,
        default="",
        help="Comma-separated 'subset:page' pairs to drop entirely (e.g. '4:7,4:49'). Useful for holdout-eval splits.",
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

    # Collect (subset_id, ann_fname) pairs across all chosen subsets
    rng = random.Random(args.seed)
    train_items = []
    val_items = []
    for subset_id in args.subsets:
        sm_dir = join(args.project_root, subset_id)
        if not os.path.isdir(join(sm_dir, "ann")):
            print(f"  skipping subset {subset_id}: no ann/ subdir")
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

            knot_boxes = bboxes_from_mask(knot_mask_from_ann(ann))
            pith_boxes = pith_bboxes_from_ann(ann, half_size=args.pith_bbox_half)
            total_knot += len(knot_boxes)
            total_pith += len(pith_boxes)

            base_stem = img_fname.replace(".tiff", "")
            stem = f"ds{subset_id}_{base_stem}"
            Image.fromarray(img).convert("RGB").save(out_dir / "images" / split / f"{stem}.jpg", quality=95)
            lines = [to_yolo_line(b, w, h, cls=0) for b in knot_boxes]
            lines += [to_yolo_line(b, w, h, cls=1) for b in pith_boxes]
            (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    yaml_path = out_dir / "knots.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnc: 2\nnames: [knot, pith]\n"
    )
    print(
        f"\nwrote {total_knot} knot bboxes + {total_pith} pith bboxes across {len(train_items) + len(val_items)} slices"
    )
    print(f"YOLO dataset at {out_dir}")
    print(f"config: {yaml_path}")


if __name__ == "__main__":
    main()
