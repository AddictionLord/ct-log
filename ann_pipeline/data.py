"""Data loading utilities for the CT-log Supervisely dataset.

Yields image arrays and GT pith points (in image coordinates) for slices that
have a Pith annotation. Pith is stored as a single (x, y) point in the
Supervisely JSON.
"""

from dataclasses import dataclass
import json
import os
from os.path import join
from typing import Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

DEFAULT_SM_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025/1"


@dataclass
class PithSlice:
    page: int
    img: np.ndarray  # (H, W) uint8 grayscale
    pith_xy: Tuple[int, int]  # ground-truth pith point (x, y) in pixel coords
    img_path: str
    ann_path: str


def _load_tiff_gray(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    return arr.astype(np.uint8)


def _extract_pith(ann: dict) -> Optional[Tuple[int, int]]:
    for obj in ann.get("objects", []):
        if obj.get("classTitle") == "Pith":
            pts = obj.get("points", {}).get("exterior", [])
            if pts:
                x, y = pts[0]
                return int(x), int(y)
    return None


def iter_pith_slices(sm_dir: str = DEFAULT_SM_DIR) -> Iterator[PithSlice]:
    """Yield every slice in `sm_dir` that has a Pith annotation."""
    img_dir = join(sm_dir, "img")
    ann_dir = join(sm_dir, "ann")
    for fname in sorted(os.listdir(ann_dir)):
        ann_path = join(ann_dir, fname)
        with open(ann_path) as f:
            ann = json.load(f)
        pith = _extract_pith(ann)
        if pith is None:
            continue
        img_fname = fname[: -len(".json")]
        img_path = join(img_dir, img_fname)
        page = int(img_fname.replace("page_", "").replace(".tiff", ""))
        yield PithSlice(
            page=page,
            img=_load_tiff_gray(img_path),
            pith_xy=pith,
            img_path=img_path,
            ann_path=ann_path,
        )


def collect_pith_slices(sm_dir: str = DEFAULT_SM_DIR) -> List[PithSlice]:
    return list(iter_pith_slices(sm_dir))
