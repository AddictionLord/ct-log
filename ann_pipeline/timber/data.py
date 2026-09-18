"""Load Timber slices and compose per-instance masks from Supervisely annotations.

Each annotated frame carries exactly 8 objects, one per numeric class title
(timber id). The class title is therefore the instance identity, persistent
across slices.
"""

import base64
import json
import os
from os.path import join
from typing import Dict, List, Tuple
import zlib

import cv2
import numpy as np
from PIL import Image

DEFAULT_PROJECT_ROOT = "/mnt/D/datasets/ct_log/382882_timber"
DEFAULT_SUBSET = "01"
NON_TIMBER_CLASSES = frozenset({"KNOT"})


def decode_bitmap(b64: str) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(b64))
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., 3] > 0
    return arr > 0


def load_gray(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    return arr.astype(np.uint8)


def timber_instances_from_ann(ann: Dict) -> Dict[str, np.ndarray]:
    """Return {class_title: binary mask}, one entry per timber instance."""
    height, width = ann["size"]["height"], ann["size"]["width"]
    instances = {}
    for obj in ann.get("objects", []):
        title = obj.get("classTitle")
        if title in NON_TIMBER_CLASSES or "bitmap" not in obj:
            continue
        mask = np.zeros((height, width), dtype=bool)
        bmp = obj["bitmap"]
        origin_x, origin_y = bmp["origin"]
        patch = decode_bitmap(bmp["data"])
        patch_h, patch_w = patch.shape
        mask[origin_y : origin_y + patch_h, origin_x : origin_x + patch_w] = patch
        if title in instances:
            instances[title] = instances[title] | mask
        else:
            instances[title] = mask
    return instances


def find_annotated_slices(subset_dir: str) -> List[str]:
    ann_dir = join(subset_dir, "ann")
    out = []
    for fname in sorted(os.listdir(ann_dir)):
        with open(join(ann_dir, fname)) as fh:
            ann = json.load(fh)
        objects = [o for o in ann.get("objects", []) if o.get("classTitle") not in NON_TIMBER_CLASSES]
        if objects:
            out.append(fname)
    return out


def load_slice(subset_dir: str, ann_fname: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    with open(join(subset_dir, "ann", ann_fname)) as fh:
        ann = json.load(fh)
    img_fname = ann_fname[: -len(".json")]
    gray = load_gray(join(subset_dir, "img", img_fname))
    return gray, timber_instances_from_ann(ann)
