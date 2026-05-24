"""OBB-augmented anchor propagation using SAM2-image-predictor seeds.

Variant of run_obb_augmented.py: instead of rasterising each YOLO-OBB to an
inscribed ellipse and feeding that as a mask seed, we run SAM2's image
predictor per-frame with:
  - box   = AABB of OBB
  - points = 5 positive points along OBB long axis + 4 negatives (2 on short
             axis edges, 2 beyond long axis tips)
to get a per-frame knot mask, then seed *that* into MedSAM2 video predictor
at the non-anchor frame.

Rationale (smoke test on subset 3 page 170): SAM image predictor with these
prompts produces natural, image-grounded knot shapes on hardwood textures
where mask_input ellipse seeds fall back to ellipse-shaped outputs.

Wood and Pith propagation are unchanged from run.py — anchor-only.

Run from repo root:
    CT_SUBSET=3 conda run -n ct-log python -m experiments.sm2025_subset4_propagate.run_obb_augmented_points
"""

import argparse
import os
from os.path import join
import pathlib
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor_npz
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy import ndimage as ndi
import torch
from ultralytics import YOLO

from experiments.sm2025_subset4_propagate.run import (
    CLASS_IDS,
    find_anchor_pages,
    list_pages,
    load_tiff_gray,
    page_ann_path,
    page_img_path,
    render_annotation,
)
from experiments.sm2025_subset4_propagate.run_obb_augmented import (
    propagate_segment_obb,
)

OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out"
DEFAULT_YOLO_OBB_WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_obb_v1/weights/best.pt"
SAM_CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
SAM_MODEL_CFG = "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"


def load_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1).astype(np.uint8)
    if arr.shape[-1] == 4:
        return arr[..., :3].astype(np.uint8)
    return arr.astype(np.uint8)


def fit_obb_params(corners: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    (cx, cy), (w, h), angle = cv2.minAreaRect(corners.astype(np.float32))
    if h > w:
        long_len, short_len = h, w
        angle = angle + 90.0
    else:
        long_len, short_len = w, h
    return (cx, cy), (long_len, short_len), angle


def axis_points(corners: np.ndarray, n_points: int, inset_frac: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    (cx, cy), (long_len, _), angle = fit_obb_params(corners)
    theta = np.deg2rad(angle)
    direction = np.array([np.cos(theta), np.sin(theta)])
    half = (long_len / 2.0) * (1.0 - inset_frac)
    if n_points == 1:
        offsets = [0.0]
    else:
        offsets = list(np.linspace(-half, half, n_points))
    pts = np.array([[cx + d * direction[0], cy + d * direction[1]] for d in offsets])
    return pts, np.ones(len(pts), dtype=np.int32)


def obb_negatives(corners: np.ndarray, outset_frac: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    (cx, cy), (long_len, short_len), angle = fit_obb_params(corners)
    theta = np.deg2rad(angle)
    long_dir = np.array([np.cos(theta), np.sin(theta)])
    short_dir = np.array([-np.sin(theta), np.cos(theta)])
    push_short = (short_len / 2.0) * (1.0 + outset_frac)
    push_long = (long_len / 2.0) * (1.0 + outset_frac)
    pts = np.array(
        [
            [cx + push_short * short_dir[0], cy + push_short * short_dir[1]],
            [cx - push_short * short_dir[0], cy - push_short * short_dir[1]],
            [cx + push_long * long_dir[0], cy + push_long * long_dir[1]],
            [cx - push_long * long_dir[0], cy - push_long * long_dir[1]],
        ]
    )
    return pts, np.zeros(len(pts), dtype=np.int32)


def aabb_of_obb(corners: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    x1 = max(0, int(np.floor(corners[:, 0].min())))
    y1 = max(0, int(np.floor(corners[:, 1].min())))
    x2 = min(w, int(np.ceil(corners[:, 0].max())))
    y2 = min(h, int(np.ceil(corners[:, 1].max())))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return lab == largest


def per_frame_sam_knot_mask(
    yolo: YOLO,
    sam_pred: SAM2ImagePredictor,
    img_rgb: np.ndarray,
    conf: float,
    nms_iou: float,
    n_points: int,
) -> np.ndarray:
    """Run YOLO-OBB on a frame, then SAM image predictor with box+pos+neg per
    detection. Returns the union of per-instance masks (clipped to AABB,
    largest-CC per instance) as a single binary mask."""
    h, w = img_rgb.shape[:2]
    out = np.zeros((h, w), dtype=bool)
    res = yolo.predict(img_rgb, conf=conf, iou=nms_iou, verbose=False)[0]
    if res.obb is None or len(res.obb) == 0:
        return out
    xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()
    sam_pred.set_image(img_rgb)
    for corners in xyxyxyxy:
        aabb = aabb_of_obb(corners, (h, w))
        x1, y1, x2, y2 = aabb.astype(int)
        pos_pts, pos_labs = axis_points(corners, n_points)
        neg_pts, neg_labs = obb_negatives(corners)
        pts = np.concatenate([pos_pts, neg_pts], axis=0).astype(np.float32)
        labs = np.concatenate([pos_labs, neg_labs], axis=0).astype(np.int32)
        m, _, _ = sam_pred.predict(
            box=aabb,
            point_coords=pts,
            point_labels=labs,
            multimask_output=False,
        )
        mask = m[0].astype(bool)
        clipped = np.zeros_like(mask)
        clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        clipped = keep_largest_cc(clipped)
        out |= clipped
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=SAM_CHECKPOINT)
    parser.add_argument("--yolo_obb_weights", default=DEFAULT_YOLO_OBB_WEIGHTS)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--n_points", type=int, default=5)
    parser.add_argument("--out_name", default="result_obb_aug_points.npz")
    args = parser.parse_args()

    pathlib.Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

    pages = list_pages()
    anchors = find_anchor_pages(pages)
    page_to_idx = {p: i for i, p in enumerate(pages)}
    anchor_idx = sorted(page_to_idx[p] for p in anchors)
    print("frames: %d, anchors: %d at pages %s" % (len(pages), len(anchors), anchors))

    sample_img = load_tiff_gray(page_img_path(pages[0]))
    h, w = sample_img.shape
    vol = np.stack([load_tiff_gray(page_img_path(p)) for p in pages])
    print("volume shape:", vol.shape)

    anchor_masks: Dict[int, Dict[str, np.ndarray]] = {}
    anchor_pith: Dict[int, List[Tuple[int, int]]] = {}
    for p in anchors:
        m, pith = render_annotation(page_ann_path(p), h, w)
        anchor_masks[p] = m
        anchor_pith[p] = pith

    print("loading YOLO-OBB: %s" % args.yolo_obb_weights)
    yolo = YOLO(args.yolo_obb_weights)
    print("loading SAM2 image predictor ...")
    sam_img = build_sam2("//" + SAM_MODEL_CFG, args.checkpoint)
    sam_pred = SAM2ImagePredictor(sam_img)

    print("precomputing per-frame SAM knot masks on non-anchor frames ...")
    obb_per_page: Dict[int, np.ndarray] = {}
    anchor_set = set(anchors)
    n_with_mask = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, p in enumerate(pages):
            if p in anchor_set:
                continue
            rgb = load_rgb(page_img_path(p))
            mask = per_frame_sam_knot_mask(yolo, sam_pred, rgb, args.conf, args.nms_iou, args.n_points)
            obb_per_page[p] = mask.astype(np.uint8)
            if mask.sum() > 0:
                n_with_mask += 1
    print("SAM knot masks on %d/%d non-anchor frames" % (n_with_mask, len(pages) - len(anchors)))

    del sam_img, sam_pred
    torch.cuda.empty_cache()

    repo_root = os.path.abspath(join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty", "MedSAM2"))
    model_cfg = "//" + join(repo_root, "sam2/configs/sam2.1_hiera_t512.yaml")
    predictor = build_sam2_video_predictor_npz(model_cfg, args.checkpoint)

    pred = np.zeros_like(vol, dtype=np.uint8)
    is_anchor = np.zeros(len(pages), dtype=bool)
    for ai in anchor_idx:
        is_anchor[ai] = True

    def obb_seeds_for_segment(start_global: int, end_global: int) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        for gi in range(start_global + 1, end_global):
            p = pages[gi]
            m = obb_per_page.get(p)
            if m is not None and m.sum() > 0:
                out[gi - start_global] = m
        return out

    first_a = anchor_idx[0]
    if first_a > 0:
        sub = vol[: first_a + 1]
        a_page = anchors[0]
        obb_seeds: Dict[int, np.ndarray] = {}
        for gi in range(first_a):
            p = pages[gi]
            m = obb_per_page.get(p)
            if m is not None and m.sum() > 0:
                obb_seeds[gi] = m
        seg = propagate_segment_obb(
            predictor,
            sub,
            seed_masks_a={c: np.zeros((h, w), np.uint8) for c in CLASS_IDS},
            seed_pith_a=[],
            seed_masks_b=anchor_masks[a_page],
            seed_pith_b=anchor_pith[a_page],
            obb_masks_by_local_frame=obb_seeds,
            size=args.size,
        )
        pred[: first_a + 1] = seg
        print("left tail: %d frames done (%d SAM seeds)" % (first_a + 1, len(obb_seeds)))

    for s in range(len(anchor_idx) - 1):
        a, b = anchor_idx[s], anchor_idx[s + 1]
        sub = vol[a : b + 1]
        pa, pb = anchors[s], anchors[s + 1]
        obb_seeds = obb_seeds_for_segment(a, b)
        seg = propagate_segment_obb(
            predictor,
            sub,
            seed_masks_a=anchor_masks[pa],
            seed_pith_a=anchor_pith[pa],
            seed_masks_b=anchor_masks[pb],
            seed_pith_b=anchor_pith[pb],
            obb_masks_by_local_frame=obb_seeds,
            size=args.size,
        )
        pred[a : b + 1] = seg
        print("segment pages %d..%d (%d frames, %d SAM seeds) done" % (pa, pb, b - a + 1, len(obb_seeds)))

    last_a = anchor_idx[-1]
    if last_a < len(pages) - 1:
        sub = vol[last_a:]
        pa = anchors[-1]
        obb_seeds = obb_seeds_for_segment(last_a, len(pages))
        seg = propagate_segment_obb(
            predictor,
            sub,
            seed_masks_a=anchor_masks[pa],
            seed_pith_a=anchor_pith[pa],
            seed_masks_b=None,
            seed_pith_b=None,
            obb_masks_by_local_frame=obb_seeds,
            size=args.size,
        )
        pred[last_a:] = seg
        print("right tail: %d frames done (%d SAM seeds)" % (len(pages) - last_a, len(obb_seeds)))

    out_path = join(OUT_DIR, args.out_name)
    np.savez_compressed(
        out_path,
        imgs=vol,
        pred=pred,
        pages=np.array(pages),
        is_anchor=is_anchor,
        anchors=np.array(anchors),
    )
    print("saved", out_path)


if __name__ == "__main__":
    main()
