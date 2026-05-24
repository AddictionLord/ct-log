"""OBB-augmented propagation across the subset 4 volume.

Same architecture as run.py (segment-by-segment between anchors), but on top
of the anchor GT seeds we add **YOLO-OBB raster masks as additional knot
seeds on every non-anchor frame** within the segment. All knot seeds collapse
into a single tracked object (obj_id=KNOT) — we keep single-class semantics
and split into per-instance bitmaps only at the final encoding stage.

Rationale: propagation alone misses knots that appear mid-segment (between
anchors). Adding YOLO-OBB hints at non-anchor frames patches this without
destroying SAM2's temporal coherence — overlapping seeds across frames are
merged into one mask, so SAM2's memory bank smooths spurious single-frame
detections away.

Wood and Pith propagation are unchanged from run.py — they only get seeded
at anchors.

Output: `result_obb_augmented.npz` (same keys as `result.npz`).

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.run_obb_augmented
"""

import argparse
import os
from os.path import join
import pathlib
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor_npz
import torch
from ultralytics import YOLO

from experiments.sm2025_subset4_propagate.run import (
    CLASS_IDS,
    find_anchor_pages,
    list_pages,
    load_tiff_gray,
    page_ann_path,
    page_img_path,
    prepare_video,
    render_annotation,
)

OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out"
DEFAULT_YOLO_OBB_WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_obb_v1/weights/best.pt"


def load_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1).astype(np.uint8)
    if arr.shape[-1] == 4:
        return arr[..., :3].astype(np.uint8)
    return arr.astype(np.uint8)


def obb_knot_mask(
    yolo: YOLO,
    img_rgb: np.ndarray,
    h: int,
    w: int,
    conf: float,
    nms_iou: float,
    seed_shape: str = "ellipse",
) -> np.ndarray:
    """Run YOLO-OBB on a frame and return the union of seed rasters as a binary mask.

    seed_shape:
        "rectangle" — fill the OBB polygon (sharp rect seed, original behaviour).
        "ellipse"   — fill the maximum-area ellipse inscribed in the OBB (soft seed).
    """
    res = yolo.predict(img_rgb, conf=conf, iou=nms_iou, verbose=False)[0]
    out = np.zeros((h, w), dtype=np.uint8)
    if res.obb is None or len(res.obb) == 0:
        return out
    xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()
    for corners in xyxyxyxy:
        if seed_shape == "rectangle":
            cv2.fillPoly(out, [corners.astype(np.int32)], 1)
        elif seed_shape == "ellipse":
            (cx, cy), (rw, rh), angle = cv2.minAreaRect(corners.astype(np.float32))
            cv2.ellipse(
                out,
                center=(int(round(cx)), int(round(cy))),
                axes=(max(1, int(rw / 2)), max(1, int(rh / 2))),
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=1,
                thickness=-1,
            )
        else:
            msg = "unknown seed_shape: %s" % seed_shape
            raise ValueError(msg)
    return out


def seed_state_with_obb(
    predictor,
    state,
    local_anchor_frame: int,
    anchor_masks: Dict[str, np.ndarray],
    anchor_pith: List[Tuple[int, int]],
    obb_masks_by_local_frame: Dict[int, np.ndarray],
) -> None:
    """Seed the video state at the anchor frame with full anchor masks, then
    on every non-anchor frame with the OBB raster (knot-class only)."""
    for cls_name, cid in CLASS_IDS.items():
        if cls_name == "Pith":
            if anchor_pith:
                arr = np.array(anchor_pith, dtype=np.float32)
                labels = np.ones(len(arr), dtype=np.int32)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=local_anchor_frame,
                    obj_id=cid,
                    points=arr,
                    labels=labels,
                )
        else:
            m = anchor_masks[cls_name]
            if m.sum() > 0:
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=local_anchor_frame,
                    obj_id=cid,
                    mask=m.astype(np.uint8),
                )

    knot_id = CLASS_IDS["Knot"]
    for local_fi, obb_mask in obb_masks_by_local_frame.items():
        if obb_mask.sum() == 0:
            continue
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=local_fi,
            obj_id=knot_id,
            mask=obb_mask.astype(np.uint8),
        )


def propagate_segment_obb(
    predictor,
    vol: np.ndarray,
    seed_masks_a: Dict[str, np.ndarray],
    seed_pith_a: List[Tuple[int, int]],
    seed_masks_b: Dict[str, np.ndarray],
    seed_pith_b: List[Tuple[int, int]],
    obb_masks_by_local_frame: Dict[int, np.ndarray],
    size: int,
) -> np.ndarray:
    """Forward+reverse propagation with OBB seeds injected at non-anchor frames.

    obb_masks_by_local_frame: keys are local indices in vol (excluding the
    anchor endpoints 0 and -1). Each value is a binary knot raster.
    """
    d, h, w = vol.shape
    video = prepare_video(vol, size=size)
    pred_fwd = np.zeros_like(vol, dtype=np.uint8)
    pred_rev = np.zeros_like(vol, dtype=np.uint8)
    has_a = seed_masks_a is not None and (any(m.sum() > 0 for m in seed_masks_a.values()) or bool(seed_pith_a))
    has_b = seed_masks_b is not None and (any(m.sum() > 0 for m in seed_masks_b.values()) or bool(seed_pith_b))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if has_a:
            state = predictor.init_state(video, h, w)
            seed_state_with_obb(predictor, state, 0, seed_masks_a, seed_pith_a, obb_masks_by_local_frame)
            for out_fi, obj_ids, out_logits in predictor.propagate_in_video(state, start_frame_idx=0, reverse=False):
                for k, oid in enumerate(obj_ids):
                    m = (out_logits[k] > 0.0).cpu().numpy()[0]
                    pred_fwd[out_fi][m] = oid
            predictor.reset_state(state)

        if d > 1 and has_b:
            state = predictor.init_state(video, h, w)
            seed_state_with_obb(predictor, state, d - 1, seed_masks_b, seed_pith_b, obb_masks_by_local_frame)
            for out_fi, obj_ids, out_logits in predictor.propagate_in_video(state, start_frame_idx=d - 1, reverse=True):
                for k, oid in enumerate(obj_ids):
                    m = (out_logits[k] > 0.0).cpu().numpy()[0]
                    pred_rev[out_fi][m] = oid
            predictor.reset_state(state)

    out = np.zeros_like(vol, dtype=np.uint8)
    if has_a:
        out[0] = pred_fwd[0]
    elif has_b:
        out[0] = pred_rev[0]
    if d > 1:
        if has_b:
            out[-1] = pred_rev[-1]
        elif has_a:
            out[-1] = pred_fwd[-1]
    for i in range(1, d - 1):
        if has_a and has_b:
            out[i] = pred_fwd[i] if i <= (d - 1 - i) else pred_rev[i]
        elif has_a:
            out[i] = pred_fwd[i]
        elif has_b:
            out[i] = pred_rev[i]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt")
    parser.add_argument("--yolo_obb_weights", default=DEFAULT_YOLO_OBB_WEIGHTS)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument(
        "--seed_shape",
        choices=["ellipse", "rectangle"],
        default="ellipse",
        help="Shape used to rasterise OBB into a SAM2 mask seed. Ellipse is softer and avoids rect-shaped outputs.",
    )
    parser.add_argument("--out_name", default="result_obb_augmented.npz")
    args = parser.parse_args()

    pathlib.Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

    pages = list_pages()
    anchors = find_anchor_pages(pages)
    page_to_idx = {p: i for i, p in enumerate(pages)}
    anchor_idx = [page_to_idx[p] for p in anchors]
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
    print("precomputing OBB rasters on non-anchor frames ...")
    obb_per_page: Dict[int, np.ndarray] = {}
    n_with_obb = 0
    for i, p in enumerate(pages):
        if i in anchor_idx:
            continue
        rgb = load_rgb(page_img_path(p))
        mask = obb_knot_mask(yolo, rgb, h, w, args.conf, args.nms_iou, seed_shape=args.seed_shape)
        obb_per_page[p] = mask
        if mask.sum() > 0:
            n_with_obb += 1
    print("OBB detections on %d/%d non-anchor frames" % (n_with_obb, len(pages) - len(anchors)))

    repo_root = os.path.abspath(join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty", "MedSAM2"))
    model_cfg = "//" + join(repo_root, "sam2/configs/sam2.1_hiera_t512.yaml")
    predictor = build_sam2_video_predictor_npz(model_cfg, args.checkpoint)

    pred = np.zeros_like(vol, dtype=np.uint8)
    is_anchor = np.zeros(len(pages), dtype=bool)
    for ai in anchor_idx:
        is_anchor[ai] = True

    def obb_seeds_for_segment(start_global: int, end_global: int) -> Dict[int, np.ndarray]:
        """Local-frame -> OBB raster, for frames strictly inside (start_global, end_global)."""
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
        obb_seeds = obb_seeds_for_segment(-1, first_a)
        obb_seeds = {k + 1: v for k, v in obb_seeds.items()}
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
        print("left tail: %d frames done (%d OBB seeds)" % (first_a + 1, len(obb_seeds)))

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
        print("segment pages %d..%d (%d frames, %d OBB seeds) done" % (pa, pb, b - a + 1, len(obb_seeds)))

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
        print("right tail: %d frames done (%d OBB seeds)" % (len(pages) - last_a, len(obb_seeds)))

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
