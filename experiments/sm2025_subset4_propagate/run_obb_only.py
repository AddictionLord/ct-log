"""Anchor-free OBB-only propagation across a CT-log volume.

For each frame: run YOLO-OBB → rasterise as ellipses → seed those into a
MedSAM2 video predictor. All seeds collapse into one tracked knot object.
No GT anchors. No wood, no pith — knot-only propagation, intended as a
"what if we had zero human annotation" test.

To fit on a 4 GB GPU we process the volume in **overlapping windows**: each
window is `window_size` frames with `overlap` frames re-seeded from the
previous window's output (so memory continuity is preserved across the join).

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.run_obb_only

Output: result_obb_only.npz with the same keys as result.npz (only the Knot
class is non-zero in `pred`).
"""

import argparse
import os
from os.path import join
import pathlib
from typing import Dict

import cv2
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor_npz
import torch
from ultralytics import YOLO

from experiments.sm2025_subset4_propagate.run import (
    CLASS_IDS,
    list_pages,
    load_tiff_gray,
    page_img_path,
    prepare_video,
)

OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out"
DEFAULT_YOLO_OBB_WEIGHTS = (
    "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_obb_2cls_holdout_v3/weights/best.pt"
)
KNOT_CLS = 0


def load_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1).astype(np.uint8)
    if arr.shape[-1] == 4:
        return arr[..., :3].astype(np.uint8)
    return arr.astype(np.uint8)


def obb_ellipse_mask(yolo: YOLO, img_rgb: np.ndarray, h: int, w: int, conf: float, nms_iou: float) -> np.ndarray:
    """YOLO-OBB → union of inscribed ellipses (binary mask)."""
    res = yolo.predict(img_rgb, conf=conf, iou=nms_iou, verbose=False)[0]
    out = np.zeros((h, w), dtype=np.uint8)
    if res.obb is None or len(res.obb) == 0:
        return out
    obb_cls = res.obb.cls.cpu().numpy().astype(int)
    xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()[obb_cls == KNOT_CLS]
    for corners in xyxyxyxy:
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
    return out


def propagate_window(
    predictor,
    vol: np.ndarray,
    obb_seeds_by_local_frame: Dict[int, np.ndarray],
    extra_seed: Dict[int, np.ndarray],
    size: int,
) -> np.ndarray:
    """Propagate a window seeded by OBB rasters at each frame plus optional
    extra seeds (from prior window's output) at the leading overlap frames.
    Returns a per-frame binary knot mask (0/255 → 0/KNOT)."""
    d, h, w = vol.shape
    video = prepare_video(vol, size=size)
    knot_id = CLASS_IDS["Knot"]
    pred = np.zeros_like(vol, dtype=np.uint8)
    if not obb_seeds_by_local_frame and not extra_seed:
        return pred
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video, h, w)
        for fi, m in obb_seeds_by_local_frame.items():
            if m.sum() == 0:
                continue
            predictor.add_new_mask(inference_state=state, frame_idx=fi, obj_id=knot_id, mask=m.astype(np.uint8))
        for fi, m in extra_seed.items():
            if m.sum() == 0:
                continue
            predictor.add_new_mask(inference_state=state, frame_idx=fi, obj_id=knot_id, mask=m.astype(np.uint8))
        for out_fi, obj_ids, out_logits in predictor.propagate_in_video(state, start_frame_idx=0, reverse=False):
            for k, oid in enumerate(obj_ids):
                m = (out_logits[k] > 0.0).cpu().numpy()[0]
                pred[out_fi][m] = oid
        predictor.reset_state(state)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt")
    parser.add_argument("--yolo_obb_weights", default=DEFAULT_YOLO_OBB_WEIGHTS)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--window_size", type=int, default=30, help="Frames per propagation window.")
    parser.add_argument("--overlap", type=int, default=5, help="Frames re-seeded from prior window's output.")
    parser.add_argument("--out_name", default="result_obb_only.npz")
    args = parser.parse_args()

    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    pages = list_pages()
    sample = load_tiff_gray(page_img_path(pages[0]))
    h, w = sample.shape
    vol = np.stack([load_tiff_gray(page_img_path(p)) for p in pages])
    print("volume shape:", vol.shape)

    print("loading YOLO-OBB:", args.yolo_obb_weights)
    yolo = YOLO(args.yolo_obb_weights)
    print("precomputing OBB ellipses on all frames ...")
    obb_per_page: Dict[int, np.ndarray] = {}
    n_with_obb = 0
    for p in pages:
        rgb = load_rgb(page_img_path(p))
        m = obb_ellipse_mask(yolo, rgb, h, w, args.conf, args.nms_iou)
        obb_per_page[p] = m
        if m.sum() > 0:
            n_with_obb += 1
    print("OBB detections on %d/%d frames" % (n_with_obb, len(pages)))

    repo_root = os.path.abspath(join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty", "MedSAM2"))
    model_cfg = "//" + join(repo_root, "sam2/configs/sam2.1_hiera_t512.yaml")
    predictor = build_sam2_video_predictor_npz(model_cfg, args.checkpoint)

    pred = np.zeros_like(vol, dtype=np.uint8)
    n_frames = len(pages)
    step = max(1, args.window_size - args.overlap)
    starts = list(range(0, n_frames, step))
    print(
        "processing %d windows of size %d (step=%d, overlap=%d)" % (len(starts), args.window_size, step, args.overlap)
    )

    for wi, s in enumerate(starts):
        e = min(s + args.window_size, n_frames)
        sub = vol[s:e]
        d = e - s
        obb_seeds: Dict[int, np.ndarray] = {}
        for li in range(d):
            gi = s + li
            m = obb_per_page[pages[gi]]
            if m.sum() > 0:
                obb_seeds[li] = m

        extra_seed: Dict[int, np.ndarray] = {}
        if wi > 0:
            overlap_take = min(args.overlap, d)
            for li in range(overlap_take):
                gi = s + li
                prior_mask = (pred[gi] == CLASS_IDS["Knot"]).astype(np.uint8)
                if prior_mask.sum() > 0:
                    extra_seed[li] = prior_mask

        win_pred = propagate_window(predictor, sub, obb_seeds, extra_seed, args.size)
        if wi == 0:
            pred[s:e] = win_pred
        else:
            pred[s + args.overlap : e] = win_pred[args.overlap :]
        print(
            "  window %d (frames %d..%d): seeded %d OBB frames + %d extras"
            % (wi, s, e - 1, len(obb_seeds), len(extra_seed))
        )

    out_path = join(OUT_DIR, args.out_name)
    np.savez_compressed(
        out_path,
        imgs=vol,
        pred=pred,
        pages=np.array(pages),
        is_anchor=np.zeros(n_frames, dtype=bool),
        anchors=np.array([], dtype=np.int64),
    )
    n_frames_w_knot = int(sum(1 for i in range(n_frames) if (pred[i] == CLASS_IDS["Knot"]).any()))
    total_knot_px = int((pred == CLASS_IDS["Knot"]).sum())
    print("saved %s — knots on %d/%d frames, total %d px" % (out_path, n_frames_w_knot, n_frames, total_knot_px))


if __name__ == "__main__":
    main()
