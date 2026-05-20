"""Hybrid YOLO -> SAM2 image -> SAM2 video pipeline on a single subset volume.

Stage 1 (per-slice prompt generation):
  For each TIFF page in the subset:
    - Run YOLO -> get knot bboxes (cls=0) and pith bboxes (cls=1)
    - Feed each bbox to SAM2 image predictor -> binary mask
    - Union all knot masks -> single class mask for the slice
    - Union all pith masks -> single class mask for the slice

Stage 2 (volumetric propagation):
  Build a (D, H, W) volume from the slice TIFFs
  For every slice with a non-empty class mask: add_new_mask(frame_idx=k, obj_id=class)
  Propagate forward + reverse from the middle seeded frame
  Output: per-class 3D mask volume

Compare against GT on the annotated slices in this subset and report:
  - per-slice knot Dice
  - per-slice pith centroid distance
  - delta vs the per-slice (image-only) baseline

Run from repo root:
    conda run -n ct-log python -m ann_pipeline.scripts.yolo_to_sam_video --subset 1
"""

import argparse
import json
import os
from os.path import join
import pathlib
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor_npz
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ann_pipeline.knot.data_prep import knot_mask_from_ann

PROJECT_ROOT = "/mnt/D/datasets/ct_log/375492_SM_2025"
WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_2cls_v1/weights/best.pt"
CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
MODEL_CFG = "//" + "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"
KNOT_ID = 1
PITH_ID = 2


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


def load_tiff_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3]
    else:
        arr = np.stack([arr] * 3, axis=-1)
    return arr.astype(np.uint8)


def load_tiff_gray(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    return arr.astype(np.uint8)


def extract_pith_point(ann: dict) -> Optional[Tuple[float, float]]:
    for obj in ann.get("objects", []):
        if obj.get("classTitle") == "Pith":
            pts = obj.get("points", {}).get("exterior", [])
            if pts:
                return float(pts[0][0]), float(pts[0][1])
    return None


def prepare_video(vol: np.ndarray, size: int = 512) -> torch.Tensor:
    """(D, H, W) uint8 -> (D, 3, size, size) normalised tensor on cuda."""
    d, h, w = vol.shape
    out = np.zeros((d, 3, size, size), dtype=np.float32)
    for i in range(d):
        im = Image.fromarray(vol[i]).convert("RGB").resize((size, size))
        out[i] = np.array(im).transpose(2, 0, 1)
    out /= 255.0
    t = torch.from_numpy(out).cuda()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].cuda()
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].cuda()
    return (t - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="1")
    parser.add_argument("--yolo_weights", default=WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/yolo_to_sam_video")
    parser.add_argument("--max_pages", type=int, default=0, help="0 = all pages")
    parser.add_argument("--page_min", type=int, default=None, help="inclusive lower page bound")
    parser.add_argument("--page_max", type=int, default=None, help="inclusive upper page bound")
    args = parser.parse_args()

    sm_dir = join(PROJECT_ROOT, args.subset)
    img_dir = join(sm_dir, "img")
    ann_dir = join(sm_dir, "ann")

    page_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".tiff"))
    if args.page_min is not None or args.page_max is not None:
        lo = args.page_min if args.page_min is not None else -1
        hi = args.page_max if args.page_max is not None else 10**9
        page_files = [f for f in page_files if lo <= int(f.replace("page_", "").replace(".tiff", "")) <= hi]
    if args.max_pages > 0:
        page_files = page_files[: args.max_pages]
    pages = [int(f.replace("page_", "").replace(".tiff", "")) for f in page_files]
    d = len(pages)

    out_dir = pathlib.Path(args.out_dir) / f"ds{args.subset}_conf{args.conf}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_slice").mkdir(exist_ok=True)
    print(f"subset {args.subset}: {d} pages, conf>={args.conf}")

    # --- Stage 1: build per-slice prompt masks via YOLO + SAM2 image predictor ---
    print("loading YOLO ...")
    yolo = YOLO(args.yolo_weights)
    print("loading SAM2 image predictor ...")
    img_sam = build_sam2(MODEL_CFG, CHECKPOINT)
    img_predictor = SAM2ImagePredictor(img_sam)

    h0, w0 = None, None
    knot_prompts: Dict[int, np.ndarray] = {}
    pith_prompts: Dict[int, np.ndarray] = {}
    pith_centroids: Dict[int, Tuple[float, float]] = {}

    vol_gray = np.zeros((d, 778, 778), dtype=np.uint8)
    for i, fname in enumerate(tqdm(page_files, desc="stage1 yolo+sam-image")):
        img_path = join(img_dir, fname)
        rgb = load_tiff_rgb(img_path)
        gray = load_tiff_gray(img_path)
        if h0 is None:
            h0, w0 = rgb.shape[:2]
        vol_gray[i] = gray

        res = yolo.predict(rgb, conf=args.conf, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.empty((0,), int)
        knot_bbs = xyxy[cls == 0]
        pith_bbs = xyxy[cls == 1]

        if len(knot_bbs) == 0 and len(pith_bbs) == 0:
            continue

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            img_predictor.set_image(rgb)
            if len(knot_bbs):
                knot_m = np.zeros((h0, w0), dtype=bool)
                for b in knot_bbs:
                    m, _, _ = img_predictor.predict(box=np.array(b, dtype=np.float32), multimask_output=False)
                    knot_m |= m[0].astype(bool)
                knot_prompts[i] = knot_m.astype(np.uint8)
            if len(pith_bbs):
                pith_m = np.zeros((h0, w0), dtype=bool)
                # for pith, prompt with the centre point (matches our 2D pipeline)
                # but also store the centroid for the centroid-distance metric
                centroids = []
                for b in pith_bbs:
                    cx = float((b[0] + b[2]) / 2.0)
                    cy = float((b[1] + b[3]) / 2.0)
                    centroids.append((cx, cy))
                    m, _, _ = img_predictor.predict(
                        point_coords=np.array([[cx, cy]], dtype=np.float32),
                        point_labels=np.array([1], dtype=np.int32),
                        multimask_output=False,
                    )
                    pith_m |= m[0].astype(bool)
                pith_prompts[i] = pith_m.astype(np.uint8)
                # one pith per slice typically — take the centroid closest to image centre
                pith_centroids[i] = centroids[0]

    # free image predictor
    del img_predictor
    del img_sam
    torch.cuda.empty_cache()
    print(f"\nstage1 done: {len(knot_prompts)} knot prompts, {len(pith_prompts)} pith prompts")

    # --- Stage 2: video propagation ---
    print("loading SAM2 video predictor ...")
    video_predictor = build_sam2_video_predictor_npz(MODEL_CFG, CHECKPOINT)

    video = prepare_video(vol_gray, size=512)
    pred_knot = np.zeros((d, h0, w0), dtype=np.uint8)
    pred_pith = np.zeros((d, h0, w0), dtype=np.uint8)

    # choose start frame as the median of seeded knot frames (fallback: middle of volume)
    seeded_frames = sorted(set(knot_prompts) | set(pith_prompts))
    if not seeded_frames:
        print("no YOLO detections at all — nothing to propagate")
        return
    start_frame = int(np.median(seeded_frames))
    print(f"propagation start frame: {start_frame}")

    def seed(state):
        for k, m in knot_prompts.items():
            video_predictor.add_new_mask(inference_state=state, frame_idx=k, obj_id=KNOT_ID, mask=m)
        for k, m in pith_prompts.items():
            video_predictor.add_new_mask(inference_state=state, frame_idx=k, obj_id=PITH_ID, mask=m)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = video_predictor.init_state(video, h0, w0)
        seed(state)
        for out_fi, obj_ids, out_logits in video_predictor.propagate_in_video(
            state, start_frame_idx=start_frame, reverse=False
        ):
            for k, oid in enumerate(obj_ids):
                m = (out_logits[k] > 0.0).cpu().numpy()[0]
                if oid == KNOT_ID:
                    pred_knot[out_fi] |= m.astype(np.uint8)
                elif oid == PITH_ID:
                    pred_pith[out_fi] |= m.astype(np.uint8)
        video_predictor.reset_state(state)
        state = video_predictor.init_state(video, h0, w0)
        seed(state)
        for out_fi, obj_ids, out_logits in video_predictor.propagate_in_video(
            state, start_frame_idx=start_frame, reverse=True
        ):
            for k, oid in enumerate(obj_ids):
                m = (out_logits[k] > 0.0).cpu().numpy()[0]
                if oid == KNOT_ID:
                    pred_knot[out_fi] |= m.astype(np.uint8)
                elif oid == PITH_ID:
                    pred_pith[out_fi] |= m.astype(np.uint8)
        video_predictor.reset_state(state)

    # --- Evaluation on the annotated slices in this subset ---
    rows = []
    for i, fname in enumerate(page_files):
        ann_path = join(ann_dir, fname + ".json")
        if not os.path.exists(ann_path):
            continue
        with open(ann_path) as f:
            ann = json.load(f)
        if not ann.get("objects"):
            continue
        page = pages[i]
        gt_knot = knot_mask_from_ann(ann)
        gt_pith_pt = extract_pith_point(ann)

        has_knot = gt_knot.sum() > 0
        has_pith = gt_pith_pt is not None
        d_knot = dice(pred_knot[i], gt_knot) if has_knot else float("nan")

        # pith centroid: predicted pith MASK -> centroid
        if has_pith and pred_pith[i].sum() > 0:
            ys, xs = np.where(pred_pith[i])
            cx, cy = xs.mean(), ys.mean()
            pith_dist = float(np.hypot(cx - gt_pith_pt[0], cy - gt_pith_pt[1]))
        else:
            pith_dist = float("nan")

        # also report the stage1 prompt's accuracy on this slice
        had_stage1_knot = i in knot_prompts
        had_stage1_pith = i in pith_prompts

        rows.append(
            {
                "page": page,
                "gt_knot_px": int(gt_knot.sum()),
                "pred_knot_px": int(pred_knot[i].sum()),
                "dice_knot": d_knot,
                "gt_has_pith": has_pith,
                "pith_centroid_dist_px": pith_dist,
                "stage1_knot_seeded": had_stage1_knot,
                "stage1_pith_seeded": had_stage1_pith,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_slice.csv", index=False)

    # summary
    knot_df = df[df.dice_knot.notna()]
    pith_df = df[df.pith_centroid_dist_px.notna()]
    print(f"\n=== ds{args.subset}, conf {args.conf}, {len(df)} annotated pages ===")
    print(
        f"knot: n={len(knot_df)}  mean Dice={knot_df.dice_knot.mean():.3f}  median={knot_df.dice_knot.median():.3f}  p10={knot_df.dice_knot.quantile(0.1):.3f}"
    )
    if len(pith_df):
        print(
            f"pith: n={len(pith_df)}  mean dist={pith_df.pith_centroid_dist_px.mean():.2f}px  median={pith_df.pith_centroid_dist_px.median():.2f}px  p90={pith_df.pith_centroid_dist_px.quantile(0.9):.2f}px"
        )

    # render per-slice overlays for annotated pages
    n_anno = len(df)
    cols = min(4, n_anno) if n_anno else 1
    gridrows = (n_anno + cols - 1) // cols if n_anno else 1
    fig, axes = plt.subplots(gridrows, cols, figsize=(5 * cols, 5 * gridrows))
    axes = np.array(axes).reshape(-1)
    for ax, (_, r) in zip(axes, df.iterrows()):
        page = int(r["page"])
        i = pages.index(page)
        ax.imshow(vol_gray[i], cmap="gray")
        # GT knot mask (orange overlay)
        ann_path = join(ann_dir, f"page_{page:03d}.tiff.json")
        with open(ann_path) as f:
            ann = json.load(f)
        gt_knot = knot_mask_from_ann(ann)
        if gt_knot.any():
            ax.imshow(
                np.ma.masked_where(~gt_knot.astype(bool), gt_knot),
                alpha=0.3,
                cmap="autumn",
                vmin=0,
                vmax=1,
            )
        # predicted knot outline (cyan)
        for c in _mask_contours(pred_knot[i]):
            ax.plot(c[:, 0], c[:, 1], color="cyan", linewidth=1.2)
        # predicted pith outline (orange)
        for c in _mask_contours(pred_pith[i]):
            ax.plot(c[:, 0], c[:, 1], color="orange", linewidth=1.2)
        title = f"p{page} dice={r['dice_knot']:.2f}" if not np.isnan(r["dice_knot"]) else f"p{page} no_knot"
        if not np.isnan(r["pith_centroid_dist_px"]):
            title += f" pith={r['pith_centroid_dist_px']:.1f}px"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[n_anno:]:
        ax.axis("off")
    fig.suptitle(
        f"YOLO->SAM-image (per-slice prompts) -> SAM-video (propagation).  ds{args.subset}, conf>={args.conf}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "overlays.png", dpi=100)
    plt.close()

    # save the volumes for later inspection
    np.savez_compressed(
        out_dir / "volume.npz",
        imgs=vol_gray,
        pred_knot=pred_knot,
        pred_pith=pred_pith,
        pages=np.array(pages),
    )
    print(f"\nresults: {out_dir / 'overlays.png'}")
    print(f"csv:     {out_dir / 'per_slice.csv'}")


def _mask_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.squeeze(1) for c in contours if len(c) > 2]


if __name__ == "__main__":
    main()
