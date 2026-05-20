"""YOLO -> SAM2 video predictor with box+point prompts on every annotated frame.

Unlike `yolo_to_sam_video.py` (which used `add_new_mask`), this script feeds
`add_new_points_or_box(box=..., points=...)` on every frame where YOLO detected
something. SAM2's video predictor then computes the per-frame masks fresh,
using cross-frame memory attention — so the temporal context can actually
influence the masks.

Run from repo root:
    conda run -n ct-log python -m ann_pipeline.scripts.yolo_to_sam_video_boxpoint \
        --subset 1 --page_min 7 --page_max 26
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
from sam2.build_sam import build_sam2_video_predictor_npz
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
    parser.add_argument("--page_min", type=int, default=7)
    parser.add_argument("--page_max", type=int, default=26)
    parser.add_argument(
        "--knot_prompt",
        choices=["box", "point", "box+point"],
        default="box+point",
    )
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/yolo_to_sam_video_boxpoint")
    args = parser.parse_args()

    sm_dir = join(PROJECT_ROOT, args.subset)
    img_dir = join(sm_dir, "img")
    ann_dir = join(sm_dir, "ann")

    all_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".tiff"))
    page_files = [
        f for f in all_files if args.page_min <= int(f.replace("page_", "").replace(".tiff", "")) <= args.page_max
    ]
    pages = [int(f.replace("page_", "").replace(".tiff", "")) for f in page_files]
    d = len(pages)

    out_dir = pathlib.Path(args.out_dir) / f"ds{args.subset}_{args.page_min}-{args.page_max}_{args.knot_prompt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"subset {args.subset}: {d} pages (p{args.page_min}-p{args.page_max}), knot_prompt={args.knot_prompt}")

    # --- Stage 1: run YOLO on every frame, collect prompts ---
    print("running YOLO ...")
    yolo = YOLO(args.yolo_weights)

    vol_gray = np.zeros((d, 778, 778), dtype=np.uint8)
    h0, w0 = 778, 778
    knot_prompts: Dict[int, np.ndarray] = {}  # frame_idx -> bboxes (N, 4)
    pith_prompts: Dict[int, np.ndarray] = {}  # frame_idx -> bboxes (M, 4)
    for i, fname in enumerate(tqdm(page_files, desc="yolo")):
        img_path = join(img_dir, fname)
        rgb = load_tiff_rgb(img_path)
        vol_gray[i] = load_tiff_gray(img_path)
        res = yolo.predict(rgb, conf=args.conf, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.empty((0,), int)
        knot_bbs = xyxy[cls == 0]
        pith_bbs = xyxy[cls == 1]
        if len(knot_bbs):
            knot_prompts[i] = knot_bbs
        if len(pith_bbs):
            pith_prompts[i] = pith_bbs

    del yolo
    torch.cuda.empty_cache()
    print(
        f"yolo done: {sum(len(v) for v in knot_prompts.values())} knot boxes, "
        f"{sum(len(v) for v in pith_prompts.values())} pith boxes"
    )

    # --- Stage 2: video predictor with box+point prompts on every detected frame ---
    print("loading SAM2 video predictor ...")
    video_predictor = build_sam2_video_predictor_npz(MODEL_CFG, CHECKPOINT)
    video = prepare_video(vol_gray, size=512)

    pred_knot = np.zeros((d, h0, w0), dtype=np.uint8)
    pred_pith = np.zeros((d, h0, w0), dtype=np.uint8)

    seeded_frames = sorted(set(knot_prompts) | set(pith_prompts))
    start_frame = int(np.median(seeded_frames))
    print(f"start frame: {start_frame}")

    def seed(state):
        # KNOT: one obj_id per bbox (single semantic class but multiple instances per frame).
        # To stay single-class as you requested, we union the per-instance masks AFTER
        # propagation — here we use a separate obj_id per bbox so SAM can compute each
        # cleanly with its own prompt, but later we merge them.
        oid = 100
        for frame_idx, bbs in knot_prompts.items():
            for b in bbs:
                cx = float((b[0] + b[2]) / 2.0)
                cy = float((b[1] + b[3]) / 2.0)
                kwargs = {"inference_state": state, "frame_idx": frame_idx, "obj_id": oid}
                if args.knot_prompt in ("box", "box+point"):
                    kwargs["box"] = np.array(b, dtype=np.float32)
                if args.knot_prompt in ("point", "box+point"):
                    kwargs["points"] = np.array([[cx, cy]], dtype=np.float32)
                    kwargs["labels"] = np.array([1], dtype=np.int32)
                video_predictor.add_new_points_or_box(**kwargs)
                oid += 1
        # PITH: same — one obj_id per bbox; we keep these in a different ID range
        oid = 1000
        for frame_idx, bbs in pith_prompts.items():
            for b in bbs:
                cx = float((b[0] + b[2]) / 2.0)
                cy = float((b[1] + b[3]) / 2.0)
                video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=oid,
                    points=np.array([[cx, cy]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )
                oid += 1

    def merge_output(out_fi: int, obj_ids: list, out_logits: torch.Tensor) -> None:
        for k, oid in enumerate(obj_ids):
            m = (out_logits[k] > 0.0).cpu().numpy()[0]
            if 100 <= oid < 1000:
                pred_knot[out_fi] |= m.astype(np.uint8)
            elif oid >= 1000:
                pred_pith[out_fi] |= m.astype(np.uint8)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = video_predictor.init_state(video, h0, w0)
        seed(state)
        for out_fi, obj_ids, out_logits in video_predictor.propagate_in_video(
            state, start_frame_idx=start_frame, reverse=False
        ):
            merge_output(out_fi, obj_ids, out_logits)
        video_predictor.reset_state(state)
        state = video_predictor.init_state(video, h0, w0)
        seed(state)
        for out_fi, obj_ids, out_logits in video_predictor.propagate_in_video(
            state, start_frame_idx=start_frame, reverse=True
        ):
            merge_output(out_fi, obj_ids, out_logits)
        video_predictor.reset_state(state)

    # --- Eval ---
    rows = []
    for i, fname in enumerate(page_files):
        ann_path = join(ann_dir, fname + ".json")
        if not os.path.exists(ann_path):
            continue
        with open(ann_path) as f:
            ann = json.load(f)
        gt_knot = knot_mask_from_ann(ann)
        gt_pith_pt = extract_pith_point(ann)
        has_knot = gt_knot.sum() > 0
        d_knot = dice(pred_knot[i], gt_knot) if has_knot else float("nan")

        if gt_pith_pt is not None and pred_pith[i].sum() > 0:
            ys, xs = np.where(pred_pith[i])
            cx, cy = xs.mean(), ys.mean()
            pith_dist = float(np.hypot(cx - gt_pith_pt[0], cy - gt_pith_pt[1]))
        else:
            pith_dist = float("nan")

        rows.append(
            {
                "page": pages[i],
                "gt_knot_px": int(gt_knot.sum()),
                "pred_knot_px": int(pred_knot[i].sum()),
                "dice_knot": d_knot,
                "gt_has_pith": gt_pith_pt is not None,
                "pith_centroid_dist_px": pith_dist,
                "had_knot_prompt": i in knot_prompts,
                "had_pith_prompt": i in pith_prompts,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_slice.csv", index=False)
    valid = df[df.dice_knot.notna()]
    print(f"\n=== ds{args.subset} pages {args.page_min}-{args.page_max} ({args.knot_prompt}) ===")
    print(
        f"knot: n={len(valid)}  mean={valid.dice_knot.mean():.3f}  "
        f"median={valid.dice_knot.median():.3f}  p10={valid.dice_knot.quantile(0.1):.3f}"
    )
    pith_df = df[df.pith_centroid_dist_px.notna()]
    if len(pith_df):
        print(
            f"pith: n={len(pith_df)}  mean={pith_df.pith_centroid_dist_px.mean():.2f}px  "
            f"median={pith_df.pith_centroid_dist_px.median():.2f}px"
        )

    # render
    n_anno = len(df)
    cols = min(4, n_anno) if n_anno else 1
    gridrows = (n_anno + cols - 1) // cols if n_anno else 1
    fig, axes = plt.subplots(gridrows, cols, figsize=(5 * cols, 5 * gridrows))
    axes = np.array(axes).reshape(-1)
    for ax, (_, r) in zip(axes, df.iterrows()):
        page = int(r["page"])
        i = pages.index(page)
        ax.imshow(vol_gray[i], cmap="gray")
        ann_path = join(ann_dir, f"page_{page:03d}.tiff.json")
        with open(ann_path) as f:
            ann = json.load(f)
        gt_knot = knot_mask_from_ann(ann)
        if gt_knot.any():
            ax.imshow(np.ma.masked_where(~gt_knot.astype(bool), gt_knot), alpha=0.3, cmap="autumn", vmin=0, vmax=1)
        for c in _mask_contours(pred_knot[i]):
            ax.plot(c[:, 0], c[:, 1], color="cyan", linewidth=1.2)
        for c in _mask_contours(pred_pith[i]):
            ax.plot(c[:, 0], c[:, 1], color="orange", linewidth=1.2)
        title = f"p{page} dice={r['dice_knot']:.2f}" if not np.isnan(r["dice_knot"]) else f"p{page} no_knot"
        if not np.isnan(r["pith_centroid_dist_px"]):
            title += f"  pith={r['pith_centroid_dist_px']:.1f}px"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[n_anno:]:
        ax.axis("off")
    fig.suptitle(
        f"YOLO -> SAM2 video (box+point prompts, propagated).  "
        f"ds{args.subset} p{args.page_min}-p{args.page_max}, knot_prompt={args.knot_prompt}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "overlays.png", dpi=100)
    plt.close()
    print(f"\nvis: {out_dir / 'overlays.png'}")


def _mask_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.squeeze(1) for c in contours if len(c) > 2]


if __name__ == "__main__":
    main()
