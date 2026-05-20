"""Phase-1 multi-anchor experiment: seed MedSAM2 from THREE annotated slices
(pages 10, 14, 18) instead of one, then compare against GT for the full
7-26 range. The hypothesis is that multiple anchors reduce false-positive
phantom knots that the single-anchor variant produced near the propagation
boundary.

Run from repo root:
    conda run -n medsam python -m experiments.sm2025_multi_anchor.run
"""

import argparse
import base64
import json
import os
from os.path import join
import pathlib
from typing import Dict, List, Tuple
import zlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor_npz
import torch

SM_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025/1"
IMG_DIR = join(SM_DIR, "img")
ANN_DIR = join(SM_DIR, "ann")
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_multi_anchor/out"

ANCHOR_PAGES: List[int] = [10, 14, 18]
PAGE_RANGE: Tuple[int, int] = (7, 26)
CLASS_IDS: Dict[str, int] = {"Knot": 1, "Pith": 2}


def page_path(page_idx: int, ext: str) -> str:
    name = f"page_{page_idx:03d}.tiff"
    if ext == "img":
        return join(IMG_DIR, name)
    return join(ANN_DIR, name + ".json")


def load_tiff_gray(path: str) -> np.ndarray:
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    return img.astype(np.uint8)


def decode_bitmap(b64: str) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(b64))
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., 3] > 0
    return arr > 0


def render_annotation(ann_path: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    with open(ann_path) as f:
        ann = json.load(f)
    h, w = ann["size"]["height"], ann["size"]["width"]
    mask = np.zeros((h, w), dtype=np.uint8)
    pith_points: List[Tuple[int, int]] = []
    for obj in ann.get("objects", []):
        cls = obj["classTitle"]
        if cls == "Knot":
            bmp = obj["bitmap"]
            ox, oy = bmp["origin"]
            patch = decode_bitmap(bmp["data"])
            ph, pw = patch.shape
            mask[oy : oy + ph, ox : ox + pw][patch] = CLASS_IDS["Knot"]
        elif cls == "Pith":
            for x, y in obj["points"]["exterior"]:
                pith_points.append((int(x), int(y)))
                cv2.circle(mask, (int(x), int(y)), 3, CLASS_IDS["Pith"], -1)
    return mask, pith_points


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


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


def seed_state(
    predictor,
    state,
    anchor_local_frames: List[int],
    knot_masks_per_anchor: Dict[int, np.ndarray],
    pith_points_per_anchor: Dict[int, List[Tuple[int, int]]],
) -> None:
    """Add knot mask + pith point at every anchor frame using the same obj_ids."""
    for frame_idx in anchor_local_frames:
        knot_mask = knot_masks_per_anchor[frame_idx]
        if knot_mask.sum() > 0:
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=1,
                mask=knot_mask,
            )
        pts = pith_points_per_anchor[frame_idx]
        if pts:
            arr = np.array(pts, dtype=np.float32)
            labels = np.ones(len(arr), dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=2,
                points=arr,
                labels=labels,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt")
    args = parser.parse_args()

    pathlib.Path(OUT_DIR).mkdir(exist_ok=True, parents=True)
    pathlib.Path(join(OUT_DIR, "overlays")).mkdir(exist_ok=True, parents=True)

    start, end = PAGE_RANGE
    pages = list(range(start, end + 1))
    d = len(pages)
    anchor_local_frames = [pages.index(p) for p in ANCHOR_PAGES]
    middle_anchor_local = anchor_local_frames[len(anchor_local_frames) // 2]
    print(f"loaded {d} frames; anchors local={anchor_local_frames} pages={ANCHOR_PAGES}")

    vol = np.stack([load_tiff_gray(page_path(p, "img")) for p in pages])
    gts = np.zeros_like(vol, dtype=np.uint8)
    pith_points_per_frame: Dict[int, List[Tuple[int, int]]] = {}
    for i, p in enumerate(pages):
        gts[i], pts = render_annotation(page_path(p, "ann"))
        pith_points_per_frame[i] = pts

    knot_masks_per_anchor = {fi: (gts[fi] == CLASS_IDS["Knot"]).astype(np.uint8) for fi in anchor_local_frames}
    pith_per_anchor = {fi: pith_points_per_frame[fi] for fi in anchor_local_frames}
    for fi in anchor_local_frames:
        print(
            f"anchor frame {fi} (page {pages[fi]}): knot px={knot_masks_per_anchor[fi].sum()}, "
            f"pith pts={len(pith_per_anchor[fi])}"
        )

    h, w = vol.shape[1], vol.shape[2]
    video = prepare_video(vol, size=512)

    pred = np.zeros_like(vol, dtype=np.uint8)
    repo_root = os.path.abspath(join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty", "MedSAM2"))
    model_cfg = "//" + join(repo_root, "sam2/configs/sam2.1_hiera_t512.yaml")
    predictor = build_sam2_video_predictor_npz(model_cfg, args.checkpoint)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # ---- forward propagate from middle anchor, all 3 anchors seeded ----
        state = predictor.init_state(video, h, w)
        seed_state(predictor, state, anchor_local_frames, knot_masks_per_anchor, pith_per_anchor)
        for out_fi, obj_ids, out_logits in predictor.propagate_in_video(
            state, start_frame_idx=middle_anchor_local, reverse=False
        ):
            for k, oid in enumerate(obj_ids):
                m = (out_logits[k] > 0.0).cpu().numpy()[0]
                pred[out_fi][m] = oid

        # ---- reverse propagate ----
        predictor.reset_state(state)
        state = predictor.init_state(video, h, w)
        seed_state(predictor, state, anchor_local_frames, knot_masks_per_anchor, pith_per_anchor)
        for out_fi, obj_ids, out_logits in predictor.propagate_in_video(
            state, start_frame_idx=middle_anchor_local, reverse=True
        ):
            for k, oid in enumerate(obj_ids):
                m = (out_logits[k] > 0.0).cpu().numpy()[0]
                pred[out_fi][m] = oid
        predictor.reset_state(state)

    np.savez_compressed(join(OUT_DIR, "result.npz"), imgs=vol, gts=gts, pred=pred, pages=np.array(pages))

    rows = []
    for i, p in enumerate(pages):
        is_anchor = i in anchor_local_frames
        for cls, cid in CLASS_IDS.items():
            rows.append({
                "page": p,
                "class": cls,
                "anchor": is_anchor,
                "dice": dice(pred[i] == cid, gts[i] == cid),
                "gt_px": int((gts[i] == cid).sum()),
                "pred_px": int((pred[i] == cid).sum()),
            })

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(vol[i], cmap="gray")
        ax[0].set_title(f"page_{p:03d}")
        ax[0].axis("off")
        ax[1].imshow(vol[i], cmap="gray")
        ax[1].imshow(np.ma.masked_where(gts[i] == 0, gts[i]), alpha=0.5, cmap="autumn", vmin=0, vmax=2)
        ax[1].set_title("GT")
        ax[1].axis("off")
        ax[2].imshow(vol[i], cmap="gray")
        ax[2].imshow(
            np.ma.masked_where(pred[i] == 0, pred[i]),
            alpha=0.5,
            cmap="autumn",
            vmin=0,
            vmax=2,
        )
        marker = " (ANCHOR)" if is_anchor else ""
        ax[2].set_title(f"Pred{marker}")
        ax[2].axis("off")
        plt.tight_layout()
        plt.savefig(join(OUT_DIR, "overlays", f"page_{p:03d}.png"), dpi=100)
        plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(join(OUT_DIR, "dice.csv"), index=False)
    print(df.pivot_table(index="page", columns="class", values="dice").round(3))


if __name__ == "__main__":
    main()
