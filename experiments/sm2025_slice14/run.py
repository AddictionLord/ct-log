"""Phase-1 experiment: seed MedSAM2 from a single annotated slice (page_014) and
propagate Knot + Pith over slices 7-26, then compare against the GT annotations
present on every slice in that range.

Run from repo root:
    conda run -n medsam python -m experiments.sm2025_slice14.run
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
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_slice14/out"

SEED_PAGE = 14
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
    """Supervisely bitmap: base64 -> zlib -> PNG (RGBA). Returns alpha as bool."""
    raw = zlib.decompress(base64.b64decode(b64))
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., 3] > 0
    return arr > 0


def render_annotation(ann_path: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Returns (semantic_mask[H,W] uint8 with values in {0, 1=Knot, 2=Pith}, pith_points[(x,y)]).

    Pith is rendered as a 3-px filled disk in the semantic mask (so we have something
    to compare model output against). The explicit point list is also returned for
    use as the SAM2 prompt.
    """
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
    """vol: (D, H, W) uint8 -> tensor (D, 3, size, size) float32, ImageNet-normed."""
    d, h, w = vol.shape
    out = np.zeros((d, 3, size, size), dtype=np.float32)
    for i in range(d):
        im = Image.fromarray(vol[i]).convert("RGB").resize((size, size))
        out[i] = np.array(im).transpose(2, 0, 1)
    out /= 255.0
    t = torch.from_numpy(out).cuda()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].cuda()
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].cuda()
    t = (t - mean) / std
    return t


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt")
    parser.add_argument("--cfg", default="//sam2/configs/sam2.1_hiera_t512.yaml")
    args = parser.parse_args()

    pathlib.Path(OUT_DIR).mkdir(exist_ok=True, parents=True)
    pathlib.Path(join(OUT_DIR, "overlays")).mkdir(exist_ok=True, parents=True)

    start, end = PAGE_RANGE
    pages = list(range(start, end + 1))
    d = len(pages)
    seed_frame = pages.index(SEED_PAGE)
    print(f"loaded {d} frames; seed frame local idx = {seed_frame} (page {SEED_PAGE})")

    vol = np.stack([load_tiff_gray(page_path(p, "img")) for p in pages])
    print("vol:", vol.shape, vol.dtype)

    gts = np.zeros_like(vol, dtype=np.uint8)
    pith_points_per_frame: Dict[int, List[Tuple[int, int]]] = {}
    for i, p in enumerate(pages):
        gts[i], pts = render_annotation(page_path(p, "ann"))
        pith_points_per_frame[i] = pts

    seed_mask = gts[seed_frame]
    seed_points = pith_points_per_frame[seed_frame]
    print(f"seed slice: knot px={(seed_mask == 1).sum()}, pith points={len(seed_points)}")

    h, w = vol.shape[1], vol.shape[2]
    video = prepare_video(vol, size=512)

    pred = np.zeros_like(vol, dtype=np.uint8)

    # absolute path so Hydra finds the YAML
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(join(script_dir, "..", "..", "thirdparty", "MedSAM2"))
    model_cfg = "//" + join(repo_root, "sam2/configs/sam2.1_hiera_t512.yaml")
    print("cfg:", model_cfg)
    predictor = build_sam2_video_predictor_npz(model_cfg, args.checkpoint)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video, h, w)

        # ---- Knot: mask prompt at seed frame, obj_id=1 ----
        knot_mask = (seed_mask == CLASS_IDS["Knot"]).astype(np.uint8)
        # rescale mask to 512 to match the predictor's working resolution? The
        # video_predictor API takes masks in original (h, w) space, it handles
        # resizing internally; verify by inspecting the call.
        if knot_mask.sum() > 0:
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=seed_frame,
                obj_id=1,
                mask=knot_mask,
            )
            print(f"added knot mask: {knot_mask.sum()} px")

        # ---- Pith: point prompt at seed frame, obj_id=2 ----
        if len(seed_points) > 0:
            pts = np.array(seed_points, dtype=np.float32)
            labels = np.ones(len(pts), dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=seed_frame,
                obj_id=2,
                points=pts,
                labels=labels,
            )
            print(f"added pith point prompt: {seed_points}")

        # ---- forward propagate from seed ----
        for out_fi, obj_ids, out_logits in predictor.propagate_in_video(
            state, start_frame_idx=seed_frame, reverse=False
        ):
            for k, oid in enumerate(obj_ids):
                m = (out_logits[k] > 0.0).cpu().numpy()[0]
                pred[out_fi][m] = oid

        # ---- reverse propagate ----
        predictor.reset_state(state)
        state = predictor.init_state(video, h, w)
        if knot_mask.sum() > 0:
            predictor.add_new_mask(inference_state=state, frame_idx=seed_frame, obj_id=1, mask=knot_mask)
        if len(seed_points) > 0:
            pts = np.array(seed_points, dtype=np.float32)
            labels = np.ones(len(pts), dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=seed_frame,
                obj_id=2,
                points=pts,
                labels=labels,
            )
        for out_fi, obj_ids, out_logits in predictor.propagate_in_video(
            state, start_frame_idx=seed_frame, reverse=True
        ):
            for k, oid in enumerate(obj_ids):
                m = (out_logits[k] > 0.0).cpu().numpy()[0]
                pred[out_fi][m] = oid
        predictor.reset_state(state)

    np.savez_compressed(join(OUT_DIR, "result.npz"), imgs=vol, gts=gts, pred=pred, pages=np.array(pages))

    # per-slice Dice + side-by-side overlay
    rows = []
    for i, p in enumerate(pages):
        for cls, cid in CLASS_IDS.items():
            rows.append({
                "page": p,
                "class": cls,
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
        seed_marker = " (SEED)" if i == seed_frame else ""
        ax[2].set_title(f"Pred{seed_marker}")
        ax[2].axis("off")
        plt.tight_layout()
        plt.savefig(join(OUT_DIR, "overlays", f"page_{p:03d}.png"), dpi=100)
        plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(join(OUT_DIR, "dice.csv"), index=False)
    print(df.pivot_table(index="page", columns="class", values="dice").round(3))


if __name__ == "__main__":
    main()
