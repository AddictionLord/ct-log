"""Phase-1 per-instance experiment: same single-anchor (page 14) seeding as
the slice14 baseline, but each connected-component knot gets its own obj_id
instead of being merged into one mask. Tests whether per-instance tracking
recovers Dice on frames where knots have heterogeneous z-extents.

Run from repo root:
    conda run -n medsam python -m experiments.sm2025_per_instance.run
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
from skimage import measure
import torch

SM_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025/1"
IMG_DIR = join(SM_DIR, "img")
ANN_DIR = join(SM_DIR, "ann")
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_per_instance/out"

SEED_PAGE = 14
PAGE_RANGE: Tuple[int, int] = (7, 26)
KNOT_CLASS_ID = 1
PITH_CLASS_ID = 2
PITH_OBJ_ID = 1000


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
            mask[oy : oy + ph, ox : ox + pw][patch] = KNOT_CLASS_ID
        elif cls == "Pith":
            for x, y in obj["points"]["exterior"]:
                pith_points.append((int(x), int(y)))
                cv2.circle(mask, (int(x), int(y)), 3, PITH_CLASS_ID, -1)
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


def split_into_instances(binary_mask: np.ndarray) -> List[np.ndarray]:
    """Return list of per-instance binary masks from a single-class binary mask."""
    labels = measure.label(binary_mask, connectivity=2)
    return [(labels == lab).astype(np.uint8) for lab in range(1, labels.max() + 1)]


def seed_state(
    predictor,
    state,
    seed_frame: int,
    knot_instance_masks: List[np.ndarray],
    pith_points: List[Tuple[int, int]],
) -> Tuple[List[int], int]:
    knot_obj_ids: List[int] = []
    for k, m in enumerate(knot_instance_masks, start=1):
        predictor.add_new_mask(inference_state=state, frame_idx=seed_frame, obj_id=k, mask=m)
        knot_obj_ids.append(k)
    if pith_points:
        arr = np.array(pith_points, dtype=np.float32)
        labels = np.ones(len(arr), dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=seed_frame,
            obj_id=PITH_OBJ_ID,
            points=arr,
            labels=labels,
        )
    return knot_obj_ids, PITH_OBJ_ID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt")
    args = parser.parse_args()

    pathlib.Path(OUT_DIR).mkdir(exist_ok=True, parents=True)
    pathlib.Path(join(OUT_DIR, "overlays")).mkdir(exist_ok=True, parents=True)

    start, end = PAGE_RANGE
    pages = list(range(start, end + 1))
    d = len(pages)
    seed_frame = pages.index(SEED_PAGE)

    vol = np.stack([load_tiff_gray(page_path(p, "img")) for p in pages])
    gts = np.zeros_like(vol, dtype=np.uint8)
    pith_points_per_frame: Dict[int, List[Tuple[int, int]]] = {}
    for i, p in enumerate(pages):
        gts[i], pts = render_annotation(page_path(p, "ann"))
        pith_points_per_frame[i] = pts

    seed_knot_binary = (gts[seed_frame] == KNOT_CLASS_ID).astype(np.uint8)
    knot_instances = split_into_instances(seed_knot_binary)
    seed_pith_points = pith_points_per_frame[seed_frame]
    print(
        f"seed frame {seed_frame} (page {SEED_PAGE}): "
        f"{len(knot_instances)} knot instances ({[int(m.sum()) for m in knot_instances]} px), "
        f"pith pts={len(seed_pith_points)}"
    )

    h, w = vol.shape[1], vol.shape[2]
    video = prepare_video(vol, size=512)

    # semantic pred for evaluation (collapsed instances back into class)
    pred = np.zeros_like(vol, dtype=np.uint8)
    # per-instance pred for inspection
    pred_inst = np.zeros_like(vol, dtype=np.uint16)

    repo_root = os.path.abspath(join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty", "MedSAM2"))
    model_cfg = "//" + join(repo_root, "sam2/configs/sam2.1_hiera_t512.yaml")
    predictor = build_sam2_video_predictor_npz(model_cfg, args.checkpoint)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # ---- forward ----
        state = predictor.init_state(video, h, w)
        knot_obj_ids, pith_obj_id = seed_state(predictor, state, seed_frame, knot_instances, seed_pith_points)
        for out_fi, obj_ids, out_logits in predictor.propagate_in_video(
            state, start_frame_idx=seed_frame, reverse=False
        ):
            for k, oid in enumerate(obj_ids):
                m = (out_logits[k] > 0.0).cpu().numpy()[0]
                pred_inst[out_fi][m] = oid
                if oid in knot_obj_ids:
                    pred[out_fi][m] = KNOT_CLASS_ID
                elif oid == pith_obj_id:
                    pred[out_fi][m] = PITH_CLASS_ID

        # ---- reverse ----
        predictor.reset_state(state)
        state = predictor.init_state(video, h, w)
        seed_state(predictor, state, seed_frame, knot_instances, seed_pith_points)
        for out_fi, obj_ids, out_logits in predictor.propagate_in_video(
            state, start_frame_idx=seed_frame, reverse=True
        ):
            for k, oid in enumerate(obj_ids):
                m = (out_logits[k] > 0.0).cpu().numpy()[0]
                pred_inst[out_fi][m] = oid
                if oid in knot_obj_ids:
                    pred[out_fi][m] = KNOT_CLASS_ID
                elif oid == pith_obj_id:
                    pred[out_fi][m] = PITH_CLASS_ID
        predictor.reset_state(state)

    np.savez_compressed(
        join(OUT_DIR, "result.npz"),
        imgs=vol,
        gts=gts,
        pred=pred,
        pred_inst=pred_inst,
        pages=np.array(pages),
    )

    rows = []
    for i, p in enumerate(pages):
        for cls_name, cid in [("Knot", KNOT_CLASS_ID), ("Pith", PITH_CLASS_ID)]:
            rows.append({
                "page": p,
                "class": cls_name,
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
            np.ma.masked_where(pred_inst[i] == 0, pred_inst[i]),
            alpha=0.6,
            cmap="tab10",
            vmin=0,
            vmax=10,
        )
        marker = " (SEED)" if i == seed_frame else ""
        ax[2].set_title(f"Pred per-instance{marker}")
        ax[2].axis("off")
        plt.tight_layout()
        plt.savefig(join(OUT_DIR, "overlays", f"page_{p:03d}.png"), dpi=100)
        plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(join(OUT_DIR, "dice.csv"), index=False)
    print(df.pivot_table(index="page", columns="class", values="dice").round(3))


if __name__ == "__main__":
    main()
