"""Phase-1 combined experiment: per-instance knot tracking + multi-anchor
seeding at pages [10, 14], dropping the empty page 18 anchor that hurt the
previous multi-anchor run. Each knot connected component on each anchor
slice gets its own obj_id; matching knots across anchor slices share obj_ids
when they overlap spatially (greedy IoU > 0 between anchor pairs).

Pith is seeded at every anchor frame as a single point prompt with obj_id
PITH_OBJ_ID.

Run from repo root:
    conda run -n medsam python -m experiments.sm2025_combined.run
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
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_combined/out"

ANCHOR_PAGES: List[int] = [10, 14]
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


def split_instances(binary_mask: np.ndarray) -> List[np.ndarray]:
    labels = measure.label(binary_mask, connectivity=2)
    return [(labels == lab).astype(np.uint8) for lab in range(1, labels.max() + 1)]


def match_instances(prev: List[np.ndarray], curr: List[np.ndarray], min_iou: float = 0.05) -> Dict[int, int]:
    """Greedy IoU-based matching: returns curr_idx -> prev_idx (or -1 if no match)."""
    matches: Dict[int, int] = {}
    used_prev = set()
    pairs = []
    for ci, cm in enumerate(curr):
        for pi, pm in enumerate(prev):
            inter = np.logical_and(cm, pm).sum()
            union = np.logical_or(cm, pm).sum()
            if union == 0:
                continue
            iou = inter / union
            if iou >= min_iou:
                pairs.append((iou, ci, pi))
    pairs.sort(reverse=True)
    for iou, ci, pi in pairs:
        if ci in matches or pi in used_prev:
            continue
        matches[ci] = pi
        used_prev.add(pi)
    for ci in range(len(curr)):
        if ci not in matches:
            matches[ci] = -1
    return matches


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

    # split each anchor's knot mask into instances and assign global obj_ids
    knot_inst_per_anchor: Dict[int, List[np.ndarray]] = {}
    for fi in anchor_local_frames:
        binary = (gts[fi] == KNOT_CLASS_ID).astype(np.uint8)
        knot_inst_per_anchor[fi] = split_instances(binary)
        print(
            f"anchor local={fi} page={pages[fi]}: "
            f"{len(knot_inst_per_anchor[fi])} knot instances "
            f"({[int(m.sum()) for m in knot_inst_per_anchor[fi]]})"
        )

    # assign global obj_ids by matching across anchors (start from the central anchor)
    central_local = middle_anchor_local
    inst_obj_ids: Dict[int, List[int]] = {}
    inst_obj_ids[central_local] = list(range(1, len(knot_inst_per_anchor[central_local]) + 1))
    next_id = max(inst_obj_ids[central_local], default=0) + 1
    for fi in anchor_local_frames:
        if fi == central_local:
            continue
        matches = match_instances(knot_inst_per_anchor[central_local], knot_inst_per_anchor[fi])
        ids = []
        for ci in range(len(knot_inst_per_anchor[fi])):
            pi = matches[ci]
            if pi == -1:
                ids.append(next_id)
                next_id += 1
            else:
                ids.append(inst_obj_ids[central_local][pi])
        inst_obj_ids[fi] = ids
        print(f"anchor local={fi} obj_ids={ids} (matched against central anchor)")

    all_knot_obj_ids = sorted({oid for ids in inst_obj_ids.values() for oid in ids})

    h, w = vol.shape[1], vol.shape[2]
    video = prepare_video(vol, size=512)

    pred = np.zeros_like(vol, dtype=np.uint8)
    pred_inst = np.zeros_like(vol, dtype=np.uint16)

    repo_root = os.path.abspath(join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty", "MedSAM2"))
    model_cfg = "//" + join(repo_root, "sam2/configs/sam2.1_hiera_t512.yaml")
    predictor = build_sam2_video_predictor_npz(model_cfg, args.checkpoint)

    def seed(state) -> None:
        for fi in anchor_local_frames:
            for inst_mask, oid in zip(knot_inst_per_anchor[fi], inst_obj_ids[fi]):
                predictor.add_new_mask(inference_state=state, frame_idx=fi, obj_id=oid, mask=inst_mask)
            pts = pith_points_per_frame[fi]
            if pts:
                arr = np.array(pts, dtype=np.float32)
                labels = np.ones(len(arr), dtype=np.int32)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=fi,
                    obj_id=PITH_OBJ_ID,
                    points=arr,
                    labels=labels,
                )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # forward
        state = predictor.init_state(video, h, w)
        seed(state)
        for out_fi, obj_ids, out_logits in predictor.propagate_in_video(
            state, start_frame_idx=middle_anchor_local, reverse=False
        ):
            for k, oid in enumerate(obj_ids):
                m = (out_logits[k] > 0.0).cpu().numpy()[0]
                pred_inst[out_fi][m] = oid
                if oid in all_knot_obj_ids:
                    pred[out_fi][m] = KNOT_CLASS_ID
                elif oid == PITH_OBJ_ID:
                    pred[out_fi][m] = PITH_CLASS_ID

        # reverse
        predictor.reset_state(state)
        state = predictor.init_state(video, h, w)
        seed(state)
        for out_fi, obj_ids, out_logits in predictor.propagate_in_video(
            state, start_frame_idx=middle_anchor_local, reverse=True
        ):
            for k, oid in enumerate(obj_ids):
                m = (out_logits[k] > 0.0).cpu().numpy()[0]
                pred_inst[out_fi][m] = oid
                if oid in all_knot_obj_ids:
                    pred[out_fi][m] = KNOT_CLASS_ID
                elif oid == PITH_OBJ_ID:
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
        is_anchor = i in anchor_local_frames
        for cls_name, cid in [("Knot", KNOT_CLASS_ID), ("Pith", PITH_CLASS_ID)]:
            rows.append({
                "page": p,
                "class": cls_name,
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
            np.ma.masked_where(pred_inst[i] == 0, pred_inst[i]),
            alpha=0.6,
            cmap="tab10",
            vmin=0,
            vmax=10,
        )
        marker = " (ANCHOR)" if is_anchor else ""
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
