"""Propagate sparse Supervisely annotations across the whole subset 4 volume.

Subset 4 has 292 frames (pages 7-298) but only 16 are annotated. We seed
MedSAM2's video predictor at every annotated frame and propagate Wood, Knot and
Pith forward+backward to fill the gaps.

Strategy (memory-safe for 4 GB GPUs):
  * Split the volume into segments between consecutive anchors.
  * Within each segment [A, B], propagate forward from A and backward from B
    using a small local video (just frames A..B). For each in-between frame,
    keep the prediction from the closer anchor (ties broken by forward direction).
  * For the left tail (frames before the first anchor) we propagate backward
    from the first anchor; for the right tail, forward from the last anchor.

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.run
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
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor_npz
import torch

SUBSET_ID = os.environ.get("CT_SUBSET", "4")
SM_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025/%s" % SUBSET_ID
IMG_DIR = join(SM_DIR, "img")
ANN_DIR = join(SM_DIR, "ann")
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out"

CLASS_IDS: Dict[str, int] = {"Wood": 1, "Knot": 2, "Pith": 3}


def list_pages() -> List[int]:
    pages = []
    for fn in os.listdir(IMG_DIR):
        if fn.startswith("page_") and fn.endswith(".tiff"):
            pages.append(int(fn[5:-5]))
    return sorted(pages)


def page_img_path(p: int) -> str:
    return join(IMG_DIR, "page_%03d.tiff" % p)


def page_ann_path(p: int) -> str:
    return join(ANN_DIR, "page_%03d.tiff.json" % p)


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


def render_annotation(ann_path: str, h: int, w: int) -> Tuple[Dict[str, np.ndarray], List[Tuple[int, int]]]:
    """Decode an annotation file into per-class binary masks + a list of pith points."""
    with open(ann_path) as f:
        ann = json.load(f)
    masks: Dict[str, np.ndarray] = {c: np.zeros((h, w), dtype=np.uint8) for c in CLASS_IDS}
    pith_points: List[Tuple[int, int]] = []
    for obj in ann.get("objects", []):
        cls = obj["classTitle"]
        if cls in {"Wood", "Knot"} and "bitmap" in obj:
            bmp = obj["bitmap"]
            ox, oy = bmp["origin"]
            patch = decode_bitmap(bmp["data"])
            ph, pw = patch.shape
            masks[cls][oy : oy + ph, ox : ox + pw][patch] = 1
        elif cls == "Pith" and "points" in obj:
            for x, y in obj["points"]["exterior"]:
                pith_points.append((int(x), int(y)))
    return masks, pith_points


def find_anchor_pages(pages: List[int]) -> List[int]:
    anchors = []
    for p in pages:
        with open(page_ann_path(p)) as f:
            ann = json.load(f)
        if ann.get("objects"):
            anchors.append(p)
    return anchors


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


def seed_state(
    predictor,
    state,
    local_frame: int,
    masks: Dict[str, np.ndarray],
    pith_points: List[Tuple[int, int]],
) -> None:
    for cls_name, cid in CLASS_IDS.items():
        if cls_name == "Pith":
            if pith_points:
                arr = np.array(pith_points, dtype=np.float32)
                labels = np.ones(len(arr), dtype=np.int32)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=local_frame,
                    obj_id=cid,
                    points=arr,
                    labels=labels,
                )
        else:
            m = masks[cls_name]
            if m.sum() > 0:
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=local_frame,
                    obj_id=cid,
                    mask=m.astype(np.uint8),
                )


def propagate_segment(
    predictor,
    vol: np.ndarray,
    seed_masks_a: Dict[str, np.ndarray],
    seed_pith_a: List[Tuple[int, int]],
    seed_masks_b: Dict[str, np.ndarray],
    seed_pith_b: List[Tuple[int, int]],
    size: int,
) -> np.ndarray:
    """Propagate within a local volume seeded at frame 0 (anchor A) and frame -1 (anchor B).

    Returns a per-frame label volume; each frame in (0, len-1) gets the prediction from
    the closer anchor.
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
            seed_state(predictor, state, 0, seed_masks_a, seed_pith_a)
            for out_fi, obj_ids, out_logits in predictor.propagate_in_video(state, start_frame_idx=0, reverse=False):
                for k, oid in enumerate(obj_ids):
                    m = (out_logits[k] > 0.0).cpu().numpy()[0]
                    pred_fwd[out_fi][m] = oid
            predictor.reset_state(state)

        if d > 1 and has_b:
            state = predictor.init_state(video, h, w)
            seed_state(predictor, state, d - 1, seed_masks_b, seed_pith_b)
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
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--n_vis", type=int, default=12, help="Number of overlay frames to write")
    args = parser.parse_args()

    pathlib.Path(OUT_DIR).mkdir(exist_ok=True, parents=True)
    pathlib.Path(join(OUT_DIR, "overlays")).mkdir(exist_ok=True, parents=True)

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
        masks, pith = render_annotation(page_ann_path(p), h, w)
        anchor_masks[p] = masks
        anchor_pith[p] = pith

    repo_root = os.path.abspath(join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty", "MedSAM2"))
    model_cfg = "//" + join(repo_root, "sam2/configs/sam2.1_hiera_t512.yaml")
    predictor = build_sam2_video_predictor_npz(model_cfg, args.checkpoint)

    pred = np.zeros_like(vol, dtype=np.uint8)
    is_anchor = np.zeros(len(pages), dtype=bool)
    for ai in anchor_idx:
        is_anchor[ai] = True

    # left tail: frames before first anchor — reverse-propagate from first anchor
    first_a = anchor_idx[0]
    if first_a > 0:
        sub = vol[: first_a + 1]
        a_page = anchors[0]
        seg = propagate_segment(
            predictor,
            sub,
            seed_masks_a={c: np.zeros((h, w), np.uint8) for c in CLASS_IDS},
            seed_pith_a=[],
            seed_masks_b=anchor_masks[a_page],
            seed_pith_b=anchor_pith[a_page],
            size=args.size,
        )
        pred[: first_a + 1] = seg
        print("left tail: %d frames done" % (first_a + 1))

    # interior segments [a_i .. a_{i+1}]
    for s in range(len(anchor_idx) - 1):
        a, b = anchor_idx[s], anchor_idx[s + 1]
        sub = vol[a : b + 1]
        pa, pb = anchors[s], anchors[s + 1]
        seg = propagate_segment(
            predictor,
            sub,
            seed_masks_a=anchor_masks[pa],
            seed_pith_a=anchor_pith[pa],
            seed_masks_b=anchor_masks[pb],
            seed_pith_b=anchor_pith[pb],
            size=args.size,
        )
        pred[a : b + 1] = seg
        print("segment pages %d..%d (%d frames) done" % (pa, pb, b - a + 1))

    # right tail
    last_a = anchor_idx[-1]
    if last_a < len(pages) - 1:
        sub = vol[last_a:]
        pa = anchors[-1]
        seg = propagate_segment(
            predictor,
            sub,
            seed_masks_a=anchor_masks[pa],
            seed_pith_a=anchor_pith[pa],
            seed_masks_b=None,
            seed_pith_b=None,
            size=args.size,
        )
        pred[last_a:] = seg
        print("right tail: %d frames done" % (len(pages) - last_a))

    np.savez_compressed(
        join(OUT_DIR, "result.npz"),
        imgs=vol,
        pred=pred,
        pages=np.array(pages),
        is_anchor=is_anchor,
        anchors=np.array(anchors),
    )
    print("saved", join(OUT_DIR, "result.npz"))

    # visualizations: pick a spread of non-anchor frames + every anchor
    nonanchor_indices = [i for i in range(len(pages)) if not is_anchor[i]]
    step = max(1, len(nonanchor_indices) // args.n_vis)
    chosen = sorted(set(nonanchor_indices[::step][: args.n_vis]) | set(anchor_idx))
    cmap = plt.get_cmap("tab10")
    for i in chosen:
        p = pages[i]
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(vol[i], cmap="gray")
        ax[0].set_title("page_%03d %s" % (p, "(ANCHOR)" if is_anchor[i] else ""))
        ax[0].axis("off")

        ax[1].imshow(vol[i], cmap="gray")
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        pith_xy: Tuple[float, float] = None
        for cls_name, cid in CLASS_IDS.items():
            mask = pred[i] == cid
            if not mask.any():
                continue
            if cls_name == "Pith":
                ys, xs = np.nonzero(mask)
                pith_xy = (float(xs.mean()), float(ys.mean()))
                continue
            colour = np.array(cmap(cid - 1))
            overlay[mask] = (*colour[:3], 0.5)
        ax[1].imshow(overlay)
        if pith_xy is not None:
            colour = cmap(CLASS_IDS["Pith"] - 1)
            ax[1].scatter([pith_xy[0]], [pith_xy[1]], s=60, c=[colour], marker="x", linewidths=2)
        ax[1].set_title("pred (W=blue, K=orange, P=green x)")
        ax[1].axis("off")
        plt.tight_layout()
        plt.savefig(join(OUT_DIR, "overlays", "page_%03d.png" % p), dpi=110)
        plt.close()
    print("wrote %d overlay PNGs to %s" % (len(chosen), join(OUT_DIR, "overlays")))


if __name__ == "__main__":
    main()
