"""Combined annotation pipeline for subset 4: YOLO + propagation hybrid (v2).

Inputs:
  * propagated label volume (`result.npz` from `run.py`)
  * YOLO 2-class weights (e.g. `yolo11n_v2_all45`)
  * SAM2 image-predictor checkpoint (turns YOLO bboxes into masks)
  * threshold wood detector (already used in `ann_pipeline.wood`)

Per-frame rule:
  * Knot = YOLO+SAM2 (NMS iou=0.5); each YOLO knot box -> SAM2 mask, one
    Supervisely Knot object per detection. Falls back to nothing if YOLO did
    not detect any knots (eval showed propagation knots have low F1 and add
    noise more than they help).
  * Pith = YOLO bbox centroid at conf=0.10 (precision is 1.0 on val at this
    threshold, recall is the bottleneck); fallback to propagation pith blob
    centroid if YOLO didn't fire.
  * Wood = largest_cc(threshold ∪ propagation_wood ∪ knot_mask ∪ pith).
  * Disagreement flag: if both YOLO and propagation produced a pith and they
    are >= τ_flag apart, prepend "[REVIEW: pith_disagreement]" to the
    annotation's description. τ_flag computed inline (mean + 3σ).

Run from repo root (dry-run, no upload):
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.build_combined_annotations \\
        --out_dir /tmp/sm2025_subset4_combined_v2

Add `--upload --project_id 376641 --dataset_name auto_4_combined_v2` to push.
"""

import argparse
import json
import os
from os.path import join
import pathlib
import shutil
from typing import Dict, List, Optional, Tuple

from ann_pipeline.wood.detectors import threshold_largest_cc
import cv2
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy import ndimage as ndi
from src.utils.mask import mask_to_base64
import torch
from tqdm import tqdm
from ultralytics import YOLO

CLASS_IDS = {"Wood": 1, "Knot": 2, "Pith": 3}

CLASS_REGISTRY = [
    {"title": "Knot", "shape": "any", "color": "#FF9D00"},
    {"title": "Wood", "shape": "any", "color": "#86FF00"},
    {"title": "Pith", "shape": "point", "color": "#FF03D6"},
]

SOURCE_IMG_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025/4/img"
DEFAULT_YOLO_WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_v2_all45/weights/best.pt"
DEFAULT_YOLO_OBB_WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_obb_v1/weights/best.pt"
SAM_CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
SAM_MODEL_CFG = "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"


def bitmap_with_origin(mask: np.ndarray) -> Optional[dict]:
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    patch = mask[y_min : y_max + 1, x_min : x_max + 1].astype(bool)
    b64 = mask_to_base64(torch.from_numpy(patch))
    return {"origin": [int(x_min), int(y_min)], "data": b64}


def make_bitmap_object(class_title: str, mask: np.ndarray) -> Optional[dict]:
    bmp = bitmap_with_origin(mask)
    if bmp is None:
        return None
    return {
        "classTitle": class_title,
        "description": "",
        "tags": [],
        "bitmap": bmp,
        "shape": "bitmap",
        "geometryType": "bitmap",
        "labelerLogin": "auto-pipeline",
    }


def make_point_object(class_title: str, x: float, y: float) -> dict:
    return {
        "classTitle": class_title,
        "description": "",
        "tags": [],
        "points": {"exterior": [[int(round(x)), int(round(y))]], "interior": []},
        "shape": "point",
        "geometryType": "point",
        "labelerLogin": "auto-pipeline",
    }


def load_gray(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    return arr.astype(np.uint8)


def load_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def yolo_pith(model: YOLO, img_rgb: np.ndarray, conf: float) -> Optional[Tuple[float, float]]:
    """Return centroid of the highest-confidence pith bbox, or None."""
    res = model.predict(img_rgb, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    pith_idx = np.where(cls == 1)[0]
    if len(pith_idx) == 0:
        return None
    best = pith_idx[np.argmax(confs[pith_idx])]
    b = xyxy[best]
    return float((b[0] + b[2]) / 2.0), float((b[1] + b[3]) / 2.0)


def yolo_sam_knots(
    yolo_model: YOLO,
    sam_predictor: SAM2ImagePredictor,
    img_rgb: np.ndarray,
    conf: float,
    nms_iou: float,
) -> List[np.ndarray]:
    """Return per-instance knot masks via YOLO (NMS iou=nms_iou) -> SAM2."""
    res = yolo_model.predict(img_rgb, conf=conf, iou=nms_iou, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    knot_boxes = [b for b, c in zip(xyxy, cls) if c == 0]
    if not knot_boxes:
        return []
    masks: List[np.ndarray] = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(img_rgb)
        for box in knot_boxes:
            m, _, _ = sam_predictor.predict(box=np.array(box, dtype=np.float32), multimask_output=False)
            masks.append(m[0].astype(bool))
    return masks


def _low_res_mask(binary_mask: np.ndarray, size: int = 128) -> np.ndarray:
    low = cv2.resize(binary_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    logits = np.where(low > 0, 10.0, -10.0).astype(np.float32)
    return logits[None]


def _clip_to_bbox(mask: np.ndarray, aabb: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    x1, y1, x2, y2 = aabb.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def yolo_obb_sam_knots(
    yolo_obb_model: YOLO,
    sam_predictor: SAM2ImagePredictor,
    img_rgb: np.ndarray,
    conf: float,
    nms_iou: float,
) -> List[np.ndarray]:
    """Return per-instance knot masks via YOLO-OBB (oriented boxes) -> SAM2 with
    rasterised-OBB mask_input prior + AABB-of-OBB box prompt + bbox clip."""
    res = yolo_obb_model.predict(img_rgb, conf=conf, iou=nms_iou, verbose=False)[0]
    if res.obb is None or len(res.obb) == 0:
        return []
    xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()
    h, w = img_rgb.shape[:2]
    masks: List[np.ndarray] = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(img_rgb)
        for corners in xyxyxyxy:
            obb_raster = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(obb_raster, [corners.astype(np.int32)], 1)
            low = _low_res_mask(obb_raster.astype(bool))
            x1 = max(0, int(np.floor(corners[:, 0].min())))
            y1 = max(0, int(np.floor(corners[:, 1].min())))
            x2 = min(w, int(np.ceil(corners[:, 0].max())))
            y2 = min(h, int(np.ceil(corners[:, 1].max())))
            aabb = np.array([x1, y1, x2, y2], dtype=np.float32)
            m, _, _ = sam_predictor.predict(box=aabb, mask_input=low, multimask_output=False)
            mask = m[0].astype(bool)
            masks.append(_clip_to_bbox(mask, aabb))
    return masks


def largest_cc(mask: np.ndarray) -> np.ndarray:
    labelled, n = ndi.label(mask.astype(np.uint8))
    if n == 0:
        return mask.astype(np.uint8)
    sizes = ndi.sum(mask, labelled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return (labelled == largest).astype(np.uint8)


def build_annotation(
    img_gray: np.ndarray,
    img_rgb: np.ndarray,
    pred_frame: np.ndarray,
    yolo_model: YOLO,
    yolo_obb_model: Optional[YOLO],
    sam_predictor: SAM2ImagePredictor,
    wood_thresh: int,
    yolo_conf_pith: float,
    yolo_conf_knot: float,
    yolo_nms_iou: float,
    tau_flag: float,
) -> Tuple[dict, Dict[str, object]]:
    """Returns (Supervisely-format ann dict, per-frame metadata for logging)."""
    h, w = img_gray.shape
    objects: List[dict] = []

    prop_wood = pred_frame == CLASS_IDS["Wood"]
    prop_pith = pred_frame == CLASS_IDS["Pith"]

    if yolo_obb_model is not None:
        knot_masks = yolo_obb_sam_knots(yolo_obb_model, sam_predictor, img_rgb, yolo_conf_knot, yolo_nms_iou)
    else:
        knot_masks = yolo_sam_knots(yolo_model, sam_predictor, img_rgb, yolo_conf_knot, yolo_nms_iou)
    combined_knot_mask = np.zeros((h, w), dtype=bool)
    for km in knot_masks:
        combined_knot_mask |= km

    thresh_wood = threshold_largest_cc(img_gray, thresh=wood_thresh).astype(bool)
    wood_union = prop_wood | combined_knot_mask | prop_pith | thresh_wood
    wood_final = largest_cc(wood_union)

    obj = make_bitmap_object("Wood", wood_final)
    if obj is not None:
        objects.append(obj)

    for km in knot_masks:
        if km.sum() < 8:
            continue
        obj = make_bitmap_object("Knot", km.astype(np.uint8))
        if obj is not None:
            objects.append(obj)

    yolo_xy = yolo_pith(yolo_model, img_rgb, conf=yolo_conf_pith)
    prop_xy: Optional[Tuple[float, float]] = None
    if prop_pith.any():
        ys, xs = np.nonzero(prop_pith)
        prop_xy = (float(xs.mean()), float(ys.mean()))

    pith_source = "none"
    description = ""
    final_pith: Optional[Tuple[float, float]] = None
    pith_disagreement_px: Optional[float] = None
    if yolo_xy is not None and prop_xy is not None:
        pith_disagreement_px = float(np.hypot(yolo_xy[0] - prop_xy[0], yolo_xy[1] - prop_xy[1]))
        final_pith = yolo_xy
        pith_source = "yolo"
        if pith_disagreement_px >= tau_flag:
            description = "[REVIEW: pith_disagreement=%.1fpx]" % pith_disagreement_px
    elif yolo_xy is not None:
        final_pith = yolo_xy
        pith_source = "yolo"
        description = "[REVIEW: pith_propagation_missing]"
    elif prop_xy is not None:
        final_pith = prop_xy
        pith_source = "propagation"
        description = "[REVIEW: pith_yolo_missing]"
    else:
        pith_source = "none"
        description = "[REVIEW: pith_missing]"

    if final_pith is not None:
        objects.append(make_point_object("Pith", *final_pith))

    ann = {
        "size": {"height": h, "width": w},
        "description": description,
        "tags": [],
        "objects": objects,
    }
    meta = {
        "yolo_xy": yolo_xy,
        "prop_xy": prop_xy,
        "pith_source": pith_source,
        "pith_disagreement_px": pith_disagreement_px,
        "n_knots": len(knot_masks),
    }
    return ann, meta


def compute_tau_flag(
    pages: np.ndarray, pred: np.ndarray, yolo_model: YOLO, yolo_conf_pith: float
) -> Tuple[float, List[Tuple[int, float]]]:
    """Compute mean+3σ of YOLO-vs-prop pith distance on frames where both fire."""
    distances: List[Tuple[int, float]] = []
    for i, p in enumerate(pages):
        prop_pith_mask = pred[i] == CLASS_IDS["Pith"]
        if not prop_pith_mask.any():
            continue
        ys, xs = np.nonzero(prop_pith_mask)
        prop_xy = (float(xs.mean()), float(ys.mean()))
        img_rgb = load_rgb(join(SOURCE_IMG_DIR, "page_%03d.tiff" % int(p)))
        yolo_xy = yolo_pith(yolo_model, img_rgb, conf=yolo_conf_pith)
        if yolo_xy is None:
            continue
        d = float(np.hypot(yolo_xy[0] - prop_xy[0], yolo_xy[1] - prop_xy[1]))
        distances.append((int(p), d))
    if not distances:
        return 20.0, distances
    arr = np.array([d for _, d in distances])
    tau = float(arr.mean() + 3 * arr.std())
    return tau, distances


def write_export(
    out_dir: pathlib.Path,
    dataset_name: str,
    pages: np.ndarray,
    pred: np.ndarray,
    yolo_model: YOLO,
    yolo_obb_model: Optional[YOLO],
    sam_predictor: SAM2ImagePredictor,
    wood_thresh: int,
    yolo_conf_pith: float,
    yolo_conf_knot: float,
    yolo_nms_iou: float,
    tau_flag: float,
) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    project_meta = {
        "classes": [
            {
                "title": c["title"],
                "description": "",
                "shape": c["shape"],
                "color": c["color"],
                "geometry_config": {},
                "hotkey": "",
            }
            for c in CLASS_REGISTRY
        ],
        "tags": [],
        "projectType": "images",
        "projectSettings": {"multiView": {"enabled": False, "tagName": None, "tagId": None, "isSynced": False}},
    }
    (out_dir / "meta.json").write_text(json.dumps(project_meta, indent=2))

    ds_dir = out_dir / dataset_name
    img_out = ds_dir / "img"
    ann_out = ds_dir / "ann"
    img_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    metas = []
    for i, p in enumerate(tqdm(pages, desc="encoding")):
        fname = "page_%03d.tiff" % int(p)
        src_img = join(SOURCE_IMG_DIR, fname)
        shutil.copy(src_img, img_out / fname)
        img_gray = load_gray(src_img)
        img_rgb = load_rgb(src_img)
        ann, meta = build_annotation(
            img_gray,
            img_rgb,
            pred[i],
            yolo_model,
            yolo_obb_model,
            sam_predictor,
            wood_thresh,
            yolo_conf_pith,
            yolo_conf_knot,
            yolo_nms_iou,
            tau_flag,
        )
        meta["page"] = int(p)
        metas.append(meta)
        (ann_out / ("%s.json" % fname)).write_text(json.dumps(ann))
    return metas


def upload(local_ds_dir: pathlib.Path, dataset_name: str, project_id: int, server: str, token: str) -> None:
    import supervisely as sly

    api = sly.Api(server_address=server, token=token)
    me = api.user.get_my_info()
    print("connected to %s as user_id=%d (login=%s)" % (server, me.id, me.login))
    project = api.project.get_info_by_id(project_id)
    if project is None:
        msg = "project %d not found" % project_id
        raise ValueError(msg)
    print("target project: id=%d name=%r" % (project.id, project.name))
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project.id))
    server_titles = {cls.name for cls in project_meta.obj_classes}
    our_titles = {c["title"] for c in CLASS_REGISTRY}
    missing = our_titles - server_titles
    if missing:
        msg = "target project is missing classes: %s" % missing
        raise ValueError(msg)
    print("class check OK; server has %s" % sorted(server_titles))
    existing = api.dataset.get_info_by_name(project.id, dataset_name)
    if existing is not None:
        msg = "dataset %r already exists in project %d (id=%d); delete via GUI first" % (
            dataset_name,
            project_id,
            existing.id,
        )
        raise ValueError(msg)
    ds = api.dataset.create(
        project.id,
        dataset_name,
        description="auto-generated combined pipeline (YOLO pith + SAM propagation wood/knot + threshold-union wood)",
    )
    print("created dataset id=%d name=%r" % (ds.id, ds.name))
    img_dir = local_ds_dir / "img"
    ann_dir = local_ds_dir / "ann"
    image_paths = sorted(img_dir.glob("*.tiff"))
    names = [p.name for p in image_paths]
    paths = [str(p) for p in image_paths]
    print("uploading %d images + annotations ..." % len(image_paths))
    img_infos = api.image.upload_paths(ds.id, names, paths)
    image_ids = [info.id for info in img_infos]
    ann_paths = [str(ann_dir / ("%s.json" % name)) for name in names]
    api.annotation.upload_paths(image_ids, ann_paths)
    print("upload done")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/result.npz")
    parser.add_argument("--out_dir", default="/tmp/sm2025_subset4_combined_v2")
    parser.add_argument("--dataset_name", default="auto_4_combined_v2")
    parser.add_argument("--yolo_weights", default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument(
        "--yolo_obb_weights",
        default=None,
        help="If set, use a YOLO-OBB model for knots (with SAM2 mask_input prior). "
        "Recommended: %s" % DEFAULT_YOLO_OBB_WEIGHTS,
    )
    parser.add_argument("--wood_thresh", type=int, default=30)
    parser.add_argument("--yolo_conf_pith", type=float, default=0.10)
    parser.add_argument("--yolo_conf_knot", type=float, default=0.25)
    parser.add_argument("--yolo_nms_iou", type=float, default=0.5)
    parser.add_argument("--tau_flag", type=float, default=None, help="If unset, computed from data (mean+3σ).")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--project_id", type=int, default=376641)
    parser.add_argument("--server", default=os.environ.get("SUPERVISELY_SERVER", "https://app.supervisely.com"))
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    pages = data["pages"]
    pred = data["pred"]
    print("loaded %d frames from %s" % (len(pages), args.npz))

    yolo_model = YOLO(args.yolo_weights)
    yolo_obb_model = YOLO(args.yolo_obb_weights) if args.yolo_obb_weights else None
    if yolo_obb_model is not None:
        print("using YOLO-OBB knot model: %s" % args.yolo_obb_weights)
    print("building SAM2 image predictor ...")
    sam = build_sam2("//" + SAM_MODEL_CFG, SAM_CHECKPOINT)
    sam_predictor = SAM2ImagePredictor(sam)

    if args.tau_flag is None:
        print("computing τ_flag from YOLO-vs-propagation distance distribution ...")
        tau_flag, distances = compute_tau_flag(pages, pred, yolo_model, args.yolo_conf_pith)
        if distances:
            arr = np.array([d for _, d in distances])
            print(
                "  n_frames_with_both=%d mean=%.2f std=%.2f median=%.2f max=%.2f px -> τ_flag=%.2f"
                % (len(distances), arr.mean(), arr.std(), float(np.median(arr)), arr.max(), tau_flag)
            )
        else:
            print("  no frames had both YOLO and propagation pith; defaulting τ_flag=%.2f" % tau_flag)
    else:
        tau_flag = args.tau_flag
        print("using user-provided τ_flag=%.2f" % tau_flag)

    out_dir = pathlib.Path(args.out_dir)
    metas = write_export(
        out_dir,
        args.dataset_name,
        pages,
        pred,
        yolo_model,
        yolo_obb_model,
        sam_predictor,
        args.wood_thresh,
        args.yolo_conf_pith,
        args.yolo_conf_knot,
        args.yolo_nms_iou,
        tau_flag,
    )

    n_yolo = sum(1 for m in metas if m["pith_source"] == "yolo")
    n_prop = sum(1 for m in metas if m["pith_source"] == "propagation")
    n_none = sum(1 for m in metas if m["pith_source"] == "none")
    n_disagree = sum(
        1 for m in metas if m["pith_disagreement_px"] is not None and m["pith_disagreement_px"] >= tau_flag
    )
    n_knots_total = sum(int(m.get("n_knots", 0)) for m in metas)
    n_frames_with_knots = sum(1 for m in metas if int(m.get("n_knots", 0)) > 0)
    print(
        "\npith summary: yolo=%d propagation=%d none=%d (flagged_disagreement=%d, τ=%.1f)"
        % (n_yolo, n_prop, n_none, n_disagree, tau_flag)
    )
    print(
        "knot summary: %d knots across %d/%d frames (mean %.2f/frame)"
        % (n_knots_total, n_frames_with_knots, len(metas), n_knots_total / max(1, len(metas)))
    )

    if args.upload:
        token = os.environ.get("SUPERVISELY_TOKEN")
        if not token:
            msg = "SUPERVISELY_TOKEN not set; source /home/mary/code/ct-log/.env first"
            raise ValueError(msg)
        upload(out_dir / args.dataset_name, args.dataset_name, args.project_id, args.server, token)


if __name__ == "__main__":
    main()
