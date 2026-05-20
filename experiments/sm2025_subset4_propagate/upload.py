"""Build Supervisely-format annotations from the propagated result.npz and
optionally upload them as a new dataset.

We treat the propagation output as the source of truth. For each frame:
  * Wood and Knot are emitted as bitmap masks (Wood mask MINUS Pith blob).
  * Pith is emitted as a single point at the centroid of the predicted Pith blob.
    If the model produced no Pith blob, no Pith object is emitted for that frame.

Multi-instance Knot: the propagated `pred` volume merges all knots into one
class label, so each connected component of the knot mask becomes a separate
Supervisely "Knot" object.

Run from repo root (dry-run -> /tmp first, then add --upload):
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.upload \\
        --out_dir /tmp/sm2025_subset4_anchors \\
        --upload --project_id 376641 --dataset_name auto_3_anchors_from_rosta
"""

import argparse
import json
import os
from os.path import join
import pathlib
import shutil
from typing import List, Optional

import numpy as np
from scipy import ndimage as ndi
from src.utils.mask import mask_to_base64
import torch
from tqdm import tqdm

CLASS_IDS = {"Wood": 1, "Knot": 2, "Pith": 3}

CLASS_REGISTRY = [
    {"title": "Knot", "shape": "any", "color": "#FF9D00"},
    {"title": "Wood", "shape": "any", "color": "#86FF00"},
    {"title": "Pith", "shape": "point", "color": "#FF03D6"},
]

SOURCE_IMG_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025/4/img"


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


def build_annotation_for_frame(pred_frame: np.ndarray, h: int, w: int) -> dict:
    objects: List[dict] = []

    pith_mask = pred_frame == CLASS_IDS["Pith"]
    knot_mask = pred_frame == CLASS_IDS["Knot"]
    wood_mask = (pred_frame == CLASS_IDS["Wood"]) | pith_mask | knot_mask

    obj = make_bitmap_object("Wood", wood_mask.astype(np.uint8))
    if obj is not None:
        objects.append(obj)

    labelled, n = ndi.label(knot_mask, structure=np.ones((3, 3), dtype=np.uint8))
    for k in range(1, n + 1):
        comp = (labelled == k).astype(np.uint8)
        if comp.sum() < 8:
            continue
        obj = make_bitmap_object("Knot", comp)
        if obj is not None:
            objects.append(obj)

    if pith_mask.any():
        ys, xs = np.nonzero(pith_mask)
        objects.append(make_point_object("Pith", float(xs.mean()), float(ys.mean())))

    return {"size": {"height": h, "width": w}, "description": "", "tags": [], "objects": objects}


def write_export(out_dir: pathlib.Path, dataset_name: str, pages: np.ndarray, pred: np.ndarray) -> None:
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

    h, w = pred.shape[1], pred.shape[2]
    for i, p in enumerate(tqdm(pages, desc="encoding")):
        fname = "page_%03d.tiff" % int(p)
        src_img = join(SOURCE_IMG_DIR, fname)
        shutil.copy(src_img, img_out / fname)
        ann = build_annotation_for_frame(pred[i], h, w)
        (ann_out / ("%s.json" % fname)).write_text(json.dumps(ann))


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

    meta_json = api.project.get_meta(project.id)
    project_meta = sly.ProjectMeta.from_json(meta_json)
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
        project.id, dataset_name, description="auto-generated via MedSAM2 propagation from anchor annotations"
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
    parser.add_argument("--out_dir", default="/tmp/sm2025_subset4_anchors")
    parser.add_argument("--dataset_name", default="auto_3_anchors_from_rosta")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--project_id", type=int, default=376641)
    parser.add_argument("--server", default=os.environ.get("SUPERVISELY_SERVER", "https://app.supervisely.com"))
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    pages = data["pages"]
    pred = data["pred"]
    print("loaded %d frames" % len(pages))

    out_dir = pathlib.Path(args.out_dir)
    write_export(out_dir, args.dataset_name, pages, pred)
    print("wrote local export to %s" % out_dir)

    if args.upload:
        token = os.environ.get("SUPERVISELY_TOKEN")
        if not token:
            msg = "SUPERVISELY_TOKEN not set; source /home/mary/code/ct-log/.env first"
            raise ValueError(msg)
        upload(out_dir / args.dataset_name, args.dataset_name, args.project_id, args.server, token)


if __name__ == "__main__":
    main()
