"""Build a Supervisely-format annotation set for a full subset volume using
the production pipeline (YOLO + SAM-image box+point for knots/pith, intensity
threshold for wood).

Two modes:
  --dry_run (default): just write the annotation JSON + image files to disk
    in the Supervisely project layout. Nothing touches the server.
  --upload: push to a Supervisely workspace as a new dataset (requires
    SUPERVISELY_TOKEN env var).

Run from repo root:
    conda run -n ct-log python -m ann_pipeline.scripts.build_supervisely_annotations \\
        --subset 3 --out_dir /tmp/sm_export_test --dry_run
"""

import argparse
import json
import os
from os.path import join
import pathlib
from typing import List, Optional

import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.utils.mask import mask_to_base64
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ann_pipeline.wood.detectors import threshold_largest_cc

PROJECT_ROOT = "/mnt/D/datasets/ct_log/375492_SM_2025"
YOLO_WEIGHTS = "/home/mary/code/ct-log/ann_pipeline/out/knot_runs/yolo11n_2cls_v1/weights/best.pt"
CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
MODEL_CFG = "//" + "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"

# Class registry — these must match the live Supervisely project. We use the
# same hex colours / class IDs the existing project uses (visible in meta.json
# at the project root).
CLASS_REGISTRY = [
    {"title": "Knot", "shape": "any", "color": "#FF9D00"},
    {"title": "Wood", "shape": "any", "color": "#86FF00"},
    {"title": "Pith", "shape": "point", "color": "#FF03D6"},
]


def load_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3]
    else:
        arr = np.stack([arr] * 3, axis=-1)
    return arr.astype(np.uint8)


def load_gray(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    return arr.astype(np.uint8)


def bitmap_with_origin(mask: np.ndarray) -> Optional[dict]:
    """Convert a full-canvas binary mask into a Supervisely bitmap object with
    a tight `origin` offset. Returns None if the mask is empty."""
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    patch = mask[y_min : y_max + 1, x_min : x_max + 1].astype(bool)
    b64 = mask_to_base64(torch.from_numpy(patch))
    return {
        "origin": [int(x_min), int(y_min)],
        "data": b64,
    }


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
        "points": {"exterior": [[int(x), int(y)]], "interior": []},
        "shape": "point",
        "geometryType": "point",
        "labelerLogin": "auto-pipeline",
    }


def predict_slice(
    img_rgb: np.ndarray,
    img_gray: np.ndarray,
    yolo: YOLO,
    sam: SAM2ImagePredictor,
    conf: float = 0.25,
    wood_thresh: int = 30,
) -> dict:
    """Run the full pipeline on a single slice. Returns a Supervisely-style
    ann dict for that slice (without the outer `size`/`description` shell)."""
    h, w = img_rgb.shape[:2]
    objects: List[dict] = []

    # 1. Wood: intensity threshold
    wood_mask = threshold_largest_cc(img_gray, thresh=wood_thresh)
    obj = make_bitmap_object("Wood", wood_mask)
    if obj is not None:
        objects.append(obj)

    # 2. YOLO -> bboxes for knot & pith
    res = yolo.predict(img_rgb, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return {"size": {"height": h, "width": w}, "objects": objects, "tags": []}
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    knot_bbs = xyxy[cls == 0]
    pith_bbs = xyxy[cls == 1]

    # 3. SAM2 image predictor for masks
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam.set_image(img_rgb)
        for b in knot_bbs:
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            m, _, _ = sam.predict(
                box=np.array(b, dtype=np.float32),
                point_coords=np.array([[cx, cy]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=False,
            )
            obj = make_bitmap_object("Knot", m[0].astype(np.uint8))
            if obj is not None:
                objects.append(obj)

        # 4. Pith: emit as a single point (the YOLO bbox centroid) — matches
        # the existing project's "Pith" class shape, which is `point`.
        for b in pith_bbs:
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            objects.append(make_point_object("Pith", cx, cy))

    return {
        "size": {"height": h, "width": w},
        "description": "",
        "tags": [],
        "objects": objects,
    }


def write_supervisely_export(
    out_dir: pathlib.Path,
    subset_name: str,
    page_files: List[str],
    img_dir: str,
    yolo: YOLO,
    sam: SAM2ImagePredictor,
    conf: float,
    wood_thresh: int,
) -> None:
    """Write a self-contained Supervisely project at out_dir."""
    # Project-level files
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

    ds_dir = out_dir / subset_name
    img_out = ds_dir / "img"
    ann_out = ds_dir / "ann"
    img_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    for fname in tqdm(page_files, desc=f"{subset_name}"):
        img_path = join(img_dir, fname)
        rgb = load_rgb(img_path)
        gray = load_gray(img_path)
        ann = predict_slice(rgb, gray, yolo, sam, conf=conf, wood_thresh=wood_thresh)
        # copy the image as-is (Supervisely accepts TIFF)
        import shutil

        shutil.copy(img_path, img_out / fname)
        (ann_out / f"{fname}.json").write_text(json.dumps(ann))


def upload_dataset_to_project(
    local_dataset_dir: pathlib.Path,
    dataset_name: str,
    project_id: int,
    server: str,
    token: str,
) -> None:
    """Push a single local dataset directory into an EXISTING Supervisely project.

    The local layout we expect:
        local_dataset_dir/
            img/<name>.tiff
            ann/<name>.tiff.json

    We do NOT modify the project's meta.json (classes etc.) — it must already
    contain the Knot/Wood/Pith class definitions matching CLASS_REGISTRY here.
    """
    import supervisely as sly

    api = sly.Api(server_address=server, token=token)
    me = api.user.get_my_info()
    print(f"connected to {server} as user_id={me.id} (login={me.login})")

    project = api.project.get_info_by_id(project_id)
    if project is None:
        raise SystemExit(f"project {project_id} not found")
    print(f"target project: id={project.id} name={project.name!r} workspace_id={project.workspace_id}")

    # confirm class set in the target project matches what we generated
    meta_json = api.project.get_meta(project.id)
    project_meta = sly.ProjectMeta.from_json(meta_json)
    our_titles = {c["title"] for c in CLASS_REGISTRY}
    server_titles = {cls.name for cls in project_meta.obj_classes}
    missing = our_titles - server_titles
    if missing:
        raise SystemExit(
            f"target project is missing classes: {missing}. add them via the GUI (matching CLASS_REGISTRY) and re-run."
        )
    print(f"class check OK; server has {sorted(server_titles)}")

    # create (or fetch) the dataset
    existing = api.dataset.get_info_by_name(project.id, dataset_name)
    if existing is not None:
        raise SystemExit(
            f"dataset '{dataset_name}' already exists in project {project_id} (id={existing.id}). "
            "delete it via the GUI or choose a different --dataset_name."
        )
    ds = api.dataset.create(project.id, dataset_name, description="auto-generated by ann_pipeline")
    print(f"created dataset id={ds.id} name={ds.name!r}")

    img_dir = local_dataset_dir / "img"
    ann_dir = local_dataset_dir / "ann"
    image_paths = sorted(img_dir.glob("*.tiff"))
    print(f"uploading {len(image_paths)} images + annotations ...")

    names = [p.name for p in image_paths]
    paths = [str(p) for p in image_paths]
    img_infos = api.image.upload_paths(ds.id, names, paths)
    image_ids = [info.id for info in img_infos]

    ann_paths = [str(ann_dir / f"{name}.json") for name in names]
    api.annotation.upload_paths(image_ids, ann_paths)
    print("upload done")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="3", help="Subset id under PROJECT_ROOT to process.")
    parser.add_argument("--out_dir", default="/tmp/ct-log-supervisely-export")
    parser.add_argument("--subset_name", default=None, help="Name for the output dataset (default: same as --subset).")
    parser.add_argument("--max_pages", type=int, default=0)
    parser.add_argument("--page_min", type=int, default=None, help="inclusive lower page bound")
    parser.add_argument("--page_max", type=int, default=None, help="inclusive upper page bound")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--wood_thresh", type=int, default=30)
    parser.add_argument("--dry_run", action="store_true", default=True)
    parser.add_argument(
        "--upload", action="store_true", help="Actually push to Supervisely after writing the local export."
    )
    parser.add_argument("--project_id", type=int, default=None, help="Existing Supervisely project ID to push into.")
    parser.add_argument(
        "--dataset_name", default=None, help="Name for the new dataset inside the project (default: auto-<subset>)."
    )
    parser.add_argument("--server", default="https://app.supervisely.com")
    args = parser.parse_args()

    if args.upload and args.project_id is None:
        raise SystemExit("--upload requires --project_id")

    subset_id = args.subset
    subset_name = args.subset_name or subset_id
    sm_dir = join(PROJECT_ROOT, subset_id)
    img_dir = join(sm_dir, "img")
    if not os.path.isdir(img_dir):
        raise SystemExit(f"no such directory: {img_dir}")
    page_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".tiff"))
    if args.page_min is not None or args.page_max is not None:
        lo = args.page_min if args.page_min is not None else -1
        hi = args.page_max if args.page_max is not None else 10**9
        page_files = [f for f in page_files if lo <= int(f.replace("page_", "").replace(".tiff", "")) <= hi]
    if args.max_pages > 0:
        page_files = page_files[: args.max_pages]
    print(f"subset {subset_id}: {len(page_files)} pages to process")

    print("loading YOLO ...")
    yolo = YOLO(YOLO_WEIGHTS)
    print("loading SAM2 image predictor ...")
    sam_model = build_sam2(MODEL_CFG, CHECKPOINT)
    sam = SAM2ImagePredictor(sam_model)

    out_dir = pathlib.Path(args.out_dir)
    write_supervisely_export(
        out_dir=out_dir,
        subset_name=subset_name,
        page_files=page_files,
        img_dir=img_dir,
        yolo=yolo,
        sam=sam,
        conf=args.conf,
        wood_thresh=args.wood_thresh,
    )
    print(f"\nlocal Supervisely export written to {out_dir}")
    print(f"  project meta: {out_dir / 'meta.json'}")
    print(f"  dataset:      {out_dir / subset_name}")

    if args.upload:
        token = os.environ.get("SUPERVISELY_TOKEN")
        if not token:
            raise SystemExit("SUPERVISELY_TOKEN env var not set")
        dataset_name = args.dataset_name or f"auto-{subset_id}"
        upload_dataset_to_project(
            local_dataset_dir=out_dir / subset_name,
            dataset_name=dataset_name,
            project_id=args.project_id,
            server=args.server,
            token=token,
        )


if __name__ == "__main__":
    main()
