"""Build and upload a Phase 1 dataset for the Timber project.

Every frame gets 8 timber objects carrying their persistent numeric id as the
class title. Anchor frames (the 32 human-annotated ones) keep the original
human masks verbatim; the rest get propagated detector masks.

Writes a local Supervisely-format export first, then uploads only with
--upload so the output can be inspected beforehand.

Auth comes from $SUPERVISELY_TOKEN (and optional $SUPERVISELY_SERVER).

Run:
    python -m ann_pipeline.timber.upload_phase1               # local export only
    python -m ann_pipeline.timber.upload_phase1 --upload
"""

import argparse
import json
import os
from os.path import join
import pathlib
import shutil
from typing import Dict, List, Optional

import numpy as np
import supervisely as sly
import torch
from tqdm import tqdm

from ann_pipeline.timber.data import (
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SUBSET,
    timber_instances_from_ann,
)
from ann_pipeline.timber.track import propagate
from src.utils.mask import mask_to_base64

DEFAULT_PROJECT_ID = 382882
DEFAULT_DATASET_NAME = "Phase 1"
HUMAN_LABELER = "human"
AUTO_LABELER = "auto-pipeline"


def bitmap_with_origin(mask: np.ndarray) -> Optional[Dict]:
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    patch = mask[y_min : y_max + 1, x_min : x_max + 1].astype(bool)
    return {"origin": [x_min, y_min], "data": mask_to_base64(torch.from_numpy(patch))}


def make_bitmap_object(class_title: str, mask: np.ndarray, labeler: str) -> Optional[Dict]:
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
        "labelerLogin": labeler,
    }


def build_annotation(masks: Dict[str, np.ndarray], height: int, width: int, labeler: str) -> Dict:
    objects = []
    for class_title in sorted(masks):
        obj = make_bitmap_object(class_title, masks[class_title], labeler)
        if obj is not None:
            objects.append(obj)
    return {
        "description": "",
        "tags": [],
        "size": {"height": height, "width": width},
        "objects": objects,
    }


def write_export(subset_dir: str, out_dir: pathlib.Path) -> List[str]:
    """Write img/ + ann/ for every frame. Returns the frame names in order."""
    labelled, df = propagate(subset_dir)
    img_out = out_dir / "img"
    ann_out = out_dir / "ann"
    img_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    anchors = set(df[df.is_anchor].frame)
    frames = sorted(labelled)
    for fname in tqdm(frames, desc="writing export"):
        src_img = join(subset_dir, "img", fname)
        shutil.copyfile(src_img, img_out / fname)

        if fname in anchors:
            with open(join(subset_dir, "ann", fname + ".json")) as fh:
                source = json.load(fh)
            masks = timber_instances_from_ann(source)
            height = source["size"]["height"]
            width = source["size"]["width"]
            labeler = HUMAN_LABELER
        else:
            masks = labelled[fname]
            any_mask = next(iter(masks.values()))
            height, width = any_mask.shape
            labeler = AUTO_LABELER

        ann = build_annotation(masks, height, width, labeler)
        with open(ann_out / f"{fname}.json", "w") as fh:
            json.dump(ann, fh)
    return frames


def ensure_classes(api: sly.Api, project_id: int, class_titles: List[str]) -> None:
    meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    missing = [t for t in class_titles if meta.get_obj_class(t) is None]
    if missing:
        msg = "project %d is missing classes: %s" % (project_id, ", ".join(missing))
        raise ValueError(msg)


def upload(
    export_dir: pathlib.Path,
    frames: List[str],
    project_id: int,
    dataset_name: str,
    server: str,
    token: str,
) -> int:
    api = sly.Api(server_address=server, token=token)
    me = api.user.get_my_info()
    print("connected to %s as %s" % (server, me.login))

    existing = api.dataset.get_info_by_name(project_id, dataset_name)
    if existing is not None:
        msg = "dataset %r already exists in project %d (id=%d); rename or remove it first" % (
            dataset_name,
            project_id,
            existing.id,
        )
        raise ValueError(msg)

    class_titles = sorted({
        obj["classTitle"]
        for fname in frames
        for obj in json.load(open(export_dir / "ann" / f"{fname}.json"))["objects"]
    })
    ensure_classes(api, project_id, class_titles)

    dataset = api.dataset.create(project_id, dataset_name)
    print("created dataset %r id=%d" % (dataset_name, dataset.id))

    batch = 20
    uploaded = 0
    for start in tqdm(range(0, len(frames), batch), desc="uploading"):
        chunk = frames[start : start + batch]
        paths = [str(export_dir / "img" / f) for f in chunk]
        infos = api.image.upload_paths(dataset.id, chunk, paths)
        anns = [json.load(open(export_dir / "ann" / f"{f}.json")) for f in chunk]
        api.annotation.upload_jsons([i.id for i in infos], anns)
        uploaded += len(chunk)
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--out_dir", default="/tmp/timber_phase1")
    parser.add_argument("--project_id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--server", default=os.environ.get("SUPERVISELY_SERVER", "https://app.supervisely.com"))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    subset_dir = join(args.project_root, args.subset)
    out_dir = pathlib.Path(args.out_dir)
    frames = write_export(subset_dir, out_dir)

    counts = {}
    for fname in frames:
        with open(out_dir / "ann" / f"{fname}.json") as fh:
            counts[fname] = len(json.load(fh)["objects"])
    n_full = sum(1 for v in counts.values() if v == 8)
    print("\nexport: %s" % out_dir)
    print("frames: %d   with 8 objects: %d" % (len(frames), n_full))
    short = {k: v for k, v in counts.items() if v != 8}
    if short:
        print("frames with != 8 objects: %s" % short)

    if not args.upload:
        print("\ndry run; pass --upload to create the dataset on the server")
        return

    token = os.environ.get("SUPERVISELY_TOKEN")
    if not token:
        msg = "SUPERVISELY_TOKEN not set; source .env first"
        raise ValueError(msg)
    n = upload(out_dir, frames, args.project_id, args.dataset_name, args.server, token)
    print("uploaded %d images to %r" % (n, args.dataset_name))


if __name__ == "__main__":
    main()
