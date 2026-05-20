"""Download images + annotations for a Supervisely dataset into the local layout.

Local layout (matches what `ann_pipeline.data` expects):

    <out_dir>/
        img/<image_name>            # original images, as stored on the server
        ann/<image_name>.json       # one annotation JSON per image

Auth comes from $SUPERVISELY_TOKEN (and optional $SUPERVISELY_SERVER).

Example:
    python -m ann_pipeline.scripts.fetch_supervisely_dataset \\
        --project_id 375492 --dataset_id 1137347 \\
        --out_dir /mnt/D/datasets/ct_log/375492_SM_2025/4 --overwrite
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import supervisely as sly
from tqdm import tqdm


def fetch_dataset(
    project_id: int,
    dataset_id: int,
    out_dir: Path,
    server: str,
    token: str,
    overwrite: bool,
) -> None:
    api = sly.Api(server_address=server, token=token)
    me = api.user.get_my_info()
    print("connected to %s as user_id=%d (login=%s)" % (server, me.id, me.login))

    project = api.project.get_info_by_id(project_id)
    if project is None:
        msg = "project %d not found" % project_id
        raise ValueError(msg)
    ds = api.dataset.get_info_by_id(dataset_id)
    if ds is None:
        msg = "dataset %d not found" % dataset_id
        raise ValueError(msg)
    if ds.project_id != project_id:
        msg = "dataset %d belongs to project %d, not %d" % (dataset_id, ds.project_id, project_id)
        raise ValueError(msg)
    print("project: id=%d name=%r" % (project.id, project.name))
    print("dataset: id=%d name=%r images=%d" % (ds.id, ds.name, ds.images_count))

    img_dir = out_dir / "img"
    ann_dir = out_dir / "ann"
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        msg = "%s is non-empty; pass --overwrite to refresh it" % out_dir
        raise ValueError(msg)
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    images: List[sly.ImageInfo] = api.image.get_list(ds.id)
    print("found %d images on the server" % len(images))

    ann_jsons = api.annotation.download_json_batch(ds.id, [img.id for img in images])
    name_to_ann = {info.name: ann for info, ann in zip(images, ann_jsons)}

    for img_info in tqdm(images, desc="downloading"):
        img_path = img_dir / img_info.name
        ann_path = ann_dir / ("%s.json" % img_info.name)
        api.image.download_path(img_info.id, str(img_path))
        with open(ann_path, "w") as f:
            json.dump(name_to_ann[img_info.name], f)

    print("done. wrote %d images to %s and %d annotations to %s" % (len(images), img_dir, len(images), ann_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=int, required=True)
    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--server", default=os.environ.get("SUPERVISELY_SERVER", "https://app.supervisely.com"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SUPERVISELY_TOKEN")
    if not token:
        msg = "SUPERVISELY_TOKEN not set; source /home/mary/code/ct-log/.env first"
        raise ValueError(msg)

    fetch_dataset(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        out_dir=args.out_dir,
        server=args.server,
        token=token,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
