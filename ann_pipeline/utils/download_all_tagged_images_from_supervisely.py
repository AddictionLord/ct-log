"""Download human-reviewed (tagged) frames from a Supervisely project to the local project root.

Writes the same on-disk layout the knot data-prep scripts consume:

    output_dir/{subset}/img/page_NNN.tiff       (raw image)
    output_dir/{subset}/ann/page_NNN.tiff.json  (Supervisely annotation JSON)

One Supervisely dataset maps to one local subset directory (dataset name ==
subset id). Only images carrying the tag (default ``bumaska``) are downloaded.
Existing local files are skipped by default so subsets already fully annotated
on disk (e.g. 1-4) are left untouched; pass --overwrite to refetch.

Auth comes from $SUPERVISELY_TOKEN (and optional $SUPERVISELY_SERVER).

Example:
    set -a && source .env && set +a
    python -m ann_pipeline.utils.download_all_tagged_images_from_supervisely \\
        --project-id 377328 \\
        --output-dir /mnt/D/datasets/ct_log/375492_SM_2025 \\
        --subsets 08 10
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import supervisely as sly

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_ID = 377328  # Phase2 (auto + human review)
DEFAULT_OUTPUT_DIR = "/mnt/D/datasets/ct_log/375492_SM_2025"
HUMAN_TAG = "bumaska"


def download_tagged_images(
    project_id: int,
    output_dir: Path,
    server: str,
    token: str,
    tag_name: str,
    subsets: Optional[List[str]],
    overwrite: bool,
) -> int:
    api = sly.Api(server_address=server, token=token)
    me = api.user.get_my_info()
    logger.info("connected to %s as user_id=%d (login=%s)", server, me.id, me.login)

    project = api.project.get_info_by_id(project_id)
    if project is None:
        msg = "project %d not found" % project_id
        raise ValueError(msg)
    logger.info("source project: id=%d name=%r", project.id, project.name)

    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    tag_meta = project_meta.get_tag_meta(tag_name)
    if tag_meta is None:
        msg = "project %r has no %r tag" % (project.name, tag_name)
        raise ValueError(msg)
    tag_id = tag_meta.sly_id

    total_saved = 0
    for ds in api.dataset.get_list(project_id):
        if subsets is not None and ds.name not in subsets:
            continue

        images = api.image.get_list(ds.id)
        tagged = [img for img in images if any(t.get("tagId") == tag_id for t in (img.tags or []))]
        logger.info("dataset %s: %d/%d %s frames", ds.name, len(tagged), len(images), tag_name)
        if not tagged:
            continue

        img_dir = output_dir / ds.name / "img"
        ann_dir = output_dir / ds.name / "ann"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        to_fetch = []
        for img in tagged:
            ann_path = ann_dir / (img.name + ".json")
            img_path = img_dir / img.name
            if not overwrite and ann_path.exists() and img_path.exists():
                continue
            to_fetch.append(img)

        if not to_fetch:
            logger.info("  all %d frames already on disk, skipping", len(tagged))
            continue

        ids = [img.id for img in to_fetch]
        anns = api.annotation.download_json_batch(ds.id, ids)
        for img, ann in zip(to_fetch, anns):
            ann_obj = ann.get("annotation", ann)
            (ann_dir / (img.name + ".json")).write_text(json.dumps(ann_obj))
            np_img = api.image.download_np(img.id)
            sly.image.write(str(img_dir / img.name), np_img)
            total_saved += 1
        logger.info("  saved %d frames to %s", len(to_fetch), output_dir / ds.name)

    logger.info("done: saved %d tagged frames under %s", total_saved, output_dir)
    return total_saved


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download tagged images from a Supervisely project to local disk.")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tag-name", type=str, default=HUMAN_TAG)
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Dataset names to download (default: all). e.g. 08 10",
    )
    parser.add_argument("--overwrite", action="store_true", help="Refetch frames even if already on disk.")
    parser.add_argument(
        "--server",
        type=str,
        default=os.environ.get("SUPERVISELY_SERVER", "https://app.supervisely.com"),
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    token = os.environ.get("SUPERVISELY_TOKEN")
    if not token:
        msg = "SUPERVISELY_TOKEN not set; source .env first"
        raise ValueError(msg)

    download_tagged_images(
        args.project_id,
        args.output_dir,
        args.server,
        token,
        args.tag_name,
        args.subsets,
        args.overwrite,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
