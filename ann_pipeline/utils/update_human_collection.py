"""Incrementally sync human-reviewed (bumaska-tagged) frames from Phase2 into Human-collection.

Unlike build_human_collection (which creates the project from scratch and errors
if it exists), this script appends only the delta: bumaska frames present in the
source (Phase2) that are not yet in the target, matched by image name. Existing
target images and annotations are never touched, overwritten, or deleted. Missing
datasets are created on demand.

Auth comes from $SUPERVISELY_TOKEN (and optional $SUPERVISELY_SERVER).

Example:
    set -a && source .env && set +a
    python -m ann_pipeline.utils.update_human_collection --dry-run
    python -m ann_pipeline.utils.update_human_collection
"""

import argparse
import logging
import os
import sys
from typing import Dict

import supervisely as sly

SOURCE_PROJECT_ID = 377328  # Phase2 (auto + human review)
TARGET_PROJECT_ID = 378193  # Human-collection
HUMAN_TAG = "bumaska"


def update_human_collection(
    source_project_id: int,
    target_project_id: int,
    server: str,
    token: str,
    dry_run: bool,
) -> int:
    api = sly.Api(server_address=server, token=token)
    me = api.user.get_my_info()
    print("connected to %s as user_id=%d (login=%s)" % (server, me.id, me.login), file=sys.stderr)

    source = api.project.get_info_by_id(source_project_id)
    if source is None:
        msg = "source project %d not found" % source_project_id
        raise ValueError(msg)
    target = api.project.get_info_by_id(target_project_id)
    if target is None:
        msg = "target project %d not found; run build_human_collection first" % target_project_id
        raise ValueError(msg)
    print("source: id=%d %r  ->  target: id=%d %r" % (source.id, source.name, target.id, target.name), file=sys.stderr)

    source_meta = sly.ProjectMeta.from_json(api.project.get_meta(source.id))
    src_tag_meta = source_meta.get_tag_meta(HUMAN_TAG)
    if src_tag_meta is None:
        msg = "source project %r has no %r tag" % (source.name, HUMAN_TAG)
        raise ValueError(msg)
    src_tag_id = src_tag_meta.sly_id

    api.project.update_meta(target.id, source_meta.to_json())

    tgt_ds_by_name: Dict[str, sly.DatasetInfo] = {d.name: d for d in api.dataset.get_list(target.id)}

    total_added = 0
    for src_ds in api.dataset.get_list(source.id):
        images = api.image.get_list(src_ds.id)
        bumaska_imgs = [img for img in images if any(t.get("tagId") == src_tag_id for t in (img.tags or []))]

        tgt_ds = tgt_ds_by_name.get(src_ds.name)
        existing_names = {img.name for img in api.image.get_list(tgt_ds.id)} if tgt_ds is not None else set()

        new_imgs = [img for img in bumaska_imgs if img.name not in existing_names]
        print(
            "dataset %-4s: %d bumaska in source, %d already in target, %d to add"
            % (src_ds.name, len(bumaska_imgs), len(existing_names), len(new_imgs)),
            file=sys.stderr,
        )

        if not new_imgs:
            continue

        if dry_run:
            total_added += len(new_imgs)
            continue

        if tgt_ds is None:
            tgt_ds = api.dataset.create(target.id, src_ds.name)
            tgt_ds_by_name[src_ds.name] = tgt_ds

        src_ids = [img.id for img in new_imgs]
        anns = api.annotation.download_json_batch(src_ds.id, src_ids)
        new_infos = api.image.copy_batch(tgt_ds.id, src_ids, change_name_if_conflict=False)
        new_ids = [info.id for info in new_infos]
        api.annotation.upload_jsons(new_ids, [a.get("annotation", a) for a in anns])
        total_added += len(new_imgs)

    verb = "would add" if dry_run else "added"
    print("done: %s %d new human-reviewed frames to %r" % (verb, total_added, target.name), file=sys.stderr)
    return total_added


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-project-id", type=int, default=SOURCE_PROJECT_ID)
    parser.add_argument("--target-project-id", type=int, default=TARGET_PROJECT_ID)
    parser.add_argument("--dry-run", action="store_true", help="Report the delta without copying anything.")
    parser.add_argument(
        "--server",
        type=str,
        default=os.environ.get("SUPERVISELY_SERVER", "https://app.supervisely.com"),
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    for handler in sly.logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stderr)

    token = os.environ.get("SUPERVISELY_TOKEN")
    if not token:
        msg = "SUPERVISELY_TOKEN not set; source .env first"
        raise ValueError(msg)

    update_human_collection(
        args.source_project_id,
        args.target_project_id,
        args.server,
        token,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
