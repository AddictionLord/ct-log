"""Collect human-reviewed (bumaska-tagged) frames from Phase2 into a new project.

Walks every dataset in the source project (default Phase2), finds images
carrying the `bumaska` image-level tag, and copies those images + their
annotations into a new `Human-collection` project. Datasets in the target
mirror the source dataset names (one per subset), so provenance is preserved.
Datasets with no bumaska frames are still created (empty).

Auth comes from $SUPERVISELY_TOKEN (and optional $SUPERVISELY_SERVER).

Example:
    set -a && source .env && set +a
    python -m ann_pipeline.utils.build_human_collection
    python -m ann_pipeline.utils.build_human_collection --source-project-id 377328 --target-name Human-collection
"""

import argparse
import logging
import os
import sys

import supervisely as sly

SOURCE_PROJECT_ID = 377328  # Phase2 (auto + human review)
TARGET_NAME = "Human-collection"
HUMAN_TAG = "bumaska"


def build_human_collection(
    source_project_id: int,
    target_name: str,
    server: str,
    token: str,
) -> int:
    api = sly.Api(server_address=server, token=token)
    me = api.user.get_my_info()
    print("connected to %s as user_id=%d (login=%s)" % (server, me.id, me.login), file=sys.stderr)

    source = api.project.get_info_by_id(source_project_id)
    if source is None:
        msg = "source project %d not found" % source_project_id
        raise ValueError(msg)
    print("source project: id=%d name=%r" % (source.id, source.name), file=sys.stderr)

    existing = api.project.get_info_by_name(source.workspace_id, target_name)
    if existing is not None:
        msg = "target project %r already exists (id=%d); delete it first" % (target_name, existing.id)
        raise ValueError(msg)

    target = api.project.create(
        source.workspace_id, target_name, description="Human-reviewed (bumaska) frames from %s" % source.name
    )
    source_meta = sly.ProjectMeta.from_json(api.project.get_meta(source.id))
    api.project.update_meta(target.id, source_meta.to_json())
    print("created target project: id=%d name=%r" % (target.id, target.name), file=sys.stderr)

    src_tag_meta = source_meta.get_tag_meta(HUMAN_TAG)
    if src_tag_meta is None:
        msg = "source project %r has no %r tag" % (source.name, HUMAN_TAG)
        raise ValueError(msg)
    src_tag_id = src_tag_meta.sly_id

    total_copied = 0
    for src_ds in api.dataset.get_list(source.id):
        images = api.image.get_list(src_ds.id)
        bumaska_imgs = [img for img in images if any(t.get("tagId") == src_tag_id for t in (img.tags or []))]

        tgt_ds = api.dataset.create(target.id, src_ds.name)
        print("dataset %s: %d/%d bumaska frames" % (src_ds.name, len(bumaska_imgs), len(images)), file=sys.stderr)

        if not bumaska_imgs:
            continue

        src_ids = [img.id for img in bumaska_imgs]
        anns = api.annotation.download_json_batch(src_ds.id, src_ids)

        new_infos = api.image.copy_batch(tgt_ds.id, src_ids, change_name_if_conflict=False)
        new_ids = [info.id for info in new_infos]
        api.annotation.upload_jsons(new_ids, [a.get("annotation", a) for a in anns])
        total_copied += len(bumaska_imgs)

    print(
        "done: copied %d human-reviewed frames into %r (id=%d)" % (total_copied, target.name, target.id),
        file=sys.stderr,
    )
    return target.id


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-project-id",
        type=int,
        default=SOURCE_PROJECT_ID,
        help="Source Supervisely project to pull bumaska frames from. Default: %d (Phase2)." % SOURCE_PROJECT_ID,
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default=TARGET_NAME,
        help="Name of the new project to create. Default: %r." % TARGET_NAME,
    )
    parser.add_argument(
        "--server",
        type=str,
        default=os.environ.get("SUPERVISELY_SERVER", "https://app.supervisely.com"),
        help="Supervisely server address.",
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

    build_human_collection(args.source_project_id, args.target_name, args.server, token)


if __name__ == "__main__":
    main()
