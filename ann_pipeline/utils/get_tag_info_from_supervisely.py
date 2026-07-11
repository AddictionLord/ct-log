import argparse
from collections import Counter
import json
import logging
import os
import sys
from typing import Dict, List

import supervisely as sly

DEFAULT_PROJECT_ID = 377328  # Phase2 (auto + human review)


def _tag_id_to_name(api: sly.Api, project_id: int) -> Dict[int, str]:
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    return {tm.sly_id: tm.name for tm in project_meta.tag_metas}


def get_tag_info_from_supervisely(
    project_id: int,
    server: str,
    token: str,
) -> dict:
    """Get tag information from a Supervisely project.

    Args:
        project_id: The ID of the Supervisely project to get tag info from.
        server: The Supervisely server address.
        token: The API token for authentication.

    Returns:
        dict: A dictionary containing the project ID, project name, and a list of datasets with their tag information.

    Raises:
        ValueError: When the project with the given ID is not found.
    """
    api = sly.Api(server_address=server, token=token)
    me = api.user.get_my_info()
    print("connected to %s as user_id=%d (login=%s)" % (server, me.id, me.login), file=sys.stderr)

    project = api.project.get_info_by_id(project_id)
    if project is None:
        msg = "project %d not found" % project_id
        raise ValueError(msg)
    print("project: id=%d name=%r" % (project.id, project.name), file=sys.stderr)

    id_to_name = _tag_id_to_name(api, project_id)
    datasets: List[dict] = []

    for ds in api.dataset.get_list(project_id):
        images = api.image.get_list(ds.id)
        total = len(images)
        per_tag: Dict[str, dict] = {}
        for img in images:
            for tag in img.tags or []:
                name = tag.get("name") or id_to_name.get(tag.get("tagId"), str(tag.get("tagId")))
                entry = per_tag.setdefault(name, {"count": 0, "labelers": Counter(), "last_tag": None})
                entry["count"] += 1
                entry["labelers"][tag.get("labelerLogin")] += 1
                ts = tag.get("updatedAt") or tag.get("createdAt")
                if ts is not None and (entry["last_tag"] is None or ts > entry["last_tag"]):
                    entry["last_tag"] = ts

        tags_summary = {
            name: {
                "count": e["count"],
                "total": total,
                "coverage": round(e["count"] / total, 4) if total else 0.0,
                "last_tag": e["last_tag"],
                "labelers": dict(e["labelers"]),
            }
            for name, e in sorted(per_tag.items())
        }
        datasets.append({
            "dataset_id": ds.id,
            "dataset_name": ds.name,
            "images_count": total,
            "tags": tags_summary,
        })
        summary = ", ".join(
            "%s=%d/%d (%.0f%%)" % (n, t["count"], t["total"], 100 * t["coverage"]) for n, t in tags_summary.items()
        )
        print("dataset %s (id=%d): %s" % (ds.name, ds.id, summary or "no tags"), file=sys.stderr)

    return {
        "project_id": project.id,
        "project_name": project.name,
        "datasets": datasets,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the script.

    Returns:
        argparse.ArgumentParser: The argument parser for the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-id",
        type=int,
        default=DEFAULT_PROJECT_ID,
        help="ID of the Supervisely project to get tag info from. Default: %d (Phase2)." % DEFAULT_PROJECT_ID,
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to also save the tag info as JSON. Always printed to stdout.",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=os.environ.get("SUPERVISELY_SERVER", "https://app.supervisely.com"),
        help="Supervisely server address.",
    )

    return parser


def main() -> None:
    """Usage below.

    ```bash
    set -a && source .env && set +a
    python -m ann_pipeline.utils.get_tag_info_from_supervisely          # defaults to Phase2, JSON to stdout
    python -m ann_pipeline.utils.get_tag_info_from_supervisely --project-id 376641 --output-file tags.json
    ```

    Raises:
        ValueError: If the token is not set in the environment variables.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    for handler in sly.logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stderr)

    token = os.environ.get("SUPERVISELY_TOKEN")
    if not token:
        msg = "SUPERVISELY_TOKEN not set; source .env first"
        raise ValueError(msg)

    tag_info = get_tag_info_from_supervisely(args.project_id, args.server, token)

    print(json.dumps(tag_info, indent=2))

    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            json.dump(tag_info, f, indent=2)


if __name__ == "__main__":
    main()
