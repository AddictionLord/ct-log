"""Propagate timber ids from annotated frames to the whole scan.

Annotated frames carry numeric class titles that are persistent board ids. The
boards barely move between adjacent slices, so ids transfer by spatial overlap:
seed at each annotated frame, then walk outward, matching each frame's detections
to the previous frame's labelled ones by IoU (falling back to centroid distance
when masks do not overlap).
"""

import argparse
import json
import os
from os.path import join
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ann_pipeline.timber.data import (
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SUBSET,
    load_gray,
    timber_instances_from_ann,
)
from ann_pipeline.timber.detectors import threshold_components_v2
from ann_pipeline.timber.eval import iou

MAX_CENTROID_DIST = 40.0


def centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


def assign_ids(
    masks: List[np.ndarray],
    prev_ids: List[str],
    prev_masks: List[np.ndarray],
    min_iou: float = 0.3,
) -> List[Optional[str]]:
    """Match masks to the previous frame's labelled masks; unmatched get None."""
    if not masks or not prev_masks:
        return [None] * len(masks)
    score = np.zeros((len(masks), len(prev_masks)), dtype=np.float64)
    for i, m in enumerate(masks):
        ci = centroid(m)
        for j, pm in enumerate(prev_masks):
            overlap = iou(m, pm)
            if overlap > 0:
                score[i, j] = overlap
            else:
                cj = centroid(pm)
                dist = float(np.hypot(ci[0] - cj[0], ci[1] - cj[1]))
                if dist < MAX_CENTROID_DIST:
                    score[i, j] = min_iou * (1.0 - dist / MAX_CENTROID_DIST)
    rows, cols = linear_sum_assignment(-score)
    out: List[Optional[str]] = [None] * len(masks)
    for r, c in zip(rows, cols):
        if score[r, c] > 0:
            out[r] = prev_ids[c]
    return out


def propagate(subset_dir: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], pd.DataFrame]:
    img_dir = join(subset_dir, "img")
    ann_dir = join(subset_dir, "ann")
    frames = sorted(os.listdir(img_dir))

    detections: Dict[str, List[np.ndarray]] = {}
    seeded: Dict[str, Dict[str, np.ndarray]] = {}
    for fname in tqdm(frames, desc="detecting"):
        gray = load_gray(join(img_dir, fname))
        detections[fname] = threshold_components_v2(gray)
        ann_path = join(ann_dir, fname + ".json")
        if os.path.exists(ann_path):
            with open(ann_path) as fh:
                instances = timber_instances_from_ann(json.load(fh))
            if instances:
                seeded[fname] = instances

    if not seeded:
        msg = "no annotated frames to seed ids from"
        raise ValueError(msg)

    labelled: Dict[str, Dict[str, np.ndarray]] = {}
    for fname, instances in seeded.items():
        matched = assign_ids(
            detections[fname],
            list(instances.keys()),
            list(instances.values()),
        )
        labelled[fname] = {
            tid: mask for tid, mask in zip(matched, detections[fname]) if tid is not None
        }

    anchors = sorted(seeded, key=frames.index)
    anchor_positions = [frames.index(a) for a in anchors]

    for position, fname in enumerate(frames):
        if fname in labelled:
            continue
        nearest = int(np.argmin([abs(position - a) for a in anchor_positions]))
        step = 1 if position > anchor_positions[nearest] else -1
        start = anchor_positions[nearest]
        for cursor in range(start + step, position + step, step):
            current = frames[cursor]
            if current in labelled:
                continue
            previous = labelled.get(frames[cursor - step], {})
            matched = assign_ids(
                detections[current], list(previous.keys()), list(previous.values())
            )
            labelled[current] = {
                tid: mask for tid, mask in zip(matched, detections[current]) if tid is not None
            }

    rows = []
    for fname in frames:
        ids = labelled.get(fname, {})
        rows.append(
            {
                "frame": fname,
                "n_det": len(detections[fname]),
                "n_labelled": len(ids),
                "ids": ",".join(sorted(ids)),
                "is_anchor": fname in seeded,
            }
        )
    return labelled, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/timber_track")
    args = parser.parse_args()

    subset_dir = join(args.project_root, args.subset)
    labelled, df = propagate(subset_dir)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tracks.csv", index=False)

    expected = df[df.is_anchor].ids.iloc[0]
    print(f"frames: {len(df)}   anchors: {int(df.is_anchor.sum())}")
    print(f"fully labelled (8 ids): {int((df.n_labelled == 8).sum())}/{len(df)}")
    print(f"id set matches anchors: {int((df.ids == expected).sum())}/{len(df)}")
    bad = df[df.ids != expected]
    if len(bad):
        print("\nframes with a different id set:")
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
